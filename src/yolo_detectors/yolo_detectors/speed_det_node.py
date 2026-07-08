import os
import warnings


warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;*.warning=false"

import rclpy
from rclpy.node import Node
from lancer_interfaces.msg import HumanPoseArray
import message_filters
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import random
import csv
import time
import numpy as np

class SpeedDetNode(Node):

    def __init__(self):
        super().__init__('speed_det_node')

        self.get_logger().info("*** Speed Throw Detection Node Launched ***")

        self.save_distance_mode = True
        self.record_mode = True
        self.annotations_mode = True

        self.bridge = CvBridge()

        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))   
        self.right_wrist_color = (241, 255, 81)
        self.left_wrist_color = (218, 110, 255)

        self.fps_camera = 30.0
        self.fx = 616.0  # Focal length in pixels (x-axis)
        self.fy = 616.0  # Focal length in pixels (y-axis)
        self.cx = 320.0  # Principal point x-coordinate (image center)      
        self.cy = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info = False  # Flag to check if camera info has been received

        self.frame_count = 1

        self.start_time = time.time()
        self.timestamp_csv = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        if self.save_distance_mode:
            self.get_logger().info("Distances save mode ON.")
            self.distance_log_dir = os.path.join(os.path.expanduser("~"), "ros2_orbbec_ws", "data", "speed_detection", "csv_distances")
            if not os.path.exists(self.distance_log_dir):
                os.makedirs(self.distance_log_dir)
                self.get_logger().info(f"Directory created: {self.distance_log_dir}")                     
            self.csv_path = os.path.join(self.distance_log_dir,                   
                                    f"distances_{self.timestamp_csv}.csv")         

            self.csv_file = open(self.csv_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['frame', 'timestamp', 'X(m)', 'Y(m)', 'Z(m)', 'object_speed_m_s', 'label'])
            self.csv_file.flush()

            self.get_logger().info(f"CSV file created : {self.csv_path}")

        else:
            self.get_logger().info("Distances save mode OFF.")

        if self.record_mode:
            self.get_logger().info("Record mode ON.")
            self.video_writer = None
            self.video_folder = "data/captures_videos"       
            if not os.path.exists(self.video_folder):
                os.makedirs(self.video_folder)
                self.get_logger().info(f"Recording directory created : {self.video_folder}")

        else:
            self.get_logger().info("Record mode OFF.")
        
        self.throw_detected = False
        self.previous_throw_detected = False
        self.throw_coordinates = None
        self.previous_object_center_meters = None
        self.previous_object_timestamp = None

        self.smoothed_object_meters = None
        self.alpha_smooth = 0.3       
        self.last_object_meters = None
        self.last_world_pos = None
        self.last_timestamp = None

        self.cooldown_duration = 8.0   
        self.last_throw_trigger_time = 0.0 
        
        self.false_frame_counter = 0      
        self.max_false_frames_allowed = 10

        self.trajectory_tracking_active = False
        self.predicted_trajectory = None          # Derniere prediction "vivante" (une seule courbe, ecrasee a chaque frame)
        self.trajectory_start_time = None
        self.trajectory_start_frame = None         # numero de frame du premier point du lancer (pour retrouver le lancer dans la video)
        self.trajectory_tracking_duration = 5.0
        self.trajectory_observed = []              # Points REELLEMENT mesures depuis le debut du lancer (t, x, y, z)
        self.trajectory_history = []                # Predictions echantillonnees (pour visualiser l'evolution, pas toutes les frames)
        self.history_sampling_interval = 0.10        # secondes entre deux predictions archivees (independant du fps)
        self.last_history_sample_time = None

        self.speed_threshold = 2.0
        self.startup_grace_period = 8.0  # s, aucun lancer ne peut etre detecte avant ce delai apres le lancement du node
        self.throw_confirm_frames = 3      # L'objet doit être rapide pendant au moins 3 frames consécutives
        self.throw_confirm_counter = 0

        # --- Garde-fous physiques : toute position/mise a jour hors de ces bornes est rejetee ---
        self.min_valid_z = 0.1        # m, en dessous : profondeur invalide/trop proche
        self.max_valid_z = 10.0        # m, au-dela : profondeur aberrante
        self.max_valid_xy = 5.0        # m, deplacement lateral/vertical plausible dans la piece
        self.min_valid_dt = 0.01       # s, en dessous : bruit temporel -> vitesse qui explose
        self.max_valid_speed = 25.0    # m/s, au-dela : vitesse non physique pour un lancer a la main

        # --- Debounce pour la detection d'atterrissage : evite un arret sur la 1ere frame ---
        self.landing_z_threshold = 0.1                  # m, objet considere "arrive" quand il est a moins de 10cm de la camera
        self.min_time_before_landing_check = 0.15       # s, temps minimum de suivi avant d'autoriser la detection d'atterrissage
        self.landing_confirm_frames = 3                 # nb de frames consecutives sous le seuil pour confirmer
        self.landing_confirm_counter = 0

        # --- Bornes physiques verticales (repere: Y positif = vers le bas, Y negatif = vers le haut) ---
        # A adapter a ta piece/setup si besoin.
        self.max_predicted_height = 2.0   # m, Y ne doit jamais monter en dessous de ca (2m au-dessus de l'origine camera)
        self.max_predicted_fall = -2.0      # m, Y ne doit jamais depasser ca vers le bas (sol/limite basse plausible)

        self.camera_tilt_deg = 0.0  # Ajuste l'angle réel de ton trépied ici
        self.theta = np.radians(self.camera_tilt_deg)

        self.sub_info = self.create_subscription(
            CameraInfo,
            '/orbbec_external/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.sub_image = message_filters.Subscriber(
            self,
            Image,
            '/orbbec_external/color/image_raw'
        )

        self.sub_depth = message_filters.Subscriber(
            self,
            Image, 
            '/orbbec_external/depth/image_raw'
        )

        self.sub_fine_tune_yolo = message_filters.Subscriber(
            self,
            Detection2DArray,
            '/yolo_detected_objects'
        )

        self.sub_yolo_pose = message_filters.Subscriber(
            self,
            HumanPoseArray,
            '/yolo_detected_poses'
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_depth, self.sub_fine_tune_yolo, self.sub_yolo_pose],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.synchronized_callback)

    def camera_info_callback(self, msg: CameraInfo):

        if not self.has_camera_info:
            self.fx = msg.k[0]  # Focal length in pixels (x-axis)
            self.fy = msg.k[4]  # Focal length in pixels (y-axis)
            self.cx = msg.k[2]  # Principal point x-coordinate (image center)      
            self.cy = msg.k[5]  # Principal point y-coordinate (image center)
            self.has_camera_info = True

            self.get_logger().info(f"Camera info received: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

            self.destroy_subscription(self.sub_info)  # Unsubscribe after receiving camera info

    def is_valid_object_position(self, position_meters):
        """
        Verifie que la position 3D (x, y, z) en metres est physiquement plausible
        avant de l'utiliser pour ancrer une trajectoire. C'est ce garde-fou qui
        manquait et qui laissait passer des positions aberrantes (ex: Y=0 par
        defaut ou Y=-120 issu d'une profondeur corrompue) directement dans les
        predictions.
        """
        if position_meters is None or None in position_meters:
            return False

        x, y, z = position_meters

        if any(v != v for v in (x, y, z)):  # test NaN (NaN != NaN)
            return False
        if any(abs(v) == float('inf') for v in (x, y, z)):
            return False

        if not (self.min_valid_z <= z <= self.max_valid_z):
            return False
        if abs(x) > self.max_valid_xy or abs(y) > self.max_valid_xy:
            return False

        return True
    
    def predict_smoothed_trajectory(self, t_since_start, dt_frame, g_const=9.81):
        """
        Calcule la trajectoire future prédictive lissée par moindres carrés (polyfit)
        à partir de la liste des points observés historiques.
        
        Retourne :
            - raw_trajectory (list ou None) : points futurs prédits [(t, x, y, z), ...]
            - vx_moy (float) : vitesse moyenne estimée sur X
            - vz_moy (float) : vitesse moyenne estimée sur Z
            - x_at_landing (float ou None) : Position X estimée au plan Z = 0.0m
            - y_at_landing (float ou None) : Hauteur Y estimée au plan Z = 0.0m
            - time_to_landing (float ou None) : Temps restant (en s) avant d'atteindre la caméra (Z = 0.0m)
            """
        if len(self.trajectory_observed) < 3:
            return None, 0.0, 0.0, 0.0, 0.0, 0.0

        ts = np.array([p[0] for p in self.trajectory_observed])
        xs = np.array([p[1] for p in self.trajectory_observed])
        ys = np.array([p[2] for p in self.trajectory_observed])
        zs = np.array([p[3] for p in self.trajectory_observed])

        # Régression linéaire (degré 1) pour X et Z (vitesse constante uniforme)
        poly_x = np.polyfit(ts, xs, 1) 
        poly_z = np.polyfit(ts, zs, 1)

        vx_moy = poly_x[0]
        vz_moy = poly_z[0]

        # Si le vecteur Z global pointe dans le mauvais sens, prédiction invalide
        if vz_moy >= -0.05:
            return None, vx_moy, vz_moy, 0.0, 0.0, 0.0

        # Ajustement gravitationnel de Y (repère Y positif vers le haut)
        ys_adjusted = ys + 0.5 * g_const * (ts ** 2)
        poly_y = np.polyfit(ts, ys_adjusted, 1)

        # ==========================================
        #    CALCUL ANALYTIQUE DE L'ATTERRISSAGE (REPÈRE WORLD)
        # ==========================================
        # Z_world(t) = poly_z[0] * t + poly_z[1]
        # On cherche t_landing tel que Z_world(t_landing) = 0.0
        target_z_world = 0.0
        t_landing = (target_z_world - poly_z[1]) / vz_moy
        
        # Le temps restant avant de franchir la ligne Z_world = 0
        time_to_landing = t_landing - t_since_start
        
        # Calcul de la hauteur Y_world exacte à cet instant précis
        x_at_landing = poly_x[0] * t_landing + poly_x[1]
        y_at_landing = -0.5 * g_const * (t_landing ** 2) + poly_y[0] * t_landing + poly_y[1]
        # ==========================================

        raw_trajectory = []
        t_future = t_since_start
        max_time = t_since_start + 2.0  # Prédiction jusqu'à 2 secondes dans le futur

        while t_future <= max_time:
            x_pred = poly_x[0] * t_future + poly_x[1]
            y_pred = -0.5 * g_const * (t_future ** 2) + poly_y[0] * t_future + poly_y[1]
            z_pred = poly_z[0] * t_future + poly_z[1]

            # Arrêt si la courbe sort des limites verticales physiques
            if y_pred > self.max_predicted_height or y_pred < self.max_predicted_fall:
                raw_trajectory.append((t_future, x_pred, y_pred, z_pred))
                break

            raw_trajectory.append((t_future, x_pred, y_pred, z_pred))

            # Arrêt si l'objet atteint virtuellement l'objectif caméra
            if z_pred <= self.landing_z_threshold and t_future > 0:
                break

            t_future += dt_frame

        # Vérification du Cliff Effect (collision prématurée du sol au loin)
        if raw_trajectory:
            last_point = raw_trajectory[-1]
            last_y = last_point[2]
            last_z = last_point[3]

            if last_y <= self.max_predicted_fall and last_z > (self.landing_z_threshold + 0.10):
                self.get_logger().warn(
                    f"Trajectory prediction rejected (cliff effect): object hit the floor at Z={last_z:.2f}m "
                    f"instead of reaching the camera | estimated vz: {vz_moy:.3f} m/s"
                )
                return None, vx_moy, vz_moy, 0.0, 0.0, 0.0

        return raw_trajectory, vx_moy, vz_moy, x_at_landing, y_at_landing, time_to_landing

    def plot_trajectory_history(self):
        if not self.trajectory_observed and not self.trajectory_history:
            self.get_logger().warn("No trajectory data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.invert_xaxis()
        ax2.invert_xaxis()

        frame_label = f"frame {self.trajectory_start_frame}" if self.trajectory_start_frame is not None else "frame ?"

        # 1) Trajectoire REELLEMENT observee depuis le premier point du lancer
        #    -> une seule courbe continue, c'est la "verite terrain"
        if self.trajectory_observed:
            xs_obs = [point[1] for point in self.trajectory_observed]
            ys_obs = [point[2] for point in self.trajectory_observed]
            zs_obs = [point[3] for point in self.trajectory_observed]
            ax1.plot(zs_obs, ys_obs, color='black',
                      marker='o', markersize=3, label="Observed points", zorder=5)
            ax2.plot(zs_obs, xs_obs, color='black',
                     marker='o', markersize=3, label="Observed points", zorder=5)

        # 2) Predictions echantillonnees (une toutes les N frames, pas toutes)
        #    -> montre l'evolution de la prediction sans que tout se superpose
        for i, traj in enumerate(self.trajectory_history):
            xs = [point[1] for point in traj]
            ys = [point[2] for point in traj]
            zs = [point[3] for point in traj]
            ax1.plot(zs, ys, alpha=0.35, linestyle='--', color='tab:blue',
                      label="Predictions" if i == 0 else None)
            ax2.plot(zs, xs, alpha=0.35, linestyle='--', color='tab:blue',
                      label="Predictions" if i == 0 else None)

        # 3) Derniere prediction "vivante" mise en avant
        if self.predicted_trajectory:
            xs_last = [point[1] for point in self.predicted_trajectory]
            ys_last = [point[2] for point in self.predicted_trajectory]
            zs_last = [point[3] for point in self.predicted_trajectory]
            ax1.plot(zs_last, ys_last, color='tab:red', linewidth=1.5,
                      label="Last prediction", zorder=4)
            ax2.plot(zs_last, xs_last, color='tab:red', linewidth=1.5,
                      label="Last prediction", zorder=4)

        if self.trajectory_observed:
            ax1.annotate(f"debut : {frame_label}",
                          xy=(zs_obs[0], ys_obs[0]),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=9, color='black',
                          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8))

        ax1.set_xlabel("Z (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title(f"Profil vertical ({frame_label})")
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True)
        ax1.set_ylim(self.max_predicted_fall, self.max_predicted_height)

        ax2.set_xlabel("Z (m)")
        ax2.set_ylabel("X (m)")
        ax2.set_title(f"Profil horizontal ({frame_label})")
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True)
        ax2.set_ylim(-1.5, 1.5)

        plt.tight_layout()

        plot_dir = os.path.join(os.path.expanduser("~"), "ros2_orbbec_ws", "data", "speed_detection", "trajectory_plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        frame_tag = f"frame{self.trajectory_start_frame}" if self.trajectory_start_frame is not None else "frameUNK"
        plot_path = os.path.join(plot_dir, f"trajectories_{self.timestamp_csv}_{frame_tag}.png")
        plt.savefig(plot_path)
        plt.close()

        self.get_logger().info(f"\nTrajectory plot saved : {plot_path}")

        self.trajectory_observed = []
        self.trajectory_history = []
        self.predicted_trajectory = None
        self.last_history_sample_time = None

    def synchronized_callback(self, color_msg, depth_msg, yolo_objects, yolo_poses):
        try: 
            annotated_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            if annotated_image is not None:
                h_color, w_color = annotated_image.shape[:2]
            if cv_depth_image is not None:
                h, w = cv_depth_image.shape[:2]

            object_center_pixels = None
            lw_pixels = None
            rw_pixels = None
            object_center_meters = None
            rx, ry, rz = None, None, None

            for detection in yolo_objects.detections:
                x_center = detection.bbox.center.position.x
                y_center = detection.bbox.center.position.y
                size_x = detection.bbox.size_x
                size_y = detection.bbox.size_y
                object_center_pixels = (int(x_center), int(y_center))

                x1 = int(x_center - size_x / 2)
                y1 = int(y_center - size_y / 2)
                x2 = int(x_center + size_x / 2)
                y2 = int(y_center + size_y / 2)

                if self.annotations_mode:
                    #cv2.rectangle(annotated_image, (x1, y1), (x2, y2), self.box_color, 2)
                    cv2.circle(annotated_image, (int(x_center), int(y_center)), 8, self.box_color, -1)

                if len(detection.results) > 0:
                    result = detection.results[0]
                    
                    if result.pose.pose.position.z >= 0.1:
                        object_center_meters = (result.pose.pose.position.x, 
                                                result.pose.pose.position.y, 
                                                result.pose.pose.position.z)
                    else:
                        object_center_meters = None

                for result in detection.results:
                    label = result.hypothesis.class_id
                    conf = result.hypothesis.score * 100
                    
                    if self.annotations_mode:
                        custom_label = f"{label} ({conf:.1f}%) | Z: {object_center_meters[2]:.3f}m" if object_center_meters is not None else f"{label} ({conf:.1f}%)"
                        cv2.putText(annotated_image, custom_label, (x1, y1 - 10),                                 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.box_color, 2)                     
                
            for pose in yolo_poses.poses:
                kpts = pose.keypoints
                if len(kpts) > 9 and kpts[9].confidence > 0.7 and cv_depth_image is not None:
                    lw_x = int(kpts[9].x)
                    lw_y = int(kpts[9].y)
                    lw_x = max(0, min(lw_x, w - 1))
                    lw_y = max(0, min(lw_y, h - 1))
                    lw_pixels = (lw_x, lw_y)

                    if self.annotations_mode:
                        cv2.circle(annotated_image, (lw_x, lw_y), 8, self.left_wrist_color, -1)
                        cv2.putText(annotated_image, "LEFT WRIST", (lw_x + 10, lw_y + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.left_wrist_color, 3)

                if len(kpts) > 10 and kpts[10].confidence > 0.7 and cv_depth_image is not None:
                    rw_x = int(kpts[10].x)
                    rw_y = int(kpts[10].y)
                    rw_x = max(0, min(rw_x, w - 1))
                    rw_y = max(0, min(rw_y, h - 1))
                    rw_pixels = (rw_x, rw_y)

                    if self.annotations_mode:
                        cv2.circle(annotated_image, (rw_x, rw_y), 8, self.right_wrist_color, -1)
                        cv2.putText(annotated_image, "RIGHT WRIST", (rw_x + 10, rw_y + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.right_wrist_color, 3)

            if object_center_pixels is not None and None not in object_center_pixels:
                hold_connections = []
                
                if lw_pixels is not None and None not in lw_pixels:
                    hold_connections.append((object_center_pixels, lw_pixels))
                    
                if rw_pixels is not None and None not in rw_pixels:
                    hold_connections.append((object_center_pixels, rw_pixels))

                if self.annotations_mode:
                    for pt1, pt2 in hold_connections:
                        cv2.line(annotated_image, pt1, pt2, (0, 255, 255), 2)

            current_timestamp = time.time() - self.start_time
            object_speed_csv = 0.0
            remaining_time = self.startup_grace_period - current_timestamp

            if remaining_time > 0 and self.annotations_mode:
                cv2.putText(annotated_image, f"WARMING UP ({remaining_time:.1f}s)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

            if object_center_meters is not None and None not in object_center_meters:
                rx, ry, rz = object_center_meters
                object_speed_m_s = 0.0
                vrx, vry, vrz = 0.0, 0.0, 0.0
                dt = None
                
                if self.smoothed_object_meters is None:
                    self.smoothed_object_meters = list(object_center_meters)
                else:
                    self.smoothed_object_meters[0] = self.alpha_smooth * object_center_meters[0] + (1 - self.alpha_smooth) * self.smoothed_object_meters[0]
                    self.smoothed_object_meters[1] = self.alpha_smooth * object_center_meters[1] + (1 - self.alpha_smooth) * self.smoothed_object_meters[1]
                    self.smoothed_object_meters[2] = self.alpha_smooth * object_center_meters[2] + (1 - self.alpha_smooth) * self.smoothed_object_meters[2]

                rx_cam, ry_cam, rz_cam = self.smoothed_object_meters

                rx_world = rx_cam
                ry_world = -ry_cam * np.cos(self.theta) + rz_cam * np.sin(self.theta)
                rz_world =  ry_cam * np.sin(self.theta) + rz_cam * np.cos(self.theta)
                current_world_pos = [rx_world, ry_world, rz_world]

                if self.last_world_pos is not None and self.last_timestamp is not None:
                    dt = current_timestamp - self.last_timestamp
                    if dt >= self.min_valid_dt:
                        vrx = (current_world_pos[0] - self.last_world_pos[0]) / dt
                        vry = (current_world_pos[1] - self.last_world_pos[1]) / dt
                        vrz = (current_world_pos[2] - self.last_world_pos[2]) / dt

                        object_speed_m_s = (vrx**2 + vry**2 + vrz**2)**0.5 

                        if self.annotations_mode:
                            cv2.putText(annotated_image, f"Speed: {object_speed_m_s:.2f} m/s", (30, 100),                                        
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            
                    else:
                        self.get_logger().warn("Cannot compute speed: dt too small or negative.")
                                        
                self.last_world_pos = list(current_world_pos)
                self.last_timestamp = current_timestamp
                object_speed_csv = round(object_speed_m_s, 4) 

                if remaining_time <= 0:
                    if object_speed_m_s >= self.speed_threshold and vrz < 0:
                        self.throw_confirm_counter += 1

                        if self.throw_confirm_counter >= self.throw_confirm_frames:
                            self.false_frame_counter = 0
                            self.throw_detected = True

                            if self.annotations_mode:
                                        cv2.putText(annotated_image, f"THROW", (30, 50),                                        
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            
                    else:   # if remaining_time <= 0 but not (object_speed_m_s >= self.speed_threshold and vrz < 0):
                        self.throw_confirm_counter = 0
                        self.false_frame_counter += 1                    
                        if self.false_frame_counter >= self.max_false_frames_allowed:
                            self.throw_detected = False

                if self.throw_detected is True and self.previous_throw_detected is not True and not self.trajectory_tracking_active:
                    time_since_last_throw = current_timestamp - self.last_throw_trigger_time

                    if time_since_last_throw >= self.cooldown_duration:
                        self.last_throw_trigger_time = current_timestamp 

                        if self.is_valid_object_position(object_center_meters):
                            self.throw_coordinates = current_world_pos

                            if self.annotations_mode:
                                coord_text = f"Obj: X:{rx_world:.3f}m, Y:{ry_world:.3f}m, Z:{rz_world:.3f}m"
                                cv2.putText(annotated_image, coord_text, (30, h_color - 50),                                            
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.box_color, 2)

                            self.trajectory_tracking_active = True
                            self.trajectory_start_time = current_timestamp
                            self.trajectory_start_frame = self.frame_count 
                            self.trajectory_observed = [(0.0,) + tuple(self.throw_coordinates)] # pour inclure le premier point observé dans la trajectoire
                            #self.trajectory_observed = []  # on ancre sur le 2eme point du lancer car le premier point utilise les données brutes potentiellement bruitées
                            self.trajectory_history = []
                            self.predicted_trajectory = None
                            self.last_history_sample_time = None
                            self.landing_confirm_counter = 0

                            print(f"\n\n----- FRAME {self.trajectory_start_frame} -----\n")
                            self.get_logger().info(f"Throw started (vrz = {vrz:.3f}m/s): trajectory prediction start !\n")
                        
                        else:
                            self.throw_coordinates = None
                            self.get_logger().warn("Throw detected but the object is invisible or its position is not physically plausible (rejected).")
                                
                    else:
                        self.get_logger().info(f"Flickering detected ! New initial point blocked by cooldown ({self.cooldown_duration - time_since_last_throw:.1f}s left)")

                elif self.throw_detected is not True and self.previous_throw_detected is True:
                    self.get_logger().info("\nThrow ended.")
                    self.throw_coordinates = None

                self.previous_throw_detected = self.throw_detected

            if self.trajectory_tracking_active:
                stop_reason = None

                if not self.throw_detected:
                    stop_reason = "Trajectory tracking stopped: object lost or throw ended."

                elif (self.is_valid_object_position(object_center_meters)
                      and self.previous_object_center_meters is not None
                      and self.previous_object_timestamp is not None):   
                    
                    if object_speed_m_s > self.max_valid_speed:
                        self.get_logger().warn(
                            f"Trajectory update rejected: object speed too high = {object_speed_m_s:.1f} m/s."
                        )

                    elif vrz > 0:
                       self.get_logger().warn(f"Trajectory update rejected: vz positive = {vrz:.3f} m/s")

                    elif object_speed_m_s >= self.speed_threshold and vrz < -0.05:
                        t_since_start = current_timestamp - self.trajectory_start_time

                        if self.trajectory_observed:
                            last_recorded_t = self.trajectory_observed[-1][0]
                            time_gap = t_since_start - last_recorded_t
                            if time_gap > 0.50:  # Plus de 500ms de silence (environ 15 frames perdues)
                                stop_reason = f"Trajectory tracking aborted: black hole detected ({time_gap:.2f}s without valid updates)."
                                self.trajectory_tracking_active = False

                        if stop_reason is None:
                            self.trajectory_observed.append((
                                t_since_start, 
                                rx_world, 
                                ry_world, 
                                rz_world
                            ))

                        if len(self.trajectory_observed) >= 4 and len(self.trajectory_observed) % 2 == 0:
                            dt_frame = dt if dt is not None else (1.0 / self.fps_camera)
                            raw_trajectory, vx_moy, vz_moy, x_at_landing, y_at_landing, time_to_landing = self.predict_smoothed_trajectory(t_since_start, dt_frame)

                            if raw_trajectory is None:
                                if vz_moy >= -0.05:
                                    self.get_logger().warn(
                                        f"Trajectory prediction rejected (global Vz positive or flat): "
                                        f"Vx moy: {vx_moy:.2f}m/s | Vz moy: {vz_moy:.2f}m/s. Anomalous point removed."
                                    )
                                    self.trajectory_observed.pop()
                                self.predicted_trajectory = None
                            else:
                                self.predicted_trajectory = raw_trajectory

                                if (self.last_history_sample_time is None
                                        or (current_timestamp - self.last_history_sample_time) >= self.history_sampling_interval):
                                    self.trajectory_history.append(self.predicted_trajectory)
                                    self.last_history_sample_time = current_timestamp

                                self.get_logger().info(
                                    f"Trajectory updated ({len(self.trajectory_observed)} obs. pts) ---> "
                                    f"Vx moy: {vx_moy:.2f}m/s | Vz moy: {vz_moy:.2f}m/s"
                                )
                                self.get_logger().info(
                                    f"[PRED] X_landing: {x_at_landing:.2f}m | Y_landing: {y_at_landing:.2f}m | Impact in: {time_to_landing:.3f}s"
                                )   

                        time_tracked = current_timestamp - self.trajectory_start_time

                        if rz < self.landing_z_threshold and time_tracked >= self.min_time_before_landing_check:
                            self.landing_confirm_counter += 1
                        else:
                            self.landing_confirm_counter = 0

                        if self.landing_confirm_counter >= self.landing_confirm_frames:
                            stop_reason = f"Object has reached the camera (Z={rz:.3f}m < {self.landing_z_threshold}m, confirmed)."
                    
                    else:
                        self.get_logger().warn("Cannot update trajectory : vz too small")

                if stop_reason is None and self.false_frame_counter >= self.max_false_frames_allowed:
                    stop_reason = "Trajectory tracking aborted: object stopped moving or lost."

                elapsed = current_timestamp - self.trajectory_start_time
                if stop_reason is None and elapsed >= self.trajectory_tracking_duration:
                    stop_reason = f"Trajectory tracking stopped after timeout ({elapsed:.2f}s)."

                if stop_reason is not None:
                    self.trajectory_tracking_active = False
                    print()
                    self.get_logger().info(stop_reason)
                    self.plot_trajectory_history()

            if self.is_valid_object_position(object_center_meters):
                self.previous_object_center_meters = object_center_meters
                self.previous_object_timestamp = current_timestamp
                    
            cv2.putText(annotated_image, f"Frame {self.frame_count}", (1150, 50),                                            
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.last_throw_trigger_time > 0.0:
                time_since_last_throw = current_timestamp - self.last_throw_trigger_time
                remaining_cooldown = self.cooldown_duration - time_since_last_throw

                if remaining_cooldown > 0 and self.annotations_mode:
                    text_cooldown = f"CD: {remaining_cooldown:.1f}s"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2.2 
                    thickness = 6
                    color_orange = (0, 140, 255) 
                    
                    text_size, _ = cv2.getTextSize(text_cooldown, font, font_scale, thickness)
                    text_w, text_h = text_size
                    
                    pos_x = w_color - text_w - 40
                    pos_y = h_color - 40
                    
                    cv2.putText(annotated_image, text_cooldown, (pos_x, pos_y), 
                                font, font_scale, color_orange, thickness)
                    
            if self.save_distance_mode:        
                x_csv = round(rx_world, 4) if rx is not None else ""
                y_csv = round(ry_world, 4) if ry is not None else ""
                z_csv = round(rz_world, 4) if rz is not None else ""
                self.csv_writer.writerow([self.frame_count, round(current_timestamp, 4), x_csv, y_csv, z_csv, object_speed_csv, self.throw_detected])
                if self.frame_count % 30 == 0: 
                    self.csv_file.flush()

            if self.record_mode:
                if self.video_writer is None:
                    self.record_path = os.path.join(self.video_folder,                   
                                f"{self.timestamp_csv}_speed_det.avi")                                     
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')                    
                    self.video_writer = cv2.VideoWriter(self.record_path, fourcc, self.fps_camera, (w_color, h_color))
                    self.get_logger().info(f"Recording started.")
                
                if self.video_writer is not None:
                    self.video_writer.write(annotated_image)
            

            self.frame_count += 1

            #cv2.putText(annotated_image, "Press ECHAP to quit", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Throw Detection With Object Speed", annotated_image)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("\n\n-----")
                self.get_logger().info("ECHAP pressed. Shutting down the node...")
                raise KeyboardInterrupt
    
        except KeyboardInterrupt:
            raise

        except Exception as e:
            self.get_logger().info(f"Error in the synchronized callback : {e}")

    def destroy_node(self):
            
            if getattr(self, 'trajectory_tracking_active', False) and self.trajectory_observed:           
                self.get_logger().info("\nNode shutting down with an active throw: saving partial trajectory plot...")
                self.trajectory_tracking_active = False
                self.plot_trajectory_history()

            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
                self.get_logger().info(f"\nRecord video saved : {self.record_path}")

            if hasattr(self, 'csv_file') and self.csv_file is not None and not self.csv_file.closed:
                self.csv_file.close()
                self.get_logger().info("\nCSV file flushed and closed correctly.")
            cv2.destroyAllWindows()
            return super().destroy_node()

def main(args=None):
    
    rclpy.init(args=args)
    node = SpeedDetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()