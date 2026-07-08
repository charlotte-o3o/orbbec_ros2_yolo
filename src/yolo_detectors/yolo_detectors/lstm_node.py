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
from datetime import datetime
import torch
import torch.nn as nn

class ThrowLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=3):
        super(ThrowLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LSTMNode(Node):

    def __init__(self):
        super().__init__('lstm_node')

        self.get_logger().info("*** LSTM Throw Detection Node Launched ***")

        self.save_distance_mode = True
        self.record_mode = True
        self.annotations_mode = True

        self.bridge = CvBridge()

        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))   
        self.line_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.circle_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
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
            self.distance_log_dir = os.path.join(os.path.expanduser("~"), "ros2_orbbec_ws", "data", "lstm", "csv_distances")
            if not os.path.exists(self.distance_log_dir):
                os.makedirs(self.distance_log_dir)
                self.get_logger().info(f"Directory created: {self.distance_log_dir}")                     
            self.csv_path = os.path.join(self.distance_log_dir,                   
                                    f"distances_{self.timestamp_csv}.csv")         

            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'timestamp', 'object_speed_m_s', 'distance_left_wrist_m', 'distance_right_wrist_m', 'label'])

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
        
        # Liste des connexions du squelette (identique à yolo_pose_node)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),           
            (3, 5), (4, 6),                           
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  
            (5, 11), (6, 12), (11, 12),               
            (11, 13), (13, 15), (12, 14), (14, 16)    
        ]

        self.sequence_length = 20
        self.lstm_buffer = []
        self.last_features = None
        self.throw_detected = False
        self.previous_throw_detected = False
        self.throw_coordinates = None
        self.previous_object_center_meters = None
        self.previous_object_timestamp = None

        self.smoothed_object_meters = None
        self.alpha_smooth = 0.6            
        self.last_object_meters = None
        self.last_timestamp = None

        self.cooldown_duration = 3.0   
        self.last_throw_trigger_time = 0.0 
        
        self.false_frame_counter = 0      
        self.max_false_frames_allowed = 10

        self.trajectory_tracking_active = False
        self.predicted_trajectory = None          # Derniere prediction "vivante" (une seule courbe, ecrasee a chaque frame)
        self.trajectory_start_time = None
        self.trajectory_start_frame = None         # numero de frame du premier point du lancer (pour retrouver le lancer dans la video)
        self.trajectory_tracking_duration = 3.0
        self.trajectory_observed = []              # Points REELLEMENT mesures depuis le debut du lancer (t, x, y, z)
        self.trajectory_history = []                # Predictions echantillonnees (pour visualiser l'evolution, pas toutes les frames)
        self.history_sampling_interval = 0.10        # secondes entre deux predictions archivees (independant du fps)
        self.last_history_sample_time = None

        # --- Garde-fous physiques : toute position/mise a jour hors de ces bornes est rejetee ---
        self.min_valid_z = 0.05        # m, en dessous : profondeur invalide/trop proche
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
        self.max_predicted_height = -2.0   # m, Y ne doit jamais monter en dessous de ca (2m au-dessus de l'origine camera)
        self.max_predicted_fall = 3.0      # m, Y ne doit jamais depasser ca vers le bas (sol/limite basse plausible)

        # Détecter si on a une carte graphique NVIDIA, sinon utiliser le processeur
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Instanciation de l'architecture (attention, input_size=3 comme pour l'entraînement)
        self.model = ThrowLSTM(input_size=6, num_classes=2).to(self.device)
        
        # Définis ici le chemin exact vers ton fichier throw_lstm.pth
        model_path = os.path.join(os.path.expanduser("~"), "ros2_orbbec_ws", "weights", "throw_lstm_v9.pth")
        
        try:
            # Chargement des poids
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            # CRUCIAL : Passer le modèle en mode "évaluation" (désactive l'apprentissage)
            self.model.eval() 
            self.get_logger().info(f"LSTM model loaded successfully on {self.device} !")
        except Exception as e:
            self.get_logger().error(f"Failed to load LSTM model : {e}")
            raise e

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

    def predict_parabolic_trajectory(self, x0, y0, z0, vx0, vy0, vz0,
                                  g=9.81, dt=None, z_camera=0.3):
        """
        Calcule les points de la trajectoire parabolique de l'objet
        à partir de sa position et vitesse initiales (repère caméra, en mètres).

        Retourne une liste de tuples (t, x, y, z) jusqu'à ce que
        l'objet atteigne la caméra.
        """
        trajectory = []
        t = 0.0
        max_time = 5.0 
        if dt is None:
            dt = 1.0 / self.fps_camera

        while t <= max_time:
            x = vx0 * t + x0
            y = 0.5 * g * (t ** 2) + vy0 * t + y0
            z = vz0 * t + z0

            # Garde-fou physique : on arrete l'integration des qu'elle sort d'une
            # plage plausible plutot que de laisser le carre en t^2 diverger vers
            # des dizaines de metres a cause d'une vitesse initiale bruitee.
            if y < self.max_predicted_height or y > self.max_predicted_fall:
                trajectory.append((t, x, y, z))
                break

            trajectory.append((t, x, y, z))

            if z <= z_camera and t > 0:
                break

            t += dt

        return trajectory
    
    def plot_trajectory_history(self):
        if not self.trajectory_observed and not self.trajectory_history:
            self.get_logger().warn("No trajectory data to plot.")
            return

        plt.figure(figsize=(8, 6))

        # 1) Trajectoire REELLEMENT observee depuis le premier point du lancer
        #    -> une seule courbe continue, c'est la "verite terrain"
        if self.trajectory_observed:
            xs_obs = [point[1] for point in self.trajectory_observed]
            ys_obs = [-point[2] for point in self.trajectory_observed]
            zs_obs = [-point[3] for point in self.trajectory_observed]
            plt.plot(zs_obs, ys_obs, color='black', linewidth=2.5,
                      marker='o', markersize=3, label="Trajectoire observee", zorder=5)

        # 2) Predictions echantillonnees (une toutes les N frames, pas toutes)
        #    -> montre l'evolution de la prediction sans que tout se superpose
        for i, traj in enumerate(self.trajectory_history):
            xs = [point[1] for point in traj]
            ys = [-point[2] for point in traj]
            zs = [-point[3] for point in traj]
            plt.plot(zs, ys, alpha=0.35, linestyle='--', color='tab:blue',
                      label="Predictions (echantillonnees)" if i == 0 else None)

        # 3) Derniere prediction "vivante" mise en avant
        if self.predicted_trajectory:
            xs_last = [point[1] for point in self.predicted_trajectory]
            ys_last = [-point[2] for point in self.predicted_trajectory]
            zs_last = [-point[3] for point in self.predicted_trajectory]
            plt.plot(zs_last, ys_last, color='tab:red', linewidth=1.5,
                      label="Derniere prediction", zorder=4)

        # Annotation du numero de frame du premier point du lancer, pour retrouver
        # facilement a quel lancer de la video ce graphique correspond.
        frame_label = f"frame {self.trajectory_start_frame}" if self.trajectory_start_frame is not None else "frame ?"
        if self.trajectory_observed:
            plt.annotate(f"debut : {frame_label}",
                          xy=(zs_obs[0], ys_obs[0]),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=9, color='black',
                          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8))

        plt.xlabel("Z (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Trajectoire observee vs predictions ({frame_label})")
        plt.legend(loc='best', fontsize=8)
        plt.grid(True)

        # Echelle verticale fixe, alignee sur les memes bornes physiques que
        # celles utilisees pour couper l'integration (avec le signe inverse
        # car ys/zs sont affiches negatifs par rapport a y/z bruts).
        plt.ylim(-self.max_predicted_fall, -self.max_predicted_height)

        plot_dir = os.path.join(os.path.expanduser("~"), "ros2_orbbec_ws", "data", "lstm", "trajectory_plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # NB: time.strftime ne supporte pas %f (microsecondes) contrairement a
        # datetime.strftime -> on utilise datetime pour avoir un nom de fichier
        # correct (auparavant ca produisait litteralement "..._%f.png").
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        frame_tag = f"frame{self.trajectory_start_frame}" if self.trajectory_start_frame is not None else "frameUNK"
        plot_path = os.path.join(plot_dir, f"trajectories_{timestamp_str}_{frame_tag}.png")
        plt.savefig(plot_path)
        plt.close()

        self.get_logger().info(f"Trajectory plot saved : {plot_path}")

        # On vide l'historique juste apres la sauvegarde pour ne jamais melanger
        # les donnees d'un lancer avec celles du suivant (et eviter un double-plot
        # avec les memes donnees si deux conditions d'arret se declenchent d'affilee).
        self.trajectory_observed = []
        self.trajectory_history = []
        self.predicted_trajectory = None
        self.last_history_sample_time = None


    def synchronized_callback(self, color_msg, depth_msg, yolo_objects, yolo_poses):
        # Process the synchronized messages here
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
            lw_x_m, lw_y_m, lw_z_m = None, None, None
            rw_x_m, rw_y_m, rw_z_m = None, None, None
            object_center_meters = None
            lw_meters = None
            rw_meters = None

            for detection in yolo_objects.detections:
                # Récupération des dimensions
                x_center = detection.bbox.center.position.x
                y_center = detection.bbox.center.position.y
                size_x = detection.bbox.size_x
                size_y = detection.bbox.size_y
                object_center_pixels = (int(x_center), int(y_center))

                # Calcul des coins supérieur gauche et inférieur droit
                x1 = int(x_center - size_x / 2)
                y1 = int(y_center - size_y / 2)
                x2 = int(x_center + size_x / 2)
                y2 = int(y_center + size_y / 2)

                if self.annotations_mode:
                    #cv2.rectangle(annotated_image, (x1, y1), (x2, y2), self.box_color, 2)
                    cv2.circle(annotated_image, (int(x_center), int(y_center)), 8, self.box_color, -1)

                if len(detection.results) > 0:
                    # On récupère le premier résultat (l'hypothèse principale de YOLO)
                    result = detection.results[0]
                    
                    # Lecture des coordonnées 3D en mètres calculées par fine_tune_yolo_node
                    object_center_meters = (result.pose.pose.position.x, 
                                           result.pose.pose.position.y, 
                                           result.pose.pose.position.z)

                for result in detection.results:
                    label = result.hypothesis.class_id
                    conf = result.hypothesis.score * 100
                    z_dist = result.pose.pose.position.z
                    
                    if self.annotations_mode:
                        custom_label = f"{label} ({conf:.1f}%) | Z: {z_dist:.3f}m"
                        cv2.putText(annotated_image, custom_label, (x1, y1 - 10),                                 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.box_color, 2)
                        
            for pose in yolo_poses.poses:
                kpts = pose.keypoints

                # Vérification de la validité de l'index 9 (Poignet droit)
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

                    lw_z_m = cv_depth_image[lw_y, lw_x] / 1000.0

                    if lw_z_m > 0:
                        lw_x_m = ((lw_x - self.cx) * lw_z_m) / self.fx
                        lw_y_m = ((lw_y - self.cy) * lw_z_m) / self.fy
                    else:
                        lw_x_m, lw_y_m = None, None

                    lw_meters = (lw_x_m, lw_y_m, lw_z_m)

                # Vérification de la validité de l'index 10 (Poignet gauche)
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

                    rw_z_m = cv_depth_image[rw_y, rw_x] / 1000.0

                    if rw_z_m > 0:
                        rw_x_m = ((rw_x - self.cx) * rw_z_m) / self.fx
                        rw_y_m = ((rw_y - self.cy) * rw_z_m) / self.fy
                    else:
                        rw_x_m, rw_y_m = None, None

                    rw_meters = (rw_x_m, rw_y_m, rw_z_m)

                """                
                # Dessiner les lignes du squelette
                for pt1_idx, pt2_idx in self.skeleton_connections:
                    if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                        kp1 = kpts[pt1_idx]
                        kp2 = kpts[pt2_idx]
                        
                        # On vérifie la confiance (seuil à 0.5)
                        if kp1.confidence > 0.5 and kp2.confidence > 0.5:
                            start_point = (int(kp1.x), int(kp1.y))
                            end_point = (int(kp2.x), int(kp2.y))
                            cv2.line(annotated_image, start_point, end_point, self.line_color, 2)

                # Dessiner les points des articulations
                for kp in kpts:
                    if kp.confidence > 0.5:
                        cv2.circle(annotated_image, (int(kp.x), int(kp.y)), 4, self.circle_color, -1)
                
                if pose.position_centre_3d.z > 0:
                    z_text = f"Human Z: {pose.position_centre_3d.z:.2f}m"
                    if len(kpts) > 0 and kpts[0].confidence > 0.5:
                        cv2.putText(annotated_image, z_text, (int(kpts[0].x), int(kpts[0].y) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)             
                """

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
            left_dist_csv = ""
            right_dist_csv = ""

            current_left_dist = None
            current_right_dist = None

            if object_center_meters is not None and None not in object_center_meters:
                
                if self.smoothed_object_meters is None:
                    self.smoothed_object_meters = list(object_center_meters)
                else:
                    self.smoothed_object_meters[0] = self.alpha_smooth * object_center_meters[0] + (1 - self.alpha_smooth) * self.smoothed_object_meters[0]
                    self.smoothed_object_meters[1] = self.alpha_smooth * object_center_meters[1] + (1 - self.alpha_smooth) * self.smoothed_object_meters[1]
                    self.smoothed_object_meters[2] = self.alpha_smooth * object_center_meters[2] + (1 - self.alpha_smooth) * self.smoothed_object_meters[2]


                object_speed_m_s = 0.0
                if self.last_object_meters is not None and self.last_timestamp is not None:
                    dt = current_timestamp - self.last_timestamp
                    if dt > 0:
                        dx = self.smoothed_object_meters[0] - self.last_object_meters[0]
                        dy = self.smoothed_object_meters[1] - self.last_object_meters[1]
                        dz = self.smoothed_object_meters[2] - self.last_object_meters[2]
                        # Distance parcourue en 3D divisée par le temps écoulé
                        object_speed_m_s = (dx**2 + dy**2 + dz**2)**0.5 / dt
                
                # Sauvegarde pour la frame suivante
                self.last_object_meters = list(self.smoothed_object_meters)
                self.last_timestamp = current_timestamp
                object_speed_csv = round(object_speed_m_s, 4)
            

                if lw_meters is not None and None not in lw_meters:
                    left_dist_m = ((object_center_meters[0] - lw_meters[0]) ** 2 + 
                                   (object_center_meters[1] - lw_meters[1]) ** 2 + 
                                   (object_center_meters[2] - lw_meters[2]) ** 2) ** 0.5
                    left_dist_csv = round(left_dist_m, 4)
                    current_left_dist = left_dist_m

                    left_line_center = ((object_center_pixels[0] + lw_pixels[0]) / 2,
                                        (object_center_pixels[1] + lw_pixels[1]) / 2)
                    
                    if self.annotations_mode:
                        cv2.circle(annotated_image, (int(left_line_center[0]), int(left_line_center[1])), 5, (0, 255, 255), -1)
                        cv2.putText(annotated_image, f"{left_dist_m:.3f} m", (int(left_line_center[0]) + 1, int(left_line_center[1]) + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if rw_meters is not None and None not in rw_meters:
                    right_dist_m = ((object_center_meters[0] - rw_meters[0]) ** 2 + 
                                    (object_center_meters[1] - rw_meters[1]) ** 2 + 
                                    (object_center_meters[2] - rw_meters[2]) ** 2) ** 0.5
                    right_dist_csv = round(right_dist_m, 4)
                    current_right_dist = right_dist_m

                    right_line_center = ((object_center_pixels[0] + rw_pixels[0]) / 2,
                                         (object_center_pixels[1] + rw_pixels[1]) / 2)
                    
                    if self.annotations_mode:
                        cv2.circle(annotated_image, (int(right_line_center[0]), int(right_line_center[1])), 5, (0, 255, 255), -1)
                        cv2.putText(annotated_image, f"{right_dist_m:.3f} m", (int(right_line_center[0]) + 1, int(right_line_center[1]) + 1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        

                    
            #########################################################
            #                                                       #
            #                         LSTM                          #
            #                                                       #
            #########################################################   


                   
            if current_left_dist is not None and current_right_dist is not None:
                    
                    # 1. Position actuelle (3 features)
                    current_features = [object_speed_m_s, current_left_dist, current_right_dist]
                    
                    # 2. Calcul des deltas (Vitesse)
                    if self.last_features is None:
                        # Première frame du nœud : pas d'historique, delta = 0
                        delta_features = [0.0, 0.0, 0.0]
                    else:
                        # Différence avec la frame précédente
                        delta_features = [
                            current_features[0] - self.last_features[0],
                            current_features[1] - self.last_features[1],
                            current_features[2] - self.last_features[2]
                        ]
                    
                    # On sauvegarde la position actuelle pour la frame suivante
                    self.last_features = current_features
                    
                    # Concaténation : On fusionne positions + vitesses -> 6 features
                    full_6_features = current_features + delta_features
                    
                    # 3. Gestion du buffer
                    self.lstm_buffer.append(full_6_features)
                    if len(self.lstm_buffer) > self.sequence_length:
                        self.lstm_buffer.pop(0)

                    # 4. Inférence
                    if len(self.lstm_buffer) == self.sequence_length:
                        input_tensor = torch.tensor([self.lstm_buffer], dtype=torch.float32).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence, predicted_class = torch.max(probabilities, 1)
                            action_id = predicted_class.item()
                            conf_score = confidence.item()
                        
                        if action_id == 1 and conf_score > 0.85:
                            self.false_frame_counter = 0  # On réinitialise, tout va bien
                            self.throw_detected = True    # Le lancer est actif

                            if self.annotations_mode:
                                cv2.putText(annotated_image, f"THROW ({conf_score*100:.0f}%)", (30, 50),                                        
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            
                        else:
                            # 🟢 ANTI-REBOND : L'IA ne voit plus de lancer, mais on attend avant de paniquer
                            self.false_frame_counter += 1
                            
                            # Seulement si on dépasse le seuil (ex: 10 frames), on officialise l'arrêt
                            if self.false_frame_counter >= self.max_false_frames_allowed:
                                self.throw_detected = False

                        # 🆕 ÉTAPE B : Gestion des flancs avec sécurité Cooldown
                        if self.throw_detected is True and self.previous_throw_detected is not True:
                            
                            # ⏱️ Calcul du temps écoulé depuis le TOUT PREMIER déclenchement du lancer précédent
                            time_since_last_throw = current_timestamp - self.last_throw_trigger_time
                            
                            if time_since_last_throw >= self.cooldown_duration:
                                # C'est un VRAI nouveau lancer (le cooldown est expiré)
                                self.last_throw_trigger_time = current_timestamp  # On verrouille le chrono
                                
                                self.get_logger().info(f"Throw started ({conf_score*100:.0f}%) : trajectory prediction start !")

                                if self.is_valid_object_position(object_center_meters):
                                    self.throw_coordinates = object_center_meters
                                    rx, ry, rz = self.throw_coordinates

                                    if self.annotations_mode:
                                        coord_text = f"Obj: X:{rx:.3f}m, Y:{ry:.3f}m, Z:{rz:.3f}m"
                                        cv2.putText(annotated_image, coord_text, (30, h_color - 50),                                            
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.box_color, 2)
                                    
                                    self.get_logger().info(f"Initial throw point (frame {self.frame_count}) ---> X: {rx:.3f}m, Y: {ry:.3f}m, Z: {rz:.3f}m")

                                    self.trajectory_tracking_active = True
                                    self.trajectory_start_time = current_timestamp
                                    self.trajectory_start_frame = self.frame_count  # pour retrouver ce lancer dans la video/CSV
                                    self.trajectory_observed = [(0.0,) + self.throw_coordinates]  # on ancre sur le VRAI premier point du lancer
                                    self.trajectory_history = []
                                    self.predicted_trajectory = None
                                    self.last_history_sample_time = None
                                    self.landing_confirm_counter = 0

                                else:
                                    self.throw_coordinates = None
                                    self.get_logger().warn("Throw detected but the object is invisible or its position is not physically plausible (rejected).")
                                    
                            else:
                                self.get_logger().info(f"Flickering detected ! New initial point blocked by cooldown ({self.cooldown_duration - time_since_last_throw:.1f}s left)")
                                self.throw_detected = False

                        elif self.throw_detected is not True and self.previous_throw_detected is True:
                            self.get_logger().info("Throw ended.")
                            self.throw_coordinates = None

                        self.previous_throw_detected = self.throw_detected

                        if self.trajectory_tracking_active:
                            stop_reason = None  # on ne s'arrete/plot qu'UNE fois par frame, quelle que soit la raison

                            if (self.is_valid_object_position(object_center_meters)
                                    and self.previous_object_center_meters is not None
                                    and self.previous_object_timestamp is not None):

                                rx, ry, rz = object_center_meters
                                dt = current_timestamp - self.previous_object_timestamp

                                if dt >= self.min_valid_dt:
                                    vrx = (rx - self.previous_object_center_meters[0]) / dt
                                    vry = (ry - self.previous_object_center_meters[1]) / dt
                                    vrz = (rz - self.previous_object_center_meters[2]) / dt
                                    speed = (vrx ** 2 + vry ** 2 + vrz ** 2) ** 0.5

                                    if speed > self.max_valid_speed:
                                        # Vitesse non physique (souvent un dt trop petit ou un saut de detection)
                                        # -> on ignore cette mise a jour plutot que de generer une parabole aberrante
                                        self.get_logger().warn(
                                            f"Trajectory update rejected: implausible speed ({speed:.1f} m/s)."
                                        )
                                    elif vrz > 0:
                                        # Dans ce repere, Z diminue quand l'objet se rapproche de la camera
                                        # (vz doit donc etre negatif pendant un lancer valide). Un vz positif
                                        # signifie que l'objet semble s'eloigner de la camera : c'est suspect
                                        # (bruit de detection/profondeur) -> on rejette ce point.
                                        self.get_logger().warn(
                                            f"Trajectory update rejected: vz positive ({vrz:.3f} m/s), object seems to move away from the camera."
                                        )
                                    else:
                                        # 1) On enregistre le point REELLEMENT observe (verite terrain)
                                        t_since_start = current_timestamp - self.trajectory_start_time
                                        self.trajectory_observed.append((t_since_start, rx, ry, rz))

                                        # 2) On calcule la nouvelle prediction, qui REMPLACE la precedente
                                        #    (une seule courbe "vivante", pas une accumulation de courbes superposees)
                                        self.predicted_trajectory = self.predict_parabolic_trajectory(
                                            x0=rx, y0=ry, z0=rz,
                                            vx0=vrx, vy0=vry, vz0=vrz,
                                            dt=dt
                                        )

                                        # 3) On n'archive qu'un echantillon des predictions (par intervalle de TEMPS,
                                        #    pas par nombre de frames) pour visualiser leur evolution sans que tout
                                        #    se superpose, meme sur un lancer court ou peu de frames valides.
                                        if (self.last_history_sample_time is None
                                                or (current_timestamp - self.last_history_sample_time) >= self.history_sampling_interval):
                                            self.trajectory_history.append(self.predicted_trajectory)
                                            self.last_history_sample_time = current_timestamp

                                        self.get_logger().info(
                                            f"Trajectory updated (frame {self.frame_count}) ---> "
                                            f"pos: X:{rx:.3f}m Y:{ry:.3f}m Z:{rz:.3f}m | "
                                            f"vel: vx:{vrx:.3f} vy:{vry:.3f} vz:{vrz:.3f} m/s | "
                                            f"{len(self.predicted_trajectory)} points predicted"
                                        )

                                        # Garde-fou : la condition "atterrissage" ne doit jamais pouvoir
                                        # se declencher sur la toute premiere frame du lancer (sinon on
                                        # n'a jamais qu'un seul point observe, comme constate en pratique).
                                        # On exige un minimum de temps de suivi ET une confirmation sur
                                        # plusieurs frames consecutives avant de conclure a un atterrissage.
                                        # NB: on se base sur rz (profondeur, se rapproche de la camera),
                                        # PAS sur ry qui est presque toujours < seuil pendant tout le vol
                                        # dans ce repere (Y negatif = vers le haut).
                                        time_tracked = current_timestamp - self.trajectory_start_time
                                        if rz < self.landing_z_threshold and time_tracked >= self.min_time_before_landing_check:
                                            self.landing_confirm_counter += 1
                                        else:
                                            self.landing_confirm_counter = 0

                                        if self.landing_confirm_counter >= self.landing_confirm_frames:
                                            stop_reason = f"Object has reached the camera (Z={rz:.3f}m < {self.landing_z_threshold}m, confirmed)."

                                else:
                                    self.get_logger().warn("Cannot update trajectory: dt too small or negative.")

                            else:
                                self.get_logger().warn("Cannot update trajectory: missing/invalid current or previous object position.")

                            elapsed = current_timestamp - self.trajectory_start_time
                            if stop_reason is None and elapsed >= self.trajectory_tracking_duration:
                                stop_reason = f"Trajectory tracking stopped after {elapsed:.2f}s."

                            if stop_reason is not None:
                                self.trajectory_tracking_active = False
                                self.get_logger().info(stop_reason)
                                self.plot_trajectory_history()
   
                    if self.is_valid_object_position(object_center_meters):
                        self.previous_object_center_meters = object_center_meters
                        self.previous_object_timestamp = current_timestamp

            self.frame_count += 1

            cv2.putText(annotated_image, f"Frame {self.frame_count}", (1150, 50),                                            
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
            if self.save_distance_mode:        
                with open(self.csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.frame_count, round(current_timestamp, 4), object_speed_csv, left_dist_csv, right_dist_csv, ""])

            if self.record_mode:
                if self.video_writer is None:
                    self.record_path = os.path.join(self.video_folder,                   
                                f"{self.timestamp_csv}_lstm_det.avi")                                     
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')                    
                    self.video_writer = cv2.VideoWriter(self.record_path, fourcc, self.fps_camera, (w_color, h_color))
                    self.get_logger().info(f"Recording started : {self.record_path}")
                
                if self.video_writer is not None:
                    self.video_writer.write(annotated_image)

            #cv2.putText(annotated_image, "Press ECHAP to quit", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Combined YOLO Detections & Poses + LSTM", annotated_image)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                self.get_logger().info("ECHAP pressed. Shutting down the node...")
                raise KeyboardInterrupt
    
        except KeyboardInterrupt:
            raise

        except Exception as e:
            self.get_logger().info(f"Error in the synchronized callback : {e}")

    def destroy_node(self):
            # Si le node s'arrete pendant qu'un lancer est encore en cours de suivi
            # (objet sorti du champ avant l'atterrissage, ECHAP/Ctrl+C premature...),
            # on sauvegarde quand meme ce qui a ete observe/predit au lieu de le perdre.
            if getattr(self, 'trajectory_tracking_active', False) and self.trajectory_observed:
                self.get_logger().info("Node shutting down with an active throw: saving partial trajectory plot...")
                self.trajectory_tracking_active = False
                self.plot_trajectory_history()

            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
                self.get_logger().info("Record video saved and closed correctly.")
            cv2.destroyAllWindows()
            return super().destroy_node()

def main(args=None):
    
    rclpy.init(args=args)
    node = LSTMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()