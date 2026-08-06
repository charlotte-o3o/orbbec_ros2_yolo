import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;*.warning=false"

import rclpy
from rclpy.node import Node
from lancer_interfaces.msg import HumanPoseArray, LandingPrediction
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

        self.get_logger().info("*** Speed Based Throw Detection Node Launched ***")

        self.bridge = CvBridge()

        self.box_color         = (random.randint(0,255), random.randint(0,255), random.randint(0,255))   
        self.right_wrist_color = (241, 255, 81)
        self.left_wrist_color  = (218, 110, 255)

        ######################################################################
        #          CAMERA PARAMETERS (to be updated from CameraInfo)         #
        ######################################################################

        self.fps_camera      = 30.0
        self.fx              = 616.0  # Focal length in pixels (x-axis)
        self.fy              = 616.0  # Focal length in pixels (y-axis)
        self.cx              = 320.0  # Principal point x-coordinate (image center)      
        self.cy              = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info = False

        self.frame_count = 1

        self.start_time    = time.time()
        self.timestamp_csv = time.strftime("%Y-%m-%d_%H-%M-%S")

        ######################################################################
        #                      CONFIGURABLE PARAMETERS                       #
        ######################################################################

        # Operating modes (Toggles / Flags)
        self.save_distance_mode                 = self.declare_parameter('save_distance_mode', False).value
        self.record_mode                        = self.declare_parameter('record_mode', False).value
        self.annotations_mode                   = self.declare_parameter('annotations_mode', True).value

        # Throw detection parameters
        self.alpha_smooth                       = self.declare_parameter('alpha_smooth', 0.4).value
        self.cooldown_duration                  = self.declare_parameter('cooldown_duration', 8.0).value
        self.max_false_frames_allowed           = self.declare_parameter('max_false_frames_allowed', 10).value
        self.trajectory_tracking_duration       = self.declare_parameter('trajectory_tracking_duration', 3.0).value
        self.history_sampling_interval          = self.declare_parameter('history_sampling_interval', 0.10).value
        self.speed_threshold                    = self.declare_parameter('speed_threshold', 2.0).value
        self.startup_grace_period               = self.declare_parameter('startup_grace_period', 8.0).value # s, no throw detection during the first seconds after node startup
        self.throw_confirm_frames               = self.declare_parameter('throw_confirm_frames', 3).value # the object must be detected as a throw for N consecutive frames to confirm the detection

        # Physical safeguards
        self.min_valid_z                        = self.declare_parameter('min_valid_z', 0.1).value
        self.max_valid_z                        = self.declare_parameter('max_valid_z', 10.0).value
        self.max_valid_xy                       = self.declare_parameter('max_valid_xy', 5.0).value
        self.min_valid_dt                       = self.declare_parameter('min_valid_dt', 0.01).value # s, below this value: temporal noise -> exploding speed
        self.max_valid_speed                    = self.declare_parameter('max_valid_speed', 25.0).value
        self.min_valid_speed                    = self.declare_parameter('min_valid_speed', -0.2).value
        self.non_approaching_velocity_threshold = self.declare_parameter('non_approaching_velocity_threshold', -0.05).value

        # Landing detection
        self.landing_z_threshold                = self.declare_parameter('landing_z_threshold', 0.5).value
        self.min_time_before_landing_check      = self.declare_parameter('min_time_before_landing_check', 0.15).value
        self.landing_confirm_frames             = self.declare_parameter('landing_confirm_frames', 3).value

        # Predicted trajectory limits
        self.max_predicted_height               = self.declare_parameter('max_predicted_height', 2.0).value
        self.max_predicted_fall                 = self.declare_parameter('max_predicted_fall', -2.0).value
        self.camera_tilt_deg                    = self.declare_parameter('camera_tilt_deg', 0.0).value
        self.camera_roll_deg                    = self.declare_parameter('camera_roll_deg', 0.0).value # 0 = landscape (default), 90 or -90 = camera rotated to vertical (requires calibration)

        # Derivated variables
        self.theta_tilt = np.radians(self.camera_tilt_deg)
        self.alpha_roll = np.radians(self.camera_roll_deg)

        ######################################################################
        #                        DATA LOGGING SETUP                          #
        ######################################################################
        
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

        ######################################################################
        #                INTERNAL TRACKING STATUSES & COUNTERS               #
        ######################################################################

        # Flags and events
        self.throw_detected                = False
        self.previous_throw_detected       = False
        self.trajectory_tracking_active    = False

        # Coordinates, measurements, and geometry
        self.throw_coordinates             = None
        self.previous_object_center_meters = None
        self.smoothed_object_meters        = None
        self.last_object_meters            = None
        self.last_world_pos                = None

        # Timestamps and timers
        self.previous_object_timestamp     = None
        self.last_timestamp                = None
        self.last_throw_trigger_time       = 0.0
        self.trajectory_start_time         = None
        self.trajectory_start_frame        = None
        self.last_history_sample_time      = None

        # Frame counters
        self.false_frame_counter           = 0
        self.throw_confirm_counter         = 0
        self.landing_confirm_counter       = 0

        # Histories and trajectories
        self.predicted_trajectory          = None
        self.trajectory_observed           = []
        self.trajectory_history            = []
        
        ######################################################################
        #                           PUBLISHERS                               #
        ######################################################################

        self.pub_landing        = self.create_publisher(
            LandingPrediction,
            '/trajectory/landing_prediction',
            10
        )

        ######################################################################
        #                           SUBSCRIBERS                              #
        ######################################################################

        self.sub_info           = self.create_subscription(
            CameraInfo,
            '/orbbec_external/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.sub_image          = message_filters.Subscriber(
            self,
            Image,
            '/orbbec_external/color/image_raw'
        )

        self.sub_depth          = message_filters.Subscriber(
            self,
            Image, 
            '/orbbec_external/depth/image_raw'
        )

        self.sub_fine_tune_yolo = message_filters.Subscriber(
            self,
            Detection2DArray,
            '/yolo_detected_objects'
        )

        self.sub_yolo_pose      = message_filters.Subscriber(
            self,
            HumanPoseArray,
            '/yolo_detected_poses'
        )

        ######################################################################
        #                          SYNCHRONIZER                              #
        ######################################################################

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_depth, self.sub_fine_tune_yolo, self.sub_yolo_pose],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.synchronized_callback)

    def camera_info_callback(self, msg: CameraInfo):

        if not self.has_camera_info:
            self.fx = msg.k[0]
            self.fy = msg.k[4] 
            self.cx = msg.k[2]
            self.cy = msg.k[5] 
            self.has_camera_info = True

            self.get_logger().info(f"Camera info received: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

            self.destroy_subscription(self.sub_info)  # Unsubscribe after receiving camera info

    def is_valid_object_position(self, position_meters):
        """
        Checks that the 3D position (x, y, z) in meters is physically 
        plausible before using it to anchor a trajectory.
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

    ######################################################################
    #                        TRAJECTORY PREDICTION                       #
    ######################################################################
    
    def predict_smoothed_trajectory(self, t_since_start, dt_frame, g_const=9.81):
        """
        Computes the smoothed predictive future trajectory using least squares polynomial fitting (polyfit)
        from the list of historical observed points.

        Returns:
            - raw_trajectory (list or None): Predicted future points [(t, x, y, z), ...]
            - vx_moy (float): Estimated average velocity along X
            - vy_moy (float): Estimated average velocity along Y at t_since_start
            - vz_moy (float): Estimated average velocity along Z
        """
        if len(self.trajectory_observed) < 3:
            return None, 0.0, 0.0, 0.0

        ts = np.array([p[0] for p in self.trajectory_observed])
        xs = np.array([p[1] for p in self.trajectory_observed])
        ys = np.array([p[2] for p in self.trajectory_observed])
        zs = np.array([p[3] for p in self.trajectory_observed])

        # Linear regression (degree 1) for X and Z (constant velocity)
        poly_x = np.polyfit(ts, xs, 1) 
        poly_z = np.polyfit(ts, zs, 1)
        
        # Gravitational adjustment of Y (Y-axis positive upwards)
        ys_adjusted = ys + 0.5 * g_const * (ts ** 2)
        poly_y = np.polyfit(ts, ys_adjusted, 1)

        vx_moy = poly_x[0]
        vy_moy = poly_y[0] - g_const * t_since_start 
        vz_moy = poly_z[0]

        # If the object is not moving towards the workspace (moving away or flat), prediction is invalid
        if vz_moy >= self.non_approaching_velocity_threshold:
            return None, vx_moy, vy_moy, vz_moy

        raw_trajectory = []
        t_future = t_since_start
        max_time = t_since_start + 2.0  # Prediction horizon (2 seconds in future)

        while t_future <= max_time:
            x_pred = poly_x[0] * t_future + poly_x[1]
            y_pred = -0.5 * g_const * (t_future ** 2) + poly_y[0] * t_future + poly_y[1]
            z_pred = poly_z[0] * t_future + poly_z[1]

            # Stop if the predicted point exits physical vertical boundaries
            if y_pred > self.max_predicted_height or y_pred < self.max_predicted_fall:
                raw_trajectory.append((t_future, x_pred, y_pred, z_pred))
                break

            raw_trajectory.append((t_future, x_pred, y_pred, z_pred))

            # Stop if the object moves past Z <= 0
            if z_pred <= 0.0 and t_future > 0:
                break

            t_future += dt_frame

        return raw_trajectory, vx_moy, vy_moy, vz_moy

    def plot_trajectory_history(self):
        if not self.trajectory_observed and not self.trajectory_history:
            self.get_logger().warn("No trajectory data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.invert_xaxis()
        ax2.invert_xaxis()

        frame_label = f"frame {self.trajectory_start_frame}" if self.trajectory_start_frame is not None else "frame ?"

        # Ground truth observed trajectory
        if self.trajectory_observed:
            xs_obs = [point[1] for point in self.trajectory_observed]
            ys_obs = [point[2] for point in self.trajectory_observed]
            zs_obs = [point[3] for point in self.trajectory_observed]
            ax1.plot(zs_obs, ys_obs, color='black',
                      marker='o', markersize=3, label="Observed points", zorder=5)
            ax2.plot(zs_obs, xs_obs, color='black',
                     marker='o', markersize=3, label="Observed points", zorder=5)

        # Sampled predictions
        for i, traj in enumerate(self.trajectory_history):
            xs = [point[1] for point in traj]
            ys = [point[2] for point in traj]
            zs = [point[3] for point in traj]
            ax1.plot(zs, ys, alpha=0.35, linestyle='--', color='tab:blue',
                      label="Predictions" if i == 0 else None)
            ax2.plot(zs, xs, alpha=0.35, linestyle='--', color='tab:blue',
                      label="Predictions" if i == 0 else None)

        # Last prediction
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

    def check_workspace_intersection(self, pos_0, vel_0, g=9.81):
        """
        Compute the time and position of entry into the robot's capture box.
        pos_0: [x0, y0, z0] at time t=0
        vel_0: [vx, vy, vz] at time t=0
        """
        x0, y0, z0 = pos_0
        vx, vy, vz = vel_0

        # Definition of the robot's capture box (in meters, in the camera's reference frame)
        """X_MIN, X_MAX = -0.25, -0.11   
        Y_MIN, Y_MAX = -0.09, 0.21   
        Z_MIN, Z_MAX =  0.0, 0.17"""   
        X_MIN, X_MAX = -0.40, 0
        Y_MIN, Y_MAX = -1, 1
        Z_MIN, Z_MAX =  0.0, 0.2

        # Sampling of the future of the parabola (e.g., over 1.5 seconds with a step of 10ms)
        time_steps = np.linspace(0.01, 1.5, 150)

        for t in time_steps:
            # Trajectory equations under gravity (Y-axis positive upwards)
            x_t = x0 + vx * t
            y_t = y0 + vy * t - 0.5 * g * (t**2)
            z_t = z0 + vz * t

            # Check if the point is inside the cube
            in_x = X_MIN <= x_t <= X_MAX
            in_y = Y_MIN <= y_t <= Y_MAX
            in_z = Z_MIN <= z_t <= Z_MAX

            if in_x and in_y and in_z:
                # First point of intersection found => capture point and time to landing
                return True, t, x_t, y_t, z_t

        # No trajectory point intersects the workspace cube
        return False, None, None, None, None

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

            ######################################################################
            #                          PROCESSING OF                             #
            #                   YOLO DETECTIONS AND POSES                        #
            ######################################################################           

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

            ######################################################################
            #                         THROW DETECTION                            #
            ######################################################################

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
                    # Smoothing the 3D position of the object to reduce noise and jitter in speed calculations
                    self.smoothed_object_meters[0] = self.alpha_smooth * object_center_meters[0] + (1 - self.alpha_smooth) * self.smoothed_object_meters[0]
                    self.smoothed_object_meters[1] = self.alpha_smooth * object_center_meters[1] + (1 - self.alpha_smooth) * self.smoothed_object_meters[1]
                    self.smoothed_object_meters[2] = self.alpha_smooth * object_center_meters[2] + (1 - self.alpha_smooth) * self.smoothed_object_meters[2]

                rx_cam, ry_cam, rz_cam = self.smoothed_object_meters

                # Step 1: compensate for the physical roll of the camera around its own
                # optical axis (Z_cam) - e.g. 90 deg when the camera is mounted vertically
                # instead of horizontally. This brings the raw camera-optical coordinates
                # back into the same "landscape" convention (x right, y down, z forward)
                # that the tilt transform below expects. With camera_roll_deg = 0 this is
                # a no-op and reduces exactly to the original behaviour.
                x_lvl = rx_cam * np.cos(self.alpha_roll) - ry_cam * np.sin(self.alpha_roll)
                y_lvl = rx_cam * np.sin(self.alpha_roll) + ry_cam * np.cos(self.alpha_roll)
                z_lvl = rz_cam
 
                # Step 2: transformation from the leveled camera frame to world coordinates 
                # (assuming camera is tilted by theta around its own X-axis)
                rx_world = - x_lvl
                ry_world = - y_lvl * np.cos(self.theta_tilt) + z_lvl * np.sin(self.theta_tilt)
                rz_world = - y_lvl * np.sin(self.theta_tilt) + z_lvl * np.cos(self.theta_tilt)

                current_world_pos = [rx_world, ry_world, rz_world]

                if self.last_world_pos is not None and self.last_timestamp is not None:
                    dt = current_timestamp - self.last_timestamp
                    if dt >= self.min_valid_dt:
                        # Compute the velocity components in the world frame
                        vrx = (current_world_pos[0] - self.last_world_pos[0]) / dt
                        vry = (current_world_pos[1] - self.last_world_pos[1]) / dt
                        vrz = (current_world_pos[2] - self.last_world_pos[2]) / dt

                        # Compute the overall speed of the object in meters per second
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
                        # The object is moving fast enough and is getting closer to the
                        # camera, which may indicate a throw
                        self.throw_confirm_counter += 1

                        if self.throw_confirm_counter >= self.throw_confirm_frames:
                            # The object has been moving fast enough for enough 
                            # consecutive frames to confirm a throw
                            self.false_frame_counter = 0
                            self.throw_detected = True

                            if self.annotations_mode:
                                        cv2.putText(annotated_image, f"THROW", (30, 50),                                        
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    # if remaining_time <= 0 but not (object_speed_m_s >= self.speed_threshold and vrz < 0):        
                    else:   
                        self.throw_confirm_counter = 0
                        self.false_frame_counter += 1                    
                        if self.false_frame_counter >= self.max_false_frames_allowed:
                            # The object has been moving too slowly or getting farther from the camera
                            # for too many consecutive frames, so we reset the throw detection
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

                            # Start trajectory tracking
                            self.trajectory_tracking_active = True
                            self.trajectory_start_time = current_timestamp
                            self.trajectory_start_frame = self.frame_count 
                            # Uncomment the following line to include the first observed point in the trajectory
                            self.trajectory_observed = [(0.0,) + tuple(self.throw_coordinates)] 
                            # Uncomment the following line to not include the first observed point in the trajectory
                            # (we could want to anchor to the second point of the throw as the first point uses 
                            # raw data that is potentially noisy)
                            #self.trajectory_observed = []  
                            self.trajectory_history = []
                            self.predicted_trajectory = None
                            self.last_history_sample_time = None
                            self.landing_confirm_counter = 0

                            self.get_logger().info(f"Throw started (vrz = {vrz:.3f}m/s): trajectory prediction start !\n")
                        
                        else:
                            # The object is moving fast enough but its position is not 
                            # physically plausible (e.g., Z < 0.1m or Z > 10m)
                            self.throw_coordinates = None
                            self.get_logger().warn("Throw detected but the object is invisible or its position is not physically plausible (rejected).")
                                
                    else:
                        # The object is moving fast enough but the cooldown period has not yet elapsed
                        self.get_logger().info(f"Flickering detected ! New initial point blocked by cooldown ({self.cooldown_duration - time_since_last_throw:.1f}s left)")

                elif self.throw_detected is not True and self.previous_throw_detected is True:
                    # The throw has ended, so we reset the throw coordinates and log the event
                    self.get_logger().info("\nThrow ended.")
                    self.throw_coordinates = None

                self.previous_throw_detected = self.throw_detected

            if self.trajectory_tracking_active:
                # The trajectory tracking is active, so we update the trajectory with the new observed point
                stop_reason = None

                if not self.throw_detected:
                    # The object is no longer detected as being thrown, so we stop the trajectory tracking
                    stop_reason = "Trajectory tracking stopped: object lost or throw ended."

                # Check if the object position is valid and if we have a previous position to compute speed
                elif (self.is_valid_object_position(object_center_meters)
                      and self.previous_object_center_meters is not None
                      and self.previous_object_timestamp is not None):   
                    
                    if object_speed_m_s > self.max_valid_speed:
                        # The object speed is too high to be physically plausible
                        self.get_logger().warn(
                            f"Trajectory update rejected: object speed too high = {object_speed_m_s:.1f} m/s."
                        )

                    elif vrz > self.non_approaching_velocity_threshold:
                       # The object is moving away from the camera (positive Z velocity)
                       self.get_logger().warn(f"Trajectory update rejected: vz positive = {vrz:.3f} m/s")

                    elif object_speed_m_s >= self.speed_threshold and vrz < self.min_valid_speed:
                        # Calculate elapsed time since the throw trajectory started
                        t_since_start = current_timestamp - self.trajectory_start_time

                        if self.trajectory_observed:
                            last_recorded_t = self.trajectory_observed[-1][0]
                            time_gap = t_since_start - last_recorded_t
                            # Detect temporary tracking loss ("black hole") if no 
                            # observations were recorded for over 500ms
                            if time_gap > 0.50:  
                                stop_reason = f"Trajectory tracking aborted: black hole detected ({time_gap:.2f}s without valid updates)."
                                self.trajectory_tracking_active = False

                        # Store the current valid 3D point in world coordinates
                        if stop_reason is None:
                            self.trajectory_observed.append((
                                t_since_start, 
                                rx_world, 
                                ry_world, 
                                rz_world
                            ))

                        # Re-calculate predictive trajectory every 2 frames 
                        # once at least 4 points have been collected
                        if len(self.trajectory_observed) >= 4 and len(self.trajectory_observed) % 2 == 0:
                           
                            dt_frame = dt if dt is not None else (1.0 / self.fps_camera)
                            raw_trajectory, vx_moy, vy_moy, vz_moy = self.predict_smoothed_trajectory(t_since_start, dt_frame)

                            if raw_trajectory is None:
                                # If prediction fails due to non-approaching velocity, 
                                # discard the last anomalous point
                                if vz_moy >= self.non_approaching_velocity_threshold:
                                    self.get_logger().warn(
                                        f"Trajectory prediction rejected (global Vz positive or flat): "
                                        f"Vx moy: {vx_moy:.2f}m/s | Vy moy: {vy_moy:.2f}m/s | Vz moy: {vz_moy:.2f}m/s. Anomalous point removed."
                                    )
                                    self.trajectory_observed.pop()
                                self.predicted_trajectory = None
                            else:
                                self.predicted_trajectory = raw_trajectory

                                # Archive trajectory predictions periodically based on history sampling interval
                                if (self.last_history_sample_time is None
                                        or (current_timestamp - self.last_history_sample_time) >= self.history_sampling_interval):
                                    self.trajectory_history.append(self.predicted_trajectory)
                                    self.last_history_sample_time = current_timestamp

                                self.get_logger().info(
                                    f"Trajectory updated ({len(self.trajectory_observed)} obs. pts) ---> "
                                    f"Vx moy: {vx_moy:.2f}m/s | Vy moy: {vy_moy:.2f}m/s | Vz moy: {vz_moy:.2f}m/s"
                                )

                                # Compute the intersection of the predicted trajectory with the robot's 3D capture box
                                has_target, t_intercept, x_target, y_target, z_target = self.check_workspace_intersection(
                                    pos_0=[rx_world, ry_world, rz_world],
                                    vel_0=[vx_moy, vy_moy, vz_moy]
                                )

                                if has_target:
                                    self.get_logger().info(
                                        f"[TARGET IN WORKSPACE] Capture point: X={x_target:.2f}m, Y={y_target:.2f}m, Z={z_target:.2f}m in {t_intercept:.3f}s"
                                    )   

                                    # Publishing of intersection point and time to landing for the robot to catch the object
                                    msg_pred = LandingPrediction()
                                    msg_pred.time_to_landing = float(t_intercept)
                                    msg_pred.x_landing = float(x_target)
                                    msg_pred.y_landing = float(y_target)
                                    msg_pred.z_landing = float(z_target)
                                    
                                    self.pub_landing.publish(msg_pred)
                                    self.get_logger().info(f"[PUBLISHED] Sent to topic -> count subscribers: {self.pub_landing.get_subscription_count()}")
                                else:
                                    self.get_logger().warn("Trajectory prediction active but DOES NOT intersect the 3D capture box!")

                        time_tracked = current_timestamp - self.trajectory_start_time

                        # Check if object has arrived close enough to camera 
                        # (landing threshold) after minimum allowed tracking time
                        if rz < self.landing_z_threshold and time_tracked >= self.min_time_before_landing_check:
                            self.landing_confirm_counter += 1
                        else:
                            self.landing_confirm_counter = 0

                        # Stop tracking when landing is confirmed over N consecutive frames
                        if self.landing_confirm_counter >= self.landing_confirm_frames:
                            stop_reason = f"Object has reached the camera (Z={rz:.3f}m < {self.landing_z_threshold}m, confirmed)."
                    
                    else:
                        # Reject trajectory update if approach speed towards the camera is too small
                        self.get_logger().warn("Cannot update trajectory : vz too small")

                # Check if object tracking was lost due to too many consecutive undetected frames
                if stop_reason is None and self.false_frame_counter >= self.max_false_frames_allowed:
                    stop_reason = "Trajectory tracking aborted: object stopped moving or lost."

                # Check if trajectory tracking exceeded maximum allowed duration
                elapsed = current_timestamp - self.trajectory_start_time
                if stop_reason is None and elapsed >= self.trajectory_tracking_duration:
                    stop_reason = f"Trajectory tracking stopped after timeout ({elapsed:.2f}s)."

                # Finalize tracking session and save trajectory plots if any stop condition was met
                if stop_reason is not None:
                    self.trajectory_tracking_active = False
                    self.get_logger().info(stop_reason)
                    self.plot_trajectory_history()

            # Cache current valid position and timestamp for speed calculations in next frame
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

            # Write 3D trajectory and tracking status to CSV file if distance logging mode is enabled
            if self.save_distance_mode:        
                x_csv = round(rx_world, 4) if rx is not None else ""
                y_csv = round(ry_world, 4) if ry is not None else ""
                z_csv = round(rz_world, 4) if rz is not None else ""
                self.csv_writer.writerow([self.frame_count, round(current_timestamp, 4), x_csv, y_csv, z_csv, object_speed_csv, self.throw_detected])
                if self.frame_count % 30 == 0: 
                    self.csv_file.flush()

            # Uncomment the following line if the camera is flipped upside down
            #display_image = cv2.flip(annotated_image, -1)

            # Record annotated video frames if recording mode is enabled
            if self.record_mode:
                if self.video_writer is None:
                    self.record_path = os.path.join(self.video_folder,                   
                                f"{self.timestamp_csv}_speed_det.avi")                                     
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')                    
                    self.video_writer = cv2.VideoWriter(self.record_path, fourcc, self.fps_camera, (w_color, h_color))
                    self.get_logger().info(f"Recording started.")
                
                if self.video_writer is not None:
                    self.video_writer.write(annotated_image)
                    # Uncomment the following line if the camera is flipped upside down
                    #self.video_writer.write(display_image)
            

            self.frame_count += 1

            #cv2.putText(annotated_image, "Press ECHAP to quit", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Throw Detection With Object Speed", annotated_image)
            # Uncomment the following line if the camera is flipped upside down
            #cv2.imshow("Throw Detection With Object Speed", display_image)
            
            key = cv2.waitKey(1) & 0xFF

            # Exit execution if ESC key (ASCII 27) is pressed
            if key == 27:
                self.get_logger().info("ECHAP pressed. Shutting down the node...")
                raise KeyboardInterrupt
    
        except KeyboardInterrupt:
            raise

        except Exception as e:
            self.get_logger().info(f"Error in the synchronized callback : {e}")

    def destroy_node(self):
            """
            Safely cleanup resources, flush log files, release video writers, 
            and save partial plot figures when shutting down the node.
            """
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