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
import cv2
import time
import numpy as np

class OpticalTrackingNode(Node):
    def __init__(self):
        super().__init__('optical_tracking_node')

        self.bridge = CvBridge()
        
        self.get_logger().info("*** Optical Stream Tracking Node Launched ***")

        self.min_valid_z              = self.declare_parameter('min_valid_z',      0.1).value
        self.max_valid_z              = self.declare_parameter('max_valid_z',      10.0).value
        self.max_valid_xy             = self.declare_parameter('max_valid_xy',     5.0).value
        self.max_valid_speed          = self.declare_parameter('max_valid_speed',  25.0).value
        self.max_dist_meter           = self.declare_parameter('max_dist_meter',   20.0).value
        self.mask_thickness           = self.declare_parameter('mask_thickness',   0.4).value


        self.tracking_state = 'IDLE'   # 'IDLE' -> 'HOLDING' -> 'THROWN'
        self.prev_stamp = None
        self.prev_gray = None
        self.prev_object_depth = None

        # Snapshot taken at the moment of launch (last reliable data before loss of contact)
        self.last_valid_bbox             = None   # (x1, y1, x2, y2) in px
        self.last_valid_object_center_m  = None   # (x, y, z) in m
        self.last_holding_arm            = None   # dict shoulder/elbow/wrist
        self.throw_start_time            = None

        # History for detecting hand-object separation
        self.hand_object_distance_history = []
        self.max_distance_history_len     = 5

        # Throw detection thresholds
        self.throw_release_distance   = self.declare_parameter('throw_release_distance', 0.15).value  # m
        self.throw_min_release_speed  = self.declare_parameter('throw_min_release_speed', 1.5).value   # m/s
        self.hold_confirm_frames      = self.declare_parameter('hold_confirm_frames', 15).value
        self.consecutive_hold_frames  = 0

        self.right_wrist_color = (241, 255, 81)
        self.left_wrist_color  = (218, 110, 255)

        self.record_mode              = self.declare_parameter('record_mode',      False).value
        self.annotations_mode         = self.declare_parameter('annotations_mode', True).value
        self.debug_mask_overlay       = self.declare_parameter('debug_mask_overlay', False).value

        ######################################################################
        #          CAMERA PARAMETERS (to be updated from CameraInfo)         #
        ######################################################################

        self.fps_camera        = 30.0
        self.fx                = 616.0  # Focal length in pixels (x-axis)
        self.fy                = 616.0  # Focal length in pixels (y-axis)
        self.cx                = 320.0  # Principal point x-coordinate (image center)      
        self.cy                = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info   = False

        self.frame_count = 1

        ######################################################################
        #                          VIDEO RECORDER                            #
        ######################################################################

        self.start_time = time.time()
        self.timestamp_csv = time.strftime("%Y-%m-%d_%H-%M-%S")

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
        #                           SUBSCRIBERS                              #
        ######################################################################

        self.sub_info       = self.create_subscription(
            CameraInfo,
            '/orbbec_external/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.sub_image      = message_filters.Subscriber(
            self,
            Image,
            '/orbbec_external/color/image_raw'
        )

        self.sub_depth      = message_filters.Subscriber(
            self,
            Image, 
            '/orbbec_external/depth/image_raw'
        )

        self.sub_yolo_world = message_filters.Subscriber(
            self,
            Detection2DArray,
            '/yolo_detected_objects'
        )

        self.sub_yolo_pose  = message_filters.Subscriber(
            self,
            HumanPoseArray,
            '/yolo_detected_poses'
        )

        ######################################################################
        #                          SYNCHRONIZER                              #
        ######################################################################

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_depth, self.sub_yolo_world, self.sub_yolo_pose],
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

    def compute_hand_object_distance_3d(
        self, obj_center_meters, wx_px, wy_px, cv_depth_image
    ):
        """
        Computes the 3D distance between the object and a given wrist.
        """
        if obj_center_meters is None or cv_depth_image is None:
            return None

        h, w = cv_depth_image.shape[:2]
        wx_px = max(0, min(int(wx_px), w - 1))
        wy_px = max(0, min(int(wy_px), h - 1))

        wrist_z = cv_depth_image[wy_px, wx_px] / 1000.0  # mm -> m

        if wrist_z < self.min_valid_z:
            return None

        # 2D -> 3D projection
        wrist_x = ((wx_px - self.cx) * wrist_z) / self.fx
        wrist_y = ((wy_px - self.cy) * wrist_z) / self.fy

        # 3D distance
        obj_x, obj_y, obj_z = obj_center_meters
        dist_3d = float(np.sqrt(
            (obj_x - wrist_x) ** 2
            + (obj_y - wrist_y) ** 2
            + (obj_z - wrist_z) ** 2
        ))

        return dist_3d

    def is_near_hand_3d(self, obj_center_meters, wx_px, wy_px, cv_depth_image):
        """
        Returns True if the distance between the object and a given wrist is inferior to the threshold.
        Returns False if the computed 3D distance is None or if it's superior to the threshold
        """
        dist = self.compute_hand_object_distance_3d(obj_center_meters, wx_px, wy_px, cv_depth_image)

        return dist is not None and dist <= self.max_dist_meter

    def build_arm_exclusion_mask(self, frame_shape, arm, mask_thickness):
        """
        Returns a mask : 
            255 = kept zone
            0 = excluded zone (holding arm)
        In full resolution frame (to be cropped by ROI)
        """
        h_frame, w_frame = frame_shape
        mask = np.full((h_frame, w_frame), 255, dtype=np.uint8)

        if arm is None:
            return mask  # If no identified holding arm

        # Upper arm
        cv2.line(mask, arm["shoulder"], arm["elbow"], 0, thickness=mask_thickness)
        # Forearm
        cv2.line(mask, arm["elbow"], arm["wrist"], 0, thickness=mask_thickness)
        # Hand
        cv2.circle(mask, arm["wrist"], mask_thickness, 0, -1)

        return mask

    def update_throw_detection(self, is_held, holding_arm, object_center_meters,
                             bbox, wx, wy, cv_depth_image, dt):
        """
        Manages the transition IDLE -> HOLDING -> THROWN.
        Returns True if this frame marks the start of the throw..
        """
        throw_triggered = False

        if self.tracking_state == 'IDLE':
            if is_held:
                self.consecutive_hold_frames += 1
                if self.consecutive_hold_frames >= self.hold_confirm_frames:
                    self.tracking_state = 'HOLDING'
                    self.get_logger().info("Objet détecté et tenu -> état HOLDING.")
            else:
                self.consecutive_hold_frames = 0

        elif self.tracking_state == 'HOLDING':
            # Continuous update of the last known state as long as the object is held.
            if is_held:
                self.last_valid_bbox = bbox
                self.last_valid_object_center_m = object_center_meters
                self.last_holding_arm = holding_arm

                dist = self.compute_hand_object_distance_3d(
                    object_center_meters, wx, wy, cv_depth_image
                )
                if dist is not None:
                    self.hand_object_distance_history.append(dist)
                    if len(self.hand_object_distance_history) > self.max_distance_history_len:
                        self.hand_object_distance_history.pop(0)
            else:
                # Contact lost: check if it is a genuine throw.
                if len(self.hand_object_distance_history) >= 2 and dt > 0:
                    separation_speed = (
                        self.throw_release_distance - self.hand_object_distance_history[0]
                    ) / (dt * len(self.hand_object_distance_history))
                    # Rough approximation; refined in the next step using optical flow
                else:
                    separation_speed = 0.0

                # Throw confirmation: contact lost + object was held steadily beforehand
                self.tracking_state = 'THROWN'
                self.throw_start_time = time.time()
                throw_triggered = True
                self.consecutive_hold_frames = 0
                self.hand_object_distance_history.clear()
                self.get_logger().info(
                    f"THROW DETECTED. Last known BB: {self.last_valid_bbox}, "
                    f"Last position: {self.last_valid_object_center_m}"
                )

        return throw_triggered

    def synchronized_callback(self, color_msg, depth_msg, yolo_objects, yolo_poses):
        try: 

            ######################################################################
            #                            CAMERA DATA                             #
            #                            ACQUISITION                             #
            ######################################################################  

            annotated_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            if annotated_image is None or cv_depth_image is None:
                self.get_logger().warn("Frame ignored: color or depth image missing.")
                return

            # Gray scale of the RBG image for optical flow
            curr_gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY) 

            h_color, w_color = annotated_image.shape[:2]
            h, w = cv_depth_image.shape[:2]

            curr_stamp = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9

            object_center_pixels = None
            object_center_meters = None

            ######################################################################
            #                          PROCESSING OF                             #
            #                   YOLO DETECTIONS AND POSES                        #
            ######################################################################          

            # Dictionary list : {'shoulder':(x,y), 'elbow':(x,y), 'wrist':(x,y), 'side':'left'/'right'}
            valid_arms = []

            for pose in yolo_poses.poses:
                kpts = pose.keypoints
                for side, shoulder_idx, elbow_idx, wrist_idx, color in [
                        ("left",  5, 7, 9,  self.left_wrist_color),
                        ("right", 6, 8, 10, self.right_wrist_color),
                    ]:
                    if len(kpts) <= wrist_idx or kpts[wrist_idx].confidence < 0.7:
                        continue

                    # Get wrist position
                    wx = max(0, min(int(kpts[wrist_idx].x), w - 1))
                    wy = max(0, min(int(kpts[wrist_idx].y), h - 1))

                    # Get elbow position
                    # Graceful degradation : if not available, fall back to the wrist
                    # rather than losing the entire arm
                    if len(kpts) > elbow_idx and kpts[elbow_idx].confidence > 0.5:
                        ex = max(0, min(int(kpts[elbow_idx].x), w - 1))
                        ey = max(0, min(int(kpts[elbow_idx].y), h - 1))
                    else:
                        ex, ey = wx, wy

                    # Get shoulder position
                    # If not available, fall back to the elbow
                    if len(kpts) > shoulder_idx and kpts[shoulder_idx].confidence > 0.5:
                        sx = max(0, min(int(kpts[shoulder_idx].x), w - 1))
                        sy = max(0, min(int(kpts[shoulder_idx].y), h - 1))
                    else:
                        sx, sy = ex, ey

                    valid_arms.append({
                        "side": side,
                        "shoulder": (sx, sy),
                        "elbow": (ex, ey),
                        "wrist": (wx, wy),
                    })

                    if self.annotations_mode:
                        cv2.circle(annotated_image, (wx, wy), 8, color, -1)
                        cv2.line(annotated_image, (sx, sy), (ex, ey), color, 2)
                        cv2.line(annotated_image, (ex, ey), (wx, wy), color, 2)
                        cv2.putText(
                            annotated_image,
                            f"{side} arm",
                            (wx + 10, wy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )

            for detection in yolo_objects.detections:
                x_center = detection.bbox.center.position.x
                y_center = detection.bbox.center.position.y
                size_x = detection.bbox.size_x
                size_y = detection.bbox.size_y
                object_center_pixels = (int(x_center), int(y_center))

                x1 = max(0, min(int(x_center - size_x / 2.0), w_color - 1))
                y1 = max(0, min(int(y_center - size_y / 2.0), h_color - 1))
                x2 = max(0, min(int(x_center + size_x / 2.0), w_color - 1))
                y2 = max(0, min(int(y_center + size_y / 2.0), h_color - 1))

                object_center_meters = None
                if len(detection.results) > 0:
                    candidate = (
                        detection.results[0].pose.pose.position.x,
                        detection.results[0].pose.pose.position.y,
                        detection.results[0].pose.pose.position.z,
                    )
                    if self.is_valid_object_position(candidate):
                        object_center_meters = candidate

                # Check if the object is held by a person
                is_held = False
                holding_arm = None

                for arm in valid_arms:
                    wx, wy = arm["wrist"]
                    if self.is_near_hand_3d(object_center_meters, wx, wy, cv_depth_image):
                        is_held = True
                        holding_arm = arm
                        break

                dt = curr_stamp - self.prev_stamp if self.prev_stamp is not None else 0.0
                wx, wy = (holding_arm["wrist"] if holding_arm is not None else (None, None))

                throw_triggered = self.update_throw_detection(
                    is_held, holding_arm, object_center_meters,
                    (x1, y1, x2, y2), wx, wy, cv_depth_image, dt
                )

                if throw_triggered:
                    # Preparation of the initial ROI
                    self.init_optical_flow_roi()

                # If the object is not validated, continue to the next one
                if not is_held:
                    continue

                # Mask construction
                band_thickness = int(self.mask_thickness * max(size_x, size_y))
                arm_mask = self.build_arm_exclusion_mask((h_color, w_color), holding_arm, band_thickness)

                if self.debug_mask_overlay:
                    debug_overlay = annotated_image.copy()
                    debug_overlay[arm_mask == 0] = (0, 0, 255)  # rouge sur la zone exclue (BGR)
                    annotated_image[:] = cv2.addWeighted(debug_overlay, 0.4, annotated_image, 0.6, 0)

                # Annotation if the object is validated 
                if self.annotations_mode and is_held and holding_arm is not None:
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.circle(annotated_image, (int(x_center), int(y_center)), 6, (0, 255, 0), -1)
                    cv2.putText(annotated_image, "[VALIDATED]", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.line(annotated_image, object_center_pixels, (wx, wy), (0, 255, 255), 2)   

            # Uncomment the following line if the camera is flipped upside down
            #display_image = cv2.flip(annotated_image, -1) 

            # Variables update for next callback
            self.prev_stamp = curr_stamp    
            self.prev_gray = curr_gray
            if object_center_meters is not None:
                self.prev_object_depth = object_center_meters[2]
            self.frame_count += 1

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

            cv2.imshow("Optical Tracking", annotated_image)

            key = cv2.waitKey(1) & 0xFF

            # Exit execution if ESC key (ASCII 27) is pressed
            if key == 27:
                print("\n\n-----")
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
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"\nRecord video saved : {self.record_path}")

        cv2.destroyAllWindows()
        return super().destroy_node()

def main(args=None):
    
    rclpy.init(args=args)
    node = OpticalTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()