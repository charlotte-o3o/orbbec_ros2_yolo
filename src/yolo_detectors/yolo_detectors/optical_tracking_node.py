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
        self.max_dist_meter           = self.declare_parameter('max_dist_meter',   0.25).value
        self.mask_thickness_arm       = self.declare_parameter('mask_thickness_arm',   0.4).value
        self.mask_thickness_hand      = self.declare_parameter('mask_thickness_hand',   0.25).value
        self.target_class_id          = self.declare_parameter('target_class_id', 'football_ball').value

        self.get_logger().info(f"Tracking node configured for target class: '{self.target_class_id}'")

        self.tracking_state    = 'IDLE'   # 'IDLE' -> 'HOLDING' -> 'THROWN'
        self.prev_stamp        = None
        self.prev_gray         = None
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
        self.throw_min_release_speed  = self.declare_parameter('throw_min_release_speed', 1.5).value   # m/s
        self.hold_confirm_frames      = self.declare_parameter('hold_confirm_frames', 15).value
        self.consecutive_hold_frames  = 0

        self.roi_margin_ratio         = self.declare_parameter('roi_margin_ratio', 0.5).value

        self.flow_roi          = None   # (x1, y1, x2, y2) in pixels, current ROI of tracking
        self.flow_arm_mask     = None   # full-frame mask, arm excluded, frozen at release
        self.flow_object_depth = None   # last known object depth (m)

        self.right_wrist_color = (241, 255, 81)
        self.left_wrist_color  = (218, 110, 255)

        self.record_mode              = self.declare_parameter('record_mode',      True).value
        self.annotations_mode         = self.declare_parameter('annotations_mode', True).value
        self.debug_mask_overlay       = self.declare_parameter('debug_mask_overlay', True).value

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

    def build_arm_exclusion_mask(self, frame_shape, arm, mask_thickness_arm, mask_thickness_hand):
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
        cv2.line(mask, arm["shoulder"], arm["elbow"], 0, thickness=mask_thickness_arm)
        # Forearm
        cv2.line(mask, arm["elbow"], arm["wrist"], 0, thickness=mask_thickness_arm)
        # Hand
        cv2.circle(mask, arm["wrist"], mask_thickness_hand, 0, -1)

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
                    self.get_logger().info("Object detected and held -> HOLDING state.")
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
                    delta_dist = self.hand_object_distance_history[-1] - self.hand_object_distance_history[0]
                    elapsed = dt * (len(self.hand_object_distance_history) - 1)
                    separation_speed = delta_dist / elapsed if elapsed > 0 else 0.0
                    # Rough approximation; refined in the next step using optical flow
                else:
                    separation_speed = 0.0

                if separation_speed >= self.throw_min_release_speed:
                    # Throw confirmation: contact lost + object was held steadily beforehand + object speed 
                    # superior to the threshold
                    self.tracking_state = 'THROWN'
                    self.throw_start_time = time.time()
                    throw_triggered = True
                    self.get_logger().info(
                        f"THROW DETECTED (speed: {separation_speed:.2f} m/s). "
                        f"Last known BB: {self.last_valid_bbox}"
                        f"Last position: {self.last_valid_object_center_m}"
                    )

                else:
                    # Contact lost but too slow to be a throw -> likely noise or a slow, intentional release;
                    # revert to IDLE instead of THROWN
                    self.tracking_state = 'IDLE'
                    self.get_logger().info(
                        f"Contact lost but speed insufficient ({separation_speed:.2f} m/s) - reverting to IDLE."
                    )

                self.consecutive_hold_frames = 0
                self.hand_object_distance_history.clear()

        return throw_triggered

    def init_optical_flow_roi(self):
        """
        Initializes the optical flow tracking state at the precise moment of the throw. 
        Uses the last known bounding box/position/arm (captured during HOLDING)
        and self.prev_gray, which corresponds to the last frame where the object was
        still being held.
        """
        if self.last_valid_bbox is None or self.prev_gray is None:
            self.get_logger().warn(
                "Launch detected but bounding box or previous frame unavailable - initialization aborted."
            )
            self.tracking_state = 'IDLE'
            return

        x1, y1, x2, y2 = self.last_valid_bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        h_frame, w_frame = self.prev_gray.shape[:2]

        # Construction of the ROI
        # We add of a margin to each side of the bbox so that the ROI is larger than it
        margin_x = int(bbox_w * self.roi_margin_ratio)
        margin_y = int(bbox_h * self.roi_margin_ratio)

        roi_x1 = max(0, x1 - margin_x)
        roi_y1 = max(0, y1 - margin_y)
        roi_x2 = min(w_frame, x2 + margin_x)
        roi_y2 = min(h_frame, y2 + margin_y)

        self.flow_roi = (roi_x1, roi_y1, roi_x2, roi_y2)

        band_thickness_arm = int(self.mask_thickness_arm * max(bbox_w, bbox_h))
        band_thickness_hand = int(self.mask_thickness_hand * max(bbox_w, bbox_h))
        self.flow_arm_mask = self.build_arm_exclusion_mask(
            (h_frame, w_frame), self.last_holding_arm, band_thickness_arm, band_thickness_hand
        )

        self.flow_object_depth = (
            self.last_valid_object_center_m[2]
            if self.last_valid_object_center_m is not None else None
        )

        self.get_logger().info(
            f"Initialized optical flow ROI: {self.flow_roi}, "
            f"reference depth: {self.flow_object_depth}"
        )

    def compute_optical_flow_in_roi(self, prev_gray, curr_gray, roi, arm_mask=None):
        """
        Computes dense Farneback optical flow restricted to the given ROI.
        Returns (vx_px, vy_px): median flow components in pixels between the two frames,
        or None if the flow cannot be computed.
        arm_mask (optional): full-frame mask (255=keep, 0=exclude), cropped to the ROI here.
        """
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return None

        prev_roi = prev_gray[y1:y2, x1:x2]
        curr_roi = curr_gray[y1:y2, x1:x2]

        if prev_roi.size == 0 or curr_roi.size == 0:
            return None

        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        if arm_mask is not None:
            mask_roi = arm_mask[y1:y2, x1:x2]
            valid = mask_roi > 0
        else:
            valid = np.ones(flow.shape[:2], dtype=bool)

        if not np.any(valid):
            return None

        # Median rather than mean: robust to leftover arm pixels or reflections on the object
        vx_px = float(np.median(flow[..., 0][valid]))
        vy_px = float(np.median(flow[..., 1][valid]))

        return vx_px, vy_px

    def estimate_object_depth_in_roi(self, cv_depth_image, roi, arm_mask, reference_depth, depth_tolerance=0.3):
        """
        Estimates the object's current depth (m) within the ROI, reading the depth
        image directly. Excludes the arm mask and rejects background pixels too far
        from the reference depth.
        """
        x1, y1, x2, y2 = roi
        depth_roi = cv_depth_image[y1:y2, x1:x2].astype(np.float32) / 1000.0  # mm -> m

        if arm_mask is not None:
            mask_roi = arm_mask[y1:y2, x1:x2]
        else:
            mask_roi = np.full(depth_roi.shape, 255, dtype=np.uint8)

        valid = (mask_roi > 0) & (depth_roi > 0)

        if reference_depth is not None:
            valid &= (np.abs(depth_roi - reference_depth) <= depth_tolerance)

        if not np.any(valid):
            return None

        return float(np.median(depth_roi[valid]))

    def pixel_flow_to_velocity(self, vx_px, vy_px, depth_z, dt):
        """
        Converts a per-frame 2D pixel displacement into a 3D velocity (m/s),
        using the pinhole camera model at the object's current depth.
        """
        if depth_z is None or dt <= 0:
            return None

        vx_m = (vx_px * depth_z) / self.fx / dt
        vy_m = (vy_px * depth_z) / self.fy / dt

        return vx_m, vy_m

    def process_thrown_object(self, curr_gray, cv_depth_image, dt, annotated_image):
        """
        Runs every frame while tracking_state == 'THROWN'. Computes optical flow
        in the current ROI, estimates 3D velocity, and propagates the ROI for
        the next frame (naive translation for now).
        """
        if self.flow_roi is None or self.prev_gray is None or dt <= 0:
            return

        flow_result = self.compute_optical_flow_in_roi(
            self.prev_gray, curr_gray, self.flow_roi, self.flow_arm_mask
        )
        if flow_result is None:
            self.get_logger().warn("Optical flow computation failed in ROI - skipping this frame.")
            return

        vx_px, vy_px = flow_result

        depth_estimate = self.estimate_object_depth_in_roi(
            cv_depth_image, self.flow_roi, self.flow_arm_mask, self.flow_object_depth
        )
        if depth_estimate is not None:
            self.flow_object_depth = depth_estimate

        velocity = self.pixel_flow_to_velocity(vx_px, vy_px, self.flow_object_depth, dt)
        if velocity is None:
            return

        vx_m, vy_m = velocity
        self.get_logger().info(
            f"Optical flow velocity estimate: vx={vx_m:.2f} m/s, vy={vy_m:.2f} m/s, "
            f"depth={self.flow_object_depth:.2f} m"
        )

        # Naive ROI propagation: shift by the measured pixel flow, same size for now
        x1, y1, x2, y2 = self.flow_roi
        dx, dy = int(round(vx_px)), int(round(vy_px))
        h_frame, w_frame = curr_gray.shape[:2]

        new_x1 = max(0, min(x1 + dx, w_frame - 1))
        new_y1 = max(0, min(y1 + dy, h_frame - 1))
        new_x2 = max(new_x1 + 1, min(x2 + dx, w_frame))
        new_y2 = max(new_y1 + 1, min(y2 + dy, h_frame))
        self.flow_roi = (new_x1, new_y1, new_x2, new_y2)

        if self.annotations_mode:
            cv2.rectangle(annotated_image, (new_x1, new_y1), (new_x2, new_y2), (255, 0, 255), 2)
            cv2.putText(annotated_image, "FLOW ROI", (new_x1, new_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)



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
            dt = curr_stamp - self.prev_stamp if self.prev_stamp is not None else 0.0

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
                if len(detection.results) == 0:
                    continue
                if detection.results[0].hypothesis.class_id != self.target_class_id:
                    continue
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
                holding_arm_dist = None

                for arm in valid_arms:
                    wx, wy = arm["wrist"]
                    dist = self.compute_hand_object_distance_3d(object_center_meters, wx, wy, cv_depth_image)
                    if dist is not None and dist <= self.max_dist_meter:
                        if holding_arm_dist is None or dist < holding_arm_dist:
                            holding_arm = arm
                            holding_arm_dist = dist
                            is_held = True

                if holding_arm is not None:
                    self.get_logger().info(f"Object held by {holding_arm['side']} arm (distance: {holding_arm_dist:.3f} m).")

                wx, wy = (holding_arm["wrist"] if holding_arm is not None else (None, None))

                throw_triggered = self.update_throw_detection(
                    is_held, holding_arm, object_center_meters,
                    (x1, y1, x2, y2), wx, wy, cv_depth_image, dt
                )

                if throw_triggered:
                    # Preparation of the initial ROI
                    self.init_optical_flow_roi()
                    if self.annotations_mode and self.flow_roi is not None:
                        rx1, ry1, rx2, ry2 = self.flow_roi
                        cv2.rectangle(annotated_image, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
                        cv2.putText(annotated_image, "FLOW ROI", (rx1, ry1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # If the object is not validated, continue to the next one
                if not is_held:
                    continue

                # Mask construction
                band_thickness_arm = int(self.mask_thickness_arm * max(size_x, size_y))
                band_thickness_hand = int(self.mask_thickness_hand * max(size_x, size_y))
                arm_mask = self.build_arm_exclusion_mask((h_color, w_color), holding_arm, band_thickness_arm, band_thickness_hand)

                if self.debug_mask_overlay:
                    debug_overlay = annotated_image.copy()
                    debug_overlay[arm_mask == 0] = (0, 0, 255) 
                    annotated_image[:] = cv2.addWeighted(debug_overlay, 0.4, annotated_image, 0.6, 0)

                # Annotation if the object is validated 
                if self.annotations_mode and is_held and holding_arm is not None:
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.circle(annotated_image, (int(x_center), int(y_center)), 6, (0, 255, 0), -1)
                    cv2.putText(annotated_image, "[VALIDATED]", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.line(annotated_image, object_center_pixels, (wx, wy), (0, 255, 255), 2)   

            if self.tracking_state == 'THROWN':
                self.process_thrown_object(curr_gray, cv_depth_image, dt, annotated_image)


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
                                f"{self.timestamp_csv}_optical_flow.avi")                                     
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