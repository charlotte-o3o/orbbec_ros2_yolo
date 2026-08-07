import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;*.warning=false"

from rclpy.node import Node
import message_filters
from sensor_msgs.msg import CameraInfo, Image
import rclpy
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import message_filters
import random
import numpy as np
import time
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

class FineTuneYoloNode(Node):

    def __init__(self):
        super().__init__('fine_tune_yolo_node')

        self.declare_parameter('model_path',  'weights/alien_plushie_v5.pt')
        self.declare_parameter('confidence',  0.50)
        self.declare_parameter('max_history', 5)
        self.declare_parameter('max_jump',    2.0)
        self.declare_parameter('bb_margin',   0.46)
        self.declare_parameter('max_velocity', 5.0)
        # Counter-clockwise rotation applied to the colour image BEFORE inference, so YOLO
        # sees an upright scene when the camera is physically mounted rotated. Detections are
        # mapped straight back to native pixels, so depth sampling, the intrinsics and the
        # published 3D points are all unaffected by this.
        self.declare_parameter('input_rotation_deg', 90)

        self.model_path           = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence').value
        self.max_history          = self.get_parameter('max_history').value
        self.max_jump             = self.get_parameter('max_jump').value
        self.bb_margin            = self.get_parameter('bb_margin').value
        self.max_velocity         = self.get_parameter('max_velocity').value
        self.input_rotation_deg   = int(self.get_parameter('input_rotation_deg').value) % 360

        # The cv2.rotate flag and the inverse box mapping are both derived from this single
        # value, so they cannot drift out of sync.
        _rotate_flags = {
            0:   None,
            90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE,
        }
        if self.input_rotation_deg not in _rotate_flags:
            raise ValueError(
                f"input_rotation_deg must be one of 0/90/180/270, got {self.input_rotation_deg}"
            )
        self.rotate_flag = _rotate_flags[self.input_rotation_deg]

        self.fx = 616.0  # Focal length in pixels (x-axis)
        self.fy = 616.0  # Focal length in pixels (y-axis)
        self.cx = 320.0  # Principal point x-coordinate (image center)      
        self.cy = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info = False  # Flag to check if camera info has been received

        self.smoothed_distance = None

        # --- Initialisation du convertisseur CvBridge
        self.bridge = CvBridge()
        self.distance_history: list[float] = []
        self.consecutive_jumps = 0
        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

        self.last_valid_time = self.get_clock().now()

        self.get_logger().info("*** YOLO Node Launched successfully ***")

        self.get_logger().info(f"Model loading : {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.get_logger().info("Model loaded successfully")
        

        self.sub_info = self.create_subscription(
            CameraInfo,
            '/orbbec_external/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.sub_color = message_filters.Subscriber(
            self,
            Image, 
            '/orbbec_external/color/image_raw'
        )
        
        self.sub_depth = message_filters.Subscriber(
            self,
            Image, 
            '/orbbec_external/depth/image_raw'
        )

        self.pub_detections = self.create_publisher(
            Detection2DArray,
            '/yolo_detected_objects',
            10
        )

        # --- Config du synchroniseur temporel approximatif
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth],
            queue_size=10,
            slop=0.05
        )

        # --- Fonction callback pour les deux messages synchronisés
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

    def unrotate_box(self, x1_up, y1_up, x2_up, y2_up, w_orig, h_orig):
        """
        Map a box from the upright (rotated) image back to native colour-image pixels.

        Forward mapping of a native pixel (x, y), per cv2.rotate mode:
            90  CCW : (x_up, y_up) = (y,              w_orig - 1 - x)
            180     : (x_up, y_up) = (w_orig - 1 - x, h_orig - 1 - y)
            270 CCW : (x_up, y_up) = (h_orig - 1 - y, x)

        Below is the inverse applied to both corners; min/max then re-normalises the box,
        since rotating swaps which corner is top-left.
        """
        if self.input_rotation_deg == 0:
            return x1_up, y1_up, x2_up, y2_up

        if self.input_rotation_deg == 90:
            corners = [(w_orig - 1 - y1_up, x1_up),
                       (w_orig - 1 - y2_up, x2_up)]
        elif self.input_rotation_deg == 180:
            corners = [(w_orig - 1 - x1_up, h_orig - 1 - y1_up),
                       (w_orig - 1 - x2_up, h_orig - 1 - y2_up)]
        else:  # 270
            corners = [(y1_up, h_orig - 1 - x1_up),
                       (y2_up, h_orig - 1 - x2_up)]

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def synchronized_callback(self, color_msg, depth_msg):
        try:
            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            h_orig, w_orig = cv_color_image.shape[:2]

            # --- 1. Straighten the image for YOLO ---
            # The models are trained on upright scenes, so a physically rotated camera costs
            # recall. Rotate for inference only: everything downstream stays native.
            if self.rotate_flag is None:
                cv_color_upright = cv_color_image
            else:
                cv_color_upright = cv2.rotate(cv_color_image, self.rotate_flag)

            start_time = time.perf_counter()
            results = self.model(cv_color_upright, stream=True, verbose=False, conf=self.confidence_threshold)
            results = list(results)
            end_time = time.perf_counter() 

            inference_time = (end_time - start_time) * 1000
            fps = 1000.0 / inference_time if inference_time > 0 else 0.0

            # L'annotation s'effectue toujours sur l'image native (couchée)
            annotated_image = cv_color_image.copy()
            boxes = results[0].boxes

            # Hauteur et largeur de l'image de profondeur native
            h, w = cv_depth_image.shape[:2]

            msg_array = Detection2DArray()
            msg_array.header = color_msg.header
            
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    confidence = float(box.conf[0]) * 100
                    
                    # --- 2. Coordinates in the upright image, as YOLO saw it ---
                    x1_up, y1_up, x2_up, y2_up  = map(int, box.xyxy[0])

                    # --- 3. Back to native colour-image pixels ---
                    # From here on (x1, y1, x2, y2) are registered on the native image, so
                    # depth sampling and deprojection need no further adjustment.
                    x1, y1, x2, y2 = self.unrotate_box(x1_up, y1_up, x2_up, y2_up,
                                                       w_orig, h_orig)

                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    x_center = max(0, min(x_center, w - 1))
                    y_center = max(0, min(y_center, h - 1))

                    margin_x = int((x2 - x1) * self.bb_margin)                         
                    margin_y = int((y2 - y1) * self.bb_margin)     

                    y1_p = max(0, y1 + margin_y)  
                    y2_p = min(cv_depth_image.shape[0], y2 - margin_y)    
                    x1_p = max(0, x1 + margin_x)                             
                    x2_p = min(cv_depth_image.shape[1], x2 - margin_x)

                    patch = cv_depth_image[y1_p:y2_p, x1_p:x2_p]                 
                    valid = patch[patch > 0]

                    if len(valid) > 0:   
                        median_val = float(np.median(valid))
                        std_val = float(np.std(valid))
                        filtered = valid[np.abs(valid - median_val) < std_val] 

                        if len(filtered) > 0:                              
                            distance = float(np.median(filtered)) / 1000.0
                        else:
                            distance = median_val / 1000.0 
                    else:                                                              
                        distance = 0.0  

                    current_time = self.get_clock().now()
                    dt = (current_time - self.last_valid_time).nanoseconds / 1e9
                    dynamic_max_jump = max(0.20, self.max_velocity * dt)

                    if distance > 0 and len(self.distance_history) > 0:
                        if abs(distance - self.distance_history[-1]) > dynamic_max_jump:
                            self.consecutive_jumps += 1

                            if self.consecutive_jumps > 5:
                                self.get_logger().warn("Blocked Z-filter detected! Resetting history.")
                                self.distance_history.clear()
                                self.consecutive_jumps = 0
                            else:
                                distance = self.distance_history[-1]  
                        else:
                            self.consecutive_jumps = 0  
                            self.last_valid_time = current_time

                    if distance > 0:                              
                        self.distance_history.append(distance)   

                        if len(self.distance_history) > self.max_history:                       
                            self.distance_history.pop(0)  

                        distance = float(np.mean(self.distance_history)) 
                   
                    if distance > 0:
                        x_meters = ((x_center - self.cx) * distance) / self.fx
                        y_meters = ((y_center - self.cy) * distance) / self.fy
                        text_coord = f"X: {x_meters:.2f}m, Y: {y_meters:.2f}m, Z: {distance:.2f}m"
                    else:
                        x_meters, y_meters = None, None
                        text_coord = "X: ---, Y: ---, Z: ---"

                    detection = Detection2D()
                    detection.bbox.center.position.x = float(x_center)
                    detection.bbox.center.position.y = float(y_center)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)

                    hyp = ObjectHypothesisWithPose() 
                    hyp.hypothesis.class_id = str(label) 
                    hyp.hypothesis.score = confidence / 100.0  
                    hyp.pose.pose.position.x = float(x_meters) if x_meters is not None else 0.0
                    hyp.pose.pose.position.y = float(y_meters) if y_meters is not None else 0.0
                    hyp.pose.pose.position.z = float(distance)  
                    
                    msg_array.detections.append(detection)
                    detection.results.append(hyp)
                    
                    custom_label = f"{label} ({confidence:.1f}%) | {text_coord}"
                    cv2.rectangle(annotated_image, (x1,y1), (x2,y2), self.box_color, 2)
                    cv2.circle(annotated_image, (x_center, y_center), 4, (0, 0, 255), -1)
                    cv2.putText(annotated_image, custom_label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.box_color, 2)
                    
                    (tw, th), _ = cv2.getTextSize(f"{distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 10.0, 14)           
                    tx = 30                                                            
                    ty = h - 30                                         
                    overlay = annotated_image.copy() 
                                        
                    cv2.rectangle(overlay, (tx - 10, ty - th - 10),              
                                (tx + tw + 10, ty + 10), (0, 0, 0), -1)        
                    cv2.addWeighted(overlay, 0.5, annotated_image, 0.5, 0, annotated_image)                             
                    cv2.putText(annotated_image, f"{distance:.2f} m", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                                10.0, (0, 255, 0), 14, cv2.LINE_AA) 
                    
            self.pub_detections.publish(msg_array)

            cv2.putText(
                annotated_image, 
                f"Inference: {inference_time:.1f} ms ({fps:.0f} FPS)", 
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            
            # Optionnel : Tu peux décommenter ces lignes pour vérifier visuellement le résultat
            cv2.imshow("BGR Image with YOLO", annotated_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Error in the synchronized callback : {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        return super().destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = FineTuneYoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


