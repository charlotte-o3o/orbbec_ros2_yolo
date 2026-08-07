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
from ultralytics import YOLOWorld
import message_filters
import random
import numpy as np
import time
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

class YoloWorldNode(Node):

    def __init__(self):
        super().__init__('yolo_world_node')

        default_classes = [
            "plastic bottle",
            "cardboard_box",
            "alien_plushie",
            "juice_bottle",
            "bottle",
            "book",
            "person",
            "computer",
            "laptop",
            "banana",
            "ball",
            "football_ball",
            "soccer_ball",
            "white_box",
            "white_cardboard_box",
            "headphones",
            "orange_strap",
        ]

        self.model_path           = self.declare_parameter('model_path',     'weights/yolov8s-world.pt').value
        self.confidence_threshold = self.declare_parameter('confidence',     0.50).value
        self.max_history          = self.declare_parameter('max_history',    5).value
        self.max_jump             = self.declare_parameter('max_jump',       0.5).value
        self.bb_margin            = self.declare_parameter('bb_margin',      0.46).value
        self.max_velocity         = self.declare_parameter('max_velocity',   5.0).value
        self.declare_parameter('custom_classes', default_classes)
        self.custom_classes       = list(self.get_parameter('custom_classes').value)
        

        self.fx = 616.0  # Focal length in pixels (x-axis)
        self.fy = 616.0  # Focal length in pixels (y-axis)
        self.cx = 320.0  # Principal point x-coordinate (image center)
        self.cy = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info = False  # Flag to check if camera info has been received

        self.get_logger().info("*** YOLO Node Launched successfully ***")

        # Initialisation du convertisseur CvBridge
        self.bridge = CvBridge()
        self.distance_history: list[float] = []
        self.box_color = (255, 0, 0)
        self.consecutive_jumps = 0

        self.last_valid_time = self.get_clock().now()

        self.get_logger().info(f"Model loading : {self.model_path}...")
        self.model = YOLOWorld(self.model_path)
        self.get_logger().info("Model loaded successfully")
        self.model.set_classes(self.custom_classes)
        self.get_logger().info(f"Defined classes ({len(self.custom_classes)}) : {self.custom_classes}")

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

        # Config du synchroniseur temporel approximatif
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth],
            queue_size=10,
            slop=0.05
        )

        # Fonction callback pour les deux messages synchronisés
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

    def synchronized_callback(self, color_msg, depth_msg):
        try:
            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # 1. Récupération des dimensions d'origine
            h_orig, w_orig = cv_color_image.shape[:2]

            # 2. Rotation à 90° (Sens Horaire / ROTATE_90_CLOCKWISE)
            # Change selon ton montage : cv2.ROTATE_90_CLOCKWISE ou cv2.ROTATE_90_COUNTERCLOCKWISE
            rot_code = cv2.ROTATE_90_COUNTERCLOCKWISE 
            cv_color_rot = cv2.rotate(cv_color_image, rot_code)
            cv_depth_rot = cv2.rotate(cv_depth_image, rot_code)

            start_time = time.perf_counter()
            # Inférence YOLO sur l'image pivotée
            results = self.model(cv_color_rot, stream=True, verbose=False, conf=self.confidence_threshold)           
            results = list(results)
            end_time = time.perf_counter() 

            inference_time = (end_time - start_time) * 1000
            fps = 1000.0 / inference_time if inference_time > 0 else 0.0

            annotated_image = cv_color_image.copy()  # On annote sur l'image d'origine
            boxes = results[0].boxes

            msg_array = Detection2DArray()
            msg_array.header = color_msg.header
            
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    confidence = float(box.conf[0]) * 100
                    
                    # Coordonnées dans l'image pivotée (ROTATED)
                    x1_r, y1_r, x2_r, y2_r = map(int, box.xyxy[0])
                    x_center_r = int((x1_r + x2_r) / 2)
                    y_center_r = int((y1_r + y2_r) / 2)

                    # --- Extraction de la profondeur sur la carte de profondeur pivotée ---
                    margin_x = int((x2_r - x1_r) * self.bb_margin)                         
                    margin_y = int((y2_r - y1_r) * self.bb_margin)     

                    y1_p = max(0, y1_r + margin_y)  
                    y2_p = min(cv_depth_rot.shape[0], y2_r - margin_y)    
                    x1_p = max(0, x1_r + margin_x)                             
                    x2_p = min(cv_depth_rot.shape[1], x2_r - margin_x)

                    patch = cv_depth_rot[y1_p:y2_p, x1_p:x2_p]                 
                    valid = patch[patch > 0]

                    if len(valid) > 0:   
                        median_val = float(np.median(valid))
                        std_val = float(np.std(valid))
                        filtered = valid[np.abs(valid - median_val) < std_val] 
                        distance = float(np.median(filtered)) / 1000.0 if len(filtered) > 0 else median_val / 1000.0
                    else:                                                              
                        distance = 0.0  

                    # Filtre temporel de Z
                    current_time = self.get_clock().now()
                    dt = (current_time - self.last_valid_time).nanoseconds / 1e9
                    dynamic_max_jump = max(0.20, self.max_velocity * dt)

                    if distance > 0 and len(self.distance_history) > 0:
                        if abs(distance - self.distance_history[-1]) > dynamic_max_jump:
                            self.consecutive_jumps += 1
                            if self.consecutive_jumps > 5:
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

                    # 3. Remapping des coordonnées vers le repère d'origine (ORIGINAL)
                    if rot_code == cv2.ROTATE_90_CLOCKWISE:
                        x_center_orig = y_center_r
                        y_center_orig = h_orig - 1 - x_center_r
                        size_x_orig = float(y2_r - y1_r)
                        size_y_orig = float(x2_r - x1_r)
                        
                        x1_orig = y1_r
                        y1_orig = h_orig - 1 - x2_r
                        x2_orig = y2_r
                        y2_orig = h_orig - 1 - x1_r
                    elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                        x_center_orig = w_orig - 1 - y_center_r
                        y_center_orig = x_center_r
                        size_x_orig = float(y2_r - y1_r)
                        size_y_orig = float(x2_r - x1_r)

                        x1_orig = w_orig - 1 - y2_r
                        y1_orig = x1_r
                        x2_orig = w_orig - 1 - y1_r
                        y2_orig = x2_r

                    # Calcul de la position 3D dans le repère caméra natif
                    if distance > 0:
                        x_meters = ((x_center_orig - self.cx) * distance) / self.fx
                        y_meters = ((y_center_orig - self.cy) * distance) / self.fy
                        text_coord = f"X: {x_meters:.2f}m, Y: {y_meters:.2f}m, Z: {distance:.2f}m"
                    else:
                        x_meters, y_meters = None, None
                        text_coord = "X: ---, Y: ---, Z: ---"

                    # Remplissage du message ROS 2 (Coordonnées dans le repère ORIGINAL)
                    detection = Detection2D()
                    detection.bbox.center.position.x = float(x_center_orig)
                    detection.bbox.center.position.y = float(y_center_orig)
                    detection.bbox.size_x = size_x_orig
                    detection.bbox.size_y = size_y_orig

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(label)
                    hyp.hypothesis.score = confidence / 100.0
                    hyp.pose.pose.position.x = float(x_meters) if x_meters is not None else 0.0
                    hyp.pose.pose.position.y = float(y_meters) if y_meters is not None else 0.0
                    hyp.pose.pose.position.z = float(distance)

                    detection.results.append(hyp)
                    msg_array.detections.append(detection)
                
                    # Tracé sur l'image d'origine pour la vérification visuelle
                    custom_label = f"{label} ({confidence:.1f}%) | {text_coord}"
                    cv2.rectangle(annotated_image, (x1_orig, y1_orig), (x2_orig, y2_orig), self.box_color, 2)
                    cv2.circle(annotated_image, (x_center_orig, y_center_orig), 4, (0, 0, 255), -1)
                    cv2.putText(annotated_image, custom_label, (x1_orig, y1_orig - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)
                    
            self.pub_detections.publish(msg_array)

            cv2.putText(
                annotated_image, 
                f"Inference: {inference_time:.1f} ms ({fps:.0f} FPS)", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            
            cv2.imshow("BGR Image with YOLO", annotated_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Error in the synchronized callback : {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        return super().destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


