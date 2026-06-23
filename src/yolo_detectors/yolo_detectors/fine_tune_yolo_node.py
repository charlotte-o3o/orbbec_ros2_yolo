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
from ultralytics import YOLO, YOLOWorld
import message_filters
import random
import numpy as np
import time
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

class FineTuneYoloNode(Node):

    def __init__(self):
        super().__init__('fine_tune_yolo_node')

        self.declare_parameter('model_path',  'weights/alien_plushie_v4.pt')
        self.declare_parameter('confidence',  0.50)
        self.declare_parameter('max_history', 5)
        self.declare_parameter('max_jump',    2.0)

        self.model_path           = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence').value
        self.max_history          = self.get_parameter('max_history').value
        self.max_jump             = self.get_parameter('max_jump').value

        self.fx = 616.0  # Focal length in pixels (x-axis)
        self.fy = 616.0  # Focal length in pixels (y-axis)
        self.cx = 320.0  # Principal point x-coordinate (image center)      
        self.cy = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info = False  # Flag to check if camera info has been received

        self.smoothed_distance = None

        # --- Initialisation du convertisseur CvBridge
        self.bridge = CvBridge()
        self.distance_history: list[float] = []
        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

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

    def synchronized_callback(self, color_msg, depth_msg):
        try:
            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            start_time = time.perf_counter()
            results = self.model(cv_color_image, stream=True, verbose=False, conf=self.confidence_threshold)           
            results = list(results)
            end_time = time.perf_counter() 

            inference_time = (end_time - start_time) * 1000
            fps = 1000.0 / inference_time if inference_time > 0 else 0.0

            annotated_image = cv_color_image.copy()
            boxes = results[0].boxes

            # --- Hauteur et largeur de l'image de profondeur pour éviter les débordements de pixels
            h, w = cv_depth_image.shape[:2]

            msg_array = Detection2DArray()
            msg_array.header = color_msg.header
            
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    confidence = float(box.conf[0]) * 100
                    x1, y1, x2, y2  = map(int, box.xyxy[0])

                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    x_center = max(0, min(x_center, w - 1))
                    y_center = max(0, min(y_center, h - 1))

                    #distance_mm = cv_depth_image[y_center, x_center]

                    # --- Valeur max pour la marge : 0.5 => pixel central uniquement
                    # --- 0.375 => 25% de la BB d'origine
                    margin_x = int((x2 - x1) * 0.375)                         
                    margin_y = int((y2 - y1) * 0.375)     

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

                    if distance > 0 and len(self.distance_history) > 0:
                        if abs(distance - self.distance_history[-1]) > self.max_jump:
                            distance = self.distance_history[-1]  

                    if distance > 0:                              
                        self.distance_history.append(distance)   

                        if len(self.distance_history) > self.max_history:                       
                            self.distance_history.pop(0)  

                        #print(f"Dist. history : {self.distance_history}")
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

                    hyp = ObjectHypothesisWithPose() # Hypothèse sur l'objet détecté et sa distance dans l'id de la pose
                    hyp.hypothesis.class_id = str(label) # Nom ou id de l'objet (alien plushie)
                    hyp.hypothesis.score = confidence / 100.0  # Confiance de la détection (0.0 à 1.0)
                    hyp.pose.pose.position.x = float(x_meters) if x_meters is not None else 0.0
                    hyp.pose.pose.position.y = float(y_meters) if y_meters is not None else 0.0
                    hyp.pose.pose.position.z = float(distance)  # Distance en mètres dans la coord z de la pose

                    
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
            
            #cv2.imshow("BGR Image with YOLO", annotated_image)

            depth_vis = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            #cv2.imshow("Depth Image (sync)", depth_colormap)

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


