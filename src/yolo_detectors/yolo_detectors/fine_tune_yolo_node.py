import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;*.warning=false"

from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
import rclpy
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import message_filters
import random
import numpy as np
import time
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance, Pose, Point

class FineTuneYoloNode(Node):

    def __init__(self):
        super().__init__('fine_tune_yolo_node')

        self.declare_parameter('model_path',  'weights/alien_plushie_v3.pt')
        self.declare_parameter('confidence',  0.50)
        self.declare_parameter('max_history', 5)
        self.declare_parameter('max_jump',    0.5)

        self.model_path           = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence').value
        self.max_history          = self.get_parameter('max_history').value
        self.max_jump             = self.get_parameter('max_jump').value

        self.get_logger().info("*** YOLO Node Launched successfully ***")

        # Initialisation du convertisseur CvBridge
        self.bridge = CvBridge()
        self.distance_history: list[float] = []
        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

        self.get_logger().info(f"Model loading : {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.get_logger().info("Model loaded successfully")


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

            # Hauteur et largeur de l'image de profondeur pour éviter les débordements de pixels
            h, w = cv_depth_image.shape[:2]

            msg_array = Detection2DArray()
            msg_array.header = color_msg.header
            
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    confie = float(box.conf[0]) * 100
                    x1, y1, x2, y2  = map(int, box.xyxy[0])

                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    x_center = max(0, min(x_center, w - 1))
                    y_center = max(0, min(y_center, h - 1))

                    #distance_mm = cv_depth_image[y_center, x_center]

                    margin_x = int((x2 - x1) * 0.35)                         
                    margin_y = int((y2 - y1) * 0.35)     

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

                    detection = Detection2D()
                    detection.bbox.center.position.x = float(x_center)
                    detection.bbox.center.position.y = float(y_center)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)

                    hyp = ObjectHypothesisWithPose() # Hypothèse sur l'objet détecté et sa distance dans l'id de la pose
                    hyp.hypothesis.class_id = str(label) # Nom ou id de l'objet (alien plushie)
                    hyp.hypothesis.score = confie / 100.0  # Confiance de la détection (0.0 à 1.0)
                    hyp.pose.pose.position.z = float(distance)  # Distance en mètres dans la coord z de la pose

                    detection.results.append(hyp)
                    msg_array.detections.append(detection)
                    
                    if distance == 0:
                        text_dist = "---"
                    else:
                        text_dist = f"{distance:.2f}m"

                    custom_label = f"{label} ({confie:.1f}%) : {text_dist}"
                    cv2.rectangle(annotated_image, (x1,y1), (x2,y2), self.box_color, 2)
                    cv2.circle(annotated_image, (x_center, y_center), 4, (0, 0, 255), -1)
                    cv2.putText(annotated_image, custom_label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.box_color, 2)
                    
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
            
            cv2.imshow("BGR Image with YOLO", annotated_image)

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


