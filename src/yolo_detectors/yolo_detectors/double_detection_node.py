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
from sensor_msgs.msg import Image
import cv2
import random

class DoubleDetectionNode(Node):

    def __init__(self):
        super().__init__('double_detection_node')

        self.bridge = CvBridge()

        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))   
        self.line_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.circle_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

        # Liste des connexions du squelette (identique à yolo_pose_node)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),           
            (3, 5), (4, 6),                           
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  
            (5, 11), (6, 12), (11, 12),               
            (11, 13), (13, 15), (12, 14), (14, 16)    
        ]

        self.sub_image = message_filters.Subscriber(
            self,
            Image,
            '/orbbec_external/color/image_raw'
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
            [self.sub_image, self.sub_fine_tune_yolo, self.sub_yolo_pose],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.synchronized_callback)
        self.get_logger().info("*** Double Detection Visualizer Node Launched ***")


    def synchronized_callback(self, color_msg, yolo_objects, yolo_poses):
        # Process the synchronized messages here
        try: 
            annotated_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')

            for detection in yolo_objects.detections:
                # Récupération des dimensions
                x_center = detection.bbox.center.position.x
                y_center = detection.bbox.center.position.y
                size_x = detection.bbox.size_x
                size_y = detection.bbox.size_y

                # Calcul des coins supérieur gauche et inférieur droit
                x1 = int(x_center - size_x / 2)
                y1 = int(y_center - size_y / 2)
                x2 = int(x_center + size_x / 2)
                y2 = int(y_center + size_y / 2)

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), self.box_color, 2)

                for result in detection.results:
                    label = result.hypothesis.class_id
                    conf = result.hypothesis.score * 100
                    z_dist = result.pose.pose.position.z
                    
                    custom_label = f"{label} ({conf:.1f}%) | Z: {z_dist:.2f}m"
                    cv2.putText(annotated_image, custom_label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.box_color, 2)
                    
            for pose in yolo_poses.poses:
                kpts = pose.keypoints
                
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # --- 3. AFFICHAGE UNIQUE ---
            cv2.imshow("Combined YOLO Detections & Poses", annotated_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Error in the synchronized callback : {e}")

def main(args=None):
    
    rclpy.init(args=args)
    node = DoubleDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
