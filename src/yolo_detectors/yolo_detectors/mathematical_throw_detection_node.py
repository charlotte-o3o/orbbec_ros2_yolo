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
import cv2
import random
import csv
import time

class MathematicalThrowDetectionNode(Node):

    def __init__(self):
        super().__init__('mathematical_throw_detection_node')

        self.bridge = CvBridge()

        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))   
        self.line_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.circle_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.right_wrist_color = (241, 255, 81)
        self.left_wrist_color = (218, 110, 255)

        self.fx = 616.0  # Focal length in pixels (x-axis)
        self.fy = 616.0  # Focal length in pixels (y-axis)
        self.cx = 320.0  # Principal point x-coordinate (image center)      
        self.cy = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info = False  # Flag to check if camera info has been received

        self.distance_log_dir = os.path.join(os.path.expanduser("~"), "ros2_orbbec_ws", "data")
        if not os.path.exists(self.distance_log_dir):
            os.makedirs(self.distance_log_dir)
            self.get_logger().info(f"Directory created: {self.distance_log_dir}")

        self.timestamp_csv = time.strftime("%Y-%m-%d_%H-%M-%S")             
        self.csv_path = os.path.join(self.distance_log_dir,                   
                                f"distances_{self.timestamp_csv}.csv") 
        
        self.start_time = time.time()

        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'dist_object_m', 'distance_left_wrist_m', 'distance_right_wrist_m'])

        self.get_logger().info(f"CSV file created : {self.csv_path}")                

        # Liste des connexions du squelette (identique à yolo_pose_node)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),           
            (3, 5), (4, 6),                           
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  
            (5, 11), (6, 12), (11, 12),               
            (11, 13), (13, 15), (12, 14), (14, 16)    
        ]

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

        self.get_logger().info("*** Mathematical Throw Detection Node Launched ***")

    def camera_info_callback(self, msg: CameraInfo):

        if not self.has_camera_info:
            self.fx = msg.k[0]  # Focal length in pixels (x-axis)
            self.fy = msg.k[4]  # Focal length in pixels (y-axis)
            self.cx = msg.k[2]  # Principal point x-coordinate (image center)      
            self.cy = msg.k[5]  # Principal point y-coordinate (image center)
            self.has_camera_info = True

            self.get_logger().info(f"Camera info received: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

            self.destroy_subscription(self.sub_info)  # Unsubscribe after receiving camera info

    def synchronized_callback(self, color_msg, depth_msg, yolo_objects, yolo_poses):
        # Process the synchronized messages here
        try: 
            annotated_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

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
                    
                    custom_label = f"{label} ({conf:.1f}%) | Z: {z_dist:.2f}m"
                    cv2.putText(annotated_image, custom_label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.box_color, 2)
                    
            for pose in yolo_poses.poses:
                kpts = pose.keypoints

                # Vérification de la validité de l'index 9 (Poignet droit)
                if len(kpts) > 9 and kpts[9].confidence > 0.7 and cv_depth_image is not None:
                    lw_x = int(kpts[9].x)
                    lw_y = int(kpts[9].y)
                    lw_pixels = (lw_x, lw_y)

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
                    rw_pixels = (rw_x, rw_y)

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

                for pt1, pt2 in hold_connections:
                    cv2.line(annotated_image, pt1, pt2, (0, 255, 255), 2)

            current_timestamp = time.time() - self.start_time
            object_z_csv = ""
            left_dist_csv = ""
            right_dist_csv = ""

            if object_center_meters is not None and None not in object_center_meters:
                object_z_csv = round(object_center_meters[2], 4)

                if lw_meters is not None and None not in lw_meters:
                    left_dist_m = ((object_center_meters[0] - lw_meters[0]) ** 2 + 
                                   (object_center_meters[1] - lw_meters[1]) ** 2 + 
                                   (object_center_meters[2] - lw_meters[2]) ** 2) ** 0.5
                    self.get_logger().info(f"Distance from object to LEFT wrist: {left_dist_m:.3f} m")

                    left_dist_csv = round(left_dist_m, 4)

                    left_line_center = ((object_center_meters[0] + lw_meters[0]) / 2,
                                        (object_center_meters[1] + lw_meters[1]) / 2)
                    cv2.circle(annotated_image, (int(left_line_center[0]), int(left_line_center[1])), 5, (0, 255, 255), -1)
                    cv2.putText(annotated_image, f"{left_dist_m:.2f} m", (int(left_line_center[0]) + 1, int(left_line_center[1]) + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if rw_meters is not None and None not in rw_meters:
                    right_dist_m = ((object_center_meters[0] - rw_meters[0]) ** 2 + 
                                    (object_center_meters[1] - rw_meters[1]) ** 2 + 
                                    (object_center_meters[2] - rw_meters[2]) ** 2) ** 0.5
                    self.get_logger().info(f"Distance from object to RIGHT wrist: {right_dist_m:.3f} m")

                    right_dist_csv = round(right_dist_m, 4)

                    right_line_center = ((object_center_meters[0] + rw_meters[0]) / 2,
                                         (object_center_meters[1] + rw_meters[1]) / 2)
                    cv2.circle(annotated_image, (int(right_line_center[0]), int(right_line_center[1])), 5, (0, 255, 255), -1)
                    cv2.putText(annotated_image, f"{right_dist_m:.2f} m", (int(right_line_center[0]) + 1, int(right_line_center[1]) + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([round(current_timestamp, 4), object_z_csv, left_dist_csv, right_dist_csv])

            cv2.imshow("Combined YOLO Detections & Poses", annotated_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Error in the synchronized callback : {e}")

def main(args=None):
    
    rclpy.init(args=args)
    node = MathematicalThrowDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
