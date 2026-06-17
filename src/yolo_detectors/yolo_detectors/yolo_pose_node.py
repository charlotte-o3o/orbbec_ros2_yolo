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
import numpy as np
import time
from lancer_interfaces.msg import HumanPoseArray, HumanPose, Keypoint2D

class YoloPoseNode(Node):

    def __init__(self):
        super().__init__('yolo_pose_node')

        self.declare_parameter('model_path',  'weights/yolo26n-pose.pt')
        self.declare_parameter('confidence',  0.50)

        self.model_path           = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence').value

        self.fx = 616.0  # Focal length in pixels (x-axis)
        self.fy = 616.0  # Focal length in pixels (y-axis)
        self.cx = 320.0  # Principal point x-coordinate (image center)      
        self.cy = 240.0  # Principal point y-coordinate (image center)
        self.has_camera_info = False  # Flag to check if camera info has been received

        self.get_logger().info("*** YOLO-Pose Node Launched successfully ***")

        # Initialisation du convertisseur CvBridge
        self.bridge = CvBridge()

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

        self.pub_pose = self.create_publisher(
            HumanPoseArray,
            '/yolo_detected_poses',
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

            start_time = time.perf_counter()
            results = self.model(cv_color_image, stream=True, verbose=False, conf=self.confidence_threshold)           
            results = list(results)
            end_time = time.perf_counter() 

            inference_time = (end_time - start_time) * 1000
            fps = 1000.0 / inference_time if inference_time > 0 else 0.0

            annotated_image = results[0].plot(labels=False)

            boxes = results[0].boxes
            keypoints_object = results[0].keypoints
            num_persons = len(boxes) if boxes is not None else 0

            # Hauteur et largeur de l'image de profondeur pour éviter les débordements de pixels
            h, w = cv_depth_image.shape[:2]

            msg_pose_array = HumanPoseArray()
            msg_pose_array.header = color_msg.header # Copie du timestamp de synchronisation
            
            if boxes is not None and keypoints_object is not None and cv_depth_image is not None:

                kpts = keypoints_object.data.cpu().numpy()

                for i, _ in enumerate(boxes):

                    if i >= len(kpts):
                        continue

                    person_kpts = kpts[i]

                    x_l_shoulder, y_l_shoulder, conf_l = person_kpts[5] # Épaule gauche
                    x_r_shoulder, y_r_shoulder, conf_r = person_kpts[6] # Épaule droite

                    x_mean_shoulder = int((x_l_shoulder + x_r_shoulder) / 2)
                    y_mean_shoulder = int((y_l_shoulder + y_r_shoulder) / 2)

                    if x_mean_shoulder == 0 and y_mean_shoulder == 0:
                        continue

                    x_mean_shoulder = max(0, min(x_mean_shoulder, w - 1))
                    y_mean_shoulder = max(0, min(y_mean_shoulder, h - 1))

                    distance_box_m = cv_depth_image[y_mean_shoulder, x_mean_shoulder] / 1000.0

                    if distance_box_m > 0:
                        x_meters = ((x_mean_shoulder - self.cx) * distance_box_m) / self.fx
                        y_meters = ((y_mean_shoulder - self.cy) * distance_box_m) / self.fy
                    else:
                        x_meters, y_meters = None, None

                    human_pose_msg = HumanPose()
                    human_pose_msg.id = int(i) # ID de la personne détectée

                    # Remplissage du centre 3D de l'humain (X et Y en pixel, Z en mètres)
                    human_pose_msg.position_centre_3d.x = float(x_meters) if x_meters is not None else 0.0
                    human_pose_msg.position_centre_3d.y = float(y_meters) if y_meters is not None else 0.0  
                    human_pose_msg.position_centre_3d.z = float(distance_box_m)

                    # Remplissage de TOUS les 17 keypoints de la personne dans la liste
                    for kp in person_kpts:
                        kp_msg = Keypoint2D()
                        kp_msg.x = float(kp[0])
                        kp_msg.y = float(kp[1])
                        kp_msg.confidence = float(kp[2]) # Score d'invisibilité/visibilité du point
                        human_pose_msg.keypoints.append(kp_msg)

                    msg_pose_array.poses.append(human_pose_msg)

                    if distance_box_m > 0:
                        text_dist = f"Z: {distance_box_m:.2f}m"
                    else:
                        text_dist = "Z: ---"

                    if x_meters is not None and y_meters is not None:
                        text_coord = f"X: {x_meters:.2f}m, Y: {y_meters:.2f}m"
                    else:
                        text_coord = "X: ---, Y: ---"

                    coord_label = f"({text_coord}, {text_dist})"
                    cv2.circle(annotated_image, (x_mean_shoulder, y_mean_shoulder), 5, (0, 0, 255), -1)
                    cv2.putText(annotated_image, coord_label, (x_mean_shoulder + 10, y_mean_shoulder - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
            self.pub_pose.publish(msg_pose_array)

            cv2.putText(
                annotated_image, 
                f"Person(s): {num_persons}", 
                (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2
                )
            cv2.putText(
                annotated_image, 
                f"Inference: {inference_time:.1f} ms ({fps:.0f} FPS)", 
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
                )
            
            cv2.imshow("BGR Image with YOLO-Pose", annotated_image)

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
    node = YoloPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


