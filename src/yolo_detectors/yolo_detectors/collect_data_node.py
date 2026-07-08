#!/usr/bin/env python3
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;*.warning=false"

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from lancer_interfaces.msg import HumanPoseArray
import cv2
from cv_bridge import CvBridge
import numpy as np

class CollectDataNode(Node):

    def __init__(self):
        super().__init__('collecte_data_node')
        
        self.get_logger().info("*** Nœud de Collecte de Données Lancé ***")
        self.get_logger().info("👉 CLIQUE GAUCHE sur la fenêtre vidéo pour un LANCER.")
        self.get_logger().info("👉 CLIQUE DROIT sur la fenêtre vidéo pour un PARASITE.")

        self.bridge = CvBridge()
        
        # --- CONFIGURATION DU DATASET CONFIGURÉ SUR DEUX DOSSIERS ---
        self.output_dir_lancer = "dataset/lancer"
        self.output_dir_parasite = "dataset/parasite"
        os.makedirs(self.output_dir_lancer, exist_ok=True)
        os.makedirs(self.output_dir_parasite, exist_ok=True)
        
        self.sequence_length = 30  # Nombre de frames par séquence
        
        # Variables d'état pour la capture
        self.is_recording = False
        self.current_sequence = []
        self.counter_lancer = 15
        self.counter_parasite = 0
        self.current_label = 1  # 1 pour lancer, 0 pour parasite
        self.max_sequences = 71  # Sécurité globale

        # Variable pour initialiser la fenêtre OpenCV une seule fois
        self.window_initialized = False
        self.window_name = "Collecte de donnees (Clic G: Lancer | Clic D: Parasite)"

        # --- PARAMÈTRES INTRINSÈQUES DE LA CAMÉRA ORBBEC ---
        self.fx = 747.1493530273438
        self.fy = 746.4198608398438
        self.cx = 635.7774658203125
        self.cy = 363.455322265625

        # --- SYNCHRONISATION DES TOPICS ---
        self.sub_poses = message_filters.Subscriber(self, HumanPoseArray, '/yolo_detected_poses')
        self.sub_objects = message_filters.Subscriber(self, Detection2DArray, '/yolo_detected_objects')
        self.sub_color_img = message_filters.Subscriber(self, Image, '/orbbec_external/color/image_raw')

        # Synchronisation temporelle des 3 flux
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_poses, self.sub_objects, self.sub_color_img],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.synchronized_callback)

    def mouse_click_callback(self, event, x, y, flags, param):
        """Déclenché automatiquement par OpenCV lors d'une action de la souris"""
        if self.is_recording:
            return  # Déjà en train d'enregistrer une séquence

        if event == cv2.EVENT_LBUTTONDOWN:  # CLIC GAUCHE = LANCER
            self.is_recording = True
            self.current_label = 1
            self.current_sequence = []
            self.get_logger().info(f"🎯 [CLIC GAUCHE] Enregistrement LANCER #{self.counter_lancer}...")

        elif event == cv2.EVENT_RBUTTONDOWN:  # CLIC DROIT = PARASITE
            self.is_recording = True
            self.current_label = 0
            self.current_sequence = []
            self.get_logger().info(f"🔄 [CLIC DROIT] Enregistrement PARASITE #{self.counter_parasite}...")

    def map_to_stgcn_frame(self, human_msg, object_msg):
        """Convertit les coordonnées ROS 2 en une frame de graphe (18, 3) en mètres"""
        frame_data = np.zeros((18, 3), dtype=np.float32)
        
        # 1. Extraction du premier humain détecté s'il existe
        if len(human_msg.poses) > 0:
            human = human_msg.poses[0]
            for idx, kp in enumerate(human.keypoints):
                if idx < 17:
                    if human.position_centre_3d.z > 0:
                        z_m = human.position_centre_3d.z
                        frame_data[idx, 0] = ((kp.x - self.cx) * z_m) / self.fx
                        frame_data[idx, 1] = ((kp.y - self.cy) * z_m) / self.fy
                        frame_data[idx, 2] = z_m
                    else:
                        frame_data[idx, :] = [kp.x, kp.y, 0.0]
                        
        # 2. Extraction de l'objet (Peluche) comme le 18ème point (Index 17)
        if len(object_msg.detections) > 0:
            obj_hyp = object_msg.detections[0].results[0]
            frame_data[17, 0] = obj_hyp.pose.pose.position.x
            frame_data[17, 1] = obj_hyp.pose.pose.position.y
            frame_data[17, 2] = obj_hyp.pose.pose.position.z
        else:
            frame_data[17, :] = 0.0
            
        return frame_data

    def synchronized_callback(self, poses_msg, objects_msg, color_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            
            # --- INITIALISATION SÉCURISÉE DE LA FENÊTRE ---
            if not self.window_initialized:
                cv2.namedWindow(self.window_name)
                cv2.setMouseCallback(self.window_name, self.mouse_click_callback)
                self.window_initialized = True

            total_sequences = self.counter_lancer + self.counter_parasite

            if self.is_recording:
                if total_sequences < self.max_sequences:
                    frame_matrix = self.map_to_stgcn_frame(poses_msg, objects_msg)
                    self.current_sequence.append(frame_matrix)
                    
                    # Interface graphique d'enregistrement
                    color_indicator = (0, 0, 255) if self.current_label == 1 else (0, 165, 255)
                    text_indicator = "REC: LANCER" if self.current_label == 1 else "REC: PARASITE"
                    
                    cv2.circle(cv_img, (30, 30), 10, color_indicator, -1)
                    cv2.putText(cv_img, f"{text_indicator} ({len(self.current_sequence)}/30)", (50, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_indicator, 2)

                    if len(self.current_sequence) == self.sequence_length:
                        if self.current_label == 1:
                            filename = os.path.join(self.output_dir_lancer, f"lancer_{self.counter_lancer:03d}.npy")
                            self.counter_lancer += 1
                        else:
                            filename = os.path.join(self.output_dir_parasite, f"parasite_{self.counter_parasite:03d}.npy")
                            self.counter_parasite += 1
                            
                        np.save(filename, np.array(self.current_sequence))
                        self.get_logger().info(f"💾 Sauvegarde réussie : {filename}")
                        self.is_recording = False
                else:
                    cv2.putText(cv_img, "⚠️ Limite max de sequences atteinte.", (50, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.is_recording = False
            else:
                # Affichage du statut au repos (sans spammer le logger)
                cv2.putText(cv_img, f"Pret. Lancer: {self.counter_lancer} | Parasite: {self.counter_parasite}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(self.window_name, cv_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Erreur callback collecte: {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CollectDataNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()