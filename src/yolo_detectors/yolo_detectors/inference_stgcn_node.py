#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from lancer_interfaces.msg import HumanPoseArray
import cv2
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. ARCHITECTURE DU MODÈLE (DOIT ÊTRE IDENTIQUE À L'ENTRAÎNEMENT)
# =====================================================================
def get_skeleton_adjacency_matrix():
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (9, 17), (10, 17)
    ]
    A = np.zeros((18, 18), dtype=np.float32)
    for i, j in connections:
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    return torch.tensor(A, dtype=torch.float32)

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        super().__init__()
        self.register_buffer('A', adjacency_matrix)
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.einsum('btnd,nj->btjd', x, self.A) 
        out = self.linear(out)
        return self.dropout(F.relu(out))

class SimpleSTGCN(nn.Module):
    def __init__(self, adjacency_matrix, num_classes=2):
        super().__init__()
        self.gcn1 = SpatialGraphConv(3, 64, adjacency_matrix)
        self.gcn2 = SpatialGraphConv(64, 128, adjacency_matrix)
        self.temporal_conv = nn.Conv1d(in_channels=128*18, out_channels=256, kernel_size=5, stride=2)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        B, T, N, C = x.shape
        x = x.view(B, T, N * C).transpose(1, 2)
        x = F.relu(self.temporal_conv(x))
        x = torch.mean(x, dim=2)
        x = self.fc_dropout(x)
        return self.fc(x)

# =====================================================================
# 2. NOEUD ROS 2 D'INFÉRENCE
# =====================================================================
class InferenceSTGCNDNode(Node):

    def __init__(self):
        super().__init__('inference_stgcn_node')
        self.get_logger().info("*** Nœud d'Inférence ST-GCN Lancé ***")

        self.bridge = CvBridge()
        
        # --- CHARGEMENT DU MODÈLE ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matrix = get_skeleton_adjacency_matrix()
        self.model = SimpleSTGCN(matrix, num_classes=2).to(self.device)
        
        # Mettre le bon chemin vers ton fichier de poids .pt
        model_path = "weights/stgcn_lancer.pt" 
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f"✅ Modèle PyTorch chargé avec succès depuis {model_path}")
        else:
            self.get_logger().error(f"❌ Fichier {model_path} introuvable ! Place-le à la racine de ton workspace.")
            return

        # --- FENÊTRE GLISSANTE DE DONNÉES ---
        self.frame_buffer = []
        self.buffer_size = 30
        
        # Paramètres de la caméra pour la déprojection intrinsèque
        self.fx, self.fy = 747.1493530273438, 746.4198608398438
        self.cx, self.cy = 635.7774658203125, 363.455322265625

        # Variables d'affichage temps réel
        self.current_prediction = "En attente..."
        self.current_confidence = 0.0
        self.detection_freeze_counter = 0 # Pour laisser le texte affiché quelques frames

        # --- SYNCHRONISATION DES TOPICS ---
        self.sub_poses = message_filters.Subscriber(self, HumanPoseArray, '/yolo_detected_poses')
        self.sub_objects = message_filters.Subscriber(self, Detection2DArray, '/yolo_detected_objects')
        self.sub_color_img = message_filters.Subscriber(self, Image, '/orbbec_external/color/image_raw')

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_poses, self.sub_objects, self.sub_color_img], queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self.synchronized_callback)

    def map_to_stgcn_frame(self, human_msg, object_msg):
        """Même extraction géométrique qu'à la collecte"""
        frame_data = np.zeros((18, 3), dtype=np.float32)
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
                        
        if len(object_msg.detections) > 0:
            obj_hyp = object_msg.detections[0].results[0]
            frame_data[17, 0] = obj_hyp.pose.pose.position.x
            frame_data[17, 1] = obj_hyp.pose.pose.position.y
            frame_data[17, 2] = obj_hyp.pose.pose.position.z
        return frame_data

    def process_and_normalize_buffer(self):
        """Piste A : Même logique de centrage sur le bassin"""
        buf = np.array(self.frame_buffer, dtype=np.float32) # Shape (30, 18, 3)
        for t in range(buf.shape[0]):
            hip_left = buf[t, 11, :]
            hip_right = buf[t, 12, :]
            if np.any(hip_left) and np.any(hip_right):
                bassin_3d = (hip_left + hip_right) / 2.0
            else:
                bassin_3d = buf[t, 0, :]
            mask = np.any(buf[t, :, :], axis=1, keepdims=True)
            buf[t, :, :] = buf[t, :, :] - (bassin_3d * mask)
        return buf

    def synchronized_callback(self, poses_msg, objects_msg, color_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            
            # 1. Extraire la frame courante et l'ajouter au buffer glissant
            current_frame = self.map_to_stgcn_frame(poses_msg, objects_msg)
            self.frame_buffer.append(current_frame)

            # Garder uniquement les 30 dernières frames
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)

            # 2. Si le buffer est plein, on lance la prédiction de l'IA
            if len(self.frame_buffer) == self.buffer_size:

                peluche_visible = sum(1 for f in self.frame_buffer if np.any(f[17, :]))

                if peluche_visible < 2:  # Si la peluche n'est pas visible dans au moins 2 frames   
                    self.current_prediction = "🔄 Peluche non visible"
                    self.current_confidence = 0.0
                    
                else:
                    self.current_prediction = "🎯 Peluche détectée"
                    # Appliquer la normalisation relative
                    input_data = self.process_and_normalize_buffer()
                    # Convertir au format Tenseur PyTorch (Batch=1, Time=30, Nodes=18, Channels=3)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        confidence, predicted_class = torch.max(probabilities, dim=1)
                        
                        prob_val = confidence.item()
                        class_idx = predicted_class.item()

                    # Seuil de sécurité : On ne valide l'action que si l'IA est sûre à plus de 80%
                    if prob_val > 0.70:
                        if class_idx == 1:
                            self.current_prediction = "🎯 LANCER DETECTE !"
                            self.detection_freeze_counter = 15 # Reste affiché pendant 15 frames
                            # 💡 C'est ICI que tu pourras publier un message ROS2 pour ordonner au robot de fermer la pince
                        else:
                            if self.detection_freeze_counter <= 0:
                                self.current_prediction = "🔄 Parasite / Mouvement"
                        self.current_confidence = prob_val * 100

            if self.detection_freeze_counter > 0:
                self.detection_freeze_counter -= 1

            # --- AFFICHAGE IHM EN DIRECT ---
            color = (0, 255, 0) if "LANCER" in self.current_prediction else (255, 255, 255)
            if "Parasite" in self.current_prediction: color = (0, 165, 255)
            
            cv2.putText(cv_img, f"Action : {self.current_prediction}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(cv_img, f"Confiance : {self.current_confidence:.1f}%", (30, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Dessiner une jauge visuelle de remplissage du buffer
            cv2.rectangle(cv_img, (30, 100), (30 + int(len(self.frame_buffer)*6.6), 110), (255, 0, 0), -1)

            cv2.imshow("Inference Live ST-GCN", cv_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Erreur Inference Callback: {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = InferenceSTGCNDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()