from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # --- Arguments pour surcharger les modèles/seuils sans toucher au code ---
        DeclareLaunchArgument(
            'pose_model_path', default_value='weights/yolo26n-pose.pt',
            description="Chemin du modèle YOLO-Pose"
        ),
        DeclareLaunchArgument(
            'pose_confidence', default_value='0.50',
            description="Seuil de confiance YOLO-Pose"
        ),
        DeclareLaunchArgument(
            'detect_model_path', default_value='weights/alien_plushie_v5.pt',
            description="Chemin du modèle de détection d'objets (fine_tune_yolo)"
        ),
        DeclareLaunchArgument(
            'detect_confidence', default_value='0.50',
            description="Seuil de confiance détection d'objets"
        ),

        # --- Détection d'objets (publie /yolo_detected_objects) ---
        Node(
            package='yolo_detectors',
            executable='fine_tune_yolo',
            name='fine_tune_yolo_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('detect_model_path'),
                'confidence': LaunchConfiguration('detect_confidence'),
            }],
        ),

        # --- Détection de pose (publie /yolo_detected_poses) ---
        Node(
            package='yolo_detectors',
            executable='yolo_pose',
            name='yolo_pose_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('pose_model_path'),
                'confidence': LaunchConfiguration('pose_confidence'),
            }],
        ),

        # --- Détection de lancer basée sur la vitesse : dépend des 2 nœuds ci-dessus
        #     (synchronise image + depth + detections + poses) ---
        Node(
            package='yolo_detectors',
            executable='speed_det',
            name='speed_det_node',
            output='screen',
        ),
    ])