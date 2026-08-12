from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

config_file = os.path.join(
    get_package_share_directory('yolo_detectors'),
    'config',
    'speed_det_params.yaml'
)

def generate_launch_description():
    return LaunchDescription([

        # --- Détection d'objets (publie /yolo_detected_objects) ---
        Node(
            package='yolo_detectors',
            executable='fine_tune_yolo',
            name='fine_tune_yolo_node',
            output='screen',
            parameters=[config_file],
        ),

        # --- Détection de pose (publie /yolo_detected_poses) ---
        # Plus utilisée par la détection de lancer : le nœud reste disponible
        # (executable 'yolo_pose') mais n'est plus lancé.
        # Node(
        #     package='yolo_detectors',
        #     executable='yolo_pose',
        #     name='yolo_pose_node',
        #     output='screen',
        #     parameters=[config_file],
        # ),

        # --- Détection de lancer basée sur la vitesse : dépend du nœud ci-dessus ---
        # ---   (consomme /yolo_detected_objects ; aucune image sauf en record)    ---
        Node(
            package='yolo_detectors',
            executable='speed_det',
            name='speed_det_node',
            output='screen',
            parameters=[config_file],
        ),
    ])