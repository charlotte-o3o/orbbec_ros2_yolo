from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

config_file = os.path.join(
    get_package_share_directory('yolo_detectors'),
    'config',
    'params.yaml'
)

def generate_launch_description():
    return LaunchDescription([

        # --- Détection d'objets (publie /yolo_detected_objects) ---
        Node(
            package='yolo_detectors',
            executable='fine_tune_yolo',
            name='fine_tune_yolo_node',
            output='screen',
            parameters=[{
                config_file,
            }],
        ),

        # --- Détection de pose (publie /yolo_detected_poses) ---
        Node(
            package='yolo_detectors',
            executable='yolo_pose',
            name='yolo_pose_node',
            output='screen',
            parameters=[{
                config_file,
            }],
        ),

        # --- Détection de lancer basée sur la vitesse : dépend des 2 nœuds ci-dessus ---
        # ---           (synchronise image + depth + detections + poses)              ---
        Node(
            package='yolo_detectors',
            executable='speed_det',
            name='speed_det_node',
            output='screen',
            parameters=[{
                config_file,
            }],
        ),
    ])