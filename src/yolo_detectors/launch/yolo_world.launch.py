from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

config_file = os.path.join(
    get_package_share_directory('yolo_detectors'),
    'config',
    'yolo_world_params.yaml'
)

def generate_launch_description():
    return LaunchDescription([

        # --- Détection d'objets (publie /yolo_detected_objects) ---
        Node(
            package='yolo_detectors',
            executable='yolo_world',
            name='yolo_world_node',
            output='screen',
            parameters=[config_file],
        ),

        Node(
            package='yolo_detectors',
            executable='yolo_pose',
            name='yolo_pose_node',
            output='screen',
            parameters=[config_file],
        ),

        Node(
            package='yolo_detectors',
            executable='optical_tracking',
            name='optical_tracking_node',
            output='screen',
            parameters=[config_file],
        ),

    ])