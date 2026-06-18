from setuptools import find_packages, setup

package_name = 'yolo_detectors'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Charlotte Di Loreto',
    maintainer_email='charlotte@example.com',
    description='YOLO26 object detection and pose estimation with Orbbec Femto Bolt',
    license='MIT',
    entry_points={
        'console_scripts': [
            'test_subscriber = yolo_detectors.test_subscriber_node:main',
            'fine_tune_yolo = yolo_detectors.fine_tune_yolo_node:main',
            'yolo_pose = yolo_detectors.yolo_pose_node:main',
            'yolo_world = yolo_detectors.yolo_world_node:main',
            'collect_data = yolo_detectors.collect_data_node:main',
            'inference_stgcn = yolo_detectors.inference_stgcn_node:main',
            'double_detection = yolo_detectors.double_detection_node:main',
        ],
    },
)
