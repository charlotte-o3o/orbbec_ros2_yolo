# orbbec_ros2_yolo26

ROS 2 nodes for running YOLO26 pose estimation and a fine-tuned YOLO26 detector on synchronized color/depth streams from an Orbbec camera, with depth-based distance estimation.

## Nodes

- **`yolo_pose_node`** — runs YOLO26 pose estimation on the color stream, overlays skeleton keypoints, and computes the distance to each detected person (from shoulder midpoint) using the synchronized depth image.
- **`fine_tune_yolo_node`** — runs a custom fine-tuned YOLO26 model (e.g. `alien_plushie_v3.pt`), draws bounding boxes with class/confidence, and computes a smoothed distance estimate per detection using a filtered depth patch. The model used is a custom model fine-tuned by myself, available in the [`weights/`](./weights) folder.

Both nodes subscribe to:
- `/orbbec_external/color/image_raw`
- `/orbbec_external/depth/image_raw`

and synchronize them with an approximate time synchronizer.

## Camera Driver

The Orbbec camera is run through a dockerized ROS 2 wrapper, available here: [hucebot/orbbec_ros2](https://github.com/hucebot/orbbec_ros2). Credit to the [Hucebot](https://github.com/hucebot) team for this driver.

## Configuration

You will need **two open terminals**.

### Terminal 1 — `/orbbec_ros2`

1. (Optional) Export the environment variables listed in the table below.
2. Deploy the camera driver:
   ```bash
   make deploy
   ```

### Terminal 2 — `/ros2_orbbec_ws`

1. Source the ROS 2 environment:
   ```bash
   source /opt/ros/<ros_distro>/setup.bash
   ```
2. (Optional) Export the environment variables listed in the table below.
3. Verify the topics are being published:
   ```bash
   ros2 topic list
   ros2 topic hz <topic_name>
   ros2 topic echo <topic_name>
   ```
4. Build the workspace:
   ```bash
   colcon build
   ```
5. Launch the desired node:
   ```bash
   ros2 run <package_name> yolo_pose_node
   # or
   ros2 run <package_name> fine_tune_yolo_node
   ```

### Environment variables

| Variable | `~/orbbec_ros2` | `~/ros2_orbbec_ws` |
|---|---|---|
| `ROS_DOMAIN_ID` | `2` | `2` |
| `CYCLONEDDS_URI` | `<CycloneDDS><Domain><General><Interfaces><NetworkInterface name='lo'/></Interfaces><AllowMulticast>false</AllowMulticast></General></Domain></CycloneDDS>` | *null* |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | *null* |

### Troubleshooting

If the topics stop being published, stop the Docker container, unplug the camera, then plug it back in and relaunch the Docker container.

## Node parameters

### `yolo_pose_node`

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `weights/yolo26n-pose.pt` | Path to the YOLO pose model weights |
| `confidence` | `0.50` | Minimum detection confidence |

### `fine_tune_yolo_node`

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `weights/alien_plushie_v3.pt` | Path to the fine-tuned YOLO model weights |
| `confidence` | `0.50` | Minimum detection confidence |
| `max_history` | `5` | Number of past distance readings used for smoothing |
| `max_jump` | `0.5` | Maximum allowed distance jump (m) between consecutive frames before it's rejected as noise |

## Requirements

- ROS 2
- `cv_bridge`, `message_filters`
```bash
sudo apt install ros-<distro>-cv-bridge
sudo apt install ros-<distro>-message-filters
```
- OpenCV (`opencv-python`)
- `ultralytics` (YOLO)
- An Orbbec camera publishing synchronized color/depth image topics

All Python dependencies with their exact required versions are listed in [`requirements.txt`](./requirements.txt). Install them with:

```bash
pip install -r requirements.txt
```