# orbbec_ros2_yolo

ROS 2 nodes for running YOLO pose estimation, a fine-tuned YOLO object detector, and an LSTM-based throwing-action classifier on synchronized color/depth streams from an Orbbec camera, with depth-based 3D distance estimation between detected objects and human wrist keypoints.

## Nodes

- **`yolo_pose_node`** — runs YOLO pose estimation on the color stream, overlays skeleton keypoints, and computes the distance to each detected person (from shoulder midpoint) using the synchronized depth image.
- **`fine_tune_yolo_node`** — runs a custom fine-tuned YOLO model (e.g. `alien_plushie_v4.pt`), draws bounding boxes with class/confidence, and computes a smoothed distance estimate per detection using a filtered depth patch. The model used is a custom model fine-tuned by myself, available in the [`weights/`](./weights) folder.
- **`lstm_node`** — fuses synchronized color/depth images, `fine_tune_yolo_node` detections (object distance), and `yolo_pose_node` keypoints (wrist positions) to compute, frame by frame, the 3D distance between the detected object and each wrist. These distances (plus their frame-to-frame deltas, 6 features total) are fed into a sliding window of 20 frames passed to a 2-layer LSTM classifier (`ThrowLSTM`), which detects the throwing action in real time, estimates the initial throw point and velocity, and can log distances to CSV and record annotated video.

`yolo_pose_node` and `fine_tune_yolo_node` both subscribe to:
- `/orbbec_external/color/image_raw`
- `/orbbec_external/depth/image_raw`

and synchronize them with an approximate time synchronizer.

`lstm_node` synchronizes four streams: the color and depth images above, plus the downstream detection topics published by the other two nodes:
- `/orbbec_external/color/image_raw`
- `/orbbec_external/depth/image_raw`
- `/yolo_detected_objects` (from `fine_tune_yolo_node`, `vision_msgs/Detection2DArray`)
- `/yolo_detected_poses` (from `yolo_pose_node`, `lancer_interfaces/HumanPoseArray`)

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
   source /opt/ros/<ros-distro>/setup.bash
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
   # or
   ros2 run <package_name> lstm_node
   ```

   > `lstm_node` depends on `/yolo_detected_objects` and `/yolo_detected_poses`, so `fine_tune_yolo_node` and `yolo_pose_node` must also be running for it to receive synchronized data.

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
| `model_path` | `weights/alien_plushie_v4.pt` | Path to the fine-tuned YOLO model weights |
| `confidence` | `0.50` | Minimum detection confidence |
| `max_history` | `5` | Number of past distance readings used for smoothing |
| `max_jump` | `2.0` | Maximum allowed distance jump (m) between consecutive frames before it's rejected as noise |

### `lstm_node`

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `weights/throw_lstm_v8.pth` | Path to the trained `ThrowLSTM` weights (hardcoded to `~/ros2_orbbec_ws/weights/throw_lstm_v8.pth`) |
| `sequence_length` | `20` | Number of consecutive frames (object/wrist distances + deltas) used as input to the LSTM |
| `input_size` | `6` | Number of input features per frame (object distance, left/right wrist distances, and their frame-to-frame deltas) |
| `num_classes` | `2` | Number of output classes (throw / no throw) |
| `confidence threshold` | `0.85` | Minimum softmax confidence required to classify a frame as "throw" |
| `max_false_frames_allowed` | `10` | Number of consecutive non-throw frames required before the throw state is officially closed (debounce) |
| `cooldown_duration` | `3.0` s | Minimum time between two consecutive throw triggers, to avoid flickering re-triggers |
| `save_distance_mode` | `True` | If enabled, logs per-frame object/wrist distances to a timestamped CSV file under `data/csv_distances/` |
| `record_mode` | `True` | If enabled, records the annotated video stream to `data/captures_videos/` |

> **Note:** this node relies on the synchronized outputs of `yolo_pose_node` (wrist keypoints) and `fine_tune_yolo_node` (object 3D position) as input to the LSTM.

## Repository Structure

```
ros2_orbbec_ws/
├── src/
│   ├── lancer_interfaces/        # Custom ROS 2 message definitions
│   │   ├── msg/
│   │   │   ├── HumanPose.msg
│   │   │   ├── HumanPoseArray.msg
│   │   │   └── Keypoint2D.msg
│   │   └── ...
│   └── yolo_detectors/           # Main package
│       ├── yolo_detectors/
│       │   ├── yolo_pose_node.py
│       │   ├── fine_tune_yolo_node.py
│       │   ├── lstm_node.py
│       ├── config/
│       ├── resource/
│       └── test/
├── weights/                      # Model weights
│   ├── yolo26n-pose.pt
│   ├── alien_plushie_v4.pt
│   └── throw_lstm_v8.pth
├── requirements.txt
├── setup_env.sh
└── cyclonedds_host.xml
```

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

