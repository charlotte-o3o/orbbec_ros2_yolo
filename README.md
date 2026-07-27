# orbbec_ros2_yolo

ROS 2 nodes for running YOLO pose estimation, a fine-tuned YOLO object detector, and a speed-based throwing-action detector on synchronized color/depth streams from an Orbbec camera, with depth-based 3D distance estimation between detected objects and human wrist keypoints.

## Nodes

- **`yolo_pose_node`** — runs YOLO pose estimation on the color stream, overlays skeleton keypoints, and computes the distance to each detected person (from shoulder midpoint) using the synchronized depth image.
- **`fine_tune_yolo_node`** — runs a custom fine-tuned YOLO model (e.g. `alien_plushie_v4.pt`), draws bounding boxes with class/confidence, and computes a smoothed distance estimate per detection using a filtered depth patch. The model used is a custom model fine-tuned by myself, available in the [`weights/`](./weights) folder.
- **`speed_det_node`** — fuses synchronized color/depth images and `fine_tune_yolo_node` detections (object 3D position) to compute, frame by frame, the object's position in a tilt-corrected world frame and its instantaneous 3D speed. A throw is detected purely from kinematics: the object's speed must stay above a threshold, with a negative velocity component towards the camera, for several consecutive frames. Once a throw is confirmed, the node tracks the observed trajectory and, from a small history of points, fits a linear (constant-velocity) model on X/Z and a gravity-corrected model on Y to predict the future trajectory and the estimated landing point/time. It can also log distances/speed to CSV, record annotated video, and plot the observed vs. predicted trajectory at the end of each throw.

`yolo_pose_node` and `fine_tune_yolo_node` both subscribe to:
- `/orbbec_external/color/image_raw`
- `/orbbec_external/depth/image_raw`

and synchronize them with an approximate time synchronizer.

`speed_det_node` synchronizes four streams: the color and depth images above, plus the downstream detection topics published by the other two nodes:
- `/orbbec_external/color/image_raw`
- `/orbbec_external/depth/image_raw`
- `/yolo_detected_objects` (from `fine_tune_yolo_node`, `vision_msgs/Detection2DArray`)
- `/yolo_detected_poses` (from `yolo_pose_node`, `lancer_interfaces/HumanPoseArray`)

It also subscribes once to `/orbbec_external/color/camera_info` to retrieve the real camera intrinsics (fx, fy, cx, cy), then unsubscribes.

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
   ros2 run <package_name> speed_det_node
   ```

   > `speed_det_node` depends on `/yolo_detected_objects` and `/yolo_detected_poses`, so `fine_tune_yolo_node` and `yolo_pose_node` must also be running for it to receive synchronized data.

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

### `speed_det_node`

> **Note:** this node relies on the detections published by `fine_tune_yolo_node` (object 3D position) and, through the synchronized message filter, on `yolo_pose_node` (wrist keypoints), even though the throw itself is detected from object speed only.

**Detection thresholds / debounce**

| Parameter | Default | Description |
|---|---|---|
| `speed_threshold` | `2.0` m/s | Minimum object speed required to start considering a throw |
| `throw_confirm_frames` | `3` | Number of consecutive frames above `speed_threshold` (with a negative velocity component towards the camera) required to confirm a throw |
| `max_false_frames_allowed` | `10` | Number of consecutive frames below threshold required before the throw state is officially closed (debounce) |
| `cooldown_duration` | `8.0` s | Minimum time between two consecutive throw triggers, to avoid flickering re-triggers |
| `startup_grace_period` | `8.0` s | Warm-up delay after the node starts, during which no throw can be detected |

**Smoothing / kinematics**

| Parameter | Default | Description |
|---|---|---|
| `alpha_smooth` | `0.3` | Exponential moving average factor used to smooth the raw object 3D position before computing speed |
| `camera_tilt_deg` | `0.0` | Camera tilt angle (degrees), used to project the smoothed camera-frame position into a gravity-aligned world frame |
| `min_valid_dt` | `0.01` s | Minimum time delta between two frames below which the speed computation is skipped (avoids noise blow-up) |
| `max_valid_speed` | `25.0` m/s | Maximum physically plausible speed for a hand throw; higher values reject the trajectory update as noise |

**Trajectory tracking & landing prediction**

| Parameter | Default | Description |
|---|---|---|
| `trajectory_tracking_duration` | `5.0` s | Maximum duration a trajectory is tracked after a throw is confirmed, before timing out |
| `history_sampling_interval` | `0.10` s | Minimum time between two archived trajectory predictions (used for the end-of-throw plot, independent of FPS) |
| `landing_z_threshold` | `0.1` m | Distance to the camera below which the object is considered to have "arrived" |
| `min_time_before_landing_check` | `0.15` s | Minimum tracking time before the landing check is allowed to trigger |
| `landing_confirm_frames` | `3` | Number of consecutive frames under `landing_z_threshold` required to confirm landing |

**Physical validity guards**

| Parameter | Default | Description |
|---|---|---|
| `min_valid_z` / `max_valid_z` | `0.1` / `10.0` m | Valid depth range for the object; positions outside are rejected |
| `max_valid_xy` | `5.0` m | Maximum plausible lateral/vertical position in the room |
| `max_predicted_height` / `max_predicted_fall` | `2.0` / `-2.0` m | Vertical bounds (world Y) beyond which a predicted trajectory point is considered non-physical and tracking stops |

**Logging / output**

| Parameter | Default | Description |
|---|---|---|
| `save_distance_mode` | `True` | If enabled, logs per-frame object position (X/Y/Z, world frame) and speed to a timestamped CSV file under `data/speed_detection/csv_distances/` |
| `record_mode` | `True` | If enabled, records the annotated video stream to `data/captures_videos/` (suffix `_speed_det.avi`) |
| `annotations_mode` | `True` | If enabled, overlays speed, throw status, cooldown timer, and object coordinates on the video feed |

At the end of each tracked throw (or on node shutdown while a throw is active), the node saves a plot comparing the observed trajectory to the predicted one.

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
│       │   ├── speed_det_node.py
│       ├── config/
│       ├── resource/
│       └── test/
├── weights/                      # Model weights
│   ├── yolo26n-pose.pt
│   └── alien_plushie_v5.pt
├── requirements.txt
└── setup_env.sh
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
- `matplotlib` (trajectory plotting)
- An Orbbec camera publishing synchronized color/depth image topics

All Python dependencies with their exact required versions are listed in [`requirements.txt`](./requirements.txt). Install them with:

```bash
pip install -r requirements.txt
```