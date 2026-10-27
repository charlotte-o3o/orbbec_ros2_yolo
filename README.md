# orbbec_ros2_yolo

ROS 2 nodes for running YOLO pose estimation, a fine-tuned YOLO object detector, and a speed-based throwing-action detector with landing/interception prediction, on synchronized color/depth streams from an Orbbec camera. Depth is used to estimate the 3D position of the detected object and of the human keypoints.

This workspace is **dockerized**: everything runs inside a container built from the provided `Dockerfile`, launched via `docker compose` (wrapped in a `Makefile` for convenience). No local ROS 2 / Python environment setup is required to run it — only Docker.

## Nodes

- **`yolo_pose_node`** — runs YOLO pose estimation on the color stream (the image is rotated 90° counter-clockwise before inference because the camera is mounted vertically, and the keypoints are mapped back to the native image), overlays the skeleton, and estimates the 3D position of each detected person from the shoulder midpoint using the synchronized depth image. Publishes the keypoints and the 3D shoulder position on `/yolo_detected_poses`.
- **`fine_tune_yolo_node`** — runs a custom fine-tuned YOLO model (default: `weights/yellow_box_v2.pt`), draws bounding boxes with class/confidence, and estimates the object's 3D position (X/Y/Z in the camera frame, using the camera intrinsics). The depth is taken from a central patch of the bounding box (see `bb_margin`), filtered around the median, then passed through a velocity-based outlier rejection and a moving average. Publishes the result on `/yolo_detected_objects`. Also logs average inference time/FPS every 1000 frames.
- **`speed_det_node`** — fuses synchronized color/depth images and the detections of the two nodes above to compute, frame by frame, the object's position in a gravity-aligned world frame and its instantaneous 3D speed. A throw is detected purely from kinematics: the object's speed must stay above a threshold, with a negative velocity component towards the camera, for several consecutive frames. Once a throw is confirmed, the node tracks the observed trajectory and, from a sliding window of the last observed points, fits a linear model on X/Z and a gravity-corrected linear model on Y to predict the future trajectory. If the predicted trajectory crosses the robot capture box, a `LandingPrediction` (interception point + time) is published on `/trajectory/landing_prediction`. It can also log position/speed to CSV, record the annotated video, and plot the observed vs. predicted trajectory at the end of each throw.

All three nodes subscribe once to `/orbbec_external/color/camera_info` to retrieve the real camera intrinsics (fx, fy, cx, cy), then unsubscribe. Until it is received, default values (fx = fy = 616, cx = 320, cy = 240) are used.

`yolo_pose_node` and `fine_tune_yolo_node` both synchronize (approximate time synchronizer, `slop = 0.05 s`):
- `/orbbec_external/color/image_raw`
- `/orbbec_external/depth/image_raw`

`speed_det_node` synchronizes four streams (`slop = 0.1 s`): the color and depth images above, plus the detection topics published by the other two nodes:
- `/orbbec_external/color/image_raw`
- `/orbbec_external/depth/image_raw`
- `/yolo_detected_objects` (from `fine_tune_yolo_node`, `vision_msgs/Detection2DArray`)
- `/yolo_detected_poses` (from `yolo_pose_node`, `lancer_interfaces/HumanPoseArray`)

> **X11:** `speed_det_node` and `fine_tune_yolo_node` both open an OpenCV/Qt debug window for live annotated video — this requires X11 access from the container (see [Launching](#launching) below). `yolo_pose_node` does not display a window by default (the `imshow` calls are commented out). Press `ESC` in the `speed_det_node` window to shut that node down.

### Topics and messages

| Topic | Type | Publisher | Description |
|---|---|---|---|
| `/yolo_detected_objects` | `vision_msgs/Detection2DArray` | `fine_tune_yolo_node` | 2D bounding box (pixels) + class/score + estimated 3D position of the object in the camera frame (`results[0].pose.pose.position`, in meters; `0.0` for X/Y and Z if no valid depth) |
| `/yolo_detected_poses` | `lancer_interfaces/HumanPoseArray` | `yolo_pose_node` | One `HumanPose` per person: `id`, `position_centre_3d` (shoulder midpoint, meters) and 17 `Keypoint2D` (`x`, `y` in native-image pixels, `confidence`) |
| `/trajectory/landing_prediction` | `lancer_interfaces/LandingPrediction` | `speed_det_node` | `time_to_landing` (s) and `x_landing`, `y_landing`, `z_landing` (m, world frame): first point where the predicted trajectory enters the robot capture box |

Custom messages are defined in `src/lancer_interfaces/msg/`:

```
# HumanPose.msg
int32 id
geometry_msgs/Point position_centre_3d
Keypoint2D[] keypoints

# HumanPoseArray.msg
std_msgs/Header header
HumanPose[] poses

# Keypoint2D.msg
float32 x
float32 y
float32 confidence

# LandingPrediction.msg
float64 time_to_landing
float64 x_landing
float64 y_landing
float64 z_landing
```

## Camera Driver

The Orbbec camera is run through a **separate, dockerized** ROS 2 wrapper, available here: [hucebot/orbbec_ros2](https://github.com/hucebot/orbbec_ros2). Credit to the [Hucebot](https://github.com/hucebot) team for this driver.

This repo does **not** vendor or modify that driver — clone it independently, following its own README/Makefile. The only thing that must match between the two is `ROS_DOMAIN_ID` and `DEPLOYMENT_ENV` (see [Environment variables](#environment-variables) below), so both DDS domains agree and can discover each other.

## Prerequisites

- Docker + Docker Compose
- NVIDIA Container Toolkit (for GPU access — YOLO inference)
- An X server on the host (for the debug windows) — standard on any Linux desktop
- The camera driver ([hucebot/orbbec_ros2](https://github.com/hucebot/orbbec_ros2)) cloned and running separately

## Setup

1. Clone this repo.
2. Copy the environment template and fill in your values:
   ```bash
   cp .env.template .env
   ```
   At minimum, check `ROS_DOMAIN_ID` and `DEPLOYMENT_ENV` match the `.env` used in your `orbbec_ros2` clone (see [Environment variables](#environment-variables)).
3. Put your model weights in `weights/` (by default `yolo26n-pose.pt` and `yellow_box_v2.pt`, see [Node parameters](#node-parameters)).

## Launching

You will need **two terminals** (or two clones running independently) — one for the camera driver, one for this vision pipeline.

### Terminal 1 — camera driver (`orbbec_ros2`)

```bash
cd /path/to/orbbec_ros2
make deploy
```

### Terminal 2 — this repo

```bash
cd /path/to/ros2_orbbec_ws
make build     # only needed the first time, or after changing Dockerfile/config/requirements.txt
make deploy    # runs `xhost +local:docker` then starts the container in the background
make logs      # follow the logs
```

The three nodes (`fine_tune_yolo_node`, `yolo_pose_node`, `speed_det_node`) are started together by `launch/speed_det.launch.py`, which loads all their parameters from `config/speed_det_params.yaml` (see [Node parameters](#node-parameters)).

`make deploy` runs `xhost +local:docker` for you, since the debug windows need X11 access as soon as the nodes start. If you're on a fresh session/reboot and see an X11/Qt error on launch, this should already be handled — if not, run `xhost +local:docker` manually once per session.

### Useful Make targets

| Command | What it does |
|---|---|
| `make build` | Build the Docker image. Required after changing `Dockerfile`, `requirements.txt`, or anything under `config/` (these are baked into the image, not volume-mounted). |
| `make deploy` | `xhost +local:docker` + start the container in the background (`--force-recreate`). |
| `make logs` | Follow the container's logs. |
| `make stop` | Stop the container. |
| `make clean` | `docker compose down --remove-orphans` (stop + remove the container) and delete local `build/`, `install/`, `log/`, `Log/` (ROS 2/colcon artifacts). |
| `make attach` | Open an interactive shell inside the running container (`ros2_vision_container`). |
| `make fix-daemon` | Kill a stale local `ros2-daemon` process, in case it's cached with the wrong `RMW_IMPLEMENTATION` and `ros2 topic list`/`ros2 node list` show nothing despite the container running fine. |
| `make help` | List the available targets. |

> Editing files under `src/` or `weights/` does **not** require a rebuild — they're volume-mounted, so `make stop && make deploy` (or just restarting the container) picks up changes. If you change a compiled/C++ package, you'll still need to rebuild the ROS 2 workspace inside the container (`colcon build`), not the Docker image itself.

### Verifying it's running

From inside the container (`make attach`), or with the ROS 2 CLI on the host if you have it installed with matching `RMW_IMPLEMENTATION`/`CYCLONEDDS_URI`/`ROS_DOMAIN_ID`:
```bash
ros2 topic list
ros2 topic hz /orbbec_external/color/image_raw
```
You should see the full set of `/orbbec_external/...` topics from the camera driver, plus `/yolo_detected_objects` and `/yolo_detected_poses` from this pipeline (and `/trajectory/landing_prediction` once a throw has been predicted to reach the capture box). If you only see `/parameter_events` and `/rosout`, try `make fix-daemon` first before assuming something is actually broken — this is very often a stale local daemon, not a real discovery issue.

### Environment variables

Set in `.env` (copy from `.env.template`). Must match the corresponding `.env` in the `orbbec_ros2` clone for `ROS_DOMAIN_ID` and `DEPLOYMENT_ENV`:

| Variable | Description | Must match `orbbec_ros2`? |
|---|---|---|
| `ROS_DOMAIN_ID` | ROS 2 DDS domain ID | ✅ Yes |
| `DEPLOYMENT_ENV` | `local` (single-machine dev, default) or `robot` (multi-machine deployment) — selects `config/cyclonedds_local.xml` or `config/cyclonedds_robot.xml`. Any other value prints a warning and falls back to `local`. | ✅ Yes |

`RMW_IMPLEMENTATION` (`rmw_cyclonedds_cpp`) and the CycloneDDS XML configs live in [`config/`](./config) and are set automatically by `entrypoint.sh` based on `DEPLOYMENT_ENV` — no need to export them manually. The entrypoint prints the selected `DEPLOYMENT_ENV` and `CYCLONEDDS_URI` at startup (visible in `make logs`).

### Troubleshooting

- **No topics visible / only `/parameter_events` and `/rosout`**: run `make fix-daemon`, then retry. This clears a stale `ros2-daemon` process that can get cached with the wrong `RMW_IMPLEMENTATION` and silently fail to see any CycloneDDS nodes.
- **`speed_det_node` or `fine_tune_yolo_node` crashes with a Qt/xcb error**: X11 isn't authorized for the container. Run `xhost +local:docker` on the host, then `make deploy` again.
- **Camera topics stop being published**: stop the camera driver container, unplug the camera, plug it back in, then relaunch the driver (`make stop && make deploy` in the `orbbec_ros2` terminal).
- **No throw detected right after launch**: this is expected during the `startup_grace_period` (the video shows `WARMING UP`).

## Node parameters

All parameters are declared in the nodes with the defaults shown below, and are **overridden by `src/yolo_detectors/config/speed_det_params.yaml`** when launched through `speed_det.launch.py`. The "Shipped config" column shows the value currently set in that file (`–` means the file doesn't override the default). Since `config/` is baked into the image, run `make build` after editing it.

### `yolo_pose_node`

| Parameter | Default | Shipped config | Description |
|---|---|---|---|
| `model_path` | `weights/yolo26n-pose.pt` | `weights/yolo26n-pose.pt` | Path to the YOLO pose model weights |
| `confidence` | `0.50` | `0.50` | Minimum detection confidence |

### `fine_tune_yolo_node`

| Parameter | Default | Shipped config | Description |
|---|---|---|---|
| `model_path` | `weights/yellow_box_v2.pt` | `weights/yellow_box_v2.pt` | Path to the fine-tuned YOLO model weights |
| `confidence` | `0.75` | `0.75` | Minimum detection confidence |
| `max_history` | `5` | `5` | Number of past distance readings used for the moving-average smoothing |
| `bb_margin` | `0.46` | `0.495` | Fraction of the bounding-box width/height trimmed on each side before sampling depth. The depth is read from the remaining central patch, so values close to 0.5 keep only a very small patch around the box center (more robust to background pixels, more sensitive to depth holes) |
| `max_velocity` | `5.0` m/s | – | Maximum plausible object velocity along Z. A new distance is rejected as an outlier if it differs from the previous one by more than `max(0.20 m, max_velocity × dt)`; after more than 5 consecutive rejections, the filter history is reset |
| `max_jump` | `2.0` m | `2.0` | **Legacy:** still declared, but no longer used — the outlier threshold is now the dynamic one computed from `max_velocity` |

### `speed_det_node`

> **Note:** this node relies on the detections published by `fine_tune_yolo_node` (object 3D position) and, through the synchronized message filter, on `yolo_pose_node` (wrist keypoints, only used for the overlay), even though the throw itself is detected from object speed only.

**Detection thresholds / debounce**

| Parameter | Default | Shipped config | Description |
|---|---|---|---|
| `speed_threshold` | `2.0` m/s | `1.5` | Minimum object speed required to start considering a throw |
| `throw_confirm_frames` | `3` | – | Number of consecutive frames above `speed_threshold` (with a negative velocity component towards the camera) required to confirm a throw |
| `max_false_frames_allowed` | `30` | – | Number of consecutive frames below threshold required before the throw state is officially closed (debounce) |
| `cooldown_duration` | `8.0` s | `5.0` | Minimum time between two consecutive throw triggers, to avoid flickering re-triggers |
| `startup_grace_period` | `8.0` s | `8.0` | Warm-up delay after the node starts, during which no throw can be detected |

**Smoothing / kinematics / camera geometry**

| Parameter | Default | Shipped config | Description |
|---|---|---|---|
| `alpha_smooth` | `0.4` | `0.4` | Exponential moving average factor used to smooth the raw object 3D position before computing speed |
| `camera_roll_deg` | `0.0` | `-90.0` | Physical roll of the camera around its optical axis. `0` = landscape; `±90` = camera mounted vertically. Compensated first, so the camera-frame position is brought back to the standard landscape convention |
| `camera_tilt_deg` | `0.0` | `15.0` | Camera tilt angle (degrees) around its X-axis, used to project the roll-compensated position into a gravity-aligned world frame |
| `min_valid_dt` | `0.01` s | – | Minimum time delta between two frames below which the speed computation is skipped (avoids noise blow-up) |
| `max_valid_speed` | `25.0` m/s | – | Maximum physically plausible speed for a hand throw; higher values reject the trajectory update as noise |
| `min_valid_speed` | `-0.2` m/s | – | Despite its name, this is a threshold on the **Z velocity**: a trajectory point is only added when `vz` is below this (negative) value, i.e. the object is clearly approaching the camera |
| `non_approaching_velocity_threshold` | `-0.05` m/s | – | If `vz` is above this value, the object is considered as moving away/flat: the trajectory update is rejected, and so is any prediction whose fitted `vz` is above it |

**Trajectory tracking & landing prediction**

| Parameter | Default | Shipped config | Description |
|---|---|---|---|
| `effective_gravity` | `8.5` m/s² | `8.0` | Gravity value used in the Y model of the prediction. Set below 9.81 to compensate for drag and depth-estimation bias; tune empirically |
| `trajectory_tracking_duration` | `3.0` s | – | Maximum duration a trajectory is tracked after a throw is confirmed, before timing out |
| `history_sampling_interval` | `0.10` s | – | Minimum time between two archived trajectory predictions (used for the end-of-throw plot, independent of FPS) |
| `landing_z_threshold` | `0.3` m | `0.3` | Distance to the camera below which the object is considered to have "arrived" |
| `min_time_before_landing_check` | `0.15` s | – | Minimum tracking time before the landing check is allowed to trigger |
| `landing_confirm_frames` | `3` | – | Number of consecutive frames under `landing_z_threshold` required to confirm landing |

**Physical validity guards**

| Parameter | Default | Description |
|---|---|---|
| `min_valid_z` / `max_valid_z` | `0.1` / `10.0` m | Valid depth range for the object; positions outside are rejected |
| `max_valid_xy` | `5.0` m | Maximum plausible lateral/vertical position in the room |
| `max_predicted_height` / `max_predicted_fall` | `2.0` / `-2.0` m | Vertical bounds (world Y) beyond which a predicted trajectory point is considered non-physical and the prediction stops |

**Logging / output**

| Parameter | Default | Shipped config | Description |
|---|---|---|---|
| `save_distance_mode` | `False` | `false` | If enabled, logs per-frame object position (X/Y/Z, world frame), speed and throw state to a timestamped CSV file (columns: `frame, timestamp, X(m), Y(m), Z(m), object_speed_m_s, label`; the last column holds the `throw_detected` flag) |
| `record_mode` | `True` | `true` | If enabled, records the annotated video stream to `data/captures_videos/` (suffix `_speed_det.avi`) |
| `annotations_mode` | `True` | `true` | If enabled, overlays speed, throw status, warm-up/cooldown timers, wrist keypoints and object coordinates on the video feed |

Output locations:

| Output | Path |
|---|---|
| CSV | `~/ros2_orbbec_ws/data/speed_detection/csv_distances/distances_<timestamp>.csv` |
| Video | `data/captures_videos/<timestamp>_speed_det.avi` |
| Trajectory plots | `~/ros2_orbbec_ws/data/speed_detection/trajectory_plots/trajectories_<timestamp>_frame<N>.png` |

### How the throw detection and prediction work

1. **World frame.** The smoothed object position is first corrected for the camera roll (`camera_roll_deg`), then rotated by the camera tilt (`camera_tilt_deg`) into a gravity-aligned frame (Y up, Z towards the camera axis direction). The speed and its components are computed in this frame.
2. **Throw confirmation.** After the warm-up, a throw is confirmed when `speed ≥ speed_threshold` and `vz < 0` for `throw_confirm_frames` consecutive frames. It is closed after `max_false_frames_allowed` consecutive frames that don't satisfy the condition. A new throw can only be triggered after `cooldown_duration`.
3. **Tracking.** Once a throw starts, valid observed points are accumulated. Tracking stops when the throw ends, when no valid point has been recorded for more than 0.5 s ("black hole"), when the object reaches the camera (`landing_z_threshold`, confirmed over `landing_confirm_frames`), or after `trajectory_tracking_duration`.
4. **Prediction.** From 4 observed points on, and every 2 new points, the last 6 points are used to fit a linear model on X and Z, and a linear model on Y after adding the gravity term (`effective_gravity`). The trajectory is extrapolated over a 2 s horizon. A prediction is rejected if the object is not approaching the camera, or if it falls below `max_predicted_fall` while still far from the camera (the "cliff effect").
5. **Interception.** The current position and the fitted mean velocities are propagated (over 1.5 s) to find the first point inside the robot capture box. If there is one, a `LandingPrediction` is published on `/trajectory/landing_prediction`. The capture box is currently **hard-coded** in `check_workspace_intersection()` (world frame, X ∈ [-1, 1] m, Y ∈ [-1, 1] m, Z ∈ [0, 0.2] m) — adjust it there to match your robot setup.
6. **Plot.** At the end of each tracked throw (or on node shutdown while a throw is active), a plot of the observed points, the sampled predictions and the last prediction is saved (vertical profile Y/Z and horizontal profile X/Z).

## Repository Structure

```
ros2_orbbec_ws/
├── config/
│   ├── cyclonedds_local.xml      # DEPLOYMENT_ENV=local
│   └── cyclonedds_robot.xml      # DEPLOYMENT_ENV=robot
├── src/
│   ├── lancer_interfaces/        # Custom ROS 2 message definitions
│   │   ├── msg/
│   │   │   ├── HumanPose.msg
│   │   │   ├── HumanPoseArray.msg
│   │   │   ├── Keypoint2D.msg
│   │   │   └── LandingPrediction.msg
│   │   └── ...
│   └── yolo_detectors/           # Main package
│       ├── yolo_detectors/
│       │   ├── yolo_pose_node.py
│       │   ├── fine_tune_yolo_node.py
│       │   └── speed_det_node.py
│       ├── config/
│       │   └── speed_det_params.yaml   # Parameters of the 3 nodes
│       ├── launch/
│       │   └── speed_det.launch.py     # Starts the 3 nodes together
│       ├── resource/
│       └── test/
├── weights/                      # Model weights
│   ├── yolo26n-pose.pt
│   └── yellow_box_v2.pt
├── data/                         # Generated outputs (videos, CSV, trajectory plots)
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh                 # Picks CYCLONEDDS_URI based on DEPLOYMENT_ENV
├── Makefile                      # make build / deploy / stop / logs / clean / attach / fix-daemon / help
├── .env.template                 # Copy to .env and fill in
└── requirements.txt
```

## Requirements

Everything below is installed automatically inside the Docker image — you don't need any of this on your host to run the pipeline via `make deploy`.

- ROS 2 Humble
- `cv_bridge`, `message_filters`, `vision_msgs`
- OpenCV (`opencv-python`)
- `numpy`
- `ultralytics` (YOLO)
- `matplotlib` (trajectory plotting, `Agg` backend)
- An Orbbec camera publishing synchronized color/depth image topics (via the separate `orbbec_ros2` driver)

All Python dependencies with their exact required versions are listed in [`requirements.txt`](./requirements.txt) and installed at image build time.