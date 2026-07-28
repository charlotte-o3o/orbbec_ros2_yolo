# orbbec_ros2_yolo

ROS 2 nodes for running YOLO pose estimation, a fine-tuned YOLO object detector, and a speed-based throwing-action detector on synchronized color/depth streams from an Orbbec camera, with depth-based 3D distance estimation between detected objects and human wrist keypoints.

This workspace is **dockerized**: everything runs inside a container built from the provided `Dockerfile`, launched via `docker compose` (wrapped in a `Makefile` for convenience). No local ROS 2 / Python environment setup is required to run it — only Docker.

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

`speed_det_node` also opens an OpenCV/Qt debug window for live annotated video — this requires X11 access from the container (see [Launching](#launching) below).

## Camera Driver

The Orbbec camera is run through a **separate, dockerized** ROS 2 wrapper, available here: [hucebot/orbbec_ros2](https://github.com/hucebot/orbbec_ros2). Credit to the [Hucebot](https://github.com/hucebot) team for this driver.

This repo does **not** vendor or modify that driver — clone it independently, following its own README/Makefile. The only thing that must match between the two is `ROS_DOMAIN_ID` and `DEPLOYMENT_ENV` (see [Environment variables](#environment-variables) below), so both DDS domains agree and can discover each other.

## Prerequisites

- Docker + Docker Compose
- NVIDIA Container Toolkit (for GPU access — YOLO inference)
- An X server on the host (for `speed_det_node`'s debug window) — standard on any Linux desktop
- The camera driver ([hucebot/orbbec_ros2](https://github.com/hucebot/orbbec_ros2)) cloned and running separately

## Setup

1. Clone this repo.
2. Copy the environment template and fill in your values:
   ```bash
   cp .env.example .env
   ```
   At minimum, check `ROS_DOMAIN_ID` and `DEPLOYMENT_ENV` match the `.env` used in your `orbbec_ros2` clone (see [Environment variables](#environment-variables)).

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
make deploy    # runs `xhost +local:docker` then starts the container
```

`make deploy` runs `xhost +local:docker` for you, since `speed_det_node` needs X11 access as soon as it starts (not just for optional debugging). If you're on a fresh session/reboot and see an X11/Qt error on launch, this should already be handled — if not, run `xhost +local:docker` manually once per session.

### Useful Make targets

| Command | What it does |
|---|---|
| `make build` | Build the Docker image. Required after changing `Dockerfile`, `requirements.txt`, or anything under `config/` (these are baked into the image, not volume-mounted). |
| `make deploy` | `xhost +local:docker` + start the container (`--force-recreate`). |
| `make logs` | Follow the container's logs. |
| `make stop` | Stop the container. |
| `make clean` | Stop + remove the container, and delete local `build/`, `install/`, `log/`, `Log/` (ROS 2/colcon artifacts). |
| `make attach` | Open an interactive shell inside the running container. |
| `make fix-daemon` | Kill a stale local `ros2-daemon` process, in case it's cached with the wrong `RMW_IMPLEMENTATION` and `ros2 topic list`/`ros2 node list` show nothing despite the container running fine. |

> Editing files under `src/` or `weights/` does **not** require a rebuild — they're volume-mounted, so `make stop && make deploy` (or just restarting the container) picks up changes. If you change a compiled/C++ package, you'll still need to rebuild the ROS 2 workspace inside the container (`colcon build`), not the Docker image itself.

### Verifying it's running

From inside the container (`make attach`), or with the ROS 2 CLI on the host if you have it installed with matching `RMW_IMPLEMENTATION`/`CYCLONEDDS_URI`/`ROS_DOMAIN_ID`:
```bash
ros2 topic list
ros2 topic hz /orbbec_external/color/image_raw
```
You should see the full set of `/orbbec_external/...` topics from the camera driver, plus `/yolo_detected_objects`, `/yolo_detected_poses` from this pipeline. If you only see `/parameter_events` and `/rosout`, try `make fix-daemon` first before assuming something is actually broken — this is very often a stale local daemon, not a real discovery issue.

### Environment variables

Set in `.env` (copy from `.env.example`). Must match the corresponding `.env` in the `orbbec_ros2` clone for `ROS_DOMAIN_ID` and `DEPLOYMENT_ENV`:

| Variable | Description | Must match `orbbec_ros2`? |
|---|---|---|
| `ROS_DOMAIN_ID` | ROS 2 DDS domain ID | ✅ Yes |
| `DEPLOYMENT_ENV` | `local` (single-machine dev) or `robot` (multi-machine deployment) — selects `config/cyclonedds_local.xml` or `config/cyclonedds_robot.xml` | ✅ Yes |

`RMW_IMPLEMENTATION` (`rmw_cyclonedds_cpp`) and the CycloneDDS XML configs live in [`config/`](./config) and are set automatically by `entrypoint.sh` based on `DEPLOYMENT_ENV` — no need to export them manually.

### Troubleshooting

- **No topics visible / only `/parameter_events` and `/rosout`**: run `make fix-daemon`, then retry. This clears a stale `ros2-daemon` process that can get cached with the wrong `RMW_IMPLEMENTATION` and silently fail to see any CycloneDDS nodes.
- **`speed_det_node` crashes with a Qt/xcb error**: X11 isn't authorized for the container. Run `xhost +local:docker` on the host, then `make deploy` again.
- **Camera topics stop being published**: stop the camera driver container, unplug the camera, plug it back in, then relaunch the driver (`make stop && make deploy` in the `orbbec_ros2` terminal).

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
├── config/
│   ├── cyclonedds_local.xml      # DEPLOYMENT_ENV=local
│   └── cyclonedds_robot.xml      # DEPLOYMENT_ENV=robot
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
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh                 # Picks CYCLONEDDS_URI based on DEPLOYMENT_ENV
├── Makefile                      # make build / deploy / stop / logs / clean / attach / fix-daemon
├── .env.example                  # Copy to .env and fill in
├── requirements.txt
└── setup_env.sh                  # Optional, for running nodes on the host without Docker
```

## Requirements

Everything below is installed automatically inside the Docker image — you don't need any of this on your host to run the pipeline via `make deploy`.

- ROS 2
- `cv_bridge`, `message_filters`
- OpenCV (`opencv-python`)
- `ultralytics` (YOLO)
- `matplotlib` (trajectory plotting)
- An Orbbec camera publishing synchronized color/depth image topics (via the separate `orbbec_ros2` driver)

All Python dependencies with their exact required versions are listed in [`requirements.txt`](./requirements.txt) and installed at image build time.

If you want to run nodes directly on the host instead (e.g. for faster iteration without rebuilding), see `setup_env.sh` — you'll need to adapt the hardcoded paths to your own machine, and manually install:
```bash
sudo apt install ros-<distro>-cv-bridge
sudo apt install ros-<distro>-message-filters
pip install -r requirements.txt
```