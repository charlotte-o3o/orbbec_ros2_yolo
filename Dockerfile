# syntax=docker/dockerfile:1
# Base image with ROS 2 Humble & PyTorch GPU support
FROM ros:humble-ros-base-jammy

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_WS=/ros2_orbbec_ws \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Install system dependencies (OpenCV, DDS, Python tools, ROS2 message packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-vision-msgs \
    ros-humble-message-filters \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR ${ROS_WS}

# Copy python dependencies and install PyTorch/YOLO requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Copy workspace source files and configuration
COPY src ./src
COPY weights ./weights
COPY config/cyclonedds_local.xml ${ROS_WS}/config/
COPY config/cyclonedds_robot.xml ${ROS_WS}/config/
COPY entrypoint.sh ${ROS_WS}
RUN chmod +x entrypoint.sh

# Build the ROS 2 workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Configure entrypoint to source ROS 2 and workspace automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc

# Entrypoint script
ENTRYPOINT ["/ros2_orbbec_ws/entrypoint.sh"]
CMD ["ros2", "run", "yolo_detectors", "yolo_pose"]