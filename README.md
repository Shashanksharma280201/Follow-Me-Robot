# Follow-Me-Robot

# Flo Base Controller

A ROS2 package implementing a differential drive base controller with person following capabilities. This package provides nodes for controlling a differential drive robot base, processing serial communication with motor controllers, and implementing person following behavior using computer vision.

## Features

- Differential drive base controller with wheel encoder odometry
- Serial communication interface for motor control
- Person following using YOLOv5 and stereo depth perception
- Configurable parameters for robot behavior and control
- RViz visualization support
- Keyboard teleoperation interface

## Prerequisites

### Hardware Requirements
- Differential drive robot base
- Serial-controlled motor drivers
- Stereo camera system compatible with ROS2 (we used oak-d camera with depthai-ros)
- Computer with CUDA-capable GPU (for YOLOv5) (we used Jetson Nano)

### Software Requirements
- ROS2 (tested on Humble & Galactic)
- Python 3.8+
- PyTorch with CUDA support
- YOLOv5
- OpenCV
- libserial-dev
- Additional ROS2 packages:
  - cv_bridge
  - image_transport
  - geometry_msgs
  - nav_msgs
  - sensor_msgs
  - visualization_msgs

## Installation

1. Create a ROS2 workspace (if you haven't already):
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Clone this repository:
```bas
git clone https://github.com/Shashanksharma280201/Follow-Me-Robot.git
```

3. Install dependencies:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

4. Build the package:
```bash
colcon build --packages-select flo_base
```

5. Source the workspace:
```bash
source ~/ros2_ws/install/setup.bash
```

## Configuration

The package behavior can be configured through the parameters in `config/base_params.yaml`. Key parameters include:

### Base Controller Parameters
- `wheel_diameter`: Diameter of the wheels in meters (default: 0.26)
- `wheel_to_wheel_separation`: Distance between wheels in meters (default: 0.62)
- `cmd_vel_timeout`: Timeout for command velocity messages (default: 0.5s)
- `odom_hz`: Odometry publication rate (default: 30Hz)

### Serial Port Parameters
- `device`: Serial port device (default: "/dev/ttyACM0")
- `baud_rate`: Serial communication baud rate (default: 115200)

### Person Follower Parameters
- `target_distance`: Desired following distance in meters (default: 1.5)
- `max_linear_speed`: Maximum linear velocity (default: 0.36 m/s)
- `max_angular_speed`: Maximum angular velocity (default: 0.5 rad/s)
- `min_distance`: Minimum following distance (default: 0.3m)
- `max_distance`: Maximum detection distance (default: 6.0m)

## Usage

### Launch the Base Controller

To start the base controller with person following:

Terminal 1
```bash
ros2 launch flo_base person_follower.launch.py
```
Terminal 2 (This is the enable the contactor)
```bash
ros2 topic pub /serial_port_drive/in std_msgs/msg/String "{data: BA 0 1}"
```

This will start:
- Serial port communication
- Base controller
- Person follower
- RViz visualization

## Node Architecture

### serial_port
- Handles serial communication with motor controllers
- Publishes: `/serial_port_drive/out`
- Subscribes: `/serial_port_drive/in`

### base_controller
- Implements differential drive control
- Publishes:
  - `/odom/wheel_encoder`: Wheel encoder-based odometry
  - `/motion/status`: Robot motion status
- Subscribes:
  - `/cmd_vel`: Velocity commands
  - `/serial_port_drive/out`: Motor feedback

### person_follower
- Implements person detection and following behavior
- Publishes:
  - `/cmd_vel`: Velocity commands for following
  - `/person_marker`: Visualization markers
  - `/debug_image`: Debug visualization
- Subscribes:
  - `/left/image_rect`: Left camera image
  - `/stereo/depth`: Depth image
  - `/stereo/camera_info`: Camera calibration info



## Acknowledgments

- YOLOv5 by Ultralytics
- ROS2 community
- OpenCV community
