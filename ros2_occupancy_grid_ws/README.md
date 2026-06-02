# ros2_occupancy_grid_ws

![ROS2 - Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)

Occupancy grid mapping.

## Setup

1. Once venv is setup, install

   ```shell
   uv pip install mujoco numpy matplotlib pyyaml
   ```

   Additionally, install

   ```shell
   sudo apt install ros-jazzy-tf2-ros ros-jazzy-tf2-tools ros-jazzy-rviz2
   ```

2. Build using `colcon`.

   ```shell
   rm -rf build/ install/ log/

   colcon build --symlink-install

   source /opt/ros/jazzy/setup.bash

   source install/setup.bash
   ```

3. Run the node.

   ```shell
   python -m occupancy_mapping.mapping_node
   ```

4. On another terminal, check:

   ```shell
   source /opt/ros/jazzy/setup.bash

   ros2 topic list

   ros2 topic hz /scan
   ros2 topic hz /odom
   ros2 topic hz /map
   ```

---
