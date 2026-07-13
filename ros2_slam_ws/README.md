# SLAM in ROS 2: CNN Visual SLAM vs. Classical EKF-SLAM

```
ros2_slam_ws/
└── src/
    ├── slam_sim_msgs/        # custom interfaces (LandmarkObservation[Array])
    ├── cnn_visual_slam/      # Demo 1: monocular VO/SLAM with an ORB "CNN" front end
    │   ├── city_sim_node          -- PyBullet city + vehicle + camera (ported CityBuilder/AutonomousVehicle)
    │   ├── cnn_feature_node       -- ORB features, essential-matrix VO, sparse map (ported+extended SimpleCNN)
    │   └── launch/visual_slam_cnn.launch.py
    ├── ekf_slam/              # Demo 2: classical range-bearing EKF-SLAM
    │   ├── landmark_sim_node      -- PyBullet landmark world + noisy odom/sensor (ported VisualSLAM_Colab env)
    │   ├── ekf_slam_node          -- full EKF-SLAM back end (ported EKF_SLAM class, unchanged math)
    │   └── launch/ekf_slam.launch.py
    └── slam_lab_bringup/      # launch/both_slam_demo.launch.py runs both demos together
```

## 1. What changed vs. the notebook (and why)

| Notebook (Colab)                                   | This workspace (ROS 2)                                                  |
|------------------------------------------------------|---------------------------------------------------------------------------|
| One monolithic script per cell, `matplotlib` redraw loop | Two decoupled nodes per demo (simulator vs. perception/estimator), talking over topics — the standard ROS separation of concerns |
| `p.ER_BULLET_HARDWARE_OPENGL` renderer (needs a GPU/X server) | `p.ER_TINY_RENDERER` (headless-safe, CPU-only, works on any VM)          |
| Feature viz only — no actual trajectory estimate from ORB features | `cnn_feature_node` now does real monocular VO: 2-view matching → 5-point essential matrix → `recoverPose` → chained trajectory → triangulated sparse map |
| EKF math                                              | Copied over **unchanged** — `predict`/`update`/`add_new_landmark` are the same equations, just re-hosted in an `rclpy` node |
| Landmark ID "given" implicitly                        | Modeled explicitly as `slam_sim_msgs/LandmarkObservation` — landmark ID is what a CNN/colour classifier would output, decoupling "which landmark is this" from "where is it" (the actual EKF problem) |

Both demos publish a **ground-truth** path/landmarks topic alongside the
**estimated** one, so RViz directly shows you the SLAM error growing (or
being corrected on loop closure / repeat observations) — that comparison
is the whole point of the lab.

## 2. VM setup

Tested against **ROS 2 Humble on Ubuntu 22.04** (adjust the distro name below
if your VM image uses a different one).

```bash
# 1. ROS 2 (skip if already installed on the lab VM)
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-tf-transformations \
                     ros-humble-cv-bridge python3-colcon-common-extensions

# 2. Python deps not in apt (pybullet has no apt package)
pip3 install --user pybullet opencv-python numpy

# 3. Clone / copy this workspace
cd ~
# (unzip the provided slam_lab_ws.zip here, or clone your fork)
cd slam_lab_ws

# 4. Build
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

> If `tf-transformations` isn't packaged for your ROS distro, `pip3 install
> --user transforms3d` and the same import will resolve via the community
> `tf_transformations` shim — either works for the quaternion helpers used
> here.

## 3. Run

**Demo 1 — CNN Visual SLAM (monocular VO + ORB front end):**
```bash
ros2 launch cnn_visual_slam visual_slam_cnn.launch.py
```
RViz opens with:
- Green path = ground-truth vehicle trajectory (from PyBullet)
- Red path = monocular VO estimate (from ORB matches → essential matrix)
- Yellow points = sparse triangulated map
- Bottom-left image panel = live ORB keypoints drawn on the camera feed

**Demo 2 — Classical EKF-SLAM (range-bearing landmarks):**
```bash
ros2 launch ekf_slam ekf_slam.launch.py
```
RViz opens with:
- Green path/spheres = ground-truth trajectory and landmark positions
- Red path + orange spheres = EKF-estimated trajectory and landmarks
- Yellow ellipses = 95% covariance ellipses per landmark — watch them
  shrink as each landmark is re-observed, the classic SLAM "uncertainty
  collapses on revisit" behaviour

**Both together (two RViz windows):**
```bash
ros2 launch slam_lab_bringup both_slam_demo.launch.py
```

### 3a. EKF-SLAM on Gazebo instead of PyBullet

`ekf_slam` now has a second, Gazebo-backed variant that doesn't touch the
EKF math at all — only *how the robot moves and how it's observed*
changes. The Gazebo world (`worlds/landmarks.sdf`) has the same six
landmarks at the same positions as the PyBullet version, plus a small
diff-drive robot with real physics (instead of a teleported kinematic
box).

Install Gazebo Harmonic + the ROS 2 bridge first (skip if already done):
```bash
sudo apt install -y ros-jazzy-ros-gz ros-jazzy-ros-gz-sim \
                     ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-interfaces
gz sim --version          # sanity check
```

Rebuild (the new world/config files need to be installed) and run:
```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch ekf_slam ekf_slam_gazebo.launch.py
```

This launches, in order: the Gazebo world, `ros_gz_bridge` (bridging
`/cmd_vel` into Gazebo and the robot's raw odometry back out),
`waypoint_driver_node` (drives the same square tour via `/cmd_vel`),
`landmark_sensor_node` (adds sensor noise + computes range-bearing
observations — this is the node that replaces the PyBullet sim's
"sensing" half), the **unchanged** `ekf_slam_node`, and RViz.

Architecturally:
```
Gazebo (physics + robot model)
   │ /model/ekf_robot/odom  (exact)
   ▼ ros_gz_bridge
/ground_truth/odom_raw ──► waypoint_driver_node ──► /cmd_vel ──► Gazebo
                       └──► landmark_sensor_node ──► /odom (noisy), /landmark_observations
                                                          │
                                                          ▼
                                                    ekf_slam_node (unchanged)
```

No camera is involved in this demo, so there's no headless-rendering risk
— if `gz sim` runs at all on your VM, this should work. If the robot
tips over or jitters, it's almost certainly the placeholder inertia/wheel
friction values in `landmarks.sdf` — nudge `wheel_separation`,
`wheel_radius`, or the `<mu>` friction values in that file.

The old `ekf_slam.launch.py` (PyBullet version) still works unchanged if
you want to A/B compare the two simulators.

To run headless (no RViz, e.g. over SSH) and just watch topics:
```bash
ros2 launch ekf_slam ekf_slam.launch.py rviz:=false
ros2 topic echo /ekf/pose
```

## 4. Things to observe / lab writeup prompts

1. **Scale drift**: `cnn_feature_node` cannot recover true metric scale from
   a single camera (fundamental limitation of monocular VO — see the
   docstring in that file for how we visually work around it). Compare
   this to the EKF, whose landmark ranges give it real metric scale.
   Which one's path error grows unbounded, and which one's stays bounded?
2. **Loop closure / re-observation**: in `ekf_slam`, watch a landmark's
   covariance ellipse shrink the second and third time it's observed.
   The visual-SLAM demo in this repo has no loop-closure/relocalization
   step — that's a natural "next step" extension to discuss.
3. **Swap the front end**: `SimpleCNN` in `cnn_feature_node.py` is
   ORB-based by design (fast, dependency-light, matches the original
   notebook). For a real "CNN" ablation, swap `cv2.ORB_create()` for a
   learned detector such as SuperPoint/DISK and re-run — the rest of the
   VO/mapping pipeline (matching → essential matrix → triangulation) is
   unchanged, which is exactly the point: the *front end* is swappable,
   the *back end math* is not.
4. **EKF vs. graph-SLAM**: this package implements the EKF variant from
   the notebook exactly. If you want the graph-SLAM alternative
   mentioned in the assignment, `slam_toolbox` (already in
   `ros-humble-desktop`) can be dropped in against the same
   `/landmark_observations`-derived `/odom` topic for a pose-graph
   comparison — ask if you'd like that wired up as a third demo.

## 5. Package summary

| Package            | Type          | Key nodes                                  |
|---------------------|---------------|---------------------------------------------|
| `slam_sim_msgs`     | `ament_cmake` (interfaces) | — |
| `cnn_visual_slam`   | `ament_python`| `city_sim_node`, `cnn_feature_node`          |
| `ekf_slam`          | `ament_python`| `landmark_sim_node`, `ekf_slam_node`         |
| `slam_lab_bringup`  | `ament_cmake` | (launch-only)                                |
