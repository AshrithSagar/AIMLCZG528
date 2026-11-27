# Sensor & Motion Modelling in ROS 2 (Webinar 1)

A differential-drive robot with a 2-D LiDAR, run in either a lightweight
kinematic simulator or Gazebo (Harmonic). Each modelling component has a
measured evaluation metric and is compared against a standard reference.

| Component                    | What is measured                                        | Reference benchmark                                                         |
| ---------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------- |
| ROS 2 graph                  | topic rate, bandwidth, delay (`ros2 topic hz/bw/delay`) | ROBOTIS LDS-01 scan rate (5 Hz)                                             |
| Simulator                    | Real-Time Factor (RTF)                                  | RTF = 1.0                                                                   |
| Visualisation / TF           | TF tree shape, TF delay (`tf2_monitor`)                 | REP-105 (`map → odom → base_link`)                                          |
| Sensor model                 | range bias, RMSE, % within spec, outliers               | ROBOTIS LDS-01 datasheet                                                    |
| Sensor model (map matching)  | likelihood peak error, sharpness, µs/pose               | Nav2 AMCL `beam` vs `likelihood_field` (default params), plus `correlation` and `feature` |
| Motion model (odometry)      | UMBmark E<sub>max,syst</sub>; ATE / RPE / drift %       | UMBmark (Borenstein & Feng 1996); TUM ATE/RPE (`evo`)                       |
| Motion model (probabilistic) | NEES, 95 % coverage                                     | Nav2 AMCL `alpha1..4 = 0.2`                                                 |

```plaintext
ros2_sensor_motion_ws/
├── expected_output/                 # figures + metrics.json (from offline_benchmark)
└── src/
    ├── mobile_bot_description/      # ament_cmake: data only
    │   ├── config/robot.yaml        #   robot + LiDAR parameters   (single source of truth)
    │   ├── config/world.yaml        #   arena + obstacles          (single source of truth)
    │   ├── config/gz_bridge.yaml    #   Gazebo <-> ROS topic bridge
    │   ├── urdf/mobile_bot.urdf.xacro
    │   ├── rviz/display.rviz
    │   └── launch/display.launch.py
    └── sensor_motion_lab/           # ament_python: code
        ├── sensor_motion_lab/core/  #   ROS-free: geometry, world, motion, sensing, features, comparison, metrics, simulation
        ├── sensor_motion_lab/ros/   #   thin rclpy nodes on top of core/
        ├── sensor_motion_lab/launch_builder.py
        ├── sensor_motion_lab/offline_benchmark.py
        ├── launch/{sim,obstacle_avoidance,motion_model,sensor_model,model_comparison}.launch.py
        ├── rviz/{lab,comparison}.rviz
        └── test/test_core.py
```

## Setup (lab VM, ROS 2 Jazzy)

```shell
conda deactivate
source /opt/ros/jazzy/setup.bash
sudo apt install -y ros-jazzy-xacro ros-jazzy-joint-state-publisher-gui \
                    ros-jazzy-teleop-twist-keyboard ros-jazzy-ros-gz   # ros-gz only for sim:=gazebo

cd ~/AIMLCZG528/ros2_sensor_motion_ws
rm -rf build/ install/ log/
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select sensor_motion_lab && colcon test-result --verbose
```

## Run

```shell
# 0. Expected output (no simulator needed; same code as the nodes)
ros2 run sensor_motion_lab offline_benchmark --out ~/w1_expected

# 1. Just the robot model + TF
ros2 launch mobile_bot_description display.launch.py

# 2. Simulator + map + RViz (drive with teleop in a 2nd terminal)
ros2 launch sensor_motion_lab sim.launch.py                    # sim:=gazebo for Gazebo
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# 2b. Lab Manual 2 experiment: odometry motion + obstacle avoidance
ros2 launch sensor_motion_lab obstacle_avoidance.launch.py duration:=60.0

# 3. Sensor model
ros2 launch sensor_motion_lab sensor_model.launch.py model:=likelihood_field
ros2 launch sensor_motion_lab sensor_model.launch.py model:=beam
ros2 launch sensor_motion_lab sensor_model.launch.py model:=correlation
ros2 launch sensor_motion_lab sensor_model.launch.py model:=feature      # line features + data association

# 3b. CS-3 worked example: the four models on the same three returns (no simulator)
ros2 launch sensor_motion_lab model_comparison.launch.py                # Case A: clear aisle
ros2 param set /model_comparison_node case B                            # Case B: a person blocks the centre beam

# 4. Motion model + UMBmark
ros2 launch sensor_motion_lab motion_model.launch.py
ros2 launch sensor_motion_lab motion_model.launch.py baseline_ratio:=1.03 runs:=5 alpha:=0.01
ros2 service call /trajectory_evaluator_node/report std_srvs/srv/Trigger   # writes TUM files
```

Metrics are published on `/diagnostics` and logged to the terminal:

```shell
ros2 topic echo /diagnostics
```

All launch files accept `sim:=kinematic|gazebo` and `rviz:=true|false`.
For Gazebo, set the odometry miscalibration in `config/robot.yaml`
(`calibration.baseline_ratio`); the DiffDrive plugin then integrates odometry
with the wrong track width.

## Topics

| Topic                                                   | Type                              | Publisher                                                  |
| ------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------- |
| `/cmd_vel`                                              | `geometry_msgs/Twist`             | teleop / `umbmark_driver_node` / `obstacle_avoidance_node` |
| `/odom`, TF `odom→base_link`                            | `nav_msgs/Odometry`               | kinematic sim / Gazebo DiffDrive                           |
| `/ground_truth/odom`                                    | `nav_msgs/Odometry`               | kinematic sim / Gazebo OdometryPublisher                   |
| `/scan`                                                 | `sensor_msgs/LaserScan`           | kinematic sim / Gazebo gpu_lidar                           |
| `/map`, `/likelihood_field`, TF `map→odom`              | `nav_msgs/OccupancyGrid`          | `world_map_node`                                           |
| `/sensor_model/{expected_scan,sampled_scan,likelihood}` |                                   | `sensor_model_node`                                        |
| `/comparison/{scan,markers}`                            | `LaserScan`, `MarkerArray`        | `model_comparison_node` (CS-3 example)                     |
| `/motion_model/{particles,mean_path}`                   |                                   | `motion_model_node`                                        |
| `/ground_truth/path`, `/odom_path`                      | `nav_msgs/Path`                   | `trajectory_evaluator_node`                                |
| `/diagnostics`                                          | `diagnostic_msgs/DiagnosticArray` | all lab nodes                                              |
