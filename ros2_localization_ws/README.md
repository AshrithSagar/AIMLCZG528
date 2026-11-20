# Localization in ROS 2 (Webinar 2)

Grid localization, augmented Monte Carlo localization (MCL), and an ML scan → pose regressor, all benchmarked against **Nav2 AMCL** on the same simulated robot.

This is an **overlay** on the W1 workspace (`ros2_sensor_motion_ws`). The simulator, map, motion model and sensor model come from W1's `sensor_motion_lab` (underlay); nothing is copied.

```plaintext
ros2_localization_ws/
├── expected_output/                    # figures + summary.md from `localization_benchmark`
└── src/localization_lab/
    ├── localization_lab/core/          # ROS-free: grid_localization, mcl, learned (NumPy MLP), scenario, evaluation
    ├── localization_lab/ros/           # localizer_node, pose_regressor_node, localization_evaluator_node
    ├── localization_lab/benchmark.py   # localization_benchmark, train_pose_regressor
    ├── launch/localization.launch.py   # includes W1's sim.launch.py
    ├── config/amcl.yaml                # Nav2 AMCL with the same models/parameters (the benchmark)
    ├── rviz/localization.rviz
    └── test/
```

## Build (underlay → overlay)

```shell
conda deactivate
source /opt/ros/jazzy/setup.bash
sudo apt install -y python3-scipy ros-jazzy-nav2-amcl ros-jazzy-nav2-lifecycle-manager   # nav2 only for method:=amcl

# 1. underlay: W1
cd ~/AIMLCZG528/ros2_sensor_motion_ws && colcon build --symlink-install
source install/setup.bash

# 2. overlay: W2 (sees W1's packages because W1 is sourced)
cd ~/AIMLCZG528/ros2_localization_ws && colcon build --symlink-install
source install/setup.bash
colcon test --packages-select localization_lab && colcon test-result --verbose
```

## Run

```shell
ros2 run localization_lab train_pose_regressor        # ~40 s on 2 CPUs -> ~/w2_models/pose_regressor.npz
ros2 run localization_lab localization_benchmark --out ~/w2_expected   # offline comparison (no ROS)

ros2 launch localization_lab localization.launch.py                    # MCL, robot starts "lost"
ros2 launch localization_lab localization.launch.py method:=grid
ros2 launch localization_lab localization.launch.py method:=mcl_ml     # ML-augmented MCL
ros2 launch localization_lab localization.launch.py method:=amcl       # Nav2 AMCL (benchmark)
ros2 launch localization_lab localization.launch.py drive:=teleop      # drive it yourself
```

While it runs:

```shell
# kidnap the robot (true pose moves, odometry does not)
ros2 topic pub --once /sim/teleport geometry_msgs/msg/PoseStamped "{pose: {position: {x: -2.0, y: -0.5}}}"

# "I am lost": spread the belief over the whole map
ros2 service call /localizer_node/global_localization std_srvs/srv/Trigger
ros2 service call /reinitialize_global_localization std_srvs/srv/Empty     # same thing for Nav2 AMCL

ros2 topic echo /diagnostics                                              # error, convergence, recovery, ms/update
ros2 service call /evaluator_mcl/report std_srvs/srv/Trigger
ros2 run tf2_ros tf2_echo map odom                                        # the correction the localizer publishes
```

In RViz, **2D Pose Estimate** publishes `/initialpose`, which both our localizer and Nav2 AMCL accept.

## Expected output (2 CPUs, 600 scans at 5 Hz, kidnapped at t = 60 s)

See `expected_output/summary.md` for the exact numbers of your run.

| method               | tracking error            | recovery after kidnap | ms / update |
| -------------------- | ------------------------- | --------------------- | ----------- |
| grid                 | ~6 cm                     | ~0.4 s                | ~115        |
| mcl (2000 particles) | ~4 cm                     | ~3 s                  | ~3          |
| mcl_ml               | ~4 cm                     | ~0.4 s                | ~3          |
| ml_only              | ~40 cm (noisy, no memory) | n/a                   | ~0.25       |

**Compute:** the regressor (118k parameters) trains on 40,000 scans in about 40 s and runs in about 25 µs per scan on CPU. Validation error: median ~23 cm, 90th percentile ~63 cm.

## Topics

| Topic                                          | Type                                      | From                                                                                  |
| ---------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------- |
| TF `map → odom`                                |                                           | `localizer_node` or Nav2 AMCL (W1's static one is switched off: `map_odom_tf:=false`) |
| `/localization/pose`, `/amcl_pose`, `/ml/pose` | `geometry_msgs/PoseWithCovarianceStamped` | ours, Nav2 AMCL, regressor                                                            |
| `/localization/particles` / `/particle_cloud`  | `PoseArray` / `nav2_msgs/ParticleCloud`   | ours / Nav2 AMCL                                                                      |
| `/localization/belief`                         | `nav_msgs/OccupancyGrid`                  | grid localizer                                                                        |
| `/sim/teleport`                                | `geometry_msgs/PoseStamped`               | you (kidnapping)                                                                      |
| `/diagnostics`                                 | `diagnostic_msgs/DiagnosticArray`         | all nodes                                                                             |
