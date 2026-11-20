# Visual SLAM in ROS 2 (RGB-D)

One SLAM node and four interchangeable camera sources. Pluggable feature front-ends: classical ORB / SIFT / AKAZE, or learned SuperPoint. Loop closure + pose-graph optimisation, scored with the TUM RGB-D benchmark metrics (ATE / RPE).

```plaintext
ros2_visual_slam_ws/
├── expected_output/            # figures + summary.md from `vslam_benchmark` (synthetic loop)
└── src/visual_slam/
    ├── visual_slam/core/       # ROS-free: se3, camera, features, tracking, backend, synthetic, datasets, metrics, gazebo
    ├── visual_slam/ros/        # rgbd_player_node, visual_slam_node, trajectory_evaluator_node, rig_driver_node
    ├── visual_slam/benchmark.py
    ├── visual_slam/launch_builder.py
    ├── launch/vslam.launch.py
    ├── config/gz_bridge.yaml
    ├── rviz/vslam.rviz
    └── test/test_core.py
```

## Setup (lab VM, ROS 2 Jazzy)

```shell
conda deactivate
source /opt/ros/jazzy/setup.bash
sudo apt install -y ros-jazzy-cv-bridge ros-jazzy-message-filters python3-opencv python3-scipy \
                    ros-jazzy-ros-gz            # ros-gz only for source:=gazebo

cd ~/AIMLCZG528/ros2_visual_slam_ws
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select visual_slam && colcon test-result --verbose
```

Optional:

```shell
# TUM RGB-D sequence (real data with motion-capture ground truth)
mkdir -p ~/datasets && cd ~/datasets
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz && tar xzf rgbd_dataset_freiburg1_xyz.tgz

# Learned front-end (SuperPoint): PyTorch + weights
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU build; use the CUDA build on a GPU VM
wget -O ~/superpoint_v1.pth https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth
```

## Run

```shell
ros2 launch visual_slam vslam.launch.py                                   # synthetic room, CPU only
ros2 launch visual_slam vslam.launch.py source:=tum path:=~/datasets/rgbd_dataset_freiburg1_xyz
ros2 launch visual_slam vslam.launch.py source:=gazebo                    # Gazebo RGB-D camera (rendering test)
ros2 launch visual_slam vslam.launch.py feature_type:=sift loop_closure:=false

ros2 service call /trajectory_evaluator_node/report std_srvs/srv/Trigger   # ATE/RPE + TUM files
ros2 topic echo /diagnostics                                              # fps, per-stage ms, dropped frames
```

Record once, replay anywhere:

```shell
ros2 bag record -o vslam_input /camera/color/image_raw /camera/color/camera_info \
    /camera/depth/image_rect_raw /ground_truth/pose /ground_truth/path /tf_static
ros2 launch visual_slam vslam.launch.py source:=bag
ros2 bag play vslam_input --clock --rate 0.5        # slower playback = what a slower machine would need
```

## Is the machine fast enough? (GPU decision)

```shell
ros2 run visual_slam vslam_benchmark --out ~/vslam_bench                                   # synthetic, 320x240
ros2 run visual_slam vslam_benchmark --source tum --path ~/datasets/rgbd_dataset_freiburg1_xyz --out ~/vslam_bench_tum
ros2 run visual_slam vslam_benchmark --features orb,superpoint --device cuda ...           # on a GPU VM
```

`summary.md` lists fps, a "real-time?" verdict against the camera rate, per-stage timings (extract / match / PnP / loop + PGO) and ATE / RPE for every front-end. If a learned front-end is real-time only with `--device cuda`, that is the case for a GPU.

For rendering, run `source:=gazebo` and compare `processed_fps` with `input_fps` on `/diagnostics`. Also check Gazebo's real-time factor:

```shell
gz topic -e -t /world/vslam_room/stats | grep real_time_factor
```

Expected output on a 2-CPU machine, synthetic 320×240 @ 15 Hz (`expected_output/summary.md`):

| front-end | fps  | ATE VO | ATE SLAM | loops |
| --------- | ---- | ------ | -------- | ----- |
| ORB       | ~100 | 7.1 cm | 4.9 cm   | 2     |
| SIFT      | ~69  | 2.6 cm | 1.8 cm   | 2     |

## Comparing against a reference SLAM (RTAB-Map)

```shell
sudo apt install ros-jazzy-rtabmap-ros
ros2 launch visual_slam vslam.launch.py source:=tum path:=... publish_tf:=false
ros2 launch rtabmap_launch rtabmap.launch.py rgb_topic:=/camera/color/image_raw \
    depth_topic:=/camera/depth/image_rect_raw camera_info_topic:=/camera/color/camera_info \
    frame_id:=camera_link approx_sync:=true rtabmap_viz:=false qos:=2
ros2 run visual_slam trajectory_evaluator_node --ros-args -r __node:=rtabmap_evaluator \
    -p estimate_topic:=/rtabmap/odom -p label:=rtabmap          # check the topic with `ros2 topic list`
```

## Topics

| Topic                                                               | Type                                | From                         |
| ------------------------------------------------------------------- | ----------------------------------- | ---------------------------- |
| `/camera/color/image_raw`, `/camera/depth/image_rect_raw`           | `sensor_msgs/Image` (rgb8, 32FC1 m) | player / Gazebo bridge / bag |
| `/camera/color/camera_info`                                         | `sensor_msgs/CameraInfo`            | player / Gazebo bridge / bag |
| TF `map → camera_link`, `/vslam/odometry`                           | `nav_msgs/Odometry`                 | `visual_slam_node`           |
| `/vslam/path`, `/vslam/optimized_path`                              | `nav_msgs/Path`                     | `visual_slam_node`           |
| `/vslam/map/dense`, `/vslam/map/points`                             | `sensor_msgs/PointCloud2`           | `visual_slam_node`           |
| `/vslam/keyframes`, `/vslam/loop_closures`, `/vslam/debug/features` |                                     | `visual_slam_node`           |
| `/ground_truth/pose` (dataset) or `/ground_truth/odom` (Gazebo)     |                                     | player / Gazebo              |
| `/diagnostics`                                                      | `diagnostic_msgs/DiagnosticArray`   | all nodes                    |
