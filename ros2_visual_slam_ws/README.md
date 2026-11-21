# Visual SLAM in ROS 2 (RGB-D)

One SLAM node and four interchangeable camera sources. Loop closure + pose-graph optimisation, scored with the TUM RGB-D benchmark metrics (ATE / RPE).

Every stage has a classical and a deep-learning option, switched by a launch argument:

| stage          | classical                           | learned                                                                     |
| -------------- | ----------------------------------- | --------------------------------------------------------------------------- |
| features       | ORB / SIFT / AKAZE                  | SuperPoint (`feature_type:=superpoint`)                                     |
| matching       | brute force + ratio test            | LightGlue (`matcher:=lightglue`)                                            |
| loop detection | pose-prior voting, image thumbnails | DINOv2 global descriptor (`place_recognition:=dinov2`)                      |
| depth          | the RGB-D sensor                    | Depth Anything V2 from colour only (`depth:=learned`)                       |
| whole system   | this pipeline                       | DPVO / DROID-SLAM, run offline on a GPU and overlaid (`dl_trajectory:=...`) |

```plaintext
ros2_visual_slam_ws/
├── expected_output/            # figures + summary.md from `vslam_benchmark` (synthetic loop)
└── src/visual_slam/
    ├── visual_slam/core/       # ROS-free: se3, camera, features, matching, tracking, place_recognition, backend,
    │                           #           depth, runtime (optional PyTorch), synthetic, datasets, metrics, gazebo
    ├── visual_slam/ros/        # rgbd_player_node, visual_slam_node, trajectory_evaluator_node, rig_driver_node,
    │                           # depth_estimator_node, trajectory_overlay_node
    ├── visual_slam/benchmark.py  # vslam_benchmark, export_synthetic
    ├── visual_slam/apps.py       # dl_check, vslam_score
    ├── visual_slam/launch_builder.py
    ├── launch/vslam.launch.py
    ├── config/gz_bridge.yaml
    ├── rviz/vslam.rviz
    └── test/                   # test_core.py, test_learned.py (no PyTorch needed)
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

# a TUM sequence with a real loop, for loop detection
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz && tar xzf rgbd_dataset_freiburg1_room.tgz
```

Deep-learning components (optional; the classical pipeline never imports PyTorch):

```shell
# user site-packages, so ROS's /usr/bin/python3 sees them; numpy<2 keeps cv_bridge working
pip install --user --break-system-packages "numpy<2" \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu      # GPU VM: drop --index-url (PyPI's Linux wheels ship CUDA)
pip install --user --break-system-packages "numpy<2" "transformers>=4.45" \
    git+https://github.com/cvg/LightGlue.git

ros2 run visual_slam dl_check                  # downloads the models once, runs each one, prints a table
ros2 run visual_slam dl_check --device cuda    # on a GPU VM
```

`dl_check` tests SuperPoint + LightGlue (relative pose error), DINOv2 (a revisit must look more similar than the far side of the room) and Depth Anything V2 (AbsRel vs sensor depth), with milliseconds per call on this machine. Models come from GitHub (LightGlue, SuperPoint) and Hugging Face (DINOv2, Depth Anything), so the VM needs access to both once.

## Run

```shell
ros2 launch visual_slam vslam.launch.py                                   # synthetic room, CPU only
ros2 launch visual_slam vslam.launch.py source:=tum path:=~/datasets/rgbd_dataset_freiburg1_xyz
ros2 launch visual_slam vslam.launch.py source:=gazebo                    # Gazebo RGB-D camera (rendering test)
ros2 launch visual_slam vslam.launch.py feature_type:=sift loop_closure:=false

# deep learning (after the pip installs above)
ros2 launch visual_slam vslam.launch.py feature_type:=superpoint matcher:=lightglue
ros2 launch visual_slam vslam.launch.py source:=tum path:=~/datasets/rgbd_dataset_freiburg1_room place_recognition:=dinov2
ros2 launch visual_slam vslam.launch.py source:=tum path:=~/datasets/rgbd_dataset_freiburg1_xyz depth:=learned rate:=2

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

A configuration is `features[+matcher][@place_recognition]`, e.g. `orb`, `superpoint+lightglue`, `orb@dinov2`.

```shell
ros2 run visual_slam vslam_benchmark --out ~/vslam_bench                                   # synthetic, 320x240
ros2 run visual_slam vslam_benchmark --source tum --path ~/datasets/rgbd_dataset_freiburg1_xyz --out ~/vslam_bench_tum
ros2 run visual_slam vslam_benchmark --source tum --path ~/datasets/rgbd_dataset_freiburg1_room \
    --configs orb,orb@thumbnail,orb@dinov2 --out ~/vslam_bench_loops                     # loop detection
ros2 run visual_slam vslam_benchmark --source tum --path ~/datasets/rgbd_dataset_freiburg1_xyz \
    --configs orb --depth learned --max-frames 300 --out ~/vslam_bench_depth               # learned depth
ros2 run visual_slam vslam_benchmark --configs orb,superpoint+lightglue --device cuda ...   # on a GPU VM
```

`summary.md` lists fps, a "real-time?" verdict against the camera rate, per-stage timings and ATE / RPE for every configuration, plus a loop-detection table (recall@1, false loops) and, with `--depth learned`, a depth table (AbsRel, RMSE, δ<1.25). Configurations whose packages are missing are listed as skipped, with the install command. If a learned component is real-time only with `--device cuda`, that is the case for a GPU.

For rendering, run `source:=gazebo` and compare `processed_fps` with `input_fps` on `/diagnostics`. Also check Gazebo's real-time factor:

```shell
gz topic -e -t /world/vslam_room/stats | grep real_time_factor
```

Expected output on a 2-CPU machine, synthetic 320×240 @ 15 Hz (`expected_output/summary.md`):

| front-end | fps | ATE VO | ATE SLAM | loops |
| --------- | --- | ------ | -------- | ----- |
| ORB       | ~80 | 9.4 cm | 4.8 cm   | 2     |
| SIFT      | ~50 | 2.8 cm | 1.7 cm   | 2     |

## End-to-end deep SLAM (DPVO) as a reference

DPVO (Teed, Lipson, Deng, NeurIPS 2023) is a whole visual-odometry system learned end to end. It needs an NVIDIA GPU with CUDA and compiles CUDA extensions, so it runs **offline, outside ROS**, in its own conda environment (follow https://github.com/princeton-vl/DPVO if a step differs):

```shell
nvidia-smi                                          # a GPU and a driver must be visible
git clone https://github.com/princeton-vl/DPVO.git --recursive && cd DPVO
conda env create -f environment.yml && conda activate dpvo
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip eigen-3.4.0.zip -d thirdparty
pip install .
./download_models_and_data.sh                       # network weights (+ sample data, ~2 GB)

# TUM fr1 calibration: fx fy cx cy k1 k2 p1 p2 k3 (DPVO undistorts with OpenCV)
echo "517.3 516.5 318.6 255.3 0.2624 -0.9531 -0.0054 0.0026 1.1633" > calib/tum_fr1.txt
python demo.py --imagedir=$HOME/datasets/rgbd_dataset_freiburg1_xyz/rgb --calib=calib/tum_fr1.txt \
    --stride=2 --name=fr1_xyz --save_trajectory     # -> saved_trajectories/fr1_xyz.txt
conda deactivate
```

DPVO stamps its trajectory with frame _indices_ (image t × stride), not dataset times. Both tools below detect that and map the indices through the sequence's `rgb.txt`:

```shell
# score it: monocular, so align with scale (Sim(3))
ros2 run visual_slam vslam_score --gt ~/datasets/rgbd_dataset_freiburg1_xyz --sim3 \
    --est ~/DPVO/saved_trajectories/fr1_xyz.txt \
    --frame-stamps ~/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt --stride 2
# ours on the same sequence (the benchmark writes dataset-stamped TUM files)
ros2 run visual_slam vslam_score --gt ~/datasets/rgbd_dataset_freiburg1_xyz --est ~/vslam_bench_tum/estimate_orb.tum

# show it in RViz next to ours (orange path), on the same sequence
ros2 launch visual_slam vslam.launch.py source:=tum path:=$HOME/datasets/rgbd_dataset_freiburg1_xyz \
    dl_trajectory:=$HOME/DPVO/saved_trajectories/fr1_xyz.txt dl_stride:=2
```

The live node's own TUM files carry ROS clock stamps, not dataset times, so score ours from the benchmark's files. No GPU on the lab VM? Run DPVO on any GPU machine and copy `fr1_xyz.txt` over; scoring and the RViz overlay run on the CPU.

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
| `/camera/depth_sensor/image_rect_raw` (`depth:=learned` only)       | `sensor_msgs/Image` (32FC1 m)       | player (reference depth)     |
| `/depth_estimator/debug`                                            | `sensor_msgs/Image` (bgr8)          | `depth_estimator_node`       |
| `/dl_slam/path`                                                     | `nav_msgs/Path`                     | `trajectory_overlay_node`    |
| `/diagnostics`                                                      | `diagnostic_msgs/DiagnosticArray`   | all nodes                    |
