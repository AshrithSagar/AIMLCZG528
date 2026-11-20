# Path planning among moving people in ROS 2 (Webinar 4)

A*, D* Lite and a genetic-algorithm (GA) waypoint planner, each replanning while people walk across the arena. Every planner is scored against the **static-map optimum** (A\* + shortcut on the map with nobody in it) and against the **plan-once baseline** (`replan:=false`).

This is an **overlay** on the W1 workspace (`ros2_sensor_motion_ws`): the simulator, robot, map and LiDAR come from W1's `sensor_motion_lab`. W1's simulator already puts the people into `/scan`. With `localization:=mcl` the planner uses W2's localizer (`ros2_localization_ws`) instead of raw odometry.

```plaintext
ros2_planning_ws/
├── expected_output/                  # figures + summary.md + metrics.json from `planning_benchmark`
└── src/planning_lab/
    ├── planning_lab/core/            # ROS-free: grid (costmap), planners (A*, D* Lite), genetic (GA),
    │                                 #           navigation (people, pure pursuit, prediction), evaluation, metrics
    ├── planning_lab/ros/             # planner_node, path_follower_node, dynamic_obstacles_node, navigation_evaluator_node
    ├── planning_lab/benchmark.py     # planning_benchmark (offline, no ROS needed)
    ├── planning_lab/launch_builder.py
    ├── launch/planning.launch.py     # includes W1's sim.launch.py
    ├── rviz/planning.rviz
    └── test/
```

## Build (underlay → overlay)

```shell
conda deactivate
source /opt/ros/jazzy/setup.bash
sudo apt install -y python3-scipy python3-matplotlib

# 1. underlay: W1 (and W2 only if you want localization:=mcl)
cd ~/AIMLCZG528/ros2_sensor_motion_ws && colcon build --symlink-install && source install/setup.bash
cd ~/AIMLCZG528/ros2_localization_ws && colcon build --symlink-install && source install/setup.bash   # optional

# 2. overlay: W4
cd ~/AIMLCZG528/ros2_planning_ws && colcon build --symlink-install && source install/setup.bash
colcon test --packages-select planning_lab && colcon test-result --verbose
```

## Run

```shell
ros2 run planning_lab planning_benchmark --out ~/w4_expected     # offline comparison, ~10 min on 2 CPUs

ros2 launch planning_lab planning.launch.py                       # D* Lite, goal (2.2, 1.8), 3 people
ros2 launch planning_lab planning.launch.py planner:=astar
ros2 launch planning_lab planning.launch.py planner:=ga           # pure GA
ros2 launch planning_lab planning.launch.py planner:=ga_seeded    # GA, first population partly seeded from A*
ros2 launch planning_lab planning.launch.py replan:=false         # baseline: plan once, safety stop only
ros2 launch planning_lab planning.launch.py goal:=none            # set goals with RViz "2D Goal Pose"
ros2 launch planning_lab planning.launch.py people_speed:=0.3     # slow motion for explaining
ros2 launch planning_lab planning.launch.py localization:=mcl     # plan from W2's MCL estimate
```

While it runs:

```shell
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: map}, pose: {position: {x: -2.3, y: 0.8}}}"
ros2 topic echo /diagnostics          # planner: ms, work, path length/clearance/turning; evaluator: per-episode score
ros2 service call /navigation_evaluator_node/report std_srvs/srv/Trigger   # -> /tmp/w4_results/<label>_episodes.json
ros2 topic hz /plan                   # how often the planner actually replans
```

## Expected output

See `expected_output/summary.md` for the exact numbers of your run (2 CPUs here).

**Static map, 30 random start/goal pairs.** Reference = A\* + shortcut.

| planner                | length / reference (median, max) | ms (mean) | work              |
| ---------------------- | -------------------------------- | --------- | ----------------- |
| A\* (8-connected grid) | 1.07, 1.16                       | ~4        | ~410 nodes        |
| D\* Lite (first plan)  | 1.07, 1.16                       | ~60       | ~380 nodes        |
| GA                     | 1.00, 1.35                       | ~650      | ~4500 evaluations |
| GA seeded from A\*     | 1.00, 1.04                       | ~650      | ~4300 evaluations |

**Three people walking, 4 start/goal pairs, constant-velocity prediction 1.5 s.**

| strategy                     | collisions (robot moving) | time to goal | ms / replan (mean) |
| ---------------------------- | ------------------------- | ------------ | ------------------ |
| plan once (safety stop only) | 3                         | ~20 s        | —                  |
| A\* replanning               | 0                         | ~30 s        | ~3                 |
| D\* Lite replanning          | 0                         | ~30 s        | ~130               |
| GA seeded, warm-started      | 2                         | ~34 s        | ~300               |

**Compute:** everything is CPU-only (2 CPUs here). The GA is the heaviest part at a few hundred ms per replan. **W4 does not need a GPU.**

## Topics

| Topic                                       | Type                               | From → to                                                                      |
| ------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------ |
| `/map`                                      | `nav_msgs/OccupancyGrid` (latched) | W1 `world_map_node` → planner, evaluator                                       |
| `/goal_pose`                                | `geometry_msgs/PoseStamped`        | RViz "2D Goal Pose" / you → planner, evaluator                                 |
| `/obstacles`                                | `geometry_msgs/PoseArray`          | `dynamic_obstacles_node` → planner, follower, evaluator, W1 sim (into `/scan`) |
| TF `map → base_link`                        |                                    | W1 (static `map → odom` + odometry) or W2 localizer → planner, follower        |
| `/plan`                                     | `nav_msgs/Path` (latched)          | `planner_node` → `path_follower_node`                                          |
| `/cmd_vel`                                  | `geometry_msgs/Twist`              | `path_follower_node` → simulator                                               |
| `/planner/costmap`                          | `nav_msgs/OccupancyGrid`           | inflation + people + predicted positions (RViz)                                |
| `/planner/population`                       | `visualization_msgs/MarkerArray`   | the GA's current population (RViz)                                             |
| `/obstacles/markers`, `/follower/lookahead` | markers / `PointStamped`           | RViz                                                                           |
| `/diagnostics`                              | `diagnostic_msgs/DiagnosticArray`  | all nodes                                                                      |

**Gazebo (`sim:=gazebo`):** the people are not spawned in Gazebo, so the Gazebo LiDAR does not see them. The planner still gets them on `/obstacles`, as it would from a people tracker. The kinematic simulator (default) does put them into `/scan`.
