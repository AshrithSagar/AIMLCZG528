# Path Planning

Sampling-based planners (RRT, PRM) vs. search-based planning (A\*)\*\* on a shared occupancy grid.

```plaintext
src/path_planning_pkg/
  path_planning_pkg/
    planner_utils.py       # OccupancyGridMap, demo map generator, shared PlannerNodeBase
    map_server_node.py      # publishes the static demo map on /map (latched)
    astar_planner_node.py   # grid-based A* (8-connected, octile heuristic)
    rrt_planner_node.py     # RRT with goal biasing, live-tunable step size
    prm_planner_node.py     # PRM roadmap + Dijkstra query
  launch/
    astar_demo.launch.py
    rrt_demo.launch.py
    prm_demo.launch.py
    all_planners_demo.launch.py   # all three at once, side-by-side comparison
  rviz/
    path_planning.rviz            # single-planner view
    path_planning_all.rviz        # 3-planner comparison view
```

The demo map is a fixed 10m x 10m grid with a narrow bottleneck corridor, an S-shaped detour, and a few freestanding obstacles — chosen so A\*'s optimal sweep, RRT's exploring tree, and PRM's roadmap all look visibly different on the same problem.

## Running each planner individually

```bash
ros2 launch path_planning_pkg astar_demo.launch.py
ros2 launch path_planning_pkg rrt_demo.launch.py
ros2 launch path_planning_pkg prm_demo.launch.py
```

Each brings up: `map_server_node` + the chosen planner + RViz2.

In RViz2:

1. Click **"2D Pose Estimate"** in the toolbar, click+drag on the map to set the **start**.
2. Click **"2D Goal Pose"**, click+drag to set the **goal**.
3. The planner runs automatically as soon as both are set, and republishes every time you move either one — so you can drag the goal around live and watch it replan.

## Running all three side-by-side

```bash
ros2 launch path_planning_pkg all_planners_demo.launch.py
```

One start/goal click drives all three planners at once (orange = A\* explored cells + path, purple = RRT tree, green = PRM roadmap).

## Live-tunable parameters

```bash
# RRT — smaller steps = more careful but slower; try 0.1 vs 0.6
ros2 param set /rrt_planner_node step_size 0.6
ros2 param set /rrt_planner_node goal_bias 0.2

# PRM — fewer samples shows disconnection risk in the bottleneck
ros2 param set /prm_planner_node num_samples 60
ros2 param set /prm_planner_node connect_radius 1.0

# A* — heuristic_weight > 1 turns it into weighted A* (faster, suboptimal)
ros2 param set /astar_planner_node heuristic_weight 2.5
```

After changing a parameter, just re-click **"2D Goal Pose"** (even at the same spot) to trigger a fresh plan with the new value.
