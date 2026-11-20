"""
Webinar 4: replanning around moving people, on top of the W1 simulator.

    ros2 launch planning_lab planning.launch.py                          # D* Lite, goal (2.2, 1.8)
    ros2 launch planning_lab planning.launch.py planner:=astar
    ros2 launch planning_lab planning.launch.py planner:=ga_seeded       # GA waypoints, A*-seeded
    ros2 launch planning_lab planning.launch.py replan:=false            # baseline: plan once
    ros2 launch planning_lab planning.launch.py goal:=none               # click "2D Goal Pose" in RViz
    ros2 launch planning_lab planning.launch.py localization:=mcl        # plan from W2's MCL estimate

New goal from the terminal:
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: map}, pose: {position: {x: -2.3, y: 0.8}}}"
"""

from launch import LaunchDescription

from planning_lab.launch_builder import PlanningLaunchBuilder


def generate_launch_description() -> LaunchDescription:
    return PlanningLaunchBuilder().build()
