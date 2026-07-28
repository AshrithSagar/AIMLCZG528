"""
Runs map_server + all three planners at once, each remapped to its own
/path and /planner_markers topics, plus RViz2 with a config that shows
all three (astar=orange explored+path, rrt=purple tree, prm=green roadmap).

Click one "2D Pose Estimate" (start) and one "2D Goal Pose" (goal) in RViz2
-- all three planners subscribe to the same /initialpose and /goal_pose,
so they all replan simultaneously on the same query.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("path_planning_pkg")
    rviz_config = os.path.join(pkg_share, "rviz", "path_planning_all.rviz")

    return LaunchDescription(
        [
            Node(
                package="path_planning_pkg",
                executable="map_server_node",
                name="map_server_node",
                output="screen",
            ),
            Node(
                package="path_planning_pkg",
                executable="astar_planner_node",
                name="astar_planner_node",
                output="screen",
                remappings=[
                    ("/path", "/astar/path"),
                    ("/planner_markers", "/astar/markers"),
                ],
            ),
            Node(
                package="path_planning_pkg",
                executable="rrt_planner_node",
                name="rrt_planner_node",
                output="screen",
                remappings=[
                    ("/path", "/rrt/path"),
                    ("/planner_markers", "/rrt/markers"),
                ],
            ),
            Node(
                package="path_planning_pkg",
                executable="prm_planner_node",
                name="prm_planner_node",
                output="screen",
                remappings=[
                    ("/path", "/prm/path"),
                    ("/planner_markers", "/prm/markers"),
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
            ),
        ]
    )
