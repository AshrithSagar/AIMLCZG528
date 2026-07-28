import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("path_planning_pkg")
    rviz_config = os.path.join(pkg_share, "rviz", "path_planning.rviz")

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
                executable="rrt_planner_node",
                name="rrt_planner_node",
                output="screen",
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
