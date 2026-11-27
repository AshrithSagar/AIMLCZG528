"""
CS-3 worked example in RViz: the four measurement models scoring the same three returns.

    ros2 launch sensor_motion_lab model_comparison.launch.py            # Case A: clear aisle
    ros2 launch sensor_motion_lab model_comparison.launch.py case:=B    # Case B: a person blocks the centre beam
    ros2 param set /model_comparison_node case B                        # switch live
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    rviz = str(Path(get_package_share_directory("sensor_motion_lab")) / "rviz" / "comparison.rviz")
    return LaunchDescription([
        DeclareLaunchArgument("case", default_value="A", choices=["A", "B"]),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(package="sensor_motion_lab", executable="model_comparison_node", output="screen",
             parameters=[{"case": LaunchConfiguration("case")}]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz], output="log",
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
