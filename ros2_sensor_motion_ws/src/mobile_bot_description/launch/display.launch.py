"""
Step 1 of the webinar: *just look at the robot*.

robot_state_publisher (URDF -> TF) + joint_state_publisher_gui (sliders for the
wheel joints) + RViz. No simulator, no physics.

    ros2 launch mobile_bot_description display.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("mobile_bot_description")
    xacro_file = os.path.join(share, "urdf", "mobile_bot.urdf.xacro")
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(share, "rviz", "display.rviz")],
            ),
        ]
    )
