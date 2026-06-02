# launch/mapping.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription(
        [
            Node(
                package="occupancy_mapping",
                executable="mapping_node",
                name="mapping_node",
                output="screen",
            ),
            Node(package="rviz2", executable="rviz2", output="screen"),
        ]
    )
