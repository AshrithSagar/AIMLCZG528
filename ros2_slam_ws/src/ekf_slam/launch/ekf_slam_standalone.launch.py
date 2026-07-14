import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_description():
    pkg_share = get_package_share_path("ekf_slam")

    # Use standard rviz configuration if package has one
    rviz_config = os.path.join(pkg_share, "rviz", "ekf_slam.rviz")

    return LaunchDescription(
        [
            Node(
                package="ekf_slam",
                executable="landmark_sensor_node",
                name="landmark_sensor_node",
                output="screen",
            ),
            Node(
                package="ekf_slam",
                executable="ekf_slam_node",
                name="ekf_slam_node",
                output="screen",
            ),
            Node(
                package="ekf_slam",
                executable="waypoint_driver_node",
                name="waypoint_driver_node",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config] if os.path.exists(rviz_config) else [],
                output="screen",
            ),
        ]
    )
