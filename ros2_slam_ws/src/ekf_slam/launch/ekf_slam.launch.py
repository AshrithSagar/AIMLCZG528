import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("ekf_slam")
    rviz_config = os.path.join(pkg_share, "config", "ekf_slam.rviz")

    use_rviz = LaunchConfiguration("rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz2 with the preset EKF-SLAM view.",
            ),
            Node(
                package="ekf_slam",
                executable="landmark_sim_node",
                name="landmark_sim_node",
                output="screen",
                parameters=[
                    {
                        "sim_rate_hz": 10.0,
                        "odom_v_noise_std": 0.03,
                        "odom_w_noise_std": 0.02,
                        "range_noise_std": 0.15,
                        "bearing_noise_std": 0.05,
                    }
                ],
            ),
            Node(
                package="ekf_slam",
                executable="ekf_slam_node",
                name="ekf_slam_node",
                output="screen",
                parameters=[
                    {
                        "q_diag": [0.1, 0.1, 0.05],
                        "r_diag": [0.2, 0.1],
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
