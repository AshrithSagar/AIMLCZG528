import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("cnn_visual_slam")
    rviz_config = os.path.join(pkg_share, "config", "visual_slam.rviz")

    use_rviz = LaunchConfiguration("rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz2 with the preset visual-SLAM view.",
            ),
            Node(
                package="cnn_visual_slam",
                executable="city_sim_node",
                name="city_sim_node",
                output="screen",
                parameters=[{"sim_rate_hz": 10.0, "loop_tour": True}],
            ),
            Node(
                package="cnn_visual_slam",
                executable="cnn_feature_node",
                name="cnn_feature_node",
                output="screen",
                parameters=[{"min_matches": 20}],
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
