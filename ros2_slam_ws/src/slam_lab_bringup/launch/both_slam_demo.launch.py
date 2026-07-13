from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    visual_slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("cnn_visual_slam"),
                    "launch",
                    "visual_slam_cnn.launch.py",
                ]
            )
        ),
    )
    ekf_slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ekf_slam"), "launch", "ekf_slam.launch.py"]
            )
        ),
    )

    return LaunchDescription(
        [
            visual_slam_launch,
            ekf_slam_launch,
        ]
    )
