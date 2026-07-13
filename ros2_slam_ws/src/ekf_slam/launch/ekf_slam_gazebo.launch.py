import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("ekf_slam")
    world_path = os.path.join(pkg_share, "worlds", "landmarks.sdf")
    bridge_config = os.path.join(pkg_share, "config", "ekf_bridge.yaml")
    rviz_config = os.path.join(pkg_share, "config", "ekf_slam.rviz")

    use_rviz = LaunchConfiguration("rviz")

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
            )
        ),
        launch_arguments={"gz_args": f"-r {world_path}"}.items(),
    )

    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{"config_file": bridge_config}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz2 with the preset EKF-SLAM view.",
            ),
            gz_sim,
            bridge_node,
            Node(
                package="ekf_slam",
                executable="waypoint_driver_node",
                name="waypoint_driver_node",
                output="screen",
                parameters=[{"control_rate_hz": 10.0}],
            ),
            Node(
                package="ekf_slam",
                executable="landmark_sensor_node",
                name="landmark_sensor_node",
                output="screen",
                parameters=[
                    {
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
