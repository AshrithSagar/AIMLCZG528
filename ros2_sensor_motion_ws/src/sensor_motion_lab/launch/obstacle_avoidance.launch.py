"""
Lab Manual 2 experiment (odometry-based motion + obstacle avoidance), on the lab stack.

    ros2 launch sensor_motion_lab obstacle_avoidance.launch.py
    ros2 launch sensor_motion_lab obstacle_avoidance.launch.py duration:=30.0 sim:=gazebo

RViz: green = ground truth, red = wheel odometry (watch them separate).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument

from sensor_motion_lab.launch_builder import FromArg, LabLaunchBuilder, NodeSpec


def generate_launch_description() -> LaunchDescription:
    return LabLaunchBuilder().build(
        extra=[NodeSpec("obstacle_avoidance_node", {"duration": FromArg("duration", float),
                                                    "threshold": FromArg("threshold", float)})],
        extra_args=[
            DeclareLaunchArgument("duration", default_value="0.0", description="stop after N s (0 = never)"),
            DeclareLaunchArgument("threshold", default_value="0.5", description="obstacle distance [m]"),
        ],
    )
