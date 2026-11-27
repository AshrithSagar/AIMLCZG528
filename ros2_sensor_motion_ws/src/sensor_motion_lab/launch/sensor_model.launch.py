"""
Sensor-model experiment: LiDAR characterisation vs. LDS-01 + map matching with the four measurement models.

    ros2 launch sensor_motion_lab sensor_model.launch.py                       # likelihood_field
    ros2 launch sensor_motion_lab sensor_model.launch.py model:=beam sim:=gazebo
    ros2 launch sensor_motion_lab sensor_model.launch.py model:=feature           # all four CS-3 models

Drive around with teleop_twist_keyboard and watch /sensor_model/likelihood in RViz.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument

from sensor_motion_lab.launch_builder import FromArg, LabLaunchBuilder, NodeSpec


def generate_launch_description() -> LaunchDescription:
    return LabLaunchBuilder().build(
        extra=[NodeSpec("sensor_model_node", {"model": FromArg("model"), "max_beams": FromArg("max_beams", int)})],
        extra_args=[
            DeclareLaunchArgument("model", default_value="likelihood_field", choices=["beam", "likelihood_field", "correlation", "feature"]),
            DeclareLaunchArgument("max_beams", default_value="60", description="beams used per evaluation (Nav2 default 60)"),
        ],
    )
