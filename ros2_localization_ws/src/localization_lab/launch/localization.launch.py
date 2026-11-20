"""
Webinar 2: localization on top of the W1 simulator.

    ros2 launch localization_lab localization.launch.py                    # MCL, global localization
    ros2 launch localization_lab localization.launch.py method:=grid
    ros2 launch localization_lab localization.launch.py method:=mcl_ml     # ML-augmented MCL
    ros2 launch localization_lab localization.launch.py method:=amcl       # Nav2 AMCL (benchmark)
    ros2 launch localization_lab localization.launch.py sim:=gazebo drive:=teleop

Kidnap the robot (kinematic sim):
    ros2 topic pub --once /sim/teleport geometry_msgs/msg/PoseStamped "{pose: {position: {x: -2.0, y: -0.5}}}"
"""

from launch import LaunchDescription

from localization_lab.launch_builder import LocalizationLaunchBuilder


def generate_launch_description() -> LaunchDescription:
    return LocalizationLaunchBuilder().build()
