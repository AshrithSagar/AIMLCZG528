"""
Bring up the simulator + robot + map + RViz (no experiment).

    ros2 launch sensor_motion_lab sim.launch.py                 # kinematic sim
    ros2 launch sensor_motion_lab sim.launch.py sim:=gazebo     # Gazebo Harmonic

Drive it:  ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

from launch import LaunchDescription

from sensor_motion_lab.launch_builder import LabLaunchBuilder


def generate_launch_description() -> LaunchDescription:
    return LabLaunchBuilder().build()
