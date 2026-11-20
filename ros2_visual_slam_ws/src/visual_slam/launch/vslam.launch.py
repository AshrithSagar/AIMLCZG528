"""
RGB-D visual SLAM, end to end.

    ros2 launch visual_slam vslam.launch.py                                  # synthetic room (CPU)
    ros2 launch visual_slam vslam.launch.py source:=tum path:=~/datasets/rgbd_dataset_freiburg1_xyz
    ros2 launch visual_slam vslam.launch.py source:=gazebo                   # Gazebo RGB-D camera
    ros2 launch visual_slam vslam.launch.py source:=bag   # then: ros2 bag play <bag> --clock
    ros2 launch visual_slam vslam.launch.py feature_type:=sift loop_closure:=false
"""

from launch import LaunchDescription

from visual_slam.launch_builder import VslamLaunchBuilder


def generate_launch_description() -> LaunchDescription:
    return VslamLaunchBuilder().build()
