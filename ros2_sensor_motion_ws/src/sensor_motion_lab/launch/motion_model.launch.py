"""
Motion-model experiment: UMBmark square drive + sampling motion model + ATE/RPE.

    ros2 launch sensor_motion_lab motion_model.launch.py
    ros2 launch sensor_motion_lab motion_model.launch.py model:=velocity alpha:=0.05
    ros2 launch sensor_motion_lab motion_model.launch.py baseline_ratio:=1.03 sim:=gazebo

(baseline_ratio only affects the kinematic sim at launch time; for Gazebo edit
 calibration.baseline_ratio in mobile_bot_description/config/robot.yaml.)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument

from sensor_motion_lab.launch_builder import FromArg, LabLaunchBuilder, NodeSpec


def generate_launch_description() -> LaunchDescription:
    a = FromArg("alpha", float)
    return LabLaunchBuilder().build(
        extra=[
            NodeSpec("umbmark_driver_node", {"side": FromArg("side", float), "runs_per_direction": FromArg("runs", int)}),
            NodeSpec("motion_model_node", {"model": FromArg("model"), "alpha1": a, "alpha2": a, "alpha3": a, "alpha4": a}),
        ],
        sim_params={"baseline_ratio": FromArg("baseline_ratio", float),
                    "diameter_ratio": FromArg("diameter_ratio", float)},
        extra_args=[
            DeclareLaunchArgument("model", default_value="odometry", choices=["odometry", "velocity"]),
            DeclareLaunchArgument("alpha", default_value="0.2", description="alpha1..4 (Nav2 AMCL default 0.2)"),
            DeclareLaunchArgument("side", default_value="2.0", description="UMBmark square side [m]"),
            DeclareLaunchArgument("runs", default_value="1", description="runs per direction (UMBmark uses 5)"),
            DeclareLaunchArgument("baseline_ratio", default_value="1.0", description="E_b for the kinematic sim"),
            DeclareLaunchArgument("diameter_ratio", default_value="1.0", description="E_d for the kinematic sim"),
        ],
    )
