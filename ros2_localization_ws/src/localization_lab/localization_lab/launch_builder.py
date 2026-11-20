"""
W2 launch composition. Reuses W1's `sim.launch.py` (an *overlay* including an
*underlay*'s launch file) and adds a localizer + evaluators on top.

    method := grid | mcl | mcl_ml | amcl
    drive  := avoid (W1's obstacle_avoidance_node wanders) | teleop (you drive)
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_MODEL = str(Path.home() / "w2_models" / "pose_regressor.npz")


class LocalizationLaunchBuilder:
    PKG = "localization_lab"
    W1_PKG = "sensor_motion_lab"
    METHODS = ("grid", "mcl", "mcl_ml", "amcl")

    def __init__(self) -> None:
        self.share = Path(get_package_share_directory(self.PKG))
        self.w1_share = Path(get_package_share_directory(self.W1_PKG))

    def build(self) -> LaunchDescription:
        args = [
            DeclareLaunchArgument("method", default_value="mcl", choices=list(self.METHODS)),
            DeclareLaunchArgument("sim", default_value="kinematic", choices=["kinematic", "gazebo"]),
            DeclareLaunchArgument("drive", default_value="avoid", choices=["avoid", "teleop"]),
            DeclareLaunchArgument("init", default_value="global", choices=["global", "origin"],
                                  description="global = robot does not know where it starts"),
            DeclareLaunchArgument("ml", default_value="true", description="also run the ML-only pose regressor"),
            DeclareLaunchArgument("model_path", default_value=DEFAULT_MODEL),
            DeclareLaunchArgument("rviz", default_value="true"),
        ]
        w1_sim = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(self.w1_share / "launch" / "sim.launch.py")),
            launch_arguments={"sim": LaunchConfiguration("sim"), "rviz": "false", "map_odom_tf": "false"}.items(),
        )
        return LaunchDescription(args + [w1_sim, OpaqueFunction(function=self._setup)])

    def _arg(self, ctx: LaunchContext, name: str) -> str:
        return LaunchConfiguration(name).perform(ctx)

    def _setup(self, ctx: LaunchContext) -> list[Action]:
        method = self._arg(ctx, "method")
        common = {"use_sim_time": self._arg(ctx, "sim") == "gazebo"}
        model = self._arg(ctx, "model_path")
        model_ok = Path(model).expanduser().is_file()
        actions: list[Action] = [
            Node(package="rviz2", executable="rviz2", output="log", parameters=[common],
                 arguments=["-d", str(self.share / "rviz" / "localization.rviz")],
                 condition=IfCondition(LaunchConfiguration("rviz"))),
        ]
        if self._arg(ctx, "drive") == "avoid":
            actions.append(Node(package=self.W1_PKG, executable="obstacle_avoidance_node", output="screen",
                                parameters=[common | {"threshold": 0.7, "front_sector_deg": 35.0, "turn_rate": 0.8,
                                                      "forward_speed": 0.25}]))
        if method == "amcl":
            actions += self._amcl(common)
            estimate = "/amcl_pose"
        else:
            actions.append(Node(package=self.PKG, executable="localizer_node", output="screen", parameters=[common | {
                "method": method, "model_path": model, "init": self._arg(ctx, "init")}]))
            estimate = "/localization/pose"
        actions.append(self._evaluator(method, estimate, common))
        if self._arg(ctx, "ml").lower() == "true" and model_ok:
            actions += [Node(package=self.PKG, executable="pose_regressor_node", output="screen",
                             parameters=[common | {"model_path": model}]),
                        self._evaluator("ml_only", "/ml/pose", common)]
        return actions

    def _evaluator(self, label: str, topic: str, common: dict) -> Node:
        return Node(package=self.PKG, executable="localization_evaluator_node", name=f"evaluator_{label}",
                    output="screen", parameters=[common | {"label": label, "estimate_topic": topic}])

    def _amcl(self, common: dict) -> list[Action]:
        """Nav2 AMCL is a lifecycle node: the lifecycle manager configures + activates it."""
        return [
            Node(package="nav2_amcl", executable="amcl", name="amcl", output="screen",
                 parameters=[str(self.share / "config" / "amcl.yaml"), common]),
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_localization",
                 output="screen", parameters=[common | {"autostart": True, "node_names": ["amcl"]}]),
        ]
