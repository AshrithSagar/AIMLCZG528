"""
Shared launch composition (DRY): every lab launch file = simulator + robot
description + map + RViz + evaluator, plus its experiment-specific nodes.

    sim := kinematic   lightweight Python simulator (no GPU needed)
    sim := gazebo      gz-sim Harmonic + ros_gz_bridge (physics + gpu_lidar)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from .core import SdfWorldWriter, WorldSpec


@dataclass(frozen=True)
class FromArg:
    """A node parameter taken from a launch argument, cast to a concrete type."""

    arg: str
    cast: Callable[[str], Any] = str


@dataclass(frozen=True)
class NodeSpec:
    """An experiment node; built once the backend (and hence use_sim_time) is known."""

    executable: str
    params: dict[str, Any] = field(default_factory=dict)

    def resolve(self, context: LaunchContext) -> dict[str, Any]:
        return {k: (v.cast(LaunchConfiguration(v.arg).perform(context)) if isinstance(v, FromArg) else v)
                for k, v in self.params.items()}


class LabLaunchBuilder:
    PKG = "sensor_motion_lab"
    DESC_PKG = "mobile_bot_description"

    def __init__(self) -> None:
        self.desc_share = Path(get_package_share_directory(self.DESC_PKG))
        self.lab_share = Path(get_package_share_directory(self.PKG))
        self.world_yaml = self.desc_share / "config" / "world.yaml"

    # ----- public API --------------------------------------------------- #
    def build(self, extra: list[NodeSpec] | None = None, extra_args: list[DeclareLaunchArgument] | None = None,
              sim_params: dict[str, Any] | None = None) -> LaunchDescription:
        args = [
            DeclareLaunchArgument("sim", default_value="kinematic", choices=["kinematic", "gazebo"],
                                  description="Simulator backend."),
            DeclareLaunchArgument("rviz", default_value="true", description="Open RViz with the lab layout."),
        ] + (extra_args or [])
        return LaunchDescription(args + [OpaqueFunction(function=self._setup, kwargs={
            "extra": extra or [], "sim": NodeSpec("kinematic_sim_node", sim_params or {})})])

    def node(self, executable: str, params: dict[str, Any] | None = None) -> Node:
        return Node(package=self.PKG, executable=executable, output="screen", parameters=[params or {}])

    # ----- composition -------------------------------------------------- #
    def _setup(self, context: LaunchContext, extra: list[NodeSpec], sim: NodeSpec) -> list[Action]:
        gazebo = LaunchConfiguration("sim").perform(context) == "gazebo"
        sim_time = {"use_sim_time": gazebo}
        actions: list[Action] = [
            self._robot_state_publisher(gazebo),
            self.node("world_map_node", sim_time),
            self.node("trajectory_evaluator_node", sim_time),
            Node(package="rviz2", executable="rviz2", output="log",
                 arguments=["-d", str(self.lab_share / "rviz" / "lab.rviz")],
                 parameters=[sim_time], condition=IfCondition(LaunchConfiguration("rviz"))),
        ]
        actions += self._gazebo() if gazebo else [self.node(sim.executable, sim.resolve(context))]
        actions += [self.node(spec.executable, spec.resolve(context) | sim_time) for spec in extra]
        return actions

    def _robot_state_publisher(self, gazebo: bool) -> Node:
        xacro_file = self.desc_share / "urdf" / "mobile_bot.urdf.xacro"
        desc = ParameterValue(Command(["xacro ", str(xacro_file), f" sim_gazebo:={str(gazebo).lower()}"]),
                              value_type=str)
        return Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
                    parameters=[{"robot_description": desc, "use_sim_time": gazebo}])

    def _gazebo(self) -> list[Action]:
        sdf_path = Path(tempfile.gettempdir()) / "sensor_motion_world.sdf"
        SdfWorldWriter().write(WorldSpec.from_yaml(self.world_yaml), sdf_path)
        gz = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"),
                                                       "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": f"-r {sdf_path}", "on_exit_shutdown": "true"}.items(),
        )
        spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                     arguments=["-topic", "robot_description", "-name", "mobile_bot", "-z", "0.06"])
        bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                      parameters=[{"config_file": str(self.desc_share / "config" / "gz_bridge.yaml"),
                                   "use_sim_time": True}])
        return [gz, spawn, bridge]
