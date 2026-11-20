"""
W4 launch composition: an *overlay* on W1 (simulator, robot, map) and, optionally,
W2 (localization). Adds people, a global planner that replans, a path follower
and an evaluator.

    planner      := astar | dstar_lite | ga | ga_seeded
    replan       := true | false          (false = plan once and hope: the baseline)
    localization := static | mcl          (static = W1's identity map->odom, odometry drift included;
                                           mcl = W2's localizer owns map->odom)
    goal         := "x,y" | none          (none = click "2D Goal Pose" in RViz)
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


class GoalArg:
    """Parses the `goal` launch argument ("x,y" or "none") into a ROS double-array parameter."""

    NONE: tuple[float, ...] = (0.0,)  # a 1-element list = "no goal" (ROS params cannot be empty typed arrays)

    @classmethod
    def parse(cls, text: str) -> list[float]:
        text = text.strip().lower()
        if text in ("", "none"):
            return list(cls.NONE)
        x, y = (float(v) for v in text.split(","))
        return [x, y]


class PlanningLaunchBuilder:
    PKG = "planning_lab"
    W1_PKG = "sensor_motion_lab"
    W2_PKG = "localization_lab"
    PLANNERS = ("astar", "dstar_lite", "ga", "ga_seeded")

    def __init__(self) -> None:
        self.share = Path(get_package_share_directory(self.PKG))
        self.w1_share = Path(get_package_share_directory(self.W1_PKG))

    def build(self) -> LaunchDescription:
        args = [
            DeclareLaunchArgument("planner", default_value="dstar_lite", choices=list(self.PLANNERS)),
            DeclareLaunchArgument("replan", default_value="true", choices=["true", "false"]),
            DeclareLaunchArgument("people", default_value="true", description="spawn the walking people"),
            DeclareLaunchArgument("people_speed", default_value="1.0", description="0 freezes them"),
            DeclareLaunchArgument("goal", default_value="2.2,1.8", description='"x,y" in map, or "none"'),
            DeclareLaunchArgument("localization", default_value="static", choices=["static", "mcl"]),
            DeclareLaunchArgument("sim", default_value="kinematic", choices=["kinematic", "gazebo"]),
            DeclareLaunchArgument("rviz", default_value="true"),
        ]
        return LaunchDescription(args + [OpaqueFunction(function=self._setup)])

    @staticmethod
    def _arg(ctx: LaunchContext, name: str) -> str:
        return LaunchConfiguration(name).perform(ctx)

    def _setup(self, ctx: LaunchContext) -> list[Action]:
        common = {"use_sim_time": self._arg(ctx, "sim") == "gazebo"}
        planner = self._arg(ctx, "planner")
        goal = GoalArg.parse(self._arg(ctx, "goal"))
        mcl = self._arg(ctx, "localization") == "mcl"
        actions: list[Action] = [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(self.w1_share / "launch" / "sim.launch.py")),
                launch_arguments={"sim": self._arg(ctx, "sim"), "rviz": "false",
                                  "map_odom_tf": "false" if mcl else "true"}.items()),
            Node(package="rviz2", executable="rviz2", output="log", parameters=[common],
                 arguments=["-d", str(self.share / "rviz" / "planning.rviz")],
                 condition=IfCondition(LaunchConfiguration("rviz"))),
            self._node("planner_node", common | {"planner": planner, "goal": goal,
                                                 "replan": self._arg(ctx, "replan") == "true"}),
            self._node("path_follower_node", common),
            self._node("navigation_evaluator_node", common | {
                "label": f"{planner}{'' if self._arg(ctx, 'replan') == 'true' else '_no_replan'}", "goal": goal}),
        ]
        if self._arg(ctx, "people") == "true":
            actions.append(self._node("dynamic_obstacles_node",
                                      common | {"speed_scale": float(self._arg(ctx, "people_speed"))}))
        if mcl:
            actions += self._localization(common)
        return actions

    def _node(self, executable: str, params: dict) -> Node:
        return Node(package=self.PKG, executable=executable, output="screen", parameters=[params])

    def _localization(self, common: dict) -> list[Action]:
        """W2's MCL localizer owns map->odom (needs the W2 workspace in the underlay chain)."""
        try:
            get_package_share_directory(self.W2_PKG)
        except PackageNotFoundError:
            return [LogInfo(msg="localization:=mcl needs the W2 workspace (localization_lab) sourced; "
                                "map->odom is missing, so the planner will wait for TF.")]
        return [Node(package=self.W2_PKG, executable="localizer_node", output="screen",
                     parameters=[common | {"method": "mcl", "init": "origin"}])]
