"""
navigation_evaluator_node — scores every navigation episode, live, against a reference.

  sub  /ground_truth/odom   nav_msgs/Odometry       (W1 sim / Gazebo: the *true* pose)
       /goal_pose           geometry_msgs/PoseStamped  (starts an episode)
       /obstacles           geometry_msgs/PoseArray
       /map                 nav_msgs/OccupancyGrid  (for the reference path)
  pub  /diagnostics         success, time, travelled / reference length, collisions, min clearance
  srv  ~/report             std_srvs/Trigger        (writes <output_dir>/episodes.json)

Reference ("benchmark") = A* + shortcut on the *static* map from the episode's start:
the shortest collision-free route if nobody were walking around. travelled / reference
is the price paid for dodging people; 1.0 is the floor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Trigger

from sensor_motion_lab.ros.common import LabNode, QoS

from ..core.evaluation import EpisodeConfig, EpisodeMonitor
from ..core.grid import PlanningGrid
from ..core.metrics import PathQuality
from ..core.planners import AStarPlanner
from .conv import PlanningConv


class NavigationEvaluatorNode(LabNode):
    def __init__(self) -> None:
        super().__init__("navigation_evaluator_node")
        self.label = self.param("label", "navigation")
        self.output_dir = Path(self.param("output_dir", "/tmp/w4_results")).expanduser()
        self.cfg = EpisodeConfig(goal_tolerance=self.param("goal_tolerance", 0.15))
        self.grid: PlanningGrid | None = None
        self.xy: tuple[float, float] | None = None
        self.obstacles = np.zeros((0, 2))
        self.monitor: EpisodeMonitor | None = None
        self.reference_m = math.nan
        self.episodes: list[dict[str, float]] = []

        self.create_subscription(OccupancyGrid, "/map", self._on_map, QoS.latched())
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_truth, 50)
        self.create_subscription(PoseArray, "/obstacles", self._on_obstacles, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_service(Trigger, "~/report", self._on_report)
        self.create_timer(self.param("report_period", 2.0), self._report)
        goal = self.param("goal", [0.0])
        self.pending_goal = (float(goal[0]), float(goal[1])) if len(goal) == 2 else None

    # ----- inputs ------------------------------------------------------------- #
    def _on_map(self, msg: OccupancyGrid) -> None:
        self.grid = PlanningConv.planning_grid(msg)  # static map only: this grid never sees the people

    def _on_obstacles(self, msg: PoseArray) -> None:
        self.obstacles = PlanningConv.points(msg)

    def _on_goal(self, msg: PoseStamped) -> None:
        self._start((msg.pose.position.x, msg.pose.position.y))

    def _on_truth(self, msg: Odometry) -> None:
        self.xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.pending_goal is not None and self.grid is not None:
            self._start(self.pending_goal)
            self.pending_goal = None
        if self.monitor is None or self.monitor.finished:
            return
        self.monitor.step(self._now(), self.xy, self.obstacles)
        if self.monitor.finished:
            self._close()

    # ----- episodes --------------------------------------------------------------- #
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _start(self, goal: tuple[float, float]) -> None:
        if self.monitor is not None and not self.monitor.finished and self.monitor.t0 is not None:
            self._close()  # a new goal aborts the running episode
        self.monitor = EpisodeMonitor(goal, self.cfg)
        self.reference_m = self._reference(goal)
        self.get_logger().info(f"[{self.label}] episode {len(self.episodes) + 1}: goal {goal}, "
                               f"reference (A*+shortcut, static map) {self.reference_m:.2f} m")

    def _reference(self, goal: tuple[float, float]) -> float:
        if self.grid is None or self.xy is None:
            return math.nan
        astar = AStarPlanner(self.grid)
        res = astar.plan(self.xy, goal)
        return PathQuality.of(astar.shortcut(res.path), astar.grid).length if res.success else math.nan

    def _row(self) -> dict[str, float]:
        assert self.monitor is not None
        rep = self.monitor.report().as_dict()
        return rep | {"reference_m": self.reference_m, "length_ratio": rep["travelled_m"] / self.reference_m
                      if self.reference_m > 0 else math.nan}

    def _close(self) -> None:
        row = self._row()
        self.episodes.append(row)
        self.diagnostics.publish(row | {"episodes": len(self.episodes)}, message=f"[{self.label}] episode finished")
        self._write()

    def _report(self) -> None:
        if self.monitor is not None and not self.monitor.finished and self.monitor.t0 is not None:
            self.diagnostics.publish(self._row(), message=f"[{self.label}] episode running", log=False)

    def _write(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.label}_episodes.json"
        path.write_text(json.dumps(self.episodes, indent=2))
        return path

    def _on_report(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        resp.success, resp.message = True, f"{len(self.episodes)} episode(s) -> {self._write()}"
        return resp
