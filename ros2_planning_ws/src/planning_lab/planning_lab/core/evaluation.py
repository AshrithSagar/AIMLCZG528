"""
Episode scoring for navigation among moving obstacles — shared by the offline
`NavigationSimulator` and the live `navigation_evaluator_node`, so the numbers in
`expected_output/` and the numbers in RViz/diagnostics mean exactly the same thing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from sensor_motion_lab.core.geometry import FloatArray


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    robot_radius: float = 0.2
    obstacle_radius: float = 0.25
    goal_tolerance: float = 0.15
    moving_speed: float = 0.02  # m/s; below this the robot counts as stopped


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    success: bool
    time_s: float
    travelled_m: float
    collisions: int  # contacts while the robot was moving (the robot's fault)
    contacts_while_stopped: int  # a person walked into the stopped robot
    min_clearance_m: float
    stopped_s: float  # time spent halted by the safety stop

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


class EpisodeMonitor:
    """Feed it the true robot position and the obstacles each step; it counts contacts, clearance and progress."""

    def __init__(self, goal: tuple[float, float], config: EpisodeConfig = EpisodeConfig()) -> None:
        self.cfg, self.goal = config, goal
        self.t0: float | None = None
        self.t = 0.0
        self.last_xy: tuple[float, float] | None = None
        self.travelled = 0.0
        self.collisions = 0
        self.contacts_while_stopped = 0
        self.min_clearance = math.inf
        self.stopped_s = 0.0
        self.success = False
        self._in_contact = False

    @property
    def finished(self) -> bool:
        return self.success

    def step(self, t: float, xy: tuple[float, float], obstacles: FloatArray, safety_stop: bool = False) -> None:
        if self.success:
            return
        if self.t0 is None:
            self.t0, self.t, self.last_xy = t, t, xy
        dt = max(t - self.t, 0.0)
        assert self.last_xy is not None
        moved = math.dist(xy, self.last_xy)
        moving = dt > 0 and moved / dt > self.cfg.moving_speed
        self.travelled += moved
        self.stopped_s += dt if safety_stop else 0.0
        self.t, self.last_xy = t, xy
        gap = self._gap(xy, obstacles)
        self.min_clearance = min(self.min_clearance, gap)
        if gap < 0 and not self._in_contact:
            if moving:
                self.collisions += 1
            else:
                self.contacts_while_stopped += 1
        self._in_contact = gap < 0
        self.success = math.dist(xy, self.goal) < self.cfg.goal_tolerance

    def report(self) -> EpisodeReport:
        elapsed = 0.0 if self.t0 is None else self.t - self.t0
        return EpisodeReport(self.success, elapsed, self.travelled, self.collisions,
                             self.contacts_while_stopped, self.min_clearance, self.stopped_s)

    def _gap(self, xy: tuple[float, float], obstacles: FloatArray) -> float:
        if not len(obstacles):
            return math.inf
        d = float(np.min(np.linalg.norm(np.asarray(obstacles) - np.asarray(xy), axis=1)))
        return d - self.cfg.obstacle_radius - self.cfg.robot_radius
