"""
Navigation among moving obstacles: obstacles, path following, safety stop, and an
offline simulator (W1's KinematicRobot) that replays the same scenario for every
replanning strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sensor_motion_lab.core import Angle, DifferentialDrive, KinematicRobot, Pose2D, SlipNoise
from sensor_motion_lab.core.config import RobotConfig
from sensor_motion_lab.core.geometry import FloatArray

from .evaluation import EpisodeConfig, EpisodeMonitor, EpisodeReport
from .grid import Cell, PlanningGrid
from .planners import Planner


# --------------------------------------------------------------------------- #
# Moving obstacles                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Walker:
    """A person walking back and forth between two points."""

    a: tuple[float, float]
    b: tuple[float, float]
    speed: float
    phase: float = 0.0  # fraction of the first leg already walked at t = 0

    def position(self, t: float) -> tuple[float, float]:
        length = math.dist(self.a, self.b)
        s = (self.phase * length + self.speed * t) % (2 * length)
        u = s / length if s <= length else 2 - s / length
        return (self.a[0] + u * (self.b[0] - self.a[0]), self.a[1] + u * (self.b[1] - self.a[1]))


@dataclass(frozen=True, slots=True)
class Crowd:
    walkers: tuple[Walker, ...]

    @classmethod
    def w1_arena(cls) -> Crowd:
        """Three people crossing the W1 arena's main routes (all inside free space)."""
        return cls((
            Walker((0.6, -2.4), (0.6, 2.4), 0.35, 0.0),
            Walker((-0.3, 1.0), (2.3, 1.0), 0.30, 0.5),
            Walker((-0.3, -1.0), (2.3, -1.0), 0.40, 0.2),
        ))

    def positions(self, t: float) -> FloatArray:
        return np.asarray([w.position(t) for w in self.walkers], dtype=np.float64).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Following                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FollowerConfig:
    lookahead: float = 0.4
    max_speed: float = 0.3
    max_turn: float = 1.5
    goal_tolerance: float = 0.15
    stop_distance: float = 0.55  # safety stop: obstacle centre this close in front -> halt
    stop_cone_deg: float = 50.0


class PurePursuit:
    """Pure-pursuit path follower with a 'collision monitor' style safety stop."""

    def __init__(self, config: FollowerConfig = FollowerConfig()) -> None:
        self.cfg = config
        self.path = np.zeros((0, 2))
        self.progress = 0
        self.carrot: FloatArray | None = None  # last look-ahead point (for visualisation)

    def set_path(self, path: FloatArray) -> None:
        self.path, self.progress = np.asarray(path, dtype=np.float64), 0

    def done(self, pose: Pose2D) -> bool:
        return len(self.path) > 0 and math.hypot(pose.x - self.path[-1, 0], pose.y - self.path[-1, 1]) < self.cfg.goal_tolerance

    def command(self, pose: Pose2D, obstacles: FloatArray) -> tuple[float, float, bool]:
        """(v, w, stopped_for_safety)."""
        if len(self.path) < 2 or self.done(pose):
            return 0.0, 0.0, False
        target = self.carrot = self._lookahead_point(np.array([pose.x, pose.y]))
        alpha = Angle.wrap(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.theta)
        if abs(alpha) > math.pi / 3:  # turn in place first
            return 0.0, math.copysign(self.cfg.max_turn * 0.6, alpha), False
        v = self.cfg.max_speed * max(0.3, math.cos(alpha))
        w = float(np.clip(2 * v * math.sin(alpha) / self.cfg.lookahead, -self.cfg.max_turn, self.cfg.max_turn))
        if self._blocked(pose, obstacles):
            return 0.0, 0.0, True
        return v, w, False

    def _lookahead_point(self, p: FloatArray) -> FloatArray:
        """Project the robot onto the path (never moving backwards), then walk `lookahead` metres along it."""
        best, best_d, best_t = self.progress, math.inf, 0.0
        for i in range(self.progress, len(self.path) - 1):
            a, b = self.path[i], self.path[i + 1]
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(a + t * ab - p))
            if d < best_d:
                best, best_d, best_t = i, d, t
        self.progress = best
        remaining = self.cfg.lookahead
        a = self.path[best] + best_t * (self.path[best + 1] - self.path[best])
        for i in range(best, len(self.path) - 1):
            b = self.path[i + 1]
            seg = float(np.linalg.norm(b - a))
            if seg >= remaining:
                return a + (b - a) * (remaining / max(seg, 1e-12))
            remaining -= seg
            a = b
        return self.path[-1]

    def _blocked(self, pose: Pose2D, obstacles: FloatArray) -> bool:
        if not len(obstacles):
            return False
        rel = obstacles - np.array([pose.x, pose.y])
        dist = np.linalg.norm(rel, axis=1)
        bearing = np.abs(Angle.wrap_array(np.arctan2(rel[:, 1], rel[:, 0]) - pose.theta))
        return bool(np.any((dist < self.cfg.stop_distance) & (bearing < math.radians(self.cfg.stop_cone_deg))))


# --------------------------------------------------------------------------- #
# Offline navigation run                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class NavConfig:
    dt: float = 0.05
    timeout_s: float = 120.0
    replan: bool = True
    horizon_m: float = 2.0  # replan if the path is blocked within this distance ahead
    min_replan_interval_s: float = 0.5
    robot_radius: float = 0.2  # for counting collisions
    obstacle_radius: float = 0.25
    prediction_s: float = 1.5  # reserve where each obstacle will be (constant-velocity prediction); 0 = off
    prediction_steps: int = 3


@dataclass
class NavLog:
    """What the offline run records: the episode score plus planner timing and the tracks for figures."""

    report: EpisodeReport
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    obstacle_tracks: list[FloatArray] = field(default_factory=list)
    replan_ms: list[float] = field(default_factory=list)
    replan_expanded: list[int] = field(default_factory=list)
    initial_ms: float = 0.0


class ObstaclePredictor:
    """Constant-velocity prediction from consecutive observations (what a tracker would give the planner)."""

    def __init__(self, horizon_s: float, steps: int) -> None:
        self.horizon_s, self.steps = horizon_s, steps
        self._prev: FloatArray | None = None
        self._t_prev = 0.0

    def predict(self, positions: FloatArray, t: float) -> FloatArray:
        """Current positions plus predicted future positions (all stamped into the cost grid)."""
        out = [positions]
        if self.horizon_s > 0 and self._prev is not None and self._prev.shape == positions.shape and t > self._t_prev:
            vel = (positions - self._prev) / (t - self._t_prev)
            for k in range(1, self.steps + 1):
                out.append(positions + vel * self.horizon_s * k / self.steps)
        self._prev, self._t_prev = positions.copy(), t
        return np.vstack(out)


class NavigationSimulator:
    """Plan once, then drive; when moving obstacles block the path ahead, ask the planner to replan."""

    def __init__(self, grid: PlanningGrid, robot: RobotConfig, crowd: Crowd, config: NavConfig = NavConfig()) -> None:
        self.grid, self.robot, self.crowd, self.cfg = grid, robot, crowd, config

    def run(self, planner: Planner, start: Pose2D, goal: tuple[float, float], seed: int = 0) -> NavLog:
        c = self.cfg
        sim = KinematicRobot(DifferentialDrive(self.robot.drive), SlipNoise(0.0), np.random.default_rng(seed), start)
        follower = PurePursuit()
        predictor = ObstaclePredictor(c.prediction_s, c.prediction_steps)
        monitor = EpisodeMonitor(goal, EpisodeConfig(c.robot_radius, c.obstacle_radius, follower.cfg.goal_tolerance))
        trajectory: list[tuple[float, float]] = []
        tracks: list[FloatArray] = []
        replan_ms: list[float] = []
        replan_expanded: list[int] = []

        self.grid.set_obstacles(predictor.predict(self.crowd.positions(0.0), 0.0))
        res = planner.plan((start.x, start.y), goal)
        pending: set[Cell] = set()
        monitor.step(0.0, (start.x, start.y), self.crowd.positions(0.0))
        if res.success:
            follower.set_path(res.path)
            t, last_replan = 0.0, -math.inf
            while t < c.timeout_s and not monitor.finished:
                obstacles = self.crowd.positions(t)
                pending |= self.grid.set_obstacles(predictor.predict(obstacles, t))
                pose = sim.true_pose
                if c.replan and t - last_replan >= c.min_replan_interval_s and \
                        not self.grid.path_free(follower.path, follower.progress, c.horizon_m):
                    r = planner.replan((pose.x, pose.y), goal, pending)
                    pending, last_replan = set(), t
                    replan_ms.append(r.ms)
                    replan_expanded.append(r.expanded)
                    if r.success:
                        follower.set_path(r.path)
                v, w, stopped = follower.command(pose, obstacles)
                sim.step(v, w, c.dt)
                t += c.dt
                xy = (sim.true_pose.x, sim.true_pose.y)
                monitor.step(t, xy, obstacles, stopped)
                trajectory.append(xy)
                tracks.append(obstacles.copy())
        return NavLog(monitor.report(), trajectory, tracks, replan_ms, replan_expanded, res.ms)
