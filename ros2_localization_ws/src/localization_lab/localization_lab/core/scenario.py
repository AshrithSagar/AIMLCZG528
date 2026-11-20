"""
A reproducible localization scenario built from W1's simulator classes:
the robot wanders the W1 arena with W1's obstacle-avoidance controller, the
wheel odometry drifts (slip), and halfway through it is *kidnapped*
(teleported) while its odometry keeps counting as if nothing happened.

The run is recorded once and replayed into every localizer, so they all see
identical data (a fair comparison — the offline equivalent of a rosbag).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sensor_motion_lab.core import (
    AvoidanceConfig,
    DifferentialDrive,
    GridMap,
    GridRasterizer,
    KinematicRobot,
    Pose2D,
    RayCaster,
    ReactiveAvoidanceController,
    Scan,
    SimulatedLidar,
    SlipNoise,
    WorldSpec,
)
from sensor_motion_lab.core.config import RobotConfig

from .common import FreeSpace, W1Assets


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    duration_s: float = 120.0
    kidnap_at_s: float | None = 60.0
    dt: float = 0.02
    scan_rate_hz: float = 5.0
    wheel_slip_std: float = 0.03
    seed: int = 1


@dataclass(frozen=True, slots=True)
class Sample:
    t: float
    odom: Pose2D
    truth: Pose2D
    scan: Scan


@dataclass
class Recording:
    samples: list[Sample] = field(default_factory=list)
    kidnap_t: float | None = None
    kidnap_to: Pose2D | None = None


@dataclass(frozen=True)
class LabWorld:
    """Everything a localizer needs about the environment, loaded from W1's config files."""

    grid: GridMap
    robot: RobotConfig
    spec: WorldSpec

    @classmethod
    def load(cls, config_dir: str | Path | None = None, resolution: float = 0.05) -> LabWorld:
        d = Path(config_dir) if config_dir else W1Assets.config_dir()
        spec = WorldSpec.from_yaml(d / "world.yaml")
        return cls(GridRasterizer(resolution).rasterize(spec), RobotConfig.from_yaml(d / "robot.yaml"), spec)


class ScenarioRecorder:
    def __init__(self, world: LabWorld, config: ScenarioConfig = ScenarioConfig()) -> None:
        self.world, self.cfg = world, config

    def record(self) -> Recording:
        c, li = self.cfg, self.world.robot.lidar
        rng = np.random.default_rng(c.seed)
        robot = KinematicRobot(DifferentialDrive(self.world.robot.drive), SlipNoise(c.wheel_slip_std), rng)
        lidar = SimulatedLidar(RayCaster(self.world.grid, li.range_max), li, rng)
        driver = ReactiveAvoidanceController(AvoidanceConfig(forward_speed=0.25, turn_rate=0.8, threshold=0.7,
                                                             front_sector=math.radians(35)))
        rec = Recording()
        every = int(round(1.0 / (c.scan_rate_hz * c.dt)))
        angles = li.angles()
        ranges = lidar.measure(robot.true_pose)
        steps = int(c.duration_s / c.dt)
        for k in range(1, steps + 1):
            t = k * c.dt
            v, w = driver.command(ranges, angles)
            robot.step(v, w, c.dt)
            if c.kidnap_at_s is not None and rec.kidnap_t is None and t >= c.kidnap_at_s:
                rec.kidnap_t, rec.kidnap_to = t, self._far_pose(robot.true_pose, rng)
                robot.true_pose = rec.kidnap_to
            if k % every == 0:
                ranges = lidar.measure(robot.true_pose)
                scan = Scan.from_raw(ranges, li.angle_min, li.angle_increment, li.range_min, li.range_max)
                rec.samples.append(Sample(t, robot.odom_pose, robot.true_pose, scan))
        return rec

    def _far_pose(self, current: Pose2D, rng: np.random.Generator) -> Pose2D:
        free = FreeSpace(self.world.grid, clearance=0.5)
        for p in free.sample(500, rng):
            if math.hypot(p[0] - current.x, p[1] - current.y) > 2.5:
                return Pose2D.from_array(p)
        return Pose2D.from_array(free.sample(1, rng)[0])
