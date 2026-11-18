"""
ROS-agnostic simulation building blocks, shared by
  * `kinematic_sim_node`   (lightweight alternative to Gazebo), and
  * `offline_benchmark`     (produces the expected-output figures without ROS).

Keeping them here means the ROS node and the offline script run *the same*
code, so the expected outputs shown at the start of the webinar are exactly
what students should see in RViz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .geometry import Angle, FloatArray, Pose2D
from .metrics import Direction
from .motion import ArcIntegrator, DifferentialDrive
from .world import RayCaster


# --------------------------------------------------------------------------- #
# Robot + LiDAR                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SlipNoise:
    """Random (non-systematic) wheel slip: multiplicative noise on each wheel speed."""

    wheel_speed_std: float = 0.02


@dataclass(frozen=True, slots=True)
class LidarConfig:
    samples: int = 360
    angle_min: float = -math.pi
    angle_max: float = math.pi
    range_min: float = 0.12
    range_max: float = 3.5
    noise_std: float = 0.01
    rate_hz: float = 5.0

    @property
    def angle_increment(self) -> float:
        return (self.angle_max - self.angle_min) / self.samples

    def angles(self) -> FloatArray:
        return self.angle_min + self.angle_increment * np.arange(self.samples)


class KinematicRobot:
    """
    A differential-drive robot whose *true* motion uses the physical (possibly
    miscalibrated) geometry, while its *wheel odometry* integrates the nominal
    geometry — exactly the mechanism UMBmark is designed to expose.
    """

    def __init__(self, drive: DifferentialDrive, slip: SlipNoise, rng: np.random.Generator,
                 start: Pose2D | None = None) -> None:
        self.drive = drive
        self.slip = slip
        self.rng = rng
        self.true_pose = start or Pose2D()
        self.odom_pose = start or Pose2D()
        self.wheel_angles = np.zeros(2)  # left, right (for /joint_states)
        self.last_twist = (0.0, 0.0)

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> None:
        wl, wr = self.drive.wheel_speeds(v_cmd, w_cmd)
        # What the ground contact actually does (random slip) ...
        wl_true = wl * (1.0 + self.rng.normal(0.0, self.slip.wheel_speed_std))
        wr_true = wr * (1.0 + self.rng.normal(0.0, self.slip.wheel_speed_std))
        v_t, w_t = self.drive.actual_twist(wl_true, wr_true)
        self.true_pose = ArcIntegrator.step(self.true_pose, v_t, w_t, dt)
        # ... and what the encoders report: the wheel *rotation* (slip is invisible
        # to them), converted to a twist with the *nominal* geometry.
        v_o, w_o = self.drive.nominal_twist(wl, wr)
        self.odom_pose = ArcIntegrator.step(self.odom_pose, v_o, w_o, dt)
        self.wheel_angles += np.array([wl, wr]) * dt
        self.last_twist = (v_o, w_o)


class SimulatedLidar:
    """Ideal ray-cast scan + Gaussian range noise (like Gazebo's gpu_lidar <noise>)."""

    def __init__(self, caster: RayCaster, config: LidarConfig, rng: np.random.Generator) -> None:
        self.caster = caster
        self.config = config
        self.rng = rng
        self._angles = config.angles()

    def expected(self, pose: Pose2D) -> FloatArray:
        return self.caster.cast(pose.as_array()[None, :], self._angles)[0]

    def measure(self, pose: Pose2D) -> FloatArray:
        z_star = self.expected(pose)
        z = z_star + self.rng.normal(0.0, self.config.noise_std, z_star.shape)
        z[z_star >= self.config.range_max] = math.inf  # REP-117: no return -> +inf
        return z


# --------------------------------------------------------------------------- #
# UMBmark square-path driver                                                  #
# --------------------------------------------------------------------------- #
class Phase(Enum):
    DRIVE = auto()
    TURN = auto()
    DONE = auto()


@dataclass(frozen=True, slots=True)
class SquarePathConfig:
    side: float = 2.0
    direction: Direction = Direction.CCW
    linear_speed: float = 0.2
    angular_speed: float = 0.5
    laps: int = 1


class SquarePathController:
    """
    Drives a square purely from *odometry* feedback (as in UMBmark): the robot
    believes it returned to the start; ground truth tells us how wrong it was.
    """

    def __init__(self, config: SquarePathConfig) -> None:
        self.config = config
        self.phase = Phase.DRIVE
        self.edges_done = 0
        self._segment_start: Pose2D | None = None
        self._sign = 1.0 if config.direction == Direction.CCW else -1.0

    @property
    def total_edges(self) -> int:
        return 4 * self.config.laps

    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE

    def command(self, odom: Pose2D) -> tuple[float, float]:
        """Return (v, w) given the current odometry pose."""
        if self.phase == Phase.DONE:
            return 0.0, 0.0
        if self._segment_start is None:
            self._segment_start = odom
        c = self.config
        if self.phase == Phase.DRIVE:
            remaining = c.side - self._segment_start.distance_to(odom)
            if remaining <= 0.0:
                self.edges_done += 1
                self._advance(Phase.TURN if self.edges_done < self.total_edges else Phase.DONE, odom)
                return 0.0, 0.0
            return min(c.linear_speed, max(remaining * 2.0, 0.02)), 0.0
        # TURN
        turned = abs(Angle.wrap(odom.theta - self._segment_start.theta))
        remaining = math.pi / 2.0 - turned
        if remaining <= 0.0:
            self._advance(Phase.DRIVE, odom)
            return 0.0, 0.0
        return 0.0, self._sign * min(c.angular_speed, max(remaining * 2.0, 0.05))

    def _advance(self, phase: Phase, odom: Pose2D) -> None:
        self.phase = phase
        self._segment_start = odom


# --------------------------------------------------------------------------- #
# Lab Manual 2 behaviour: reactive obstacle avoidance                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AvoidanceConfig:
    """
    Defaults reproduce Lab Manual 2's `moving_bot.py`: both wheels at 4 rad/s
    (r = 0.05 m -> v = 0.2 m/s); rotate in place (wheels -4 / +4 rad/s) when the
    front reading is <= 0.5 m.
    """

    forward_speed: float = 0.2
    turn_rate: float = 1.11
    threshold: float = 0.5
    front_sector: float = math.radians(10.0)  # half-width of the "front" cone (manual: centre beam only)


class ReactiveAvoidanceController:
    """Drive forward; rotate in place while something is within `threshold` in front."""

    def __init__(self, config: AvoidanceConfig) -> None:
        self.config = config

    def front_distance(self, ranges: FloatArray, angles: FloatArray) -> float:
        r = np.where(np.isfinite(ranges), ranges, np.inf)
        cone = np.abs(Angle.wrap_array(angles)) <= self.config.front_sector
        return float(np.min(r[cone])) if cone.any() else math.inf

    def command(self, ranges: FloatArray, angles: FloatArray) -> tuple[float, float]:
        if self.front_distance(ranges, angles) <= self.config.threshold:
            return 0.0, self.config.turn_rate
        return self.config.forward_speed, 0.0
