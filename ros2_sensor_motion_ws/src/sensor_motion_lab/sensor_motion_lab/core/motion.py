"""
Motion modelling.

* Differential-drive kinematics, with Borenstein's systematic-error
  parameters (E_d, E_b) so we can reproduce the UMBmark experiment.
* Probabilistic motion models from Probabilistic Robotics (Thrun et al.), Ch. 5:
    - VelocityMotionModel  (Table 5.3, sample_motion_model_velocity)
    - OdometryMotionModel  (Table 5.6, sample_motion_model_odometry)
  Both draw *samples* x_t ~ p(x_t | u_t, x_{t-1}) for a whole particle set at once.

The noise parameters use the same alpha1..alpha4 convention as Nav2 AMCL's
`nav2_amcl::DifferentialMotionModel`, so values can be compared 1:1.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from .geometry import Angle, FloatArray, Pose2D


# --------------------------------------------------------------------------- #
# Kinematics                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DriveGeometry:
    wheel_radius: float
    wheel_separation: float


@dataclass(frozen=True, slots=True)
class DriveCalibrationError:
    """
    Systematic odometry errors in Borenstein & Feng's (UMBmark) notation.

    diameter_ratio  E_d = D_R / D_L  (unequal wheel diameters  -> curved straight lines)
    baseline_ratio  E_b = b_actual / b_nominal (wheelbase error -> wrong turn angles)
    """

    diameter_ratio: float = 1.0
    baseline_ratio: float = 1.0

    def actual_geometry(self, nominal: DriveGeometry) -> tuple[float, float, float]:
        """(r_left, r_right, separation) of the *physical* robot; mean diameter is preserved."""
        r = nominal.wheel_radius
        r_left = 2.0 / (self.diameter_ratio + 1.0) * r
        r_right = 2.0 / (1.0 / self.diameter_ratio + 1.0) * r
        return r_left, r_right, nominal.wheel_separation * self.baseline_ratio


class DifferentialDrive:
    """Twist <-> wheel-speed conversions for a nominal or miscalibrated drive."""

    def __init__(self, nominal: DriveGeometry, error: DriveCalibrationError | None = None) -> None:
        self.nominal = nominal
        self.error = error or DriveCalibrationError()
        self.r_left, self.r_right, self.separation = self.error.actual_geometry(nominal)

    def wheel_speeds(self, v: float, w: float) -> tuple[float, float]:
        """Inverse kinematics using the *nominal* model (what the controller believes)."""
        b, r = self.nominal.wheel_separation, self.nominal.wheel_radius
        return (v - w * b / 2.0) / r, (v + w * b / 2.0) / r

    def nominal_twist(self, omega_l: float, omega_r: float) -> tuple[float, float]:
        """Forward kinematics with the nominal model (= what wheel odometry reports)."""
        b, r = self.nominal.wheel_separation, self.nominal.wheel_radius
        return r * (omega_r + omega_l) / 2.0, r * (omega_r - omega_l) / b

    def actual_twist(self, omega_l: float, omega_r: float) -> tuple[float, float]:
        """Forward kinematics with the *physical* geometry (= what the robot really does)."""
        v_l, v_r = self.r_left * omega_l, self.r_right * omega_r
        return (v_r + v_l) / 2.0, (v_r - v_l) / self.separation


class ArcIntegrator:
    """Exact constant-twist (circular-arc) integration on SE(2)."""

    EPS: float = 1e-9

    @classmethod
    def step(cls, pose: Pose2D, v: float, w: float, dt: float) -> Pose2D:
        if abs(w) < cls.EPS:
            return Pose2D(
                pose.x + v * dt * math.cos(pose.theta),
                pose.y + v * dt * math.sin(pose.theta),
                pose.theta,
            )
        r = v / w
        th1 = pose.theta + w * dt
        return Pose2D(
            pose.x - r * math.sin(pose.theta) + r * math.sin(th1),
            pose.y + r * math.cos(pose.theta) - r * math.cos(th1),
            Angle.wrap(th1),
        )


# --------------------------------------------------------------------------- #
# Probabilistic motion models                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MotionNoise:
    """
    alpha1: rot  <- rot      alpha2: rot  <- trans
    alpha3: trans<- trans    alpha4: trans<- rot
    alpha5/alpha6: final-rotation noise (velocity model only).
    """

    alpha1: float = 0.2
    alpha2: float = 0.2
    alpha3: float = 0.2
    alpha4: float = 0.2
    alpha5: float = 0.0
    alpha6: float = 0.0

    @classmethod
    def nav2_amcl_defaults(cls) -> MotionNoise:
        """Defaults of Nav2 AMCL (docs.nav2.org, 'Configuring AMCL')."""
        return cls(0.2, 0.2, 0.2, 0.2)


@dataclass(frozen=True, slots=True)
class VelocityControl:
    v: float
    w: float
    dt: float


@dataclass(frozen=True, slots=True)
class OdometryControl:
    previous: Pose2D
    current: Pose2D


ControlT = TypeVar("ControlT", VelocityControl, OdometryControl)


class MotionModel(ABC, Generic[ControlT]):
    """Samples x_t ~ p(x_t | u_t, x_{t-1}) for an (N, 3) particle array."""

    def __init__(self, noise: MotionNoise, rng: np.random.Generator | None = None) -> None:
        self.noise = noise
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def sample(self, particles: FloatArray, control: ControlT) -> FloatArray: ...

    def _gauss(self, variance: FloatArray | float, n: int) -> FloatArray:
        return self.rng.normal(0.0, np.sqrt(np.maximum(variance, 0.0)), size=n)


class VelocityMotionModel(MotionModel[VelocityControl]):
    """Probabilistic Robotics, Table 5.3."""

    def sample(self, particles: FloatArray, control: VelocityControl) -> FloatArray:
        a, n = self.noise, particles.shape[0]
        v, w, dt = control.v, control.w, control.dt
        v_hat = v + self._gauss(a.alpha1 * v**2 + a.alpha2 * w**2, n)
        w_hat = w + self._gauss(a.alpha3 * v**2 + a.alpha4 * w**2, n)
        g_hat = self._gauss(a.alpha5 * v**2 + a.alpha6 * w**2, n)

        x, y, th = particles[:, 0], particles[:, 1], particles[:, 2]
        straight = np.abs(w_hat) < 1e-6
        safe_w = np.where(straight, 1.0, w_hat)
        r = v_hat / safe_w
        th1 = th + w_hat * dt
        x_new = np.where(straight, x + v_hat * dt * np.cos(th), x - r * np.sin(th) + r * np.sin(th1))
        y_new = np.where(straight, y + v_hat * dt * np.sin(th), y + r * np.cos(th) - r * np.cos(th1))
        th_new = Angle.wrap_array(th1 + g_hat * dt)
        return np.column_stack((x_new, y_new, th_new))


class OdometryMotionModel(MotionModel[OdometryControl]):
    """Probabilistic Robotics, Table 5.6 (same decomposition as Nav2's DifferentialMotionModel)."""

    MIN_TRANS: float = 0.01  # below this, rot1 is ill-defined -> treat as pure rotation

    def sample(self, particles: FloatArray, control: OdometryControl) -> FloatArray:
        a, n = self.noise, particles.shape[0]
        p0, p1 = control.previous, control.current
        dx, dy = p1.x - p0.x, p1.y - p0.y
        trans = math.hypot(dx, dy)
        rot1 = 0.0 if trans < self.MIN_TRANS else Angle.wrap(math.atan2(dy, dx) - p0.theta)
        rot2 = Angle.wrap(p1.theta - p0.theta - rot1)

        # Treat driving backwards like driving forwards for noise purposes (as AMCL does).
        r1n = min(abs(rot1), abs(Angle.wrap(rot1 - math.pi)))
        r2n = min(abs(rot2), abs(Angle.wrap(rot2 - math.pi)))

        rot1_hat = rot1 - self._gauss(a.alpha1 * r1n**2 + a.alpha2 * trans**2, n)
        trans_hat = trans - self._gauss(a.alpha3 * trans**2 + a.alpha4 * (r1n**2 + r2n**2), n)
        rot2_hat = rot2 - self._gauss(a.alpha1 * r2n**2 + a.alpha2 * trans**2, n)

        th = particles[:, 2]
        return np.column_stack(
            (
                particles[:, 0] + trans_hat * np.cos(th + rot1_hat),
                particles[:, 1] + trans_hat * np.sin(th + rot1_hat),
                Angle.wrap_array(th + rot1_hat + rot2_hat),
            )
        )
