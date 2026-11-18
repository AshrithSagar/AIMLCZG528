"""
Evaluation metrics and the reference benchmarks they are compared against.

Motion / odometry
  * ATE, RPE           -- TUM RGB-D benchmark definitions (Sturm et al., IROS 2012);
                          the same numbers `evo_ape` / `evo_rpe` report.
  * Drift (% of path)  -- KITTI odometry style translational error.
  * UMBmark E_max,syst -- Borenstein & Feng (1996) square-path odometry benchmark.
  * NEES / 95% coverage-- consistency of a *probabilistic* motion model.

Sensing
  * Range bias / RMSE / precision vs. a LiDAR datasheet (ROBOTIS LDS-01,
    the TurtleBot3 LiDAR).
  * Likelihood-surface peak error and sharpness (map-matching quality).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .geometry import Angle, FloatArray


# --------------------------------------------------------------------------- #
# Trajectories                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Trajectory:
    stamps: FloatArray  # (T,) seconds
    poses: FloatArray  # (T, 3) x, y, theta

    @classmethod
    def from_lists(cls, stamps: list[float], poses: list[tuple[float, float, float]]) -> Trajectory:
        return cls(np.asarray(stamps, dtype=np.float64), np.asarray(poses, dtype=np.float64).reshape(-1, 3))

    def __len__(self) -> int:
        return int(self.stamps.size)

    @property
    def path_length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1)))

    def interpolate(self, stamps: FloatArray) -> Trajectory:
        x = np.interp(stamps, self.stamps, self.poses[:, 0])
        y = np.interp(stamps, self.stamps, self.poses[:, 1])
        th = np.interp(stamps, self.stamps, np.unwrap(self.poses[:, 2]))
        return Trajectory(stamps, np.column_stack((x, y, Angle.wrap_array(th))))

    def write_tum(self, path: str | Path) -> Path:
        """TUM format: 'stamp tx ty tz qx qy qz qw' -> directly usable with `evo`."""
        out = Path(path)
        qz, qw = np.sin(self.poses[:, 2] / 2.0), np.cos(self.poses[:, 2] / 2.0)
        rows = np.column_stack(
            (self.stamps, self.poses[:, 0], self.poses[:, 1], np.zeros(len(self)),
             np.zeros(len(self)), np.zeros(len(self)), qz, qw)
        )
        np.savetxt(out, rows, fmt="%.6f")
        return out


@dataclass(frozen=True, slots=True)
class TrajectoryReport:
    ate_rmse: float
    ate_max: float
    rpe_trans_rmse: float
    rpe_rot_rmse_deg: float
    final_error: float
    path_length: float
    drift_percent: float

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]


class TrajectoryEvaluator:
    """Compares an estimated trajectory to ground truth."""

    def __init__(self, align: bool = False, rpe_delta_m: float = 0.5) -> None:
        self.align = align
        self.rpe_delta_m = rpe_delta_m

    def evaluate(self, ground_truth: Trajectory, estimate: Trajectory) -> TrajectoryReport:
        gt, est = self._associate(ground_truth, estimate)
        est_xy = self._aligned_xy(gt.poses[:, :2], est.poses[:, :2]) if self.align else est.poses[:, :2]
        err = np.linalg.norm(gt.poses[:, :2] - est_xy, axis=1)
        rpe_t, rpe_r = self._rpe(gt, est)
        length = gt.path_length
        return TrajectoryReport(
            ate_rmse=float(np.sqrt(np.mean(err**2))),
            ate_max=float(np.max(err)),
            rpe_trans_rmse=rpe_t,
            rpe_rot_rmse_deg=rpe_r,
            final_error=float(err[-1]),
            path_length=length,
            drift_percent=float(100.0 * err[-1] / length) if length > 0 else float("nan"),
        )

    @staticmethod
    def _associate(gt: Trajectory, est: Trajectory) -> tuple[Trajectory, Trajectory]:
        t0, t1 = max(gt.stamps[0], est.stamps[0]), min(gt.stamps[-1], est.stamps[-1])
        mask = (gt.stamps >= t0) & (gt.stamps <= t1)
        gt_c = Trajectory(gt.stamps[mask], gt.poses[mask])
        return gt_c, est.interpolate(gt_c.stamps)

    @staticmethod
    def _aligned_xy(gt_xy: FloatArray, est_xy: FloatArray) -> FloatArray:
        """2-D Umeyama/Horn alignment (rotation + translation, no scale)."""
        mu_g, mu_e = gt_xy.mean(axis=0), est_xy.mean(axis=0)
        cov = (gt_xy - mu_g).T @ (est_xy - mu_e)
        u, _, vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(u @ vt))
        rot = u @ np.diag([1.0, d]) @ vt
        return (rot @ (est_xy - mu_e).T).T + mu_g

    def _rpe(self, gt: Trajectory, est: Trajectory) -> tuple[float, float]:
        """RPE over pairs separated by `rpe_delta_m` of travelled distance."""
        dist = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(gt.poses[:, :2], axis=0), axis=1))))
        j = np.searchsorted(dist, dist + self.rpe_delta_m)
        valid = j < len(gt)
        i, j = np.nonzero(valid)[0], j[valid]
        if i.size == 0:
            return float("nan"), float("nan")
        t_err, r_err = [], []
        for a, b in zip(i, j):
            dg = self._relative(gt.poses[a], gt.poses[b])
            de = self._relative(est.poses[a], est.poses[b])
            t_err.append(math.hypot(dg[0] - de[0], dg[1] - de[1]))
            r_err.append(Angle.wrap(dg[2] - de[2]))
        return float(np.sqrt(np.mean(np.square(t_err)))), float(np.degrees(np.sqrt(np.mean(np.square(r_err)))))

    @staticmethod
    def _relative(p: FloatArray, q: FloatArray) -> FloatArray:
        c, s = math.cos(p[2]), math.sin(p[2])
        dx, dy = q[0] - p[0], q[1] - p[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, Angle.wrap(q[2] - p[2])])


# --------------------------------------------------------------------------- #
# UMBmark                                                                     #
# --------------------------------------------------------------------------- #
class Direction(str, Enum):
    CW = "cw"
    CCW = "ccw"


@dataclass(frozen=True, slots=True)
class UMBmarkRun:
    direction: Direction
    error_x: float  # (true return position) - (start position), metres
    error_y: float


@dataclass(frozen=True, slots=True)
class UMBmarkReport:
    centroid_cw: tuple[float, float]
    centroid_ccw: tuple[float, float]
    r_cw: float
    r_ccw: float
    e_max_syst: float

    @classmethod
    def from_runs(cls, runs: list[UMBmarkRun]) -> UMBmarkReport:
        def centroid(d: Direction) -> tuple[float, float]:
            sel = [r for r in runs if r.direction == d]
            if not sel:
                return (float("nan"), float("nan"))
            return (float(np.mean([r.error_x for r in sel])), float(np.mean([r.error_y for r in sel])))

        c_cw, c_ccw = centroid(Direction.CW), centroid(Direction.CCW)
        r_cw, r_ccw = math.hypot(*c_cw), math.hypot(*c_ccw)
        return cls(c_cw, c_ccw, r_cw, r_ccw, float(np.nanmax([r_cw, r_ccw])))


# --------------------------------------------------------------------------- #
# Probabilistic consistency                                                   #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ConsistencyTracker:
    """
    Tracks whether ground truth is explained by the particle cloud.

    NEES (position) = e^T Σ^-1 e ; for a consistent 2-D estimator E[NEES] = 2,
    and ground truth should fall inside the 95 % ellipse (NEES <= 5.991) ~95 % of the time.
    """

    CHI2_95_2DOF: float = 5.991
    nees: list[float] = field(default_factory=list)

    def update(self, particles: FloatArray, gt_xy: tuple[float, float]) -> float:
        mu = particles[:, :2].mean(axis=0)
        cov = np.cov(particles[:, :2].T) + 1e-9 * np.eye(2)
        e = np.asarray(gt_xy) - mu
        value = float(e @ np.linalg.solve(cov, e))
        self.nees.append(value)
        return value

    @property
    def mean_nees(self) -> float:
        return float(np.mean(self.nees)) if self.nees else float("nan")

    @property
    def coverage_95(self) -> float:
        return float(np.mean(np.asarray(self.nees) <= self.CHI2_95_2DOF)) if self.nees else float("nan")


# --------------------------------------------------------------------------- #
# Range sensor characterisation                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LidarSpec:
    """Two-band datasheet accuracy spec (absolute below `split`, relative above)."""

    name: str
    range_min: float
    range_max: float
    split: float
    accuracy_abs: float  # metres, for range < split
    accuracy_rel: float  # fraction, for range >= split
    precision_abs: float
    precision_rel: float
    scan_rate_hz: float

    @classmethod
    def lds01(cls) -> LidarSpec:
        """ROBOTIS LDS-01 (TurtleBot3) — emanual.robotis.com, 'LDS-01' appendix."""
        return cls("ROBOTIS LDS-01", 0.12, 3.5, 0.5, 0.015, 0.05, 0.010, 0.035, 5.0)

    def accuracy_at(self, r: FloatArray) -> FloatArray:
        return np.where(r < self.split, self.accuracy_abs, self.accuracy_rel * r)

    def precision_at(self, r: FloatArray) -> FloatArray:
        return np.where(r < self.split, self.precision_abs, self.precision_rel * r)


@dataclass(frozen=True, slots=True)
class RangeErrorReport:
    n: int
    bias: float
    rmse: float
    precision_std: float
    within_spec_accuracy: float  # fraction of readings within the datasheet accuracy band
    outlier_fraction: float      # |error| > 3x accuracy band  (short/random/max returns)

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in self.__slots__}  # type: ignore[attr-defined]


class RangeErrorAnalyzer:
    """Measured vs. expected (ray-cast ground truth) ranges, judged against a datasheet."""

    def __init__(self, spec: LidarSpec) -> None:
        self.spec = spec

    def analyze(self, measured: FloatArray, expected: FloatArray) -> RangeErrorReport:
        m, e = np.ravel(measured), np.ravel(expected)
        ok = (e >= self.spec.range_min) & (e < self.spec.range_max)
        m, e = m[ok], e[ok]
        err = m - e
        band = self.spec.accuracy_at(e)
        inlier = np.abs(err) <= 3 * band
        return RangeErrorReport(
            n=int(err.size),
            bias=float(np.mean(err[inlier])) if inlier.any() else float("nan"),
            rmse=float(np.sqrt(np.mean(err[inlier] ** 2))) if inlier.any() else float("nan"),
            precision_std=float(np.std(err[inlier])) if inlier.any() else float("nan"),
            within_spec_accuracy=float(np.mean(np.abs(err) <= band)) if err.size else float("nan"),
            outlier_fraction=float(np.mean(~inlier)) if err.size else float("nan"),
        )


# --------------------------------------------------------------------------- #
# Likelihood surface (map matching)                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LikelihoodSurfaceReport:
    peak_error_m: float     # distance between argmax pose and the true pose
    sharpness_m: float      # posterior std (sqrt of trace of cov) of the normalised surface
    compute_ms: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in self.__slots__}  # type: ignore[attr-defined]


class LikelihoodSurfaceAnalyzer:
    @staticmethod
    def analyze(xy: FloatArray, log_lik: FloatArray, true_xy: tuple[float, float],
                compute_ms: float) -> LikelihoodSurfaceReport:
        w = np.exp(log_lik - np.max(log_lik))
        w /= w.sum()
        best = xy[int(np.argmax(log_lik))]
        mu = (w[:, None] * xy).sum(axis=0)
        var = (w[:, None] * (xy - mu) ** 2).sum(axis=0)
        return LikelihoodSurfaceReport(
            peak_error_m=float(np.hypot(*(best - np.asarray(true_xy)))),
            sharpness_m=float(np.sqrt(var.sum())),
            compute_ms=compute_ms,
        )
