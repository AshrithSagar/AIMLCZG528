"""
Grid (histogram) localization — Probabilistic Robotics, Ch. 8.2.

The belief is an explicit 3-D histogram  bel[k, i, j]  over heading bins (k)
and map cells (i, j). Every cell is a hypothesis, so it can represent *any*
multi-modal belief (great for global localization), at a cost that grows with
the number of cells — which is exactly what MCL fixes.

  predict  shift every heading slice by the odometry motion, then blur (motion noise)
  correct  multiply by the likelihood-field model evaluated at *every* cell
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift

from sensor_motion_lab.core import Angle, GridMap, LikelihoodFieldModel, LikelihoodFieldParams, MotionNoise, Pose2D, Scan
from sensor_motion_lab.core.geometry import FloatArray

from .common import FreeSpace, Localizer, PoseEstimate, WeightedPoses


@dataclass(frozen=True, slots=True)
class GridConfig:
    resolution: float = 0.1  # m per cell
    heading_bins: int = 36  # 10 degrees
    clearance: float = 0.15  # cells closer than this to an obstacle are impossible
    temperature: float = 6.0  # likelihood ** (1/T): the coarse grid cannot represent sharp peaks
    floor: float = 1e-4  # mix in a little uniform belief -> recovers from kidnapping
    max_beams: int = 30
    noise: MotionNoise = MotionNoise(0.2, 0.2, 0.2, 0.2)


class GridLocalizer(Localizer):
    name = "grid"

    def __init__(self, grid: GridMap, range_max: float, config: GridConfig = GridConfig()) -> None:
        super().__init__()
        self.cfg = config
        r = config.resolution
        self.x0, self.y0 = grid.origin_x, grid.origin_y
        self.w = int(round(grid.width * grid.resolution / r))
        self.h = int(round(grid.height * grid.resolution / r))
        xs = self.x0 + (np.arange(self.w) + 0.5) * r
        ys = self.y0 + (np.arange(self.h) + 0.5) * r
        self.cx, self.cy = np.meshgrid(xs, ys)  # (H, W)
        self.headings = -math.pi + (np.arange(config.heading_bins) + 0.5) * (2 * math.pi / config.heading_bins)
        self.bin = 2 * math.pi / config.heading_bins
        self.free = FreeSpace(grid).contains(np.column_stack((self.cx.ravel(), self.cy.ravel())),
                                             config.clearance).reshape(self.h, self.w)
        self.sensor = LikelihoodFieldModel(grid, LikelihoodFieldParams.nav2_amcl_defaults(), range_max, config.max_beams)
        # candidate poses for the correction step: every free cell x every heading
        fy, fx = np.nonzero(self.free)
        k = np.repeat(np.arange(config.heading_bins), fy.size)
        self._cand_idx = (k, np.tile(fy, config.heading_bins), np.tile(fx, config.heading_bins))
        self._cand = np.column_stack((self.cx[fy, fx][None].repeat(config.heading_bins, 0).ravel(),
                                      self.cy[fy, fx][None].repeat(config.heading_bins, 0).ravel(), self.headings[k]))
        self.bel = np.zeros((config.heading_bins, self.h, self.w))
        self._pending = Pose2D()  # odometry accumulated but not yet applied (avoids sub-cell smearing)
        self.reset_global()

    @property
    def num_cells(self) -> int:
        return int(self._cand.shape[0])

    # ----- initialisation ------------------------------------------------- #
    def reset_global(self) -> None:
        self.bel[:] = self.free[None]
        self._normalise()

    def reset_at(self, pose: Pose2D, std_xy: float = 0.25, std_theta: float = 0.2) -> None:
        d2 = (self.cx - pose.x) ** 2 + (self.cy - pose.y) ** 2
        dth = Angle.wrap_array(self.headings - pose.theta)
        self.bel = np.exp(-0.5 * dth[:, None, None] ** 2 / std_theta**2) * np.exp(-0.5 * d2 / std_xy**2)[None]
        self.bel *= self.free[None]
        self._normalise()

    # ----- Bayes filter ---------------------------------------------------- #
    def predict(self, odom_prev: Pose2D, odom_curr: Pose2D) -> None:
        self._pending = self._pending.compose(odom_prev.between(odom_curr))
        p = self._pending
        trans = math.hypot(p.x, p.y)
        if trans < self.cfg.resolution and abs(p.theta) < self.bin / 2:
            return
        rot1 = math.atan2(p.y, p.x) if trans > 1e-3 else 0.0
        rot2 = Angle.wrap(p.theta - rot1)
        out = np.empty_like(self.bel)
        for k, th in enumerate(self.headings):
            dx, dy = trans * math.cos(th + rot1), trans * math.sin(th + rot1)
            out[k] = shift(self.bel[k], (dy / self.cfg.resolution, dx / self.cfg.resolution), order=1, mode="constant")
        out = np.roll(out, int(round((rot1 + rot2) / self.bin)), axis=0)
        a = self.cfg.noise
        rot = abs(rot1) + abs(rot2)
        s_xy = max(math.sqrt(a.alpha3 * trans**2 + a.alpha4 * rot**2) / self.cfg.resolution, 0.5)
        s_th = max(math.sqrt(a.alpha1 * rot**2 + a.alpha2 * trans**2) / self.bin, 0.3)
        self.bel = gaussian_filter(out, sigma=(s_th, s_xy, s_xy), mode=("wrap", "constant", "constant"))
        self.bel *= self.free[None]
        self._pending = Pose2D()
        self._normalise()

    def correct(self, scan: Scan) -> None:
        ll = self.sensor.log_likelihood(scan, self._cand) / self.cfg.temperature
        lik = np.zeros_like(self.bel)
        lik[self._cand_idx] = np.exp(ll - ll.max())
        self.bel *= lik
        self._normalise()

    def estimate(self) -> PoseEstimate:
        k, i, j = np.unravel_index(int(np.argmax(self.bel)), self.bel.shape)
        poses = np.column_stack((self.cx[self._cand_idx[1:]], self.cy[self._cand_idx[1:]], self.headings[self._cand_idx[0]]))
        weights = self.bel[self._cand_idx]
        full = WeightedPoses.estimate(poses, weights + 1e-300)
        # report the mode (robust to multi-modal beliefs), refined by its neighbourhood
        near = (np.hypot(poses[:, 0] - self.cx[i, j], poses[:, 1] - self.cy[i, j]) < 0.4) & \
               (np.abs(Angle.wrap_array(poses[:, 2] - self.headings[k])) < 3 * self.bin)
        local = WeightedPoses.estimate(poses[near], weights[near] + 1e-300)
        return PoseEstimate(local.pose, full.covariance)

    # ----- helpers --------------------------------------------------------- #
    def _normalise(self) -> None:
        s = self.bel.sum()
        uniform = self.free[None] / (self.free.sum() * self.cfg.heading_bins)
        self.bel = uniform if s <= 0 or not np.isfinite(s) else (1 - self.cfg.floor) * self.bel / s + self.cfg.floor * uniform

    def belief_xy(self) -> FloatArray:
        """(H, W) marginal over heading, for visualisation."""
        return self.bel.sum(axis=0)

    @property
    def geometry(self) -> tuple[float, float, float, int, int]:
        """origin x, origin y, resolution, width, height (for an OccupancyGrid)."""
        return self.x0, self.y0, self.cfg.resolution, self.w, self.h
