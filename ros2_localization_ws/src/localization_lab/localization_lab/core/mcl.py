"""
Monte Carlo Localization — Probabilistic Robotics, Ch. 8.3 (Augmented MCL, Table 8.3).

  predict   W1's OdometryMotionModel samples every particle (Nav2 AMCL's alpha1..4)
  correct   W1's LikelihoodFieldModel (or beam model) weights every particle
  resample  low-variance resampling when the effective sample size drops
  recover   augmented MCL: when the short-term average likelihood falls below the
            long-term one (w_fast << w_slow), inject random particles — either
            uniformly over free space, or (ML-augmented) around a *learned* pose guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sensor_motion_lab.core import (
    GridMap,
    LikelihoodFieldModel,
    LikelihoodFieldParams,
    MotionNoise,
    OdometryControl,
    OdometryMotionModel,
    Pose2D,
    Scan,
)
from sensor_motion_lab.core.geometry import FloatArray

from .common import FreeSpace, Localizer, PoseEstimate, WeightedPoses


class PoseProposal(Protocol):
    """Anything that can guess a pose from a single scan (e.g. the learned regressor)."""

    def propose(self, scan: Scan) -> Pose2D | None: ...


@dataclass(frozen=True, slots=True)
class MclConfig:
    num_particles: int = 2000
    noise: MotionNoise = MotionNoise.nav2_amcl_defaults()
    max_beams: int = 60
    resample_threshold: float = 0.5  # resample when N_eff < threshold * N
    alpha_slow: float = 0.001  # augmented MCL = Nav2 AMCL recovery_alpha_slow/fast (Nav2 default 0.0 = off;
                               # 0.001 / 0.1 are the values its docs suggest)
    alpha_fast: float = 0.1
    proposal_std_xy: float = 0.3  # spread of particles injected around a learned guess
    proposal_std_theta: float = 0.3
    likelihood_temperature: float = 1.0


class MonteCarloLocalizer(Localizer):
    name = "mcl"

    def __init__(self, grid: GridMap, range_max: float, config: MclConfig = MclConfig(),
                 proposal: PoseProposal | None = None, seed: int = 0) -> None:
        super().__init__()
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.free = FreeSpace(grid)
        self.motion = OdometryMotionModel(config.noise, self.rng)
        self.sensor = LikelihoodFieldModel(grid, LikelihoodFieldParams.nav2_amcl_defaults(), range_max,
                                           config.max_beams)
        self.proposal = proposal
        if proposal is not None:
            self.name = "mcl_ml"
        n = config.num_particles
        self.particles = np.zeros((n, 3))
        self.weights = np.full(n, 1.0 / n)
        self.w_slow = self.w_fast = 0.0
        self.injected = 0
        self.reset_global()

    # ----- initialisation ------------------------------------------------- #
    def reset_global(self) -> None:
        self.particles = self.free.sample(self.cfg.num_particles, self.rng)
        self.weights[:] = 1.0 / self.cfg.num_particles
        self.w_slow = self.w_fast = 0.0

    def reset_at(self, pose: Pose2D, std_xy: float = 0.25, std_theta: float = 0.2) -> None:
        self.particles = self._gaussian(pose, self.cfg.num_particles, std_xy, std_theta)
        self.weights[:] = 1.0 / self.cfg.num_particles
        self.w_slow = self.w_fast = 0.0

    # ----- Bayes filter ---------------------------------------------------- #
    def predict(self, odom_prev: Pose2D, odom_curr: Pose2D) -> None:
        self.particles = self.motion.sample(self.particles, OdometryControl(odom_prev, odom_curr))

    def correct(self, scan: Scan) -> None:
        ll = self.sensor.log_likelihood(scan, self.particles) / self.cfg.likelihood_temperature
        # particles inside obstacles are impossible
        ll[~self.free.contains(self.particles[:, :2])] = -np.inf
        m = np.max(ll)
        if not np.isfinite(m):  # every particle is in a wall -> start over
            self.reset_global()
            return
        lik = np.exp(ll - m)
        w = self.weights * lik
        # average (un-normalised) likelihood drives augmented MCL; per-beam geometric mean keeps it O(1)
        avg = float(np.exp(m / max(self.cfg.max_beams, 1)) * (self.weights @ lik) ** (1.0 / max(self.cfg.max_beams, 1)))
        if self.w_slow == 0.0:
            self.w_slow = self.w_fast = avg
        self.w_slow += self.cfg.alpha_slow * (avg - self.w_slow)
        self.w_fast += self.cfg.alpha_fast * (avg - self.w_fast)
        self.weights = w / w.sum()
        if self.n_eff < self.cfg.resample_threshold * self.cfg.num_particles:
            self._resample(scan)

    def estimate(self) -> PoseEstimate:
        return WeightedPoses.estimate(self.particles, self.weights)

    @property
    def n_eff(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    # ----- resampling / recovery --------------------------------------------- #
    def _resample(self, scan: Scan) -> None:
        n = self.cfg.num_particles
        # low-variance (systematic) resampling: one random number, n evenly spaced pointers
        positions = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), positions), n - 1)
        self.particles = self.particles[idx]
        p_random = max(0.0, 1.0 - self.w_fast / self.w_slow) if self.w_slow > 0 and self.cfg.alpha_slow > 0 else 0.0
        n_inj = int(self.rng.binomial(n, min(p_random, 0.5)))
        self.injected = n_inj
        if n_inj:
            slots = self.rng.choice(n, n_inj, replace=False)
            self.particles[slots] = self._recovery_particles(n_inj, scan)
        self.weights[:] = 1.0 / n

    def _recovery_particles(self, n: int, scan: Scan) -> np.ndarray:
        guess = self.proposal.propose(scan) if self.proposal is not None else None
        if guess is None:
            return self.free.sample(n, self.rng)
        half = n // 2  # half around the learned guess, half uniform (the guess can be wrong)
        return np.vstack((self._gaussian(guess, half, self.cfg.proposal_std_xy, self.cfg.proposal_std_theta),
                          self.free.sample(n - half, self.rng)))

    def _gaussian(self, pose: Pose2D, n: int, std_xy: float, std_th: float) -> FloatArray:
        return np.column_stack((self.rng.normal(pose.x, std_xy, n), self.rng.normal(pose.y, std_xy, n),
                                self.rng.normal(pose.theta, std_th, n)))

    def global_localize_with_proposal(self, scan: Scan) -> bool:
        """ML 'one-shot' initialisation: seed the cloud around the learned guess (plus some uniform)."""
        if self.proposal is None:
            return False
        guess = self.proposal.propose(scan)
        if guess is None:
            return False
        self.particles = self._recovery_particles(self.cfg.num_particles, scan)
        self.weights[:] = 1.0 / self.cfg.num_particles
        return True

    @property
    def spread(self) -> float:
        return math.sqrt(float(np.trace(np.cov(self.particles[:, :2].T))))
