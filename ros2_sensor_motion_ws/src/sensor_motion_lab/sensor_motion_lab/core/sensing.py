"""
Sensor (measurement) modelling for 2-D range finders.

* BeamRangeFinderModel      Probabilistic Robotics Table 6.1 (4-component mixture)
* LikelihoodFieldModel      Probabilistic Robotics Table 6.3 (end-point model)
* CorrelationModel          Probabilistic Robotics §6.5 (map matching)
* FeatureBasedModel         Probabilistic Robotics §6.6 (line features + data association)

Both evaluate log p(z | x, m) for many candidate poses x at once (the
"inverse" use inside a filter); the beam model can also *sample* scans
(the "forward"/generative use, i.e. a realistic noise injector).

Parameter names and defaults mirror Nav2 AMCL (`z_hit`, `z_short`, `z_max`,
`z_rand`, `sigma_hit`, `lambda_short`, `laser_likelihood_max_dist`,
`max_beams`) so results can be compared against that reference config.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .features import LineExtractor, LineExtractorParams, LineMap
from .geometry import Angle, FloatArray, Pose2D
from .world import GridMap, RayCaster, WorldSpec


@dataclass(frozen=True, slots=True)
class Scan:
    """A single planar scan (ranges already cleaned: inf/NaN -> range_max)."""

    ranges: FloatArray
    angles: FloatArray
    range_min: float
    range_max: float

    @classmethod
    def from_raw(cls, ranges: FloatArray, angle_min: float, angle_increment: float,
                 range_min: float, range_max: float) -> Scan:
        r = np.asarray(ranges, dtype=np.float64).copy()
        r[~np.isfinite(r)] = range_max
        r = np.clip(r, 0.0, range_max)
        angles = angle_min + angle_increment * np.arange(r.size)
        return cls(r, angles, range_min, range_max)

    def subsample(self, max_beams: int) -> Scan:
        if max_beams <= 0 or self.ranges.size <= max_beams:
            return self
        idx = np.linspace(0, self.ranges.size - 1, max_beams).round().astype(int)
        return Scan(self.ranges[idx], self.angles[idx], self.range_min, self.range_max)


@dataclass(frozen=True, slots=True)
class BeamModelParams:
    z_hit: float = 0.5
    z_short: float = 0.05
    z_max: float = 0.05
    z_rand: float = 0.5
    sigma_hit: float = 0.2
    lambda_short: float = 0.1

    @classmethod
    def nav2_amcl_defaults(cls) -> BeamModelParams:
        """Values from the Nav2 AMCL example configuration (nav2_params.yaml)."""
        return cls(0.5, 0.05, 0.05, 0.5, 0.2, 0.1)

    @classmethod
    def realistic_lidar(cls) -> BeamModelParams:
        """A 'real-world-ish' generative setting for noise injection (mostly hits)."""
        return cls(z_hit=0.90, z_short=0.05, z_max=0.02, z_rand=0.03, sigma_hit=0.02, lambda_short=1.0)

    def weights(self) -> FloatArray:
        w = np.array([self.z_hit, self.z_short, self.z_max, self.z_rand], dtype=np.float64)
        return w / w.sum()


@dataclass(frozen=True, slots=True)
class LikelihoodFieldParams:
    z_hit: float = 0.5
    z_rand: float = 0.5
    sigma_hit: float = 0.2
    max_dist: float = 2.0  # laser_likelihood_max_dist

    @classmethod
    def nav2_amcl_defaults(cls) -> LikelihoodFieldParams:
        return cls(0.5, 0.5, 0.2, 2.0)


class RangeSensorModel(ABC):
    """Common interface: log p(z | x, m) for a batch of robot poses."""

    def __init__(self, grid: GridMap, max_beams: int = 60, mount: Pose2D | None = None) -> None:
        self.grid = grid
        self.max_beams = max_beams
        self.mount = mount or Pose2D()

    def sensor_poses(self, robot_poses: FloatArray) -> FloatArray:
        """Apply the (fixed) sensor mounting transform to every robot pose."""
        m = self.mount
        c, s = np.cos(robot_poses[:, 2]), np.sin(robot_poses[:, 2])
        return np.column_stack(
            (
                robot_poses[:, 0] + c * m.x - s * m.y,
                robot_poses[:, 1] + s * m.x + c * m.y,
                robot_poses[:, 2] + m.theta,
            )
        )

    def log_likelihood(self, scan: Scan, robot_poses: FloatArray) -> FloatArray:
        poses = self.sensor_poses(np.atleast_2d(robot_poses))
        return self._log_likelihood(scan.subsample(self.max_beams), poses)

    @abstractmethod
    def _log_likelihood(self, scan: Scan, sensor_poses: FloatArray) -> FloatArray: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class BeamRangeFinderModel(RangeSensorModel):
    """Mixture of hit / short / max / rand, needs ray casting per pose (expensive)."""

    MAX_TOL: float = 1e-3

    def __init__(self, grid: GridMap, params: BeamModelParams, range_max: float,
                 max_beams: int = 60, mount: Pose2D | None = None) -> None:
        super().__init__(grid, max_beams, mount)
        self.params = params
        self.range_max = range_max
        self.caster = RayCaster(grid, range_max)

    @property
    def name(self) -> str:
        return "beam"

    # ----- component densities -------------------------------------------
    def component_densities(self, z: FloatArray, z_star: FloatArray) -> FloatArray:
        """Stacked (4, ...) array: p_hit, p_short, p_max, p_rand."""
        p, zmax = self.params, self.range_max
        sig = p.sigma_hit
        z, z_star = np.broadcast_arrays(np.asarray(z, dtype=np.float64), np.asarray(z_star, dtype=np.float64))
        # p_hit: Gaussian truncated to [0, zmax]
        norm_cdf = 0.5 * (
            _ERF((zmax - z_star) / (sig * math.sqrt(2))) - _ERF((0.0 - z_star) / (sig * math.sqrt(2)))
        )
        p_hit = np.exp(-0.5 * ((z - z_star) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
        p_hit = np.where((z >= 0) & (z <= zmax), p_hit / np.maximum(norm_cdf, 1e-12), 0.0)
        # p_short: truncated exponential on [0, z*]
        eta = 1.0 / np.maximum(1.0 - np.exp(-p.lambda_short * z_star), 1e-12)
        p_short = np.where((z >= 0) & (z <= z_star), eta * p.lambda_short * np.exp(-p.lambda_short * z), 0.0)
        # p_max: point mass at zmax ; p_rand: uniform
        p_max = (z >= zmax - self.MAX_TOL).astype(np.float64)
        p_rand = np.where((z >= 0) & (z < zmax), 1.0 / zmax, 0.0)
        return np.stack((p_hit, p_short, p_max, p_rand))

    def density(self, z: FloatArray, z_star: FloatArray) -> FloatArray:
        w = np.array([self.params.z_hit, self.params.z_short, self.params.z_max, self.params.z_rand])
        comps = self.component_densities(z, z_star)
        return np.tensordot(w, comps, axes=1)

    def _log_likelihood(self, scan: Scan, sensor_poses: FloatArray) -> FloatArray:
        z_star = self.caster.cast(sensor_poses, scan.angles)  # (N, B)
        p = self.density(scan.ranges[None, :], z_star)
        return np.sum(np.log(np.maximum(p, 1e-300)), axis=1)

    # ----- generative use --------------------------------------------------
    def sample(self, z_star: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Draw z ~ p(z | z*) for every expected range in `z_star` (any shape)."""
        p, zmax = self.params, self.range_max
        flat = np.asarray(z_star, dtype=np.float64).ravel()
        comp = rng.choice(4, size=flat.size, p=p.weights())
        out = np.empty_like(flat)
        hit = comp == 0
        out[hit] = np.clip(flat[hit] + rng.normal(0.0, p.sigma_hit, hit.sum()), 0.0, zmax)
        short = comp == 1
        u = rng.random(short.sum())
        zs = flat[short]
        out[short] = -np.log(1.0 - u * (1.0 - np.exp(-p.lambda_short * zs))) / p.lambda_short
        out[comp == 2] = zmax
        rnd = comp == 3
        out[rnd] = rng.uniform(0.0, zmax, rnd.sum())
        return out.reshape(np.shape(z_star))


class LikelihoodFieldModel(RangeSensorModel):
    """End-point model: only a distance-field lookup per beam (cheap, smooth)."""

    def __init__(self, grid: GridMap, params: LikelihoodFieldParams, range_max: float,
                 max_beams: int = 60, mount: Pose2D | None = None) -> None:
        super().__init__(grid, max_beams, mount)
        self.params = params
        self.range_max = range_max
        _ = grid.distance_field  # pre-compute once (it is the "map" of this model)

    @property
    def name(self) -> str:
        return "likelihood_field"

    def _log_likelihood(self, scan: Scan, sensor_poses: FloatArray) -> FloatArray:
        p = self.params
        valid = scan.ranges < scan.range_max - 1e-3  # max-range readings are discarded
        r, a = scan.ranges[valid], scan.angles[valid]
        heading = sensor_poses[:, 2:3] + a[None, :]
        ex = sensor_poses[:, 0:1] + r[None, :] * np.cos(heading)
        ey = sensor_poses[:, 1:2] + r[None, :] * np.sin(heading)
        d = self.grid.distance_to_nearest(ex, ey, cap=p.max_dist)
        return np.sum(np.log(self.endpoint_density(d)), axis=1)

    def endpoint_density(self, d: FloatArray) -> FloatArray:
        """p(z | x, m) of one endpoint at distance d from the nearest obstacle: z_hit N(d; 0, σ) + z_rand / z_max."""
        p = self.params
        p_hit = np.exp(-0.5 * (np.asarray(d) / p.sigma_hit) ** 2) / (p.sigma_hit * math.sqrt(2 * math.pi))
        return p.z_hit * p_hit + p.z_rand / self.range_max


@dataclass(frozen=True, slots=True)
class CorrelationParams:
    sigma: float = 0.1                                 # smoothing of the global map (m)
    free_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)  # where along each beam to sample free space


class CorrelationModel(RangeSensorModel):
    """
    Map matching (Probabilistic Robotics §6.5): build a local map from the scan
    (endpoints = occupied, points along the beam = free), project it into the
    global map at pose x, and score with the correlation coefficient
        p(z | x, m) ∝ max(ρ, 0).
    """

    EPS: float = 1e-6

    def __init__(self, grid: GridMap, params: CorrelationParams, range_max: float,
                 max_beams: int = 60, mount: Pose2D | None = None) -> None:
        super().__init__(grid, max_beams, mount)
        self.params = params
        self.range_max = range_max
        _ = grid.distance_field

    @property
    def name(self) -> str:
        return "correlation"

    def _log_likelihood(self, scan: Scan, sensor_poses: FloatArray) -> FloatArray:
        valid = scan.ranges < scan.range_max - 1e-3
        r, a = scan.ranges[valid], scan.angles[valid]
        fr = np.asarray(self.params.free_fractions)
        # local map: (range, value) pairs per beam -> occupied endpoints + free samples
        ranges = np.concatenate((r, np.outer(fr, r).ravel()))
        angles = np.concatenate((a, np.tile(a, fr.size)))
        local = np.concatenate((np.ones(r.size), np.zeros(r.size * fr.size)))
        heading = sensor_poses[:, 2:3] + angles[None, :]
        px = sensor_poses[:, 0:1] + ranges[None, :] * np.cos(heading)
        py = sensor_poses[:, 1:2] + ranges[None, :] * np.sin(heading)
        d = self.grid.distance_to_nearest(px, py, cap=4 * self.params.sigma)
        global_ = np.exp(-0.5 * (d / self.params.sigma) ** 2)  # smoothed occupancy
        lc = local - local.mean()
        gc = global_ - global_.mean(axis=1, keepdims=True)
        rho = (gc @ lc) / (np.sqrt((gc**2).sum(axis=1) * (lc**2).sum()) + self.EPS)
        return np.log(np.maximum(rho, self.EPS))


@dataclass(frozen=True, slots=True)
class FeatureModelParams:
    sigma_rho: float = 0.10  # m
    sigma_alpha: float = math.radians(5.0)
    gate: float = 5.99  # chi-square, 2 dof, 95 %: farther correspondences are rejected
    extractor: LineExtractorParams = LineExtractorParams()


class FeatureBasedModel(RangeSensorModel):
    """
    Feature-based model: extract line features from the scan once, then for each pose
    predict the map's lines in the sensor frame, associate each observed line with the
    closest prediction (Mahalanobis distance d² on (rho, alpha), gated), and score

        p(z | x, m) ∝ Π_i exp(-d_i² / 2)        (rejected features get the gate's value)
    """

    def __init__(self, grid: GridMap, world: WorldSpec, params: FeatureModelParams, range_max: float,
                 mount: Pose2D | None = None) -> None:
        super().__init__(grid, max_beams=1_000_000, mount=mount)  # extraction wants every beam
        self.params = params
        self.range_max = range_max
        self.line_map = LineMap.from_world(world)
        self.extractor = LineExtractor(params.extractor)

    @property
    def name(self) -> str:
        return "feature"

    def observed_features(self, scan: Scan) -> list:  # noqa: ANN401 - list[LineFeature]
        valid = scan.ranges < scan.range_max - 1e-3
        r, a = scan.ranges[valid], scan.angles[valid]
        return self.extractor.extract(np.column_stack((r * np.cos(a), r * np.sin(a))))

    def mahalanobis(self, rho: float, alpha: float, poses: FloatArray) -> FloatArray:
        """d² from one observed line to every map line predicted at each pose: (N, M)."""
        p = self.params
        rho_hat, alpha_hat = self.line_map.in_sensor_frame(poses)
        return ((rho - rho_hat) / p.sigma_rho) ** 2 + (Angle.wrap_array(alpha - alpha_hat) / p.sigma_alpha) ** 2

    def _log_likelihood(self, scan: Scan, sensor_poses: FloatArray) -> FloatArray:
        out = np.zeros(len(sensor_poses))
        for f in self.observed_features(scan):
            d2 = self.mahalanobis(f.rho, f.alpha, sensor_poses).min(axis=1)  # best correspondence per pose
            out += -0.5 * np.minimum(d2, self.params.gate)
        return out


class RangeSensorModelFactory:
    """Builds a model from Nav2's `laser_model_type` string (Open/Closed: register new models here)."""

    KINDS = ("beam", "likelihood_field", "correlation", "feature")

    @classmethod
    def create(cls, kind: str, grid: GridMap, beam: BeamModelParams, lf: LikelihoodFieldParams,
               range_max: float, max_beams: int = 60, mount: Pose2D | None = None,
               world: WorldSpec | None = None) -> RangeSensorModel:
        if kind == "beam":
            return BeamRangeFinderModel(grid, beam, range_max, max_beams, mount)
        if kind == "likelihood_field":
            return LikelihoodFieldModel(grid, lf, range_max, max_beams, mount)
        if kind == "correlation":
            return CorrelationModel(grid, CorrelationParams(), range_max, max_beams, mount)
        if kind == "feature":
            if world is None:
                raise ValueError("the feature model needs the world (its feature map)")
            return FeatureBasedModel(grid, world, FeatureModelParams(), range_max, mount)
        raise ValueError(f"unknown sensor model '{kind}' (expected one of {cls.KINDS})")


class _Erf:
    """Vectorised erf without SciPy."""

    _f = np.vectorize(math.erf, otypes=[np.float64])

    def __call__(self, x: FloatArray | float) -> FloatArray:
        return self._f(x)


_ERF = _Erf()
