"""
"Replace it by ML": a learned scan -> pose regressor.

Global localization is a *search* for the pose whose expected scan matches the
measured one. With a known map we can generate unlimited labelled data
(random free pose -> ray-cast scan) and learn that search as a function:

    scan (72 sector minima)  --MLP-->  (x, y, cos θ, sin θ)

Uses:
  * standalone ML localizer (one forward pass per scan, no filter, no memory)
  * proposal for MCL: seed / re-seed particles around the guess (ML-augmented MCL)

The MLP is plain NumPy (Adam, mini-batches): no PyTorch, no GPU, and every
line of the training loop is readable in class.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sensor_motion_lab.core import GridMap, LidarConfig, Pose2D, RayCaster, Scan
from sensor_motion_lab.core.geometry import FloatArray

from .common import FreeSpace


# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ScanFeaturizer:
    """Robot-frame scan -> fixed-length vector: min range per angular sector, scaled to [0, 1]."""

    sectors: int = 72
    range_max: float = 3.5

    def transform(self, ranges: FloatArray, angles: FloatArray) -> FloatArray:
        """ranges/angles: (B,) or (N, B) with angles in the robot frame."""
        r = np.atleast_2d(np.where(np.isfinite(ranges), ranges, self.range_max))
        r = np.clip(r, 0.0, self.range_max)
        a = np.mod(np.asarray(angles) + math.pi, 2 * math.pi)
        sector = np.minimum((a / (2 * math.pi) * self.sectors).astype(int), self.sectors - 1)
        out = np.full((r.shape[0], self.sectors), self.range_max)
        for s in range(self.sectors):
            m = sector == s
            if m.any():
                out[:, s] = r[:, m].min(axis=1)
        return out / self.range_max

    def from_scan(self, scan: Scan) -> FloatArray:
        return self.transform(scan.ranges, scan.angles)


# --------------------------------------------------------------------------- #
# A small MLP, NumPy only                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0


class Mlp:
    """Fully connected ReLU network trained with Adam on mean-squared error."""

    def __init__(self, sizes: tuple[int, ...], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W = [rng.normal(0, math.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.x_mean = np.zeros(sizes[0])
        self.x_std = np.ones(sizes[0])

    @property
    def num_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def predict(self, X: FloatArray) -> FloatArray:
        h = (X - self.x_mean) / self.x_std
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
        return h

    def fit(self, X: FloatArray, Y: FloatArray, epochs: int = 60, batch: int = 256, lr: float = 1e-3,
            val_fraction: float = 0.1, seed: int = 0, verbose: bool = False) -> TrainingHistory:
        rng = np.random.default_rng(seed)
        self.x_mean, self.x_std = X.mean(axis=0), X.std(axis=0) + 1e-6
        n_val = int(len(X) * val_fraction)
        perm = rng.permutation(len(X))
        Xv, Yv, Xt, Yt = X[perm[:n_val]], Y[perm[:n_val]], X[perm[n_val:]], Y[perm[n_val:]]
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        hist, t0, step = TrainingHistory(), time.perf_counter(), 0
        for epoch in range(epochs):
            order = rng.permutation(len(Xt))
            losses = []
            for s in range(0, len(order), batch):
                idx = order[s : s + batch]
                grads, loss = self._backprop(Xt[idx], Yt[idx])
                losses.append(loss)
                step += 1
                for i, (p, g) in enumerate(zip(self.W + self.b, grads)):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    mh, vh = m[i] / (1 - 0.9**step), v[i] / (1 - 0.999**step)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
            hist.train_loss.append(float(np.mean(losses)))
            hist.val_loss.append(float(np.mean((self.predict(Xv) - Yv) ** 2)) if n_val else float("nan"))
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:3d}  train {hist.train_loss[-1]:.4f}  val {hist.val_loss[-1]:.4f}")
        hist.seconds = time.perf_counter() - t0
        return hist

    def _backprop(self, X: FloatArray, Y: FloatArray) -> tuple[list[FloatArray], float]:
        acts = [(X - self.x_mean) / self.x_std]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            acts.append(np.maximum(z, 0.0) if i < len(self.W) - 1 else z)
        err = acts[-1] - Y
        loss = float(np.mean(err**2))
        delta = 2.0 * err / err.size
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, acts[i].T @ delta)
            gb.insert(0, delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return gW + gb, loss

    def to_arrays(self) -> dict[str, FloatArray]:
        d = {f"W{i}": w for i, w in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}
        return d | {"x_mean": self.x_mean, "x_std": self.x_std, "sizes": np.asarray(self.sizes)}

    @classmethod
    def from_arrays(cls, d: dict[str, FloatArray]) -> Mlp:
        net = cls(tuple(int(s) for s in d["sizes"]))
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        net.x_mean, net.x_std = d["x_mean"], d["x_std"]
        return net


# --------------------------------------------------------------------------- #
# Scan -> pose                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RegressorReport:
    samples: int
    params: int
    train_seconds: float
    val_median_pos_err_m: float
    val_p90_pos_err_m: float
    val_median_heading_err_deg: float
    inference_us: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in self.__slots__}  # type: ignore[attr-defined]


class ScanPoseRegressor:
    """Learned global localizer. Implements the `PoseProposal` protocol used by MCL."""

    def __init__(self, featurizer: ScanFeaturizer, net: Mlp | None = None) -> None:
        self.featurizer = featurizer
        self.net = net or Mlp((featurizer.sectors, 256, 256, 128, 4))

    # ----- inference ---------------------------------------------------------- #
    def predict_features(self, F: FloatArray) -> FloatArray:
        y = self.net.predict(F)
        return np.column_stack((y[:, 0], y[:, 1], np.arctan2(y[:, 3], y[:, 2])))

    def propose(self, scan: Scan) -> Pose2D | None:
        return Pose2D.from_array(self.predict_features(self.featurizer.from_scan(scan))[0])

    # ----- training ----------------------------------------------------------- #
    def fit_from_map(self, grid: GridMap, lidar: LidarConfig, samples: int = 40000, epochs: int = 60,
                     range_noise: float = 0.02, dropout: float = 0.05, seed: int = 0, verbose: bool = False
                     ) -> RegressorReport:
        rng = np.random.default_rng(seed)
        poses = FreeSpace(grid, clearance=0.2).sample(samples, rng)
        angles = lidar.angles()
        caster = RayCaster(grid, lidar.range_max)
        F = np.empty((samples, self.featurizer.sectors))
        for s in range(0, samples, 2000):  # ray cast in chunks (memory)
            z = caster.cast(poses[s : s + 2000], angles)
            z = z + rng.normal(0, range_noise, z.shape)
            z[rng.random(z.shape) < dropout] = lidar.range_max  # simulated missing returns
            F[s : s + 2000] = self.featurizer.transform(z, angles)
        Y = np.column_stack((poses[:, 0], poses[:, 1], np.cos(poses[:, 2]), np.sin(poses[:, 2])))
        hist = self.net.fit(F, Y, epochs=epochs, seed=seed, verbose=verbose)
        return self._validate(grid, lidar, caster, rng, hist.seconds, samples)

    def _validate(self, grid: GridMap, lidar: LidarConfig, caster: RayCaster, rng: np.random.Generator,
                  seconds: float, samples: int) -> RegressorReport:
        poses = FreeSpace(grid, clearance=0.2).sample(2000, rng)
        angles = lidar.angles()
        z = caster.cast(poses, angles) + rng.normal(0, 0.01, (len(poses), angles.size))
        F = self.featurizer.transform(z, angles)
        t0 = time.perf_counter()
        for f in F[:200]:
            self.predict_features(f[None])
        inference_us = 1e6 * (time.perf_counter() - t0) / 200
        pred = self.predict_features(F)
        pos_err = np.hypot(pred[:, 0] - poses[:, 0], pred[:, 1] - poses[:, 1])
        th_err = np.degrees(np.abs(np.arctan2(np.sin(pred[:, 2] - poses[:, 2]), np.cos(pred[:, 2] - poses[:, 2]))))
        return RegressorReport(samples, self.net.num_params, seconds, float(np.median(pos_err)),
                               float(np.percentile(pos_err, 90)), float(np.median(th_err)), inference_us)

    # ----- persistence ---------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, sectors=self.featurizer.sectors, range_max=self.featurizer.range_max, **self.net.to_arrays())
        return out if out.suffix == ".npz" else out.with_suffix(out.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> ScanPoseRegressor:
        d = dict(np.load(Path(path).expanduser()))
        return cls(ScanFeaturizer(int(d["sectors"]), float(d["range_max"])), Mlp.from_arrays(d))
