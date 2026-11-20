"""
Trajectory evaluation with the TUM RGB-D benchmark definitions
(Sturm et al., IROS 2012) — the numbers `evo_ape` / `evo_rpe` report.

  ATE  absolute trajectory error after rigid (SE(3)) alignment, RMSE in metres
  RPE  relative pose error over a fixed time delta (default 1 s): translational
       RMSE in m and rotational RMSE in degrees
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .se3 import SE3, SO3, FloatArray


@dataclass
class Trajectory:
    stamps: list[float]
    poses: list[SE3]

    @classmethod
    def empty(cls) -> Trajectory:
        return cls([], [])

    def add(self, stamp: float, pose: SE3) -> None:
        if self.stamps and stamp <= self.stamps[-1]:
            return
        self.stamps.append(stamp)
        self.poses.append(pose)

    def __len__(self) -> int:
        return len(self.stamps)

    @property
    def positions(self) -> FloatArray:
        return np.array([p.translation for p in self.poses]) if self.poses else np.zeros((0, 3))

    @property
    def length(self) -> float:
        p = self.positions
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0

    def write_tum(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for t, p in zip(self.stamps, self.poses):
                q = p.quaternion()
                fh.write(f"{t:.6f} " + " ".join(f"{v:.6f}" for v in (*p.translation, *q)) + "\n")
        return out


@dataclass(frozen=True, slots=True)
class TrajectoryReport:
    pairs: int
    ate_rmse_m: float
    ate_max_m: float
    rpe_trans_rmse_m: float
    rpe_rot_rmse_deg: float
    path_length_m: float

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


class TrajectoryEvaluator:
    def __init__(self, max_dt: float = 0.02, rpe_delta_s: float = 1.0, with_scale: bool = False) -> None:
        self.max_dt = max_dt
        self.rpe_delta_s = rpe_delta_s
        self.with_scale = with_scale  # Sim(3) alignment for monocular methods

    def associate(self, gt: Trajectory, est: Trajectory) -> list[tuple[int, int]]:
        if not len(gt) or not len(est):
            return []
        g = np.asarray(gt.stamps)
        pairs = []
        for j, t in enumerate(est.stamps):
            i = int(np.argmin(np.abs(g - t)))
            if abs(g[i] - t) <= self.max_dt:
                pairs.append((i, j))
        return pairs

    def evaluate(self, gt: Trajectory, est: Trajectory) -> TrajectoryReport | None:
        pairs = self.associate(gt, est)
        if len(pairs) < 3:
            return None
        G = np.array([gt.poses[i].translation for i, _ in pairs])
        E = np.array([est.poses[j].translation for _, j in pairs])
        r, t, s = self.umeyama(E, G, self.with_scale)
        aligned = s * E @ r.T + t
        err = np.linalg.norm(aligned - G, axis=1)
        rpe_t, rpe_r = self._rpe([gt.poses[i] for i, _ in pairs], [est.poses[j] for _, j in pairs],
                                 [est.stamps[j] for _, j in pairs], s)
        return TrajectoryReport(len(pairs), float(np.sqrt(np.mean(err**2))), float(err.max()), rpe_t, rpe_r,
                                Trajectory([gt.stamps[i] for i, _ in pairs], [gt.poses[i] for i, _ in pairs]).length)

    def aligned_positions(self, gt: Trajectory, est: Trajectory) -> tuple[FloatArray, FloatArray]:
        pairs = self.associate(gt, est)
        G = np.array([gt.poses[i].translation for i, _ in pairs])
        E = np.array([est.poses[j].translation for _, j in pairs])
        r, t, s = self.umeyama(E, G, self.with_scale)
        return G, s * E @ r.T + t

    @staticmethod
    def umeyama(src: FloatArray, dst: FloatArray, with_scale: bool) -> tuple[FloatArray, FloatArray, float]:
        """Least-squares R, t, s with dst ≈ s R src + t (Umeyama 1991)."""
        mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
        xs, xd = src - mu_s, dst - mu_d
        cov = xd.T @ xs / len(src)
        u, d, vt = np.linalg.svd(cov)
        sgn = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            sgn[2, 2] = -1
        r = u @ sgn @ vt
        s = float(np.trace(np.diag(d) @ sgn) / np.mean(np.sum(xs**2, axis=1))) if with_scale else 1.0
        return r, mu_d - s * r @ mu_s, s

    def _rpe(self, gts: list[SE3], ests: list[SE3], stamps: list[float], scale: float) -> tuple[float, float]:
        st = np.asarray(stamps)
        terr, rerr = [], []
        for a in range(len(st)):
            b = int(np.searchsorted(st, st[a] + self.rpe_delta_s))
            if b >= len(st):
                break
            dg = gts[a].inverse() @ gts[b]
            de = ests[a].inverse() @ ests[b]
            de = SE3(de.rotation, de.translation * scale)
            e = dg.inverse() @ de
            terr.append(e.translation_norm)
            rerr.append(math.degrees(SO3.angle(e.rotation)))
        if not terr:
            return float("nan"), float("nan")
        return float(np.sqrt(np.mean(np.square(terr)))), float(np.sqrt(np.mean(np.square(rerr))))
