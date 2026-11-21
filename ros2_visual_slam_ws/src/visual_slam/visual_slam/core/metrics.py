"""
Evaluation, each against a standard benchmark definition:

  trajectories   TUM RGB-D benchmark (Sturm et al., IROS 2012) — what `evo_ape` / `evo_rpe` report
                   ATE  absolute trajectory error after SE(3) (or Sim(3), monocular) alignment, RMSE in m
                   RPE  relative pose error over 1 s: translational RMSE in m, rotational RMSE in degrees
  place recog.   recall@1 over queries that have a true revisit, and false loops among accepted ones
  depth          NYU-Depth-v2 / KITTI protocol (Eigen et al., NeurIPS 2014): AbsRel, RMSE, δ < 1.25
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

    def pose_at(self, stamp: float, max_dt: float = 0.02) -> SE3 | None:
        if not self.stamps:
            return None
        st = np.asarray(self.stamps)
        i = int(np.argmin(np.abs(st - stamp)))
        return self.poses[i] if abs(st[i] - stamp) <= max_dt else None

    @classmethod
    def read_tum(cls, path: str | Path) -> Trajectory:
        """`timestamp tx ty tz qx qy qz qw` per line (# comments allowed) — what TUM, evo, DPVO, ORB-SLAM write."""
        traj = cls.empty()
        for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
            vals = line.split()
            if len(vals) < 8 or line.lstrip().startswith("#"):
                continue
            v = np.asarray(vals[:8], dtype=np.float64)
            traj.add(float(v[0]), SE3.from_quaternion(v[4:8], v[1:4]))
        return traj

    def write_tum(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for t, p in zip(self.stamps, self.poses):
                q = p.quaternion()
                fh.write(f"{t:.6f} " + " ".join(f"{v:.6f}" for v in (*p.translation, *q)) + "\n")
        return out


class FrameIndexStamps:
    """
    Maps a trajectory stamped with frame *indices* to dataset times. DPVO's demo does this:
    its stamp t is the t-th image of the sorted, strided image list (image t * stride).
    """

    def __init__(self, rgb_list: str | Path, stride: int = 1) -> None:
        rows = [ln.split() for ln in Path(rgb_list).expanduser().read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")]
        self.stamps = np.array([float(r[0]) for r in rows])
        self.stride = stride

    @staticmethod
    def looks_like_indices(traj: Trajectory) -> bool:
        """Dataset stamps are Unix times (~1e9); indices are small integers."""
        return bool(len(traj)) and max(traj.stamps) < 1e6 and all(float(s).is_integer() for s in traj.stamps)

    def apply(self, traj: Trajectory) -> Trajectory:
        out = Trajectory.empty()
        for s, p in zip(traj.stamps, traj.poses):
            k = int(round(s)) * self.stride
            if 0 <= k < len(self.stamps):
                out.add(float(self.stamps[k]), p)
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

    def align(self, gt: Trajectory, est: Trajectory) -> Trajectory:
        """The whole estimate (rotations too) moved into the ground-truth frame (for display)."""
        pairs = self.associate(gt, est)
        if len(pairs) < 3:
            raise ValueError("fewer than 3 matching timestamps between the estimate and the ground truth")
        g = np.array([gt.poses[i].translation for i, _ in pairs])
        e = np.array([est.poses[j].translation for _, j in pairs])
        r, t, s = self.umeyama(e, g, self.with_scale)
        out = Trajectory.empty()
        for stamp, p in zip(est.stamps, est.poses):
            out.add(stamp, SE3(r @ p.rotation, s * r @ p.translation + t))
        return out

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


# --------------------------------------------------------------------------- #
# Place recognition                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PlaceRecognitionReport:
    queries: int
    queries_with_revisit: int  # ground truth says an old keyframe shows this place
    recall_at_1: float  # top-ranked candidate is a true revisit (before verification)
    loops_accepted: int  # passed the geometric check
    false_loops: int  # accepted, but ground truth says it is a different place
    ms_per_query: float

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


class PlaceRecognitionEvaluator:
    """Ground truth: two keyframes show the same place if their true poses are within `radius` and `max_angle`."""

    def __init__(self, radius_m: float = 1.0, max_angle_deg: float = 30.0, min_separation: int = 30) -> None:
        self.radius, self.max_angle, self.min_separation = radius_m, max_angle_deg, min_separation

    def same_place(self, a: SE3, b: SE3) -> bool:
        rel = a.inverse() @ b
        return rel.translation_norm < self.radius and math.degrees(rel.rotation_angle) < self.max_angle

    def evaluate(self, queries: list, keyframe_stamps: list[float], gt: Trajectory) -> PlaceRecognitionReport:  # noqa: ANN001
        """`queries`: LoopQuery records (query_id, top_id, verified_id, ms); ids index `keyframe_stamps`."""
        poses = [gt.pose_at(t) for t in keyframe_stamps]
        with_revisit = hits = accepted = false = 0
        for q in queries:
            pq = poses[q.query_id]
            if pq is None:
                continue
            old = [p for p in poses[: max(q.query_id - self.min_separation + 1, 0)] if p is not None]
            if any(self.same_place(pq, p) for p in old):
                with_revisit += 1
                top = poses[q.top_id] if q.top_id is not None else None
                hits += int(top is not None and self.same_place(pq, top))
            if q.verified_id is not None:
                accepted += 1
                pv = poses[q.verified_id]
                false += int(pv is None or not self.same_place(pq, pv))
        ms = float(np.mean([q.ms for q in queries])) if queries else 0.0
        return PlaceRecognitionReport(len(queries), with_revisit, hits / with_revisit if with_revisit else float("nan"),
                                      accepted, false, ms)


# --------------------------------------------------------------------------- #
# Depth                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DepthReport:
    pixels: int
    abs_rel: float  # mean |d - d*| / d*
    rmse_m: float
    delta_1: float  # fraction with max(d/d*, d*/d) < 1.25
    median_scale: float  # median(d*) / median(d): 1.0 = metric scale is right

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


class DepthEvaluator:
    """Standard monocular-depth metrics against a sensor depth image (invalid pixels ignored)."""

    def __init__(self, min_depth: float = 0.1, max_depth: float = 10.0) -> None:
        self.min_depth, self.max_depth = min_depth, max_depth

    def evaluate(self, pred: FloatArray, gt: FloatArray, median_align: bool = False) -> DepthReport:
        ok = np.isfinite(gt) & (gt > self.min_depth) & (gt < self.max_depth) & np.isfinite(pred) & (pred > 0)
        if not ok.any():
            return DepthReport(0, float("nan"), float("nan"), float("nan"), float("nan"))
        p, g = pred[ok].astype(np.float64), gt[ok].astype(np.float64)
        scale = float(np.median(g) / np.median(p))
        if median_align:
            p = p * scale
        ratio = np.maximum(p / g, g / p)
        return DepthReport(int(ok.sum()), float(np.mean(np.abs(p - g) / g)), float(np.sqrt(np.mean((p - g) ** 2))),
                           float(np.mean(ratio < 1.25)), scale)
