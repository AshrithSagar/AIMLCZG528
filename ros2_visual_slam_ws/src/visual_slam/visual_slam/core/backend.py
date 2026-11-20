"""
Back-end: keyframe map, loop-closure detection and pose-graph optimisation.

  KeyframeMap         keyframes + odometry edges, map point cloud
  LoopDetector        place recognition (descriptor voting) + PnP geometric check
  PoseGraphOptimizer  SE(3) pose graph, sparse least squares (SciPy)

This is what turns visual odometry (drifts forever) into SLAM (drift is
corrected when a place is revisited).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .camera import PinholeCamera
from .se3 import SE3, SO3, FloatArray
from .tracking import DescriptorMatcher, Keyframe, PnpSolver


@dataclass(frozen=True, slots=True)
class Edge:
    i: int
    j: int
    T_i_j: SE3  # measured pose of keyframe j in keyframe i's frame
    loop: bool = False


@dataclass
class KeyframeMap:
    keyframes: list[Keyframe] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add(self, kf: Keyframe) -> None:
        if self.keyframes:
            prev = self.keyframes[-1]
            self.edges.append(Edge(prev.id, kf.id, prev.pose.inverse() @ kf.pose))
        self.keyframes.append(kf)

    def by_id(self, kid: int) -> Keyframe:
        return self.keyframes[kid]  # ids are assigned sequentially

    @property
    def loop_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.loop]

    def sparse_points(self) -> FloatArray:
        """World-frame landmarks: every keyframe's features with valid depth."""
        pts = [kf.pose.apply(kf.points[kf.valid]) for kf in self.keyframes if kf.valid.any()]
        return np.vstack(pts) if pts else np.zeros((0, 3))

    def dense_points(self, camera: PinholeCamera, stride: int = 8, max_depth: float = 5.0
                     ) -> tuple[FloatArray, np.ndarray]:
        """Sub-sampled coloured cloud from every keyframe's depth image (for RViz)."""
        out_p, out_c = [], []
        for kf in self.keyframes:
            d = kf.frame.depth[::stride, ::stride]
            v, u = np.mgrid[0 : kf.frame.depth.shape[0] : stride, 0 : kf.frame.depth.shape[1] : stride]
            ok = (d > 0.1) & (d < max_depth) & np.isfinite(d)
            if not ok.any():
                continue
            uv = np.column_stack((u[ok], v[ok])).astype(np.float64)
            out_p.append(kf.pose.apply(camera.backproject(uv, d[ok].astype(np.float64))))
            out_c.append(kf.frame.rgb[::stride, ::stride][ok])
        if not out_p:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        return np.vstack(out_p), np.vstack(out_c)


@dataclass(frozen=True, slots=True)
class LoopConfig:
    min_separation: int = 30  # only consider keyframes at least this many keyframes old
    candidates: int = 3  # best-scoring candidates to verify geometrically
    min_score: float = 0.08  # fraction of current features voting for a candidate
    min_inliers: int = 40
    search_radius: float = 2.0  # m, around the (drifting) odometry estimate
    max_checked: int = 15  # nearest old keyframes scored by descriptor voting
    cooldown: int = 2  # keyframes to skip after a successful closure


@dataclass(frozen=True, slots=True)
class LoopCandidate:
    edge: Edge
    inliers: int
    score: float


class LoopDetector:
    """
    Appearance-based place recognition by direct descriptor voting against old
    keyframes (a tiny stand-in for DBoW2/NetVLAD), then PnP geometric verification.
    """

    def __init__(self, camera: PinholeCamera, matcher: DescriptorMatcher, config: LoopConfig) -> None:
        self.config = config
        self.matcher = matcher
        self.pnp = PnpSolver(camera, reproj_error_px=3.0, iterations=300, min_inliers=config.min_inliers)

    def detect(self, kf: Keyframe, kmap: KeyframeMap) -> LoopCandidate | None:
        c = self.config
        old = [k for k in kmap.keyframes if k.id <= kf.id - c.min_separation]
        if not old:
            return None
        dist = np.array([np.linalg.norm(k.pose.translation - kf.pose.translation) for k in old])
        order = np.argsort(dist)[: c.max_checked]
        old = [old[i] for i in order if dist[i] < c.search_radius]
        scored = []
        for cand in old:
            pairs = self.matcher.match(kf.features, cand.features)
            scored.append((len(pairs) / max(len(kf.features), 1), cand, pairs))
        scored.sort(key=lambda s: -s[0])
        for score, cand, pairs in scored[: c.candidates]:
            if score < c.min_score:
                break
            pairs = pairs[cand.valid[pairs[:, 1]]]
            res = self.pnp.solve(cand.points[pairs[:, 1]], kf.features.keypoints[pairs[:, 0]])
            if res.ok:
                # res.T_cur_ref = T_kf_cand  ->  edge cand -> kf
                return LoopCandidate(Edge(cand.id, kf.id, res.T_cur_ref.inverse(), loop=True), len(res.inliers), score)
        return None


class BatchSO3:
    """Vectorised exp/log maps over many rotations at once (keeps the optimiser fast in pure NumPy)."""

    @staticmethod
    def exp(w: FloatArray) -> FloatArray:
        theta = np.linalg.norm(w, axis=1)[:, None, None]
        k = np.zeros((len(w), 3, 3))
        safe = np.maximum(theta[:, 0, 0], 1e-12)
        a = w / safe[:, None]
        k[:, 0, 1], k[:, 0, 2], k[:, 1, 0] = -a[:, 2], a[:, 1], a[:, 2]
        k[:, 1, 2], k[:, 2, 0], k[:, 2, 1] = -a[:, 0], -a[:, 1], a[:, 0]
        return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)

    @staticmethod
    def log(r: FloatArray) -> FloatArray:
        cos = np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1)
        theta = np.arccos(cos)
        v = np.stack((r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]), axis=1)
        scale = np.where(theta < 1e-8, 0.5, theta / (2 * np.maximum(np.sin(theta), 1e-12)))
        return v * scale[:, None]


class PoseGraphOptimizer:
    """
    Minimises  Σ || log( Z_ij^-1 · T_i^-1 · T_j ) ||²_W  over keyframe poses,
    with the first keyframe fixed (robust Huber loss, sparse Jacobian).
    Loop edges are what pull the accumulated drift out.
    """

    def __init__(self, rot_weight: float = 10.0, trans_weight: float = 1.0, max_nfev: int = 100) -> None:
        self.w = np.array([rot_weight] * 3 + [trans_weight] * 3)
        self.max_nfev = max_nfev
        self.last_ms = 0.0

    def optimize(self, kmap: KeyframeMap) -> float:
        """Optimises keyframe poses in place; returns the final cost."""
        from scipy.optimize import least_squares  # noqa: PLC0415 - only needed here
        from scipy.sparse import lil_matrix  # noqa: PLC0415

        t0 = time.perf_counter()
        kfs, edges = kmap.keyframes, kmap.edges
        n = len(kfs)
        if n < 3 or not kmap.loop_edges:
            return 0.0
        I = np.array([e.i for e in edges])
        J = np.array([e.j for e in edges])
        Rz = np.array([e.T_i_j.rotation for e in edges])
        tz = np.array([e.T_i_j.translation for e in edges])
        R0, t0_ = kfs[0].pose.rotation, kfs[0].pose.translation
        x0 = np.concatenate([np.concatenate((SO3.log(k.pose.rotation), k.pose.translation)) for k in kfs[1:]])

        def unpack(x: FloatArray) -> tuple[FloatArray, FloatArray]:
            x = x.reshape(-1, 6)
            R = np.concatenate((R0[None], BatchSO3.exp(x[:, :3])))
            t = np.concatenate((t0_[None], x[:, 3:]))
            return R, t

        def residuals(x: FloatArray) -> FloatArray:
            R, t = unpack(x)
            RiT = np.transpose(R[I], (0, 2, 1))
            R_ij = RiT @ R[J]
            t_ij = np.einsum("nab,nb->na", RiT, t[J] - t[I])
            RzT = np.transpose(Rz, (0, 2, 1))
            r_rot = BatchSO3.log(RzT @ R_ij)
            r_tr = np.einsum("nab,nb->na", RzT, t_ij - tz)
            return (np.hstack((r_rot, r_tr)) * self.w).ravel()

        sparsity = lil_matrix((6 * len(edges), 6 * (n - 1)), dtype=int)
        for k, e in enumerate(edges):
            for node in (e.i, e.j):
                if node > 0:
                    sparsity[6 * k : 6 * k + 6, 6 * (node - 1) : 6 * node] = 1
        sol = least_squares(residuals, x0, jac_sparsity=sparsity, method="trf", loss="huber",
                            f_scale=1.0, max_nfev=self.max_nfev)
        R, t = unpack(sol.x)
        for i in range(1, n):
            kfs[i].pose = SE3(R[i], t[i])
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return float(sol.cost)
