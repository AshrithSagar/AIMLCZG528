"""
Front-end: feature matching + RGB-D visual odometry (frame-to-keyframe PnP).

For every new frame:
  1. match its features against the current keyframe's features,
  2. lift the keyframe's matched keypoints to 3-D with the keyframe depth,
  3. solve PnP + RANSAC (3-D keyframe points <-> 2-D current pixels) -> T_cur_kf,
  4. decide whether the frame becomes a new keyframe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

from .camera import PinholeCamera, RgbdFrame
from .features import Features
from .matching import DescriptorMatcher, FeatureMatcher  # noqa: F401 - DescriptorMatcher re-exported (used to live here)
from .se3 import SE3, FloatArray


@dataclass(frozen=True, slots=True)
class PnpResult:
    T_cur_ref: SE3
    inliers: NDArray[np.intp]  # indices into the matched pairs
    ok: bool


class PnpSolver:
    """
    3-D/2-D pose estimation: RANSAC with a minimal solver (EPnP), Levenberg-Marquardt refinement on the
    inliers, then a re-count of inliers under the refined pose.

    The RANSAC stage deliberately does *not* use the motion prior (`useExtrinsicGuess`): how OpenCV
    treats that flag inside RANSAC differs between versions (4.6 vs 5.x gave different trajectories on
    the same frames). The prior is only scored as one extra hypothesis, which is version-independent.
    """

    def __init__(self, camera: PinholeCamera, reproj_error_px: float = 3.0, iterations: int = 200,
                 min_inliers: int = 20) -> None:
        self.camera = camera
        self.reproj_error_px = reproj_error_px
        self.iterations = iterations
        self.min_inliers = min_inliers

    def solve(self, pts3d: FloatArray, pts2d: FloatArray, guess: SE3 | None = None) -> PnpResult:
        fail = PnpResult(SE3.identity(), np.zeros(0, dtype=np.intp), False)
        if pts3d.shape[0] < max(self.min_inliers, 6):
            return fail
        p3, p2 = np.ascontiguousarray(pts3d, np.float64), np.ascontiguousarray(pts2d, np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            p3, p2, self.camera.k, None, iterationsCount=self.iterations,
            reprojectionError=self.reproj_error_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        candidates = [(rvec, tvec)] if ok and inl is not None and len(inl) >= 6 else []
        if guess is not None:
            candidates.append((guess.rvec().reshape(3, 1).astype(np.float64),
                               guess.translation.reshape(3, 1).astype(np.float64)))
        best: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        for rv, tv in candidates:
            inliers = self._inliers(p3, p2, rv, tv)
            if len(inliers) < 6:
                continue
            rv, tv = cv2.solvePnPRefineLM(p3[inliers], p2[inliers], self.camera.k, None, rv.copy(), tv.copy())
            inliers = self._inliers(p3, p2, rv, tv)  # re-count under the refined pose
            if best is None or len(inliers) > best[0]:
                best = (len(inliers), rv, tv, inliers)
        if best is None or best[0] < self.min_inliers:
            return fail
        return PnpResult(SE3.from_rvec_tvec(best[1], best[2]), best[3].astype(np.intp), True)

    def _inliers(self, p3: FloatArray, p2: FloatArray, rvec: np.ndarray, tvec: np.ndarray) -> NDArray[np.intp]:
        proj, _ = cv2.projectPoints(p3, rvec, tvec, self.camera.k, None)
        in_front = (p3 @ cv2.Rodrigues(rvec)[0].T + tvec.ravel())[:, 2] > 0
        err = np.linalg.norm(proj.reshape(-1, 2) - p2, axis=1)
        return np.flatnonzero(in_front & (err < self.reproj_error_px))


@dataclass(slots=True)
class Keyframe:
    id: int
    stamp: float
    pose: SE3  # T_world_kf (optimised in place by the back-end)
    features: Features
    points: FloatArray  # (N, 3) per-feature 3-D point in the keyframe optical frame (NaN if no depth)
    frame: RgbdFrame

    @property
    def valid(self) -> NDArray[np.bool_]:
        return np.isfinite(self.points[:, 2])


class TrackingState(str, Enum):
    INITIALIZING = "initializing"
    OK = "ok"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class OdometryConfig:
    min_depth: float = 0.2
    max_depth: float = 6.0
    min_inliers: int = 25
    reproj_error_px: float = 3.0
    kf_translation: float = 0.25  # m
    kf_rotation_deg: float = 15.0
    kf_min_inlier_ratio: float = 0.2  # inliers / keyframe features with depth


@dataclass(slots=True)
class TrackResult:
    pose: SE3  # T_world_cam
    state: TrackingState
    matches: int
    inliers: int
    new_keyframe: Keyframe | None
    timings_ms: dict[str, float] = field(default_factory=dict)


class RgbdOdometry:
    """Frame-to-keyframe RGB-D visual odometry."""

    def __init__(self, camera: PinholeCamera, matcher: FeatureMatcher, config: OdometryConfig) -> None:
        self.camera = camera
        self.matcher = matcher
        self.config = config
        self.pnp = PnpSolver(camera, config.reproj_error_px, min_inliers=config.min_inliers)
        self.keyframe: Keyframe | None = None
        self.T_cur_kf = SE3.identity()  # last relative pose (motion prior)
        self._next_id = 0

    def make_keyframe(self, frame: RgbdFrame, feats: Features, pose: SE3) -> Keyframe:
        d = frame.depth_at(feats.keypoints)
        ok = (d > self.config.min_depth) & (d < self.config.max_depth)
        pts = np.full((len(feats), 3), np.nan)
        if ok.any():
            pts[ok] = self.camera.backproject(feats.keypoints[ok], d[ok])
        kf = Keyframe(self._next_id, frame.stamp, pose, feats, pts, frame)
        self._next_id += 1
        self.keyframe, self.T_cur_kf = kf, SE3.identity()
        return kf

    def track(self, frame: RgbdFrame, feats: Features) -> TrackResult:
        if self.keyframe is None:
            kf = self.make_keyframe(frame, feats, SE3.identity())
            return TrackResult(kf.pose, TrackingState.INITIALIZING, 0, 0, kf)
        kf = self.keyframe
        t0 = time.perf_counter()
        pairs = self.matcher.match(feats, kf.features)
        pairs = pairs[kf.valid[pairs[:, 1]]] if len(pairs) else pairs
        t1 = time.perf_counter()
        res = self.pnp.solve(kf.points[pairs[:, 1]], feats.keypoints[pairs[:, 0]], guess=self.T_cur_kf) \
            if len(pairs) else PnpResult(SE3.identity(), np.zeros(0, dtype=np.intp), False)
        t2 = time.perf_counter()
        timings = {"match_ms": 1000 * (t1 - t0), "pnp_ms": 1000 * (t2 - t1)}

        if not res.ok:  # tracking lost: re-anchor a keyframe at the last known pose
            pose = kf.pose @ self.T_cur_kf.inverse()
            new_kf = self.make_keyframe(frame, feats, pose)
            return TrackResult(pose, TrackingState.LOST, len(pairs), 0, new_kf, timings)

        self.T_cur_kf = res.T_cur_ref
        pose = kf.pose @ res.T_cur_ref.inverse()
        new_kf = None
        if self._needs_keyframe(res, kf):
            new_kf = self.make_keyframe(frame, feats, pose)
        return TrackResult(pose, TrackingState.OK, len(pairs), len(res.inliers), new_kf, timings)

    def _needs_keyframe(self, res: PnpResult, kf: Keyframe) -> bool:
        c = self.config
        rel = res.T_cur_ref
        ratio = len(res.inliers) / max(int(kf.valid.sum()), 1)
        return (rel.translation_norm > c.kf_translation
                or np.degrees(rel.rotation_angle) > c.kf_rotation_deg
                or ratio < c.kf_min_inlier_ratio)
