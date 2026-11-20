"""
`VisualSlamPipeline` — the façade both the ROS node and the offline benchmark use.

    frame ──► features ──► RGB-D odometry ──► (new keyframe?) ──► map
                                                   └──► loop detection ──► pose-graph optimisation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backend import KeyframeMap, LoopConfig, LoopDetector, PoseGraphOptimizer
from .camera import PinholeCamera, RgbdFrame
from .features import FeatureExtractor, FeatureExtractorFactory
from .metrics import Trajectory
from .se3 import SE3
from .tracking import DescriptorMatcher, OdometryConfig, RgbdOdometry, TrackingState


@dataclass(frozen=True, slots=True)
class SlamConfig:
    feature_type: str = "orb"
    max_features: int = 1000
    superpoint_weights: str = ""
    device: str = "cpu"
    match_ratio: float = 0.8
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    loop_closure: bool = True


@dataclass(slots=True)
class FrameResult:
    stamp: float
    pose: SE3  # T_world_cam (optical frame), as estimated *now*
    state: TrackingState
    features: int
    matches: int
    inliers: int
    new_keyframe: bool
    loop_closed: bool
    timings_ms: dict[str, float]


class VisualSlamPipeline:
    def __init__(self, camera: PinholeCamera, config: SlamConfig, extractor: FeatureExtractor | None = None) -> None:
        self.camera = camera
        self.config = config
        self.extractor = extractor or FeatureExtractorFactory.create(
            config.feature_type, config.max_features, config.superpoint_weights, config.device)
        matcher = DescriptorMatcher(config.match_ratio)
        self.odometry = RgbdOdometry(camera, matcher, config.odometry)
        self.map = KeyframeMap()
        self.loops = LoopDetector(camera, matcher, config.loop)
        self.optimizer = PoseGraphOptimizer()
        # every frame is stored relative to its reference keyframe, so pose-graph
        # corrections propagate to the whole trajectory (as in ORB-SLAM)
        self._frames: list[tuple[float, int, SE3]] = []
        self._loop_cooldown = 0
        self.last_features = None  # most recent frame's features (for debug visualisation)  # skip detection right after a closure (same place, same loop)

    def process(self, frame: RgbdFrame) -> FrameResult:
        t0 = time.perf_counter()
        feats = self.extractor.extract(frame.gray)
        self.last_features = feats
        timings = {"extract_ms": self.extractor.last_ms}
        res = self.odometry.track(frame, feats)
        timings.update(res.timings_ms)
        loop_closed = False
        if res.new_keyframe is not None:
            self.map.add(res.new_keyframe)
            if self.config.loop_closure:
                loop_closed = self._close_loops(res.new_keyframe, timings)
        kf = self.odometry.keyframe
        assert kf is not None
        pose = kf.pose @ self.odometry.T_cur_kf.inverse()
        self._frames.append((frame.stamp, kf.id, self.odometry.T_cur_kf.inverse()))
        timings["total_ms"] = 1000 * (time.perf_counter() - t0)
        return FrameResult(frame.stamp, pose, res.state, len(feats), res.matches, res.inliers,
                           res.new_keyframe is not None, loop_closed, timings)

    def _close_loops(self, kf, timings: dict[str, float]) -> bool:  # noqa: ANN001 - Keyframe
        if self._loop_cooldown > 0:
            self._loop_cooldown -= 1
            return False
        t0 = time.perf_counter()
        cand = self.loops.detect(kf, self.map)
        timings["loop_ms"] = 1000 * (time.perf_counter() - t0)
        if cand is None:
            return False
        self.map.edges.append(cand.edge)
        self.optimizer.optimize(self.map)
        self._loop_cooldown = self.config.loop.cooldown
        timings["pgo_ms"] = self.optimizer.last_ms
        return True

    def trajectory(self) -> Trajectory:
        """Full (loop-corrected) trajectory, one pose per processed frame."""
        traj = Trajectory.empty()
        for stamp, kid, T_kf_cur in self._frames:
            traj.add(stamp, self.map.by_id(kid).pose @ T_kf_cur)
        return traj
