"""
Tests for the deep-learning plumbing that do *not* need PyTorch: configuration,
metrics, retrieval with the classical global descriptor, and a stand-in depth
model. The real networks are exercised on the lab machine by `dl_check`.
"""

from __future__ import annotations

import numpy as np
import pytest

from visual_slam.core.metrics import FrameIndexStamps
from visual_slam.core.backend import LoopQuery
from visual_slam.core.camera import RgbdFrame
from visual_slam.core.datasets import SyntheticSource
from visual_slam.core.depth import DepthEstimator
from visual_slam.core.matching import DescriptorMatcher, MatcherFactory
from visual_slam.core.metrics import DepthEvaluator, PlaceRecognitionEvaluator, Trajectory, TrajectoryEvaluator
from visual_slam.core.pipeline import SlamConfig, VisualSlamPipeline
from visual_slam.core.place_recognition import (GlobalRetrieval, PlaceRecognizerFactory, ProximityVoting,
                                                RecognitionConfig, ThumbnailDescriptor)
from visual_slam.core.se3 import SE3, SO3


@pytest.fixture(scope="module")
def source() -> SyntheticSource:
    return SyntheticSource()


# ----- configuration --------------------------------------------------------------------
@pytest.mark.parametrize("label", ["orb", "superpoint+lightglue", "orb@dinov2", "superpoint+lightglue@thumbnail"])
def test_config_label_roundtrip(label: str) -> None:
    assert SlamConfig.from_label(label).label == label


def test_lightglue_rejects_classical_features() -> None:
    with pytest.raises(ValueError, match="learned features"):  # fails before any torch import
        MatcherFactory.create("lightglue", "orb")
    assert isinstance(MatcherFactory.create("bruteforce", "orb"), DescriptorMatcher)


def test_recognizer_factory() -> None:
    cfg, m = RecognitionConfig(), DescriptorMatcher()
    assert isinstance(PlaceRecognizerFactory.create("proximity", cfg, m), ProximityVoting)
    assert isinstance(PlaceRecognizerFactory.create("thumbnail", cfg, m), GlobalRetrieval)
    with pytest.raises(ValueError):
        PlaceRecognizerFactory.create("netvlad", cfg, m)


# ----- trajectories -----------------------------------------------------------------------
def test_tum_read_and_sim3_alignment(tmp_path) -> None:  # noqa: ANN001
    rng = np.random.default_rng(0)
    gt = Trajectory.empty()
    for i in range(40):
        gt.add(float(i), SE3(SO3.exp(rng.normal(0, 0.2, 3)), rng.normal(0, 1, 3)))
    # a monocular method: right shape, wrong scale, different world frame
    world = SE3(SO3.exp(np.array([0.3, -0.2, 0.5])), np.array([1.0, 2.0, 3.0]))
    est = Trajectory.empty()
    for t, p in zip(gt.stamps, gt.poses):
        q = world @ p
        est.add(t, SE3(q.rotation, 0.4 * q.translation))
    back = Trajectory.read_tum(est.write_tum(tmp_path / "est.tum"))
    assert len(back) == len(est)
    se3 = TrajectoryEvaluator().evaluate(gt, back)
    sim3 = TrajectoryEvaluator(with_scale=True).evaluate(gt, back)
    assert se3 is not None and sim3 is not None
    assert sim3.ate_rmse_m < 1e-4 < se3.ate_rmse_m
    aligned = TrajectoryEvaluator(with_scale=True).align(gt, back)
    assert np.allclose(aligned.positions, gt.positions, atol=1e-4)


def test_frame_index_stamps(tmp_path) -> None:  # noqa: ANN001
    rgb = tmp_path / "rgb.txt"
    rgb.write_text("# timestamp filename\n" + "".join(f"{100 + 0.1 * i:.6f} rgb/{i}.png\n" for i in range(10)))
    traj = Trajectory([0.0, 1.0, 2.0], [SE3.identity()] * 3)
    assert FrameIndexStamps.looks_like_indices(traj)
    mapped = FrameIndexStamps(rgb, stride=2).apply(traj)
    assert np.allclose(mapped.stamps, [100.0, 100.2, 100.4])
    assert not FrameIndexStamps.looks_like_indices(mapped)


# ----- place recognition ------------------------------------------------------------------
def test_thumbnail_descriptor_prefers_the_revisit(source: SyntheticSource) -> None:
    per_loop = int(source.seq.trajectory.period * source.seq.trajectory.rate_hz)
    d = ThumbnailDescriptor()
    a, revisit, opposite = (d.describe(source.frame(i).rgb) for i in (0, per_loop, per_loop // 2))
    assert abs(float(a @ a) - 1.0) < 1e-5
    assert float(a @ revisit) > float(a @ opposite)


def test_place_recognition_metrics() -> None:
    # keyframes 0..5 along a line, then 6 and 7 back at the positions of 0 and 1
    xs = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 0.1, 2.1]
    stamps = [float(i) for i in range(len(xs))]
    gt = Trajectory(stamps, [SE3(np.eye(3), np.array([x, 0.0, 0.0])) for x in xs])
    queries = [LoopQuery(6, 0, 0.9, 0, 1.0),  # right place, accepted
               LoopQuery(7, 3, 0.5, 3, 1.0),  # wrong place, accepted anyway -> false loop
               LoopQuery(5, 1, 0.2, None, 1.0)]  # no true revisit exists for 5
    rep = PlaceRecognitionEvaluator(radius_m=0.5, min_separation=3).evaluate(queries, stamps, gt)
    assert (rep.queries_with_revisit, rep.loops_accepted, rep.false_loops) == (2, 2, 1)
    assert rep.recall_at_1 == pytest.approx(0.5)


def test_pipeline_with_global_retrieval_runs(source: SyntheticSource) -> None:
    pipe = VisualSlamPipeline(source.camera, SlamConfig.from_label("orb@thumbnail"))
    for i in range(90):
        pipe.process(source.frame(i))
    assert isinstance(pipe.loops.recognizer, GlobalRetrieval)
    assert len(pipe.loops.recognizer.db) == len(pipe.map.keyframes)  # every keyframe is in the database


# ----- depth ---------------------------------------------------------------------------------
def test_depth_metrics() -> None:
    gt = np.full((20, 30), 2.0, dtype=np.float32)
    gt[0, 0] = 0.0  # invalid pixel, ignored
    perfect = DepthEvaluator().evaluate(gt.copy(), gt)
    assert perfect.abs_rel == 0.0 and perfect.delta_1 == 1.0 and perfect.pixels == 599
    double = DepthEvaluator().evaluate(2 * gt, gt)
    assert double.abs_rel == pytest.approx(1.0) and double.delta_1 == 0.0 and double.median_scale == pytest.approx(0.5)
    assert DepthEvaluator().evaluate(2 * gt, gt, median_align=True).abs_rel == pytest.approx(0.0)


class _NoisySensorDepth(DepthEstimator):
    """Stand-in for a network: the sensor depth with 3 % multiplicative noise."""

    name = "noisy_sensor"

    def __init__(self, source: SyntheticSource) -> None:
        super().__init__()
        self.source, self.rng, self.i = source, np.random.default_rng(1), 0

    def _estimate(self, rgb: np.ndarray) -> np.ndarray:
        d = self.source.frame(self.i).depth
        self.i += 1
        return d * self.rng.normal(1.0, 0.03, d.shape).astype(np.float32)


def test_slam_runs_on_predicted_depth(source: SyntheticSource) -> None:
    est = _NoisySensorDepth(source)
    pipe = VisualSlamPipeline(source.camera, SlamConfig())
    gt, traj = Trajectory.empty(), Trajectory.empty()
    for i in range(90):
        f = est.frame(source.frame(i))
        assert isinstance(f, RgbdFrame) and f.depth.dtype == np.float32
        r = pipe.process(f)
        gt.add(f.stamp, source.ground_truth(i))
        traj.add(f.stamp, r.pose)
    rep = TrajectoryEvaluator().evaluate(gt, traj)
    assert rep is not None and rep.ate_rmse_m < 0.1
