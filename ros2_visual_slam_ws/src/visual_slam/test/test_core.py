"""Unit tests for the ROS-free core (pytest / colcon test)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from visual_slam.core.backend import Edge, KeyframeMap, PoseGraphOptimizer
from visual_slam.core.camera import PinholeCamera
from visual_slam.core.datasets import SequenceFactory, SyntheticSource, TumExporter
from visual_slam.core.features import FeatureExtractorFactory
from visual_slam.core.metrics import Trajectory, TrajectoryEvaluator
from visual_slam.core.pipeline import SlamConfig, VisualSlamPipeline
from visual_slam.core.se3 import SE3, SO3
from visual_slam.core.synthetic import LoopTrajectory, SyntheticSequence
from visual_slam.core.tracking import Keyframe, PnpSolver
from visual_slam.core.features import Features
from visual_slam.core.runtime import MissingDependency, TorchRuntime


@pytest.fixture(scope="module")
def source() -> SyntheticSource:
    return SyntheticSource()


class _Prefix:
    """First n frames of a sequence (keeps tests fast)."""

    def __init__(self, base: SyntheticSource, n: int) -> None:
        self.base, self.n, self.name, self.camera = base, n, base.name, base.camera

    def __len__(self) -> int:
        return self.n

    def frame(self, i: int):  # noqa: ANN201
        return self.base.frame(i)

    def ground_truth(self, i: int) -> SE3:
        return self.base.ground_truth(i)


# ----- geometry ---------------------------------------------------------------
def test_so3_exp_log_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.normal(size=3)
        w *= rng.uniform(0, 3.0) / np.linalg.norm(w)
        np.testing.assert_allclose(SO3.log(SO3.exp(w)), w, atol=1e-8)


def test_quaternion_roundtrip() -> None:
    r = SO3.exp(np.array([0.3, -1.2, 2.0]))
    np.testing.assert_allclose(SO3.from_quaternion(SO3.to_quaternion(r)), r, atol=1e-10)


def test_se3_inverse_and_apply() -> None:
    t = SE3(SO3.exp(np.array([0.1, 0.2, 0.3])), np.array([1.0, -2.0, 0.5]))
    p = np.random.default_rng(1).normal(size=(10, 3))
    np.testing.assert_allclose(t.inverse().apply(t.apply(p)), p, atol=1e-12)
    np.testing.assert_allclose((t @ t.inverse()).matrix(), np.eye(4), atol=1e-12)


def test_camera_project_backproject() -> None:
    cam = PinholeCamera.tum_freiburg1()
    uv = np.array([[100.0, 50.0], [320.0, 240.0], [600.0, 400.0]])
    d = np.array([1.0, 2.5, 4.0])
    np.testing.assert_allclose(cam.project(cam.backproject(uv, d)), uv, atol=1e-9)


# ----- synthetic source ---------------------------------------------------------
def test_synthetic_depth_matches_geometry() -> None:
    """Camera at the origin looking along +x sees the wall at x = 4 m (room is 8 m wide)."""
    seq = SyntheticSequence()
    r_body_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    frame = seq.renderer.render(SE3(r_body_opt, np.array([0.0, 0.0, 1.5])))
    c = seq.camera
    assert frame.depth[int(c.cy), int(c.cx)] == pytest.approx(4.0, abs=0.05)


def test_trajectory_is_a_loop() -> None:
    traj = LoopTrajectory(loops=1.0)
    a, b = traj.pose(0.0), traj.pose(traj.period)
    assert np.linalg.norm(a.translation - b.translation) < 1e-9


def test_tum_export_roundtrip(tmp_path, source: SyntheticSource) -> None:  # noqa: ANN001
    TumExporter.write(_Prefix(source, 5), tmp_path)
    tum = SequenceFactory.create("tum", str(tmp_path))
    assert len(tum) == 5 and tum.camera == source.camera
    a, b = source.frame(3), tum.frame(3)
    assert (a.rgb == b.rgb).all()
    assert np.abs(a.depth - b.depth).max() < 2e-4  # uint16 / 5000 quantisation
    np.testing.assert_allclose(tum.ground_truth(3).matrix(), source.ground_truth(3).matrix(), atol=1e-5)


# ----- front-end --------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["orb", "sift", "akaze"])
def test_classical_extractors(kind: str, source: SyntheticSource) -> None:
    try:
        extractor = FeatureExtractorFactory.create(kind, 500)
    except RuntimeError as exc:  # e.g. AKAZE is not in OpenCV 5's main module
        pytest.skip(str(exc))
    feats = extractor.extract(source.frame(0).gray)
    assert 100 <= len(feats) <= 500


def test_unknown_or_unavailable_extractor() -> None:
    with pytest.raises(ValueError):
        FeatureExtractorFactory.create("harris")
    if not (TorchRuntime.available("torch") and TorchRuntime.available("lightglue")):
        with pytest.raises(MissingDependency, match="pip install"):  # a clear message, not a crash
            FeatureExtractorFactory.create("superpoint")


def test_pnp_recovers_known_motion() -> None:
    cam = PinholeCamera.tum_freiburg1()
    rng = np.random.default_rng(2)
    pts = np.column_stack((rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200), rng.uniform(2, 4, 200)))
    truth = SE3(SO3.exp(np.array([0.02, -0.05, 0.01])), np.array([0.1, 0.0, -0.05]))
    uv = cam.project(truth.apply(pts)) + rng.normal(0, 0.5, (200, 2))
    res = PnpSolver(cam).solve(pts, uv)
    assert res.ok
    err = truth.inverse() @ res.T_cur_ref
    assert err.translation_norm < 0.01 and math.degrees(err.rotation_angle) < 0.2


# ----- back-end ------------------------------------------------------------------------
def test_pose_graph_closes_a_drifting_square() -> None:
    """Odometry with a heading bias drifts; one correct loop edge pulls the end back."""
    step = SE3(SO3.exp(np.array([0, 0, math.pi / 8])), np.array([0.5, 0, 0]))  # 16 steps = a closed loop
    biased = SE3(SO3.exp(np.array([0, 0, math.pi / 8 + 0.02])), np.array([0.5, 0, 0]))
    dummy = Features(np.zeros((0, 2)), np.zeros((0, 32), np.uint8), True)
    kmap, pose = KeyframeMap(), SE3.identity()
    for i in range(17):
        kmap.add(Keyframe(i, float(i), pose, dummy, np.zeros((0, 3)), None))  # type: ignore[arg-type]
        pose = pose @ biased
    drift_before = kmap.keyframes[16].pose.translation_norm
    kmap.edges.append(Edge(0, 16, SE3.identity(), loop=True))  # truth: kf 16 is back at the start
    PoseGraphOptimizer().optimize(kmap)
    assert drift_before > 0.3
    assert kmap.keyframes[16].pose.translation_norm < 0.05
    assert np.linalg.norm(step.translation) > 0  # (step only documents the true motion)


def test_ate_is_zero_and_alignment_invariant() -> None:
    poses = [SE3(SO3.exp(np.array([0, 0, 0.1 * i])), np.array([i * 0.1, np.sin(i * 0.1), 0])) for i in range(50)]
    gt = Trajectory([float(i) for i in range(50)], poses)
    moved = SE3(SO3.exp(np.array([0.1, 0.2, 0.3])), np.array([3.0, -1.0, 2.0]))
    est = Trajectory([float(i) for i in range(50)], [moved @ p for p in poses])
    rep = TrajectoryEvaluator().evaluate(gt, est)
    assert rep is not None and rep.ate_rmse_m < 1e-9 and rep.rpe_trans_rmse_m < 1e-9


# ----- end to end ------------------------------------------------------------------------
def test_pipeline_tracks_first_segment(source: SyntheticSource) -> None:
    pipe = VisualSlamPipeline(source.camera, SlamConfig(feature_type="orb"))
    gt, est = Trajectory.empty(), Trajectory.empty()
    for i in range(120):
        f = source.frame(i)
        r = pipe.process(f)
        gt.add(f.stamp, source.ground_truth(i))
        est.add(f.stamp, r.pose)
        assert r.state.value != "lost"
    rep = TrajectoryEvaluator().evaluate(gt, est)
    assert rep is not None and rep.ate_rmse_m < 0.05


def test_gazebo_world_is_valid_xml(tmp_path) -> None:  # noqa: ANN001
    import xml.dom.minidom

    from visual_slam.core.gazebo import GazeboWorldWriter

    sdf = GazeboWorldWriter().write(tmp_path)
    doc = xml.dom.minidom.parse(str(sdf))
    names = {m.getAttribute("name") for m in doc.getElementsByTagName("model")}
    assert {"camera_rig", "room_floor", "crate_a"} <= names
    assert len(list((tmp_path / "textures").glob("*.png"))) >= 6
