"""Unit tests for the ROS-free W2 core."""

from __future__ import annotations

import math

import numpy as np
import pytest

from localization_lab.core.evaluation import LocalizationEvaluator, LocalizationLog
from localization_lab.core.grid_localization import GridLocalizer
from localization_lab.core.learned import Mlp, ScanFeaturizer, ScanPoseRegressor
from localization_lab.core.mcl import MclConfig, MonteCarloLocalizer
from localization_lab.core.common import FreeSpace
from localization_lab.core.scenario import LabWorld, ScenarioConfig, ScenarioRecorder
from sensor_motion_lab.core import Pose2D


@pytest.fixture(scope="module")
def world() -> LabWorld:
    return LabWorld.load()


@pytest.fixture(scope="module")
def recording(world: LabWorld):  # noqa: ANN201
    return ScenarioRecorder(world, ScenarioConfig(duration_s=30.0, kidnap_at_s=15.0, seed=3)).record()


def _run(loc, rec):  # noqa: ANN001, ANN202
    log = LocalizationLog()
    prev = rec.samples[0].odom
    for s in rec.samples:
        est = loc.update(prev, s.odom, s.scan)
        prev = s.odom
        log.add(s.t, est.pose, s.truth)
    return log


def test_free_space_samples_are_free(world: LabWorld) -> None:
    fs = FreeSpace(world.grid, clearance=0.2)
    p = fs.sample(500, np.random.default_rng(0))
    assert fs.contains(p[:, :2], clearance=0.15).all()


def test_mlp_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    net = Mlp((5, 7, 3), seed=1)
    X, Y = rng.normal(size=(8, 5)), rng.normal(size=(8, 3))
    grads, _ = net._backprop(X, Y)  # noqa: SLF001
    w = net.W[0]
    eps = 1e-6
    i, j = 2, 3
    w[i, j] += eps
    lp = float(np.mean((net.predict(X) - Y) ** 2))
    w[i, j] -= 2 * eps
    lm = float(np.mean((net.predict(X) - Y) ** 2))
    w[i, j] += eps
    assert grads[0][i, j] == pytest.approx((lp - lm) / (2 * eps), rel=1e-4)


def test_mlp_learns_a_simple_function() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (2000, 2))
    Y = np.column_stack((np.sin(2 * X[:, 0]), X[:, 0] * X[:, 1]))
    hist = Mlp((2, 32, 32, 2)).fit(X, Y, epochs=40)
    assert hist.val_loss[-1] < 0.1 * hist.val_loss[0]


def test_regressor_save_load_roundtrip(tmp_path, world: LabWorld) -> None:  # noqa: ANN001
    reg = ScanPoseRegressor(ScanFeaturizer(36, world.robot.lidar.range_max), Mlp((36, 16, 4)))
    path = reg.save(tmp_path / "m.npz")
    again = ScanPoseRegressor.load(path)
    f = np.random.default_rng(0).random((3, 36))
    np.testing.assert_allclose(reg.predict_features(f), again.predict_features(f))


def test_grid_localizer_global_then_tracks(world: LabWorld, recording) -> None:  # noqa: ANN001
    log = _run(GridLocalizer(world.grid, world.robot.lidar.range_max), recording)
    rep = LocalizationEvaluator().report(log, recording.samples[0].t, recording.kidnap_t)
    assert rep.converged and rep.tracking_rmse_m < 0.15


def test_mcl_global_localization_and_kidnap_recovery(world: LabWorld, recording) -> None:  # noqa: ANN001
    # with too few particles (e.g. 1500) recovery is seed-dependent: global localization needs coverage
    mcl = MonteCarloLocalizer(world.grid, world.robot.lidar.range_max, MclConfig(num_particles=2000), seed=0)
    log = _run(mcl, recording)
    rep = LocalizationEvaluator().report(log, recording.samples[0].t, recording.kidnap_t)
    assert rep.converged and rep.tracking_rmse_m < 0.1
    assert rep.recovered, "augmented MCL should re-localize after the kidnapping"


def test_mcl_without_recovery_stays_lost(world: LabWorld, recording) -> None:  # noqa: ANN001
    cfg = MclConfig(num_particles=1500, alpha_slow=0.0, alpha_fast=0.0)
    log = _run(MonteCarloLocalizer(world.grid, world.robot.lidar.range_max, cfg, seed=0), recording)
    after = np.asarray(log.stamps) > recording.kidnap_t + 5
    assert np.median(np.asarray(log.pos_err)[after]) > 0.5


def test_tracking_from_known_pose(world: LabWorld, recording) -> None:  # noqa: ANN001
    mcl = MonteCarloLocalizer(world.grid, world.robot.lidar.range_max, MclConfig(num_particles=500), seed=1)
    first = recording.samples[0]
    mcl.reset_at(first.truth, 0.1, 0.05)
    est = mcl.update(first.odom, first.odom, first.scan)
    assert math.hypot(est.pose.x - first.truth.x, est.pose.y - first.truth.y) < 0.1


def test_convergence_criterion() -> None:
    log = LocalizationLog()
    for k in range(100):
        t = 0.1 * k
        err = 1.0 if t < 2.0 else 0.05
        log.add(t, Pose2D(err, 0, 0), Pose2D(0, 0, 0))
    rep = LocalizationEvaluator().report(log, 0.0)
    assert rep.converged and rep.convergence_s == pytest.approx(2.0, abs=0.11)


def test_free_space_excludes_outside_the_walls(world: LabWorld) -> None:
    fs = FreeSpace(world.grid, clearance=0.2)
    p = fs.sample(2000, np.random.default_rng(1))
    half = world.spec.size_x / 2
    assert (np.abs(p[:, 0]) < half).all() and (np.abs(p[:, 1]) < world.spec.size_y / 2).all()
    assert not fs.contains(np.array([[half + 0.25, 0.0]]))[0]  # in the margin outside the east wall
