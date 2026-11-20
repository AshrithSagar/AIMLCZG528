"""Unit tests for the ROS-agnostic core (run with `pytest` or `colcon test`)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sensor_motion_lab.core import (
    BeamModelParams,
    BeamRangeFinderModel,
    ConsistencyTracker,
    DifferentialDrive,
    Direction,
    DriveCalibrationError,
    DriveGeometry,
    GridRasterizer,
    KinematicRobot,
    LidarSpec,
    LikelihoodFieldParams,
    MotionNoise,
    OdometryControl,
    OdometryMotionModel,
    Pose2D,
    RangeErrorAnalyzer,
    RangeSensorModelFactory,
    ReactiveAvoidanceController,
    AvoidanceConfig,
    RayCaster,
    Scan,
    SdfWorldWriter,
    SlipNoise,
    SquarePathConfig,
    SquarePathController,
    Trajectory,
    TrajectoryEvaluator,
    UMBmarkReport,
    UMBmarkRun,
    VelocityControl,
    VelocityMotionModel,
    WorldSpec,
)
from sensor_motion_lab.core.config import RobotConfig

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x on Ubuntu 24.04
DESC = Path(__file__).resolve().parents[2] / "mobile_bot_description" / "config"


@pytest.fixture(scope="module")
def world() -> WorldSpec:
    return WorldSpec.from_yaml(DESC / "world.yaml")


@pytest.fixture(scope="module")
def grid(world: WorldSpec):
    return GridRasterizer(0.05).rasterize(world)


# ----- geometry ------------------------------------------------------------
def test_pose_compose_inverse_roundtrip() -> None:
    a, b = Pose2D(1.0, 2.0, 0.7), Pose2D(-0.3, 0.5, -2.0)
    c = a.compose(b)
    back = a.between(c)
    assert back.x == pytest.approx(b.x) and back.y == pytest.approx(b.y) and back.theta == pytest.approx(b.theta)


# ----- world ----------------------------------------------------------------
def test_raycast_hits_east_wall(grid) -> None:
    caster = RayCaster(grid, max_range=10.0)
    r = caster.cast(np.array([[0.0, 1.0, 0.0]]), np.array([0.0]))[0, 0]
    assert r == pytest.approx(3.0, abs=0.06)  # inner face of the east wall at x = 3


def test_sdf_contains_all_models(world: WorldSpec) -> None:
    sdf = SdfWorldWriter().render(world)
    for box in world.all_boxes():
        assert f'<model name="{box.name}">' in sdf
    assert "gz::sim::systems::Sensors" in sdf


def test_distance_field_zero_on_obstacles(grid) -> None:
    df = grid.distance_field
    assert np.all(df[grid.occupied] == 0.0)
    assert df.max() > 0.5


def test_robot_config_loads() -> None:
    cfg = RobotConfig.from_yaml(DESC / "robot.yaml")
    assert cfg.drive.wheel_radius > 0 and cfg.lidar.samples == 360


# ----- motion ---------------------------------------------------------------
def test_calibration_preserves_mean_diameter() -> None:
    rl, rr, _ = DriveCalibrationError(1.01, 1.0).actual_geometry(DriveGeometry(0.05, 0.36))
    assert (rl + rr) / 2 == pytest.approx(0.05)
    assert rr / rl == pytest.approx(1.01)


def test_zero_noise_odometry_model_is_deterministic() -> None:
    model = OdometryMotionModel(MotionNoise(0, 0, 0, 0), np.random.default_rng(0))
    p0, p1 = Pose2D(0, 0, 0), Pose2D(1.0, 0.5, 0.3)
    out = model.sample(np.zeros((5, 3)), OdometryControl(p0, p1))
    np.testing.assert_allclose(out, np.tile(p1.as_array(), (5, 1)), atol=1e-12)


def test_zero_noise_velocity_model_matches_arc() -> None:
    model = VelocityMotionModel(MotionNoise(0, 0, 0, 0), np.random.default_rng(0))
    out = model.sample(np.zeros((1, 3)), VelocityControl(1.0, math.pi / 2, 1.0))[0]
    np.testing.assert_allclose(out, [2 / math.pi, 2 / math.pi, math.pi / 2], atol=1e-9)


def test_noise_grows_spread() -> None:
    rng = np.random.default_rng(1)
    small = OdometryMotionModel(MotionNoise(0.01, 0.01, 0.01, 0.01), rng)
    big = OdometryMotionModel(MotionNoise.nav2_amcl_defaults(), rng)
    u = OdometryControl(Pose2D(), Pose2D(1.0, 0.0, 0.0))
    s_small = small.sample(np.zeros((2000, 3)), u)[:, :2].std(axis=0).sum()
    s_big = big.sample(np.zeros((2000, 3)), u)[:, :2].std(axis=0).sum()
    assert s_big > 2 * s_small


# ----- UMBmark --------------------------------------------------------------
def _run_square(error: DriveCalibrationError, direction: Direction) -> tuple[float, float]:
    robot = KinematicRobot(DifferentialDrive(DriveGeometry(0.05, 0.36), error), SlipNoise(0.0),
                           np.random.default_rng(0))
    ctl = SquarePathController(SquarePathConfig(side=2.0, direction=direction))
    for _ in range(20000):
        if ctl.done:
            break
        robot.step(*ctl.command(robot.odom_pose), 0.01)
    assert ctl.done
    return robot.true_pose.x, robot.true_pose.y


def test_umbmark_detects_baseline_error() -> None:
    good = UMBmarkReport.from_runs(
        [UMBmarkRun(d, *_run_square(DriveCalibrationError(), d)) for d in Direction]
    )
    bad = UMBmarkReport.from_runs(
        [UMBmarkRun(d, *_run_square(DriveCalibrationError(1.0, 1.05), d)) for d in Direction]
    )
    assert good.e_max_syst < 0.05
    assert bad.e_max_syst > 4 * good.e_max_syst


# ----- metrics --------------------------------------------------------------
def test_ate_zero_for_identical_trajectories() -> None:
    t = np.linspace(0, 10, 50)
    poses = np.column_stack((t, np.sin(t), np.zeros_like(t)))
    traj = Trajectory(t, poses)
    rep = TrajectoryEvaluator().evaluate(traj, traj)
    assert rep.ate_rmse == pytest.approx(0.0, abs=1e-12)
    assert rep.rpe_trans_rmse == pytest.approx(0.0, abs=1e-12)


def test_ate_alignment_removes_rigid_offset() -> None:
    t = np.linspace(0, 10, 50)
    poses = np.column_stack((t, np.sin(t), np.zeros_like(t)))
    shifted = poses.copy()
    shifted[:, 0] += 0.5
    rep = TrajectoryEvaluator(align=True).evaluate(Trajectory(t, poses), Trajectory(t, shifted))
    assert rep.ate_rmse == pytest.approx(0.0, abs=1e-9)


def test_consistency_tracker_calibrated() -> None:
    rng = np.random.default_rng(3)
    tracker = ConsistencyTracker()
    for _ in range(400):
        particles = rng.normal(0.0, 0.1, (500, 3))
        tracker.update(particles, tuple(rng.normal(0.0, 0.1, 2)))
    assert tracker.mean_nees == pytest.approx(2.0, rel=0.2)
    assert tracker.coverage_95 == pytest.approx(0.95, abs=0.04)


def test_range_error_within_spec() -> None:
    rng = np.random.default_rng(4)
    expected = rng.uniform(0.2, 3.4, 5000)
    measured = expected + rng.normal(0, 0.005, expected.size)
    rep = RangeErrorAnalyzer(LidarSpec.lds01()).analyze(measured, expected)
    assert rep.within_spec_accuracy > 0.95 and abs(rep.bias) < 1e-3


# ----- sensor models --------------------------------------------------------
def test_beam_density_integrates_to_one(grid) -> None:
    model = BeamRangeFinderModel(grid, BeamModelParams.nav2_amcl_defaults(), range_max=3.5)
    z = np.linspace(0, 3.5 - 2e-3, 20001)  # stay below the z_max point mass
    p = model.density(z, np.full_like(z, 1.7))
    prm = model.params
    integral = _TRAPZ(p, z) + prm.z_max  # continuous part + point mass at z_max
    assert integral == pytest.approx(prm.z_hit + prm.z_short + prm.z_max + prm.z_rand, rel=0.01)


@pytest.mark.parametrize("which", ["beam", "likelihood_field", "correlation"])
def test_likelihood_peaks_near_true_pose(grid, which: str) -> None:
    rng = np.random.default_rng(5)
    true = np.array([[0.5, 0.3, 0.2]])
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    z = RayCaster(grid, 3.5).cast(true, angles)[0] + rng.normal(0, 0.01, 360)
    scan = Scan(np.clip(z, 0, 3.5), angles, 0.12, 3.5)
    model = RangeSensorModelFactory.create(which, grid, BeamModelParams.nav2_amcl_defaults(),
                                           LikelihoodFieldParams.nav2_amcl_defaults(), 3.5)
    xs = np.arange(-0.3, 0.31, 0.05)
    cand = np.array([[0.5 + dx, 0.3 + dy, 0.2] for dx in xs for dy in xs])
    best = cand[int(np.argmax(model.log_likelihood(scan, cand)))]
    assert np.hypot(*(best[:2] - true[0, :2])) <= 0.08


def test_beam_sampling_mixture_proportions(grid) -> None:
    params = BeamModelParams(0.7, 0.1, 0.1, 0.1, 0.02, 1.0)
    model = BeamRangeFinderModel(grid, params, 3.5)
    z = model.sample(np.full(20000, 2.0), np.random.default_rng(6))
    assert np.mean(z >= 3.5 - 1e-6) == pytest.approx(0.1 + 0.1 * 0.0, abs=0.02)
    assert np.mean(np.abs(z - 2.0) < 0.06) == pytest.approx(0.7, abs=0.03)


def test_avoidance_turns_only_when_blocked() -> None:
    ctl = ReactiveAvoidanceController(AvoidanceConfig())
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    clear = np.full(360, 2.0)
    blocked = clear.copy()
    blocked[180] = 0.4  # straight ahead (angle 0)
    assert ctl.command(clear, angles) == (0.2, 0.0)
    assert ctl.command(blocked, angles) == (0.0, AvoidanceConfig().turn_rate)
    assert ctl.command(np.full(360, np.inf), angles)[0] > 0  # no returns -> keep driving


def test_circle_obstacles_ray_hits() -> None:
    from sensor_motion_lab.core import CircleObstacles

    r = CircleObstacles.ranges(Pose2D(0, 0, 0), np.array([0.0, math.pi / 2]), np.array([[2.0, 0.0]]), 0.25, 3.5)
    assert r[0] == pytest.approx(1.75) and math.isinf(r[1])
