"""Unit tests for the ROS-free W4 core."""

from __future__ import annotations

import math

import numpy as np
import pytest

from planning_lab.core.genetic import GaConfig, GeneticPlanner
from planning_lab.core.grid import PlanningGrid
from planning_lab.core.metrics import PathQuality
from planning_lab.core.navigation import Crowd, NavConfig, NavigationSimulator, ObstaclePredictor, PurePursuit, Walker
from planning_lab.core.planners import AStarPlanner, DStarLitePlanner
from planning_lab.core.w1 import W1Assets
from sensor_motion_lab.core import GridMap, GridRasterizer, Pose2D, WorldSpec
from sensor_motion_lab.core.config import RobotConfig


@pytest.fixture(scope="module")
def world() -> GridMap:
    return GridRasterizer(0.05).rasterize(WorldSpec.from_yaml(W1Assets.config_dir() / "world.yaml"))


@pytest.fixture()
def grid(world: GridMap) -> PlanningGrid:
    return PlanningGrid(world)


def _queries(grid: PlanningGrid, n: int, seed: int = 0) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        s, t = grid.random_free_xy(rng), grid.random_free_xy(rng)
        if math.dist(s, t) > 2.5:
            out.append((s, t))
    return out


def test_astar_paths_are_collision_free(grid: PlanningGrid) -> None:
    for s, t in _queries(grid, 10):
        res = AStarPlanner(grid).plan(s, t)
        assert res.success and grid.path_free(res.path[1:-1]) and res.length >= math.dist(s, t) - 1e-9


def test_dstar_lite_matches_astar_initially_and_after_changes(grid: PlanningGrid) -> None:
    for s, t in _queries(grid, 5, seed=1):
        grid.set_obstacles(np.zeros((0, 2)))
        a, d = AStarPlanner(grid), DStarLitePlanner(grid)
        ra, rd = a.plan(s, t), d.plan(s, t)
        assert ra.cost == pytest.approx(rd.cost)
        mid = rd.path[len(rd.path) // 2]
        start2 = tuple(rd.path[2])
        changed = grid.set_obstacles(np.array([mid]))
        ra2, rd2 = a.plan(start2, t), d.replan(start2, t, changed)
        assert ra2.success == rd2.success
        if ra2.success:
            assert ra2.cost == pytest.approx(rd2.cost)


def test_ga_finds_a_collision_free_path(grid: PlanningGrid) -> None:
    (s, t), = _queries(grid, 1, seed=2)
    res = GeneticPlanner(grid, GaConfig(seed=0, generations=60)).plan(s, t)
    assert res.success


def test_seeded_ga_is_close_to_reference(grid: PlanningGrid) -> None:
    (s, t), = _queries(grid, 1, seed=3)
    astar = AStarPlanner(grid)
    ref = PathQuality.of(astar.shortcut(astar.plan(s, t).path), grid).length
    res = GeneticPlanner(grid, GaConfig(seed=0, seed_fraction=0.25, generations=60)).plan(s, t)
    assert res.success and res.length <= 1.1 * ref


def test_footprint_clearing_lets_the_robot_escape(grid: PlanningGrid) -> None:
    (s, t), = _queries(grid, 1, seed=4)
    grid.set_obstacles(np.array([[s[0] + 0.3, s[1]]]))  # an obstacle right next to the robot
    assert not grid.is_free(grid.to_cell(*s))
    assert AStarPlanner(grid).plan(s, t).success


def test_walker_ping_pong() -> None:
    w = Walker((0.0, 0.0), (2.0, 0.0), 1.0)
    assert w.position(1.0) == pytest.approx((1.0, 0.0))
    assert w.position(3.0) == pytest.approx((1.0, 0.0))  # on the way back


def test_predictor_extrapolates() -> None:
    p = ObstaclePredictor(1.0, 2)
    p.predict(np.array([[0.0, 0.0]]), 0.0)
    out = p.predict(np.array([[0.1, 0.0]]), 0.1)  # 1 m/s along x
    np.testing.assert_allclose(out[:, 0], [0.1, 0.6, 1.1])


def test_pure_pursuit_never_targets_behind() -> None:
    pp = PurePursuit()
    pp.set_path(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    v, w, _ = pp.command(Pose2D(1.6, 0.0, 0.0), np.zeros((0, 2)))
    assert v > 0 and abs(w) < 1e-6


def test_navigation_reaches_goal_with_replanning(world: GridMap) -> None:
    robot = RobotConfig.from_yaml(W1Assets.config_dir() / "robot.yaml")
    g = PlanningGrid(world)
    log = NavigationSimulator(g, robot, Crowd.w1_arena(), NavConfig(timeout_s=90)).run(
        AStarPlanner(g), Pose2D(-2.4, 0.0, 0.3), (2.2, 1.8))
    assert log.report.success and log.report.collisions <= 1


def test_episode_monitor_blames_only_a_moving_robot() -> None:
    from planning_lab.core.evaluation import EpisodeConfig, EpisodeMonitor

    m = EpisodeMonitor((5.0, 0.0), EpisodeConfig(robot_radius=0.2, obstacle_radius=0.25))
    far, near = np.array([[3.0, 3.0]]), np.array([[0.3, 0.0]])
    m.step(0.0, (0.0, 0.0), far)
    m.step(1.0, (0.0, 0.0), near)  # walker bumps a stopped robot
    m.step(2.0, (0.0, 0.0), far)
    m.step(3.0, (0.5, 0.0), np.array([[0.8, 0.0]]))  # robot drives into a walker
    r = m.report()
    assert (r.collisions, r.contacts_while_stopped) == (1, 1)
    assert r.travelled_m == 0.5 and r.time_s == 3.0 and r.min_clearance_m < 0 and not r.success
