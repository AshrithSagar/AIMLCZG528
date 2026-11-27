"""
Offline benchmark — the "expected output" generator.

Runs every experiment of Webinar 1 *without ROS* (same core code the nodes
use) and writes figures + a metrics table. Show these at the start of the
session, then reproduce them live in RViz / Gazebo.

    ros2 run sensor_motion_lab offline_benchmark --out ~/w1_expected
    # or, without ROS:
    python -m sensor_motion_lab.offline_benchmark --config-dir <.../mobile_bot_description/config>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402

from .core import (  # noqa: E402
    ArcIntegrator,
    BeamModelParams,
    BeamRangeFinderModel,
    ConsistencyTracker,
    DifferentialDrive,
    Direction,
    DriveCalibrationError,
    GridMap,
    GridRasterizer,
    KinematicRobot,
    LidarSpec,
    LikelihoodFieldParams,
    LikelihoodSurfaceAnalyzer,
    MotionNoise,
    OdometryControl,
    OdometryMotionModel,
    Pose2D,
    RangeErrorAnalyzer,
    RangeSensorModelFactory,
    RayCaster,
    Scan,
    SimulatedLidar,
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
from .core.comparison import AisleScenario, FourModelComparison  # noqa: E402
from .core.config import RobotConfig  # noqa: E402

Metrics = dict[str, Any]


# --------------------------------------------------------------------------- #
# Shared context                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BenchmarkContext:
    world: WorldSpec
    grid: GridMap
    robot: RobotConfig
    seed: int

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    @classmethod
    def load(cls, config_dir: Path, seed: int) -> BenchmarkContext:
        world = WorldSpec.from_yaml(config_dir / "world.yaml")
        return cls(world, GridRasterizer(0.05).rasterize(world), RobotConfig.from_yaml(config_dir / "robot.yaml"), seed)


class MapPlotter:
    """Draws the occupancy grid as a background (shared by several figures)."""

    @staticmethod
    def draw(ax: plt.Axes, grid: GridMap, alpha: float = 1.0) -> None:
        extent = (grid.origin_x, grid.origin_x + grid.width * grid.resolution,
                  grid.origin_y, grid.origin_y + grid.height * grid.resolution)
        ax.imshow(grid.occupied, origin="lower", extent=extent, cmap="Greys", alpha=alpha, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")


class Experiment(ABC):
    """One figure + its metrics."""

    slug: str = "experiment"
    title: str = ""

    def __init__(self, ctx: BenchmarkContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def run(self) -> tuple[Figure, Metrics]: ...


# --------------------------------------------------------------------------- #
# Sensor experiments                                                          #
# --------------------------------------------------------------------------- #
class ScanGenerationExperiment(Experiment):
    slug, title = "01_scan_ideal_vs_beam_model", "Ideal scan vs. beam-model sampled scan"

    def run(self) -> tuple[Figure, Metrics]:
        ctx, li = self.ctx, self.ctx.robot.lidar
        pose = Pose2D(0.4, -0.6, 0.6)
        angles = li.angles()
        z_star = RayCaster(ctx.grid, li.range_max).cast(pose.as_array()[None], angles)[0]
        beam = BeamRangeFinderModel(ctx.grid, BeamModelParams.realistic_lidar(), li.range_max)
        z = beam.sample(z_star, ctx.rng(1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, ranges, label in ((axes[0], z_star, "ideal (ray-cast z*)"), (axes[1], z, "beam-model sample z ~ p(z|z*)")):
            MapPlotter.draw(ax, ctx.grid)
            hit = ranges < li.range_max - 1e-3
            ex = pose.x + ranges * np.cos(pose.theta + angles)
            ey = pose.y + ranges * np.sin(pose.theta + angles)
            ax.scatter(ex[hit], ey[hit], s=4, c="tab:red", label="returns")
            ax.scatter(ex[~hit], ey[~hit], s=4, c="tab:orange", label="max-range (no return)")
            ax.plot(pose.x, pose.y, "o", c="tab:blue", ms=8)
            ax.arrow(pose.x, pose.y, 0.4 * math.cos(pose.theta), 0.4 * math.sin(pose.theta), color="tab:blue", width=0.03)
            ax.set_title(label)
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(self.title)
        err = z - z_star
        return fig, {
            "short_readings_pct": float(100 * np.mean(err < -0.1)),
            "max_readings_pct": float(100 * np.mean(z >= li.range_max - 1e-3)),
        }


class BeamPdfExperiment(Experiment):
    slug, title = "02_beam_model_pdf", "Beam model p(z | z* = 2 m): Nav2 AMCL defaults vs. realistic LiDAR"

    def run(self) -> tuple[Figure, Metrics]:
        zmax = self.ctx.robot.lidar.range_max
        z = np.linspace(0, zmax - 1e-3, 1500)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
        for ax, (name, prm) in zip(axes, (("Nav2 AMCL defaults", BeamModelParams.nav2_amcl_defaults()),
                                         ("realistic LiDAR (sim noise)", BeamModelParams.realistic_lidar()))):
            model = BeamRangeFinderModel(self.ctx.grid, prm, zmax)
            comps = model.component_densities(z, np.full_like(z, 2.0))
            w = (prm.z_hit, prm.z_short, prm.z_max, prm.z_rand)
            for c, wi, lab in zip(comps, w, ("hit", "short", "max", "rand")):
                if lab != "max":
                    ax.plot(z, wi * c, lw=1, ls="--", label=f"{lab} (w={wi})")
            ax.plot(z, model.density(z, np.full_like(z, 2.0)), "k", lw=2, label="mixture")
            ax.axvline(zmax, c="tab:orange", lw=3, alpha=0.6, label=f"max point mass (w={prm.z_max})")
            ax.set_title(name)
            ax.set_xlabel("measured range z [m]")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("p(z | z*)")
        fig.suptitle(self.title)
        return fig, {}


class LikelihoodSurfaceExperiment(Experiment):
    slug, title = "03_likelihood_surfaces", "Map matching with the four measurement models (beam / likelihood field: Nav2 AMCL defaults)"

    def run(self) -> tuple[Figure, Metrics]:
        ctx, li = self.ctx, self.ctx.robot.lidar
        true = Pose2D(0.4, -0.6, 0.6)
        lidar = SimulatedLidar(RayCaster(ctx.grid, li.range_max), li, ctx.rng(2))
        scan = Scan.from_raw(lidar.measure(true), li.angle_min, li.angle_increment, li.range_min, li.range_max)

        half, step = 0.5, 0.025
        offs = np.arange(-half, half + 1e-9, step)
        gx, gy = np.meshgrid(true.x + offs, true.y + offs)
        cand = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, true.theta)))
        models = tuple(
            RangeSensorModelFactory.create(kind, ctx.grid, BeamModelParams.nav2_amcl_defaults(),
                                           LikelihoodFieldParams.nav2_amcl_defaults(), li.range_max, 60, world=ctx.world)
            for kind in RangeSensorModelFactory.KINDS
        )
        fig, axes = plt.subplots(1, 5, figsize=(26, 5))
        df = ctx.grid.distance_field
        extent = (ctx.grid.origin_x, ctx.grid.origin_x + ctx.grid.width * ctx.grid.resolution,
                  ctx.grid.origin_y, ctx.grid.origin_y + ctx.grid.height * ctx.grid.resolution)
        lf_img = np.exp(-0.5 * (np.minimum(df, 2.0) / 0.2) ** 2)
        axes[0].imshow(lf_img, origin="lower", extent=extent, cmap="magma")
        axes[0].set_title("likelihood field  exp(-d²/2σ²)")
        axes[0].set_aspect("equal")
        metrics: Metrics = {}
        for ax, model in zip(axes[1:], models):
            t0 = time.perf_counter()
            ll = model.log_likelihood(scan, cand)
            ms = 1000 * (time.perf_counter() - t0)
            rep = LikelihoodSurfaceAnalyzer.analyze(cand[:, :2], ll, (true.x, true.y), ms)
            metrics[model.name] = rep.as_dict() | {"us_per_pose": 1000 * ms / len(cand)}
            img = np.exp(ll - ll.max()).reshape(gx.shape)
            ax.imshow(img, origin="lower", extent=(gx.min(), gx.max(), gy.min(), gy.max()), cmap="viridis")
            ax.plot(true.x, true.y, "r+", ms=14, mew=2, label="true pose")
            best = cand[int(np.argmax(ll))]
            ax.plot(best[0], best[1], "wx", ms=10, mew=2, label="argmax")
            ax.set_title(f"{model.name}\npeak err {rep.peak_error_m * 100:.1f} cm, "
                         f"σ {rep.sharpness_m * 100:.1f} cm, {1000 * ms / len(cand):.1f} µs/pose", fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle(self.title)
        return fig, metrics


class FourModelExperiment(Experiment):
    """The CS-3 worked example: one wall, three beams, every model; Case A clear vs Case B a person in front."""

    slug, title = "08_four_models_cs3", "CS-3 example: the four measurement models on the same three returns"

    def run(self) -> tuple[Figure, Metrics]:
        clear = FourModelComparison(AisleScenario.case("A"))
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        metrics: Metrics = {}
        for ax, case in zip(axes, ("A", "B")):
            comp = FourModelComparison(AisleScenario.case(case))
            self._draw(ax, comp, clear if case == "B" else None)
            metrics[f"case_{case}"] = comp.scores()
        fig.suptitle(self.title)
        return fig, metrics

    def _draw(self, ax: plt.Axes, comp: FourModelComparison, clear: FourModelComparison | None) -> None:
        s = comp.s
        MapPlotter.draw(ax, GridRasterizer(0.05).rasterize(s.world()), alpha=0.5)
        cell = s.cell
        for col in range(comp.wall_cell - 0, comp.wall_cell + 1):
            ax.add_patch(Rectangle((col * cell, -2.0), cell, 4.0, color="0.6", alpha=0.25, lw=0))
        for c, e in zip(comp.scan_cells(), s.endpoints):
            ax.add_patch(Rectangle((c * cell, np.floor(e[1] / cell) * cell), cell, cell, color="tab:blue", alpha=0.5, lw=0))
        for zs, z, a in zip(s.z_star, s.z, s.angles):
            ax.plot([0, zs * math.cos(a)], [0, zs * math.sin(a)], color="0.5", lw=1, ls="--")
            ax.plot([0, z * math.cos(a)], [0, z * math.sin(a)], color="tab:red", lw=2)
        ends = s.endpoints
        ax.scatter(ends[:, 0], ends[:, 1], c="gold", s=60, zorder=5, edgecolors="k")
        for e in ends:
            ax.plot([e[0], s.wall_x], [e[1], e[1]], color="m", lw=2)
        line = comp.fitted_line()
        ax.plot([line.p0[0], line.p1[0]], [line.p0[1], line.p1[1]], color="tab:green", lw=3, label=f"fitted wall rho = {line.rho:.3f} m")
        if s.person_range is not None:
            ax.add_patch(Circle((s.person_range + 0.2, 0.0), 0.2, color="tab:orange", alpha=0.8))
            blind = comp.blind_line()
            ax.plot([blind.p0[0], blind.p1[0]], [blind.p0[1], blind.p1[1]], color="tab:red", ls=":", lw=2,
                    label="blind least-squares fit")
        ax.plot(0, 0, "o", c="tab:blue", ms=8)
        ax.set_xlim(-0.5, 6.0)
        ax.set_ylim(-2.4, 2.4)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(f"Case {s.label}\n" + "\n".join(comp.short_lines(clear)), fontsize=9, family="monospace",
                     loc="left")


class LidarSpecExperiment(Experiment):
    slug, title = "04_lidar_vs_lds01", "Simulated LiDAR range error vs. ROBOTIS LDS-01 datasheet"

    def run(self) -> tuple[Figure, Metrics]:
        ctx, li, spec = self.ctx, self.ctx.robot.lidar, LidarSpec.lds01()
        caster = RayCaster(ctx.grid, li.range_max)
        rng = ctx.rng(3)
        poses = np.column_stack((rng.uniform(-0.2, 2.2, 40), rng.uniform(-2.2, 2.2, 40), rng.uniform(-np.pi, np.pi, 40)))
        z_star = caster.cast(poses, li.angles()).ravel()
        gauss = z_star + rng.normal(0, li.noise_std, z_star.shape)
        beam = BeamRangeFinderModel(ctx.grid, BeamModelParams.realistic_lidar(), li.range_max).sample(z_star, rng)
        analyzer = RangeErrorAnalyzer(spec)
        metrics = {"gazebo_gaussian": analyzer.analyze(gauss, z_star).as_dict(),
                   "beam_model_realistic": analyzer.analyze(beam, z_star).as_dict()}

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        r = np.linspace(spec.range_min, spec.range_max, 200)
        for ax, (name, meas) in zip(axes, (("Gaussian σ = %.0f mm (Gazebo <noise>)" % (1000 * li.noise_std), gauss),
                                           ("beam-model sample (realistic)", beam))):
            ok = (z_star >= spec.range_min) & (z_star < spec.range_max)
            ax.scatter(z_star[ok][::7], (meas - z_star)[ok][::7] * 100, s=2, alpha=0.4)
            ax.fill_between(r, -spec.accuracy_at(r) * 100, spec.accuracy_at(r) * 100, color="tab:green", alpha=0.2,
                            label="LDS-01 accuracy band")
            m = metrics["gazebo_gaussian" if meas is gauss else "beam_model_realistic"]
            ax.set_title(f"{name}\nwithin spec: {100 * m['within_spec_accuracy']:.1f} %, "
                         f"outliers: {100 * m['outlier_fraction']:.1f} %")
            ax.set_xlabel("true range z* [m]")
            ax.set_ylim(-30, 30)
            ax.legend(loc="lower left", fontsize=8)
        axes[0].set_ylabel("error z - z* [cm]")
        fig.suptitle(self.title)
        return fig, metrics


# --------------------------------------------------------------------------- #
# Motion experiments                                                          #
# --------------------------------------------------------------------------- #
class MotionBananaExperiment(Experiment):
    slug, title = "05_motion_model_samples", "Sampling motion models (500 particles, every 1 s)"

    def run(self) -> tuple[Figure, Metrics]:
        n, dt = 500, 0.1
        controls = [VelocityControl(0.25, 0.0, dt)] * 40 + [VelocityControl(0.25, 0.35, dt)] * 50
        configs = (
            ("odometry model, α = 0.05 (well calibrated)", OdometryMotionModel, MotionNoise(0.05, 0.05, 0.05, 0.05)),
            ("odometry model, α = 0.2 (Nav2 AMCL default)", OdometryMotionModel, MotionNoise.nav2_amcl_defaults()),
            ("velocity model, α = 0.2, α5 = α6 = 0.02", VelocityMotionModel, MotionNoise(0.2, 0.2, 0.2, 0.2, 0.02, 0.02)),
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
        metrics: Metrics = {}
        for ax, (label, cls, noise) in zip(axes, configs):
            model = cls(noise, self.ctx.rng(4))
            particles = np.zeros((n, 3))
            odom = Pose2D()
            for k, u in enumerate(controls, start=1):
                nxt = ArcIntegrator.step(odom, u.v, u.w, u.dt)
                particles = model.sample(particles, OdometryControl(odom, nxt) if cls is OdometryMotionModel else u)
                odom = nxt
                if k % 10 == 0:
                    ax.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.4)
                    self._ellipse(ax, particles)
            ax.plot(odom.x, odom.y, "k*", ms=12)
            ax.set_title(label, fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            cov = np.cov(particles[:, :2].T)
            metrics[label] = {"final_pos_std_m": float(np.sqrt(np.trace(cov))),
                              "final_heading_std_deg": float(np.degrees(np.std(particles[:, 2])))}
        axes[0].set_ylabel("y [m]")
        fig.suptitle(self.title)
        return fig, metrics

    @staticmethod
    def _ellipse(ax: plt.Axes, particles: np.ndarray) -> None:
        mu, cov = particles[:, :2].mean(axis=0), np.cov(particles[:, :2].T)
        vals, vecs = np.linalg.eigh(cov)
        ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        w, h = 2 * np.sqrt(5.991 * vals[::-1])
        ax.add_patch(Ellipse(tuple(mu), w, h, angle=ang, fill=False, lw=1, color="k"))


class UMBmarkExperiment(Experiment):
    slug, title = "06_umbmark", "UMBmark (Borenstein & Feng 1996): 2 m square, 5 runs CW + 5 runs CCW"

    RUNS = 5
    DT = 0.02

    def run(self) -> tuple[Figure, Metrics]:
        cases = (
            ("calibrated", DriveCalibrationError(1.0, 1.0)),
            ("miscalibrated (E_d = 1.005, E_b = 1.03)", DriveCalibrationError(1.005, 1.03)),
        )
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        metrics: Metrics = {}
        for idx, (name, err) in enumerate(cases):
            runs: list[UMBmarkRun] = []
            for d in Direction:
                for k in range(self.RUNS):
                    true_xy, odom_xy, end = self._drive(err, d, seed=100 * idx + 10 * k + (d == Direction.CW))
                    runs.append(UMBmarkRun(d, end.x, end.y))
                    if k == 0 and idx == 1:
                        ax = axes[0]
                        ax.plot(*odom_xy.T, "--", c="tab:gray", lw=1, label="odometry belief" if d == Direction.CW else None)
                        ax.plot(*true_xy.T, c="tab:red" if d == Direction.CW else "tab:blue", label=f"true path ({d.value})")
            rep = UMBmarkReport.from_runs(runs)
            metrics[name] = {"E_max_syst_m": rep.e_max_syst, "r_cw_m": rep.r_cw, "r_ccw_m": rep.r_ccw}
            ax = axes[1 + idx]
            for d, c in ((Direction.CW, "tab:red"), (Direction.CCW, "tab:blue")):
                pts = np.array([(r.error_x, r.error_y) for r in runs if r.direction == d])
                ax.scatter(pts[:, 0] * 100, pts[:, 1] * 100, c=c, label=d.value)
                ax.scatter(pts[:, 0].mean() * 100, pts[:, 1].mean() * 100, c=c, marker="X", s=150, edgecolors="k")
            ax.axhline(0, c="k", lw=0.5)
            ax.axvline(0, c="k", lw=0.5)
            ax.set_title(f"{name}\nE_max,syst = {rep.e_max_syst * 100:.1f} cm")
            ax.set_xlabel("return error x [cm]")
            ax.set_ylabel("return error y [cm]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend()
        MapPlotter.draw(axes[0], self.ctx.grid, alpha=0.3)
        axes[0].set_title("miscalibrated robot: belief vs. truth")
        axes[0].legend(fontsize=8)
        fig.suptitle(self.title)
        return fig, metrics

    def _drive(self, err: DriveCalibrationError, d: Direction, seed: int) -> tuple[np.ndarray, np.ndarray, Pose2D]:
        robot = KinematicRobot(DifferentialDrive(self.ctx.robot.drive, err), SlipNoise(0.02), self.ctx.rng(seed))
        ctl = SquarePathController(SquarePathConfig(side=2.0, direction=d))
        true_xy, odom_xy = [], []
        while not ctl.done:
            robot.step(*ctl.command(robot.odom_pose), self.DT)
            true_xy.append((robot.true_pose.x, robot.true_pose.y))
            odom_xy.append((robot.odom_pose.x, robot.odom_pose.y))
        return np.array(true_xy), np.array(odom_xy), robot.true_pose


class ConsistencyExperiment(Experiment):
    slug, title = "07_motion_model_consistency", "Is the motion model's uncertainty honest? (miscalibrated robot, NEES, 95 % coverage)"

    DT, MODEL_RATE_DIV = 0.02, 5  # integrate at 50 Hz, propagate particles at 10 Hz

    def run(self) -> tuple[Figure, Metrics]:
        robot = KinematicRobot(DifferentialDrive(self.ctx.robot.drive, DriveCalibrationError(1.005, 1.03)),
                               SlipNoise(0.02), self.ctx.rng(7))
        ctl = SquarePathController(SquarePathConfig(side=2.0, direction=Direction.CCW))
        gt, od, stamps = [], [], []
        t = 0.0
        while not ctl.done:
            robot.step(*ctl.command(robot.odom_pose), self.DT)
            t += self.DT
            stamps.append(t)
            gt.append(robot.true_pose)
            od.append(robot.odom_pose)
        gt_t = Trajectory.from_lists(stamps, [(p.x, p.y, p.theta) for p in gt])
        od_t = Trajectory.from_lists(stamps, [(p.x, p.y, p.theta) for p in od])
        traj = TrajectoryEvaluator().evaluate(gt_t, od_t)

        alphas = (0.0005, 0.002, 0.01, 0.05, 0.2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        metrics: Metrics = {"dead_reckoning": traj.as_dict()}
        for a in alphas:
            tracker = self._track(MotionNoise(a, a, a, a), gt, od)
            label = f"α = {a}" + ("  (Nav2 default)" if a == 0.2 else "")
            axes[0].semilogy(np.arange(len(tracker.nees)) * self.DT * self.MODEL_RATE_DIV, tracker.nees, label=label, lw=1)
            metrics[label] = {"mean_nees": tracker.mean_nees, "coverage_95": tracker.coverage_95}
        axes[0].axhline(2.0, c="k", ls="--", label="ideal E[NEES] = 2")
        axes[0].set_xlabel("time [s]")
        axes[0].set_ylabel("position NEES")
        axes[0].legend(fontsize=8)
        labels = list(metrics)[1:]
        axes[1].bar(labels, [100 * metrics[k]["coverage_95"] for k in labels], color="tab:blue")
        axes[1].axhline(95, c="k", ls="--")
        axes[1].set_ylabel("% of time truth inside 95 % ellipse")
        axes[1].tick_params(axis="x", labelsize=8)
        fig.suptitle(f"{self.title}\ndead-reckoning ATE {traj.ate_rmse * 100:.1f} cm, drift {traj.drift_percent:.2f} %")
        return fig, metrics

    def _track(self, noise: MotionNoise, gt: list[Pose2D], od: list[Pose2D]) -> ConsistencyTracker:
        model = OdometryMotionModel(noise, self.ctx.rng(8))
        particles = np.zeros((500, 3))
        tracker = ConsistencyTracker()
        prev = od[0]
        for k in range(self.MODEL_RATE_DIV, len(od), self.MODEL_RATE_DIV):
            particles = model.sample(particles, OdometryControl(prev, od[k]))
            prev = od[k]
            tracker.update(particles, (gt[k].x, gt[k].y))
        return tracker


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #
class OfflineBenchmark:
    EXPERIMENTS: tuple[type[Experiment], ...] = (
        ScanGenerationExperiment,
        BeamPdfExperiment,
        LikelihoodSurfaceExperiment,
        LidarSpecExperiment,
        MotionBananaExperiment,
        UMBmarkExperiment,
        ConsistencyExperiment,
        FourModelExperiment,
    )

    def __init__(self, ctx: BenchmarkContext, out_dir: Path) -> None:
        self.ctx = ctx
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Metrics:
        results: Metrics = {}
        for cls in self.EXPERIMENTS:
            exp = cls(self.ctx)
            fig, metrics = exp.run()
            fig.tight_layout()
            fig.savefig(self.out_dir / f"{exp.slug}.png", dpi=130)
            plt.close(fig)
            results[exp.slug] = metrics
            print(f"[ok] {exp.slug}")
        (self.out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results


class ConfigLocator:
    """Finds mobile_bot_description/config via ament (installed) or the source tree."""

    @staticmethod
    def find(explicit: str | None) -> Path:
        if explicit:
            return Path(explicit)
        try:
            from ament_index_python.packages import get_package_share_directory

            return Path(get_package_share_directory("mobile_bot_description")) / "config"
        except Exception:  # noqa: BLE001 - ROS not sourced: fall back to the source tree
            return Path(__file__).resolve().parents[2] / "mobile_bot_description" / "config"


class OfflineBenchmarkApp:
    """Command-line entry point (`offline_benchmark`)."""

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--out", default="w1_expected_output", help="output directory for figures + metrics.json")
        ap.add_argument("--config-dir", default=None, help="folder containing world.yaml and robot.yaml")
        ap.add_argument("--seed", type=int, default=42)
        args = ap.parse_args()
        ctx = BenchmarkContext.load(ConfigLocator.find(args.config_dir), args.seed)
        OfflineBenchmark(ctx, Path(args.out).expanduser()).run()


if __name__ == "__main__":
    OfflineBenchmarkApp.main()
