"""
Offline path-planning benchmark (no ROS) — the W4 "expected output".

    ros2 run planning_lab planning_benchmark --out ~/w4_expected

Part 1  static map, 30 random queries:
        A* (optimal on the grid), A* + shortcut (the *reference*), D* Lite, GA, GA seeded from A*
        -> success, length vs reference, clearance, smoothness, planning time
Part 2  moving obstacles (3 people), 4 start/goal pairs, W1's robot + pure pursuit:
        plan once vs replan with A* / D* Lite / GA (warm start), all with obstacle prediction
        -> success, robot-caused collisions, time to goal, replanning cost
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sensor_motion_lab.core import GridRasterizer, Pose2D, WorldSpec  # noqa: E402
from sensor_motion_lab.core.config import RobotConfig  # noqa: E402

from .core.genetic import GaConfig, GeneticPlanner  # noqa: E402
from .core.grid import PlanningGrid  # noqa: E402
from .core.metrics import PathQuality  # noqa: E402
from .core.navigation import Crowd, NavConfig, NavigationSimulator, NavLog  # noqa: E402
from .core.planners import AStarPlanner, DStarLitePlanner, Planner, PlanResult  # noqa: E402
from .core.w1 import W1Assets  # noqa: E402

PlannerFactory = Callable[[PlanningGrid], Planner]

PLANNERS: dict[str, PlannerFactory] = {
    "astar": AStarPlanner,
    "dstar_lite": DStarLitePlanner,
    "ga": lambda g: GeneticPlanner(g, GaConfig(seed=0)),
    "ga_seeded": lambda g: GeneticPlanner(g, GaConfig(seed=0, seed_fraction=0.25)),
}
DYNAMIC_PAIRS = (((-2.4, 0.0), (2.2, 1.8)), ((2.2, -2.2), (-2.3, 0.8)),
                 ((-0.3, 2.4), (1.4, -2.3)), ((2.3, 2.3), (-0.2, -2.4)))


class StaticSuite:
    def __init__(self, grid_factory: Callable[[], PlanningGrid], queries: int, seed: int) -> None:
        self.grid_factory, self.queries, self.seed = grid_factory, queries, seed

    def run(self) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        grid = self.grid_factory()
        rng = np.random.default_rng(self.seed)
        pairs = []
        while len(pairs) < self.queries:
            s, t = grid.random_free_xy(rng), grid.random_free_xy(rng)
            if math.dist(s, t) > 3.0:
                pairs.append((s, t))
        rows: dict[str, list[dict[str, float]]] = {}
        # illustrate with the query that needs the biggest detour (a straight-line query shows nothing)
        detour = [AStarPlanner(grid).plan(s, t).length / max(math.dist(s, t), 1e-9) for s, t in pairs]
        k_example = int(np.argmax(detour))
        example: dict[str, Any] = {"pair": pairs[k_example], "paths": {}, "ga_history": []}
        for k, (s, t) in enumerate(pairs):
            astar = AStarPlanner(grid)
            ref = astar.plan(s, t)
            ref_path = astar.shortcut(ref.path) if ref.success else ref.path
            ref_len = PathQuality.of(ref_path, grid).length if ref.success else math.nan
            for name, factory in PLANNERS.items():
                p = factory(grid)
                res = p.plan(s, t)
                self._record(rows, name, res, grid, ref_len)
                if k == k_example:
                    example["paths"][name] = res.path.tolist()
                    if isinstance(p, GeneticPlanner):
                        example["ga_history"].append((name, p.history))
            res_sc = PlanResult(ref.success, ref_path, ref.cost, ref.expanded, ref.ms)
            self._record(rows, "astar+shortcut", res_sc, grid, ref_len)
            if k == k_example:
                example["paths"]["astar+shortcut"] = ref_path.tolist()
        return {n: self._aggregate(r) for n, r in rows.items()}, example

    @staticmethod
    def _record(rows: dict[str, list[dict[str, float]]], name: str, res: PlanResult, grid: PlanningGrid,
                ref_len: float) -> None:
        q = PathQuality.of(res.path, grid) if res.success else None
        rows.setdefault(name, []).append({
            "success": float(res.success), "ms": res.ms, "work": float(res.expanded),
            "length_ratio": q.length / ref_len if q else math.nan,
            "min_clearance_m": q.min_clearance if q else math.nan,
            "turning_deg": q.turning_deg if q else math.nan,
        })

    @staticmethod
    def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        return {"success_rate": float(a["success"].mean()), "ms_mean": float(a["ms"].mean()),
                "work_mean": float(a["work"].mean()),
                "length_ratio_median": float(np.nanmedian(a["length_ratio"])),
                "length_ratio_max": float(np.nanmax(a["length_ratio"])),
                "min_clearance_m_median": float(np.nanmedian(a["min_clearance_m"])),
                "turning_deg_median": float(np.nanmedian(a["turning_deg"]))}


class DynamicSuite:
    STRATEGIES: dict[str, tuple[PlannerFactory, bool]] = {
        "static_plan": (AStarPlanner, False),
        "astar": (AStarPlanner, True),
        "dstar_lite": (DStarLitePlanner, True),
        "ga_seeded": (PLANNERS["ga_seeded"], True),
    }

    def __init__(self, grid_factory: Callable[[], PlanningGrid], robot: RobotConfig) -> None:
        self.grid_factory, self.robot = grid_factory, robot

    def run(self) -> tuple[dict[str, dict[str, float]], dict[str, NavLog]]:
        results, example = {}, {}
        for name, (factory, replan) in self.STRATEGIES.items():
            logs = []
            for k, (s, g) in enumerate(DYNAMIC_PAIRS):
                grid = self.grid_factory()
                sim = NavigationSimulator(grid, self.robot, Crowd.w1_arena(), NavConfig(replan=replan))
                logs.append(sim.run(factory(grid), Pose2D(s[0], s[1], math.atan2(g[1] - s[1], g[0] - s[0])), g))
                if k == 0:
                    example[name] = logs[-1]
            ms = [m for lg in logs for m in lg.replan_ms]
            results[name] = {
                "success_rate": float(np.mean([lg.report.success for lg in logs])),
                "collisions": float(sum(lg.report.collisions for lg in logs)),
                "walked_into_while_stopped": float(sum(lg.report.contacts_while_stopped for lg in logs)),
                "time_to_goal_s": float(np.mean([lg.report.time_s for lg in logs])),
                "min_gap_m": float(min(lg.report.min_clearance_m for lg in logs)),
                "replans": float(sum(len(lg.replan_ms) for lg in logs)),
                "ms_per_replan": float(np.mean(ms)) if ms else 0.0,
                "ms_per_replan_max": float(np.max(ms)) if ms else 0.0,
            }
            print(f"  {name}: {results[name]}", flush=True)
        return results, example


class Figures:
    COLORS = {"astar": "tab:blue", "astar+shortcut": "tab:cyan", "dstar_lite": "tab:purple", "ga": "tab:orange",
              "ga_seeded": "tab:red", "static_plan": "tab:gray"}

    def __init__(self, out: Path, grid: PlanningGrid) -> None:
        self.out, self.grid = out, grid
        g = grid
        self.extent = (g.x0, g.x0 + g.cols * g.res, g.y0, g.y0 + g.rows * g.res)

    def _map(self, ax: plt.Axes) -> None:
        # light grey = inflated (lethal for the robot centre), dark = the obstacles themselves
        shade = np.where(self.grid.static_dist < 0.05, 1.0, np.where(self.grid.static_lethal, 0.35, 0.0))
        ax.imshow(shade, origin="lower", extent=self.extent, cmap="Greys", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def static(self, example: dict[str, Any]) -> None:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self._map(ax)
        for name, path in example["paths"].items():
            p = np.asarray(path)
            if len(p):
                ax.plot(p[:, 0], p[:, 1], "--" if name == "astar" else "-", lw=2, color=self.COLORS[name], label=name)
        s, t = example["pair"]
        ax.plot(*s, "go", ms=10)
        ax.plot(*t, "r*", ms=14)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)
        ax.set_title("Static map: one query, every planner (grey = inflated obstacles)")
        for name, hist in example["ga_history"]:
            ax2.plot(hist, color=self.COLORS[name], label=name)
        ax2.set_xlabel("generation")
        ax2.set_ylabel("best fitness (cost-weighted length + penalties)")
        ax2.set_title("GA convergence: seeding from A* starts closer to the optimum")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(self.out / "static_paths.png", dpi=120)
        plt.close(fig)

    def dynamic(self, example: dict[str, NavLog]) -> None:
        fig, axes = plt.subplots(1, len(example), figsize=(5 * len(example), 5.2))
        for ax, (name, log) in zip(axes, example.items()):
            self._map(ax)
            tracks = np.asarray(log.obstacle_tracks)
            if len(tracks):
                for j in range(tracks.shape[1]):
                    ax.plot(tracks[:, j, 0], tracks[:, j, 1], "-", color="tab:green", alpha=0.25, lw=4)
            tr = np.asarray(log.trajectory)
            if len(tr):
                ax.plot(tr[:, 0], tr[:, 1], "-", color=self.COLORS[name], lw=2)
            s, g = DYNAMIC_PAIRS[0]
            ax.plot(*s, "go", ms=9)
            ax.plot(*g, "r*", ms=13)
            ax.set_title(f"{name}\n{'reached' if log.report.success else 'FAILED'} in {log.report.time_s:.0f} s, "
                         f"{log.report.collisions} collision(s), {len(log.replan_ms)} replans", fontsize=10)
        fig.suptitle("Moving obstacles (green = people's walking lanes): plan once vs replan", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(self.out / "dynamic.png", dpi=110)
        plt.close(fig)


class Summary:
    @staticmethod
    def render(p: dict[str, Any]) -> str:
        f = lambda v, d=2: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"  # noqa: E731
        lines = ["# W4 path-planning benchmark", "",
                 f"Machine: {p['system']['cpu_count']} CPUs ({p['system']['platform']})", "",
                 f"## Static map ({p['queries']} random queries; reference = A* + shortcut)", "",
                 "| planner | success | length / reference (median, max) | min clearance m (median) | turning deg (median) | ms (mean) | work (nodes / evaluations) |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for n, m in p["static"].items():
            lines.append(f"| {n} | {m['success_rate'] * 100:.0f} % | {f(m['length_ratio_median'], 3)}, {f(m['length_ratio_max'], 3)} | "
                         f"{f(m['min_clearance_m_median'])} | {f(m['turning_deg_median'], 0)} | {f(m['ms_mean'], 1)} | {f(m['work_mean'], 0)} |")
        lines += ["", "## Moving obstacles (3 people, 4 start/goal pairs, constant-velocity prediction 1.5 s)", "",
                  "| strategy | success | collisions (robot moving) | walked into while stopped | time to goal s | replans | ms / replan (mean, max) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for n, m in p["dynamic"].items():
            lines.append(f"| {n} | {m['success_rate'] * 100:.0f} % | {m['collisions']:.0f} | {m['walked_into_while_stopped']:.0f} | "
                         f"{f(m['time_to_goal_s'], 1)} | {m['replans']:.0f} | {f(m['ms_per_replan'], 1)}, {f(m['ms_per_replan_max'], 0)} |")
        return "\n".join(lines) + "\n"


class BenchmarkApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> dict[str, Any]:
        d = Path(self.args.config_dir) if self.args.config_dir else W1Assets.config_dir()
        world = GridRasterizer(0.05).rasterize(WorldSpec.from_yaml(d / "world.yaml"))
        robot = RobotConfig.from_yaml(d / "robot.yaml")
        out = Path(self.args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        print("[static] planning ...", flush=True)
        static, example = StaticSuite(lambda: PlanningGrid(world), self.args.queries, self.args.seed).run()
        print("[dynamic] driving among moving obstacles ...", flush=True)
        dynamic, dyn_example = DynamicSuite(lambda: PlanningGrid(world), robot).run()
        payload = {"system": {"cpu_count": os.cpu_count(), "platform": platform.platform()},
                   "queries": self.args.queries, "static": static, "dynamic": dynamic}
        (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        figs = Figures(out, PlanningGrid(world))
        figs.static(example)
        figs.dynamic(dyn_example)
        (out / "summary.md").write_text(Summary.render(payload), encoding="utf-8")
        print((out / "summary.md").read_text())
        return payload

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--out", default="w4_planning_benchmark")
        ap.add_argument("--config-dir", default=None, help="W1 mobile_bot_description/config")
        ap.add_argument("--queries", type=int, default=30)
        ap.add_argument("--seed", type=int, default=0)
        BenchmarkApp(ap.parse_args()).run()


if __name__ == "__main__":
    BenchmarkApp.main()
