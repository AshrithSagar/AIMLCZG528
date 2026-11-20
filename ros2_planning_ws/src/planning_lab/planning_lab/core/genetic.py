"""
Genetic-algorithm path planner (continuous waypoints).

  chromosome   0..K intermediate waypoints (x, y) between start and goal
  fitness      length + λ_col · (length of path inside obstacles) + λ_turn · Σ|turning angle|   (minimise)
  operators    tournament selection, one-point crossover on the waypoint lists,
               mutation = jitter / insert / delete a waypoint, elitism
  dynamic      `replan` keeps the population (warm start): when obstacles move, the
               survivors are re-scored and evolution continues from where it was —
               "choose waypoints dynamically" instead of starting over
  seeded       optional "AI-augmented" start: part of the first population is built from a
               coarse A* path, so evolution refines a good guess instead of searching blindly
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from sensor_motion_lab.core.geometry import FloatArray

from .grid import Cell, PlanningGrid
from .planners import AStarPlanner, Planner, PlanResult

Chromosome = list[tuple[float, float]]


@dataclass(frozen=True, slots=True)
class GaConfig:
    population: int = 60
    generations: int = 120
    replan_generations: int = 15
    max_waypoints: int = 8
    tournament: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.4
    jitter_std: float = 0.3  # m
    elitism: int = 2
    collision_weight: float = 50.0
    turn_weight: float = 0.1
    patience: int = 30  # stop after this many generations without improvement
    max_extra_generations: int = 60  # keep evolving (up to this many more) while the best path still collides
    start_tolerance: float = 0.35  # m around the robot ignored by collision checks (footprint clearing)
    seed_fraction: float = 0.0  # share of the first population seeded from A* (0 = pure GA)
    seed: int = 0


class GeneticPlanner(Planner):
    name = "ga"

    def __init__(self, grid: PlanningGrid, config: GaConfig = GaConfig()) -> None:
        super().__init__(grid)
        self.cfg = config
        if config.seed_fraction > 0:
            self.name = "ga_seeded"
        self.rng = np.random.default_rng(config.seed)
        self.population: list[Chromosome] = []
        self.history: list[float] = []  # best fitness per generation (for plots)
        self.evaluations = 0

    # ----- public API ------------------------------------------------------------ #
    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanResult:
        t0 = time.perf_counter()
        self.history, self.evaluations = [], 0
        self.population = self._initial(start, goal)
        best = self._evolve_until_free(start, goal, self.cfg.generations)
        return self._result(best, start, goal, t0)

    def replan(self, start: tuple[float, float], goal: tuple[float, float], changed: set[Cell]) -> PlanResult:
        """Warm start: keep the evolved population, re-score it on the new map, evolve a little more."""
        if not self.population:
            return self.plan(start, goal)
        t0 = time.perf_counter()
        self.evaluations = 0
        # drop waypoints the robot has already passed (closer to the goal than... no: behind the robot)
        self.population = [self._trim(c, start, goal) for c in self.population]
        n_fresh = self.cfg.population // 4  # some new blood in case the old routes are all blocked
        self.population[-n_fresh:] = [self._random(start, goal) for _ in range(n_fresh)]
        best = self._evolve_until_free(start, goal, self.cfg.replan_generations)
        return self._result(best, start, goal, t0)

    def _evolve_until_free(self, start: tuple[float, float], goal: tuple[float, float], generations: int) -> Chromosome:
        best = self._evolve(start, goal, generations)
        extra = 0
        while not self.collision_free(best, start, goal) and extra < self.cfg.max_extra_generations:
            best = self._evolve(start, goal, 10)
            extra += 10
        return best

    # ----- evolution ---------------------------------------------------------------- #
    def _evolve(self, start: tuple[float, float], goal: tuple[float, float], generations: int) -> Chromosome:
        c = self.cfg
        fit = np.array([self.fitness(ch, start, goal) for ch in self.population])
        best_f, stale = float(fit.min()), 0
        for _ in range(generations):
            order = np.argsort(fit)
            nxt: list[Chromosome] = [list(self.population[i]) for i in order[: c.elitism]]
            while len(nxt) < c.population:
                a, b = self._select(fit), self._select(fit)
                child = self._crossover(a, b) if self.rng.random() < c.crossover_rate else list(a)
                if self.rng.random() < c.mutation_rate:
                    child = self._mutate(child, start, goal)
                nxt.append(child)
            self.population = nxt
            fit = np.array([self.fitness(ch, start, goal) for ch in self.population])
            self.history.append(float(fit.min()))
            if fit.min() < best_f - 1e-6:
                best_f, stale = float(fit.min()), 0
            else:
                stale += 1
                if stale >= c.patience:
                    break
        self._fitness_cache = fit
        return self.population[int(np.argmin(fit))]

    def _select(self, fit: FloatArray) -> Chromosome:
        idx = self.rng.integers(0, len(fit), self.cfg.tournament)
        return self.population[int(idx[np.argmin(fit[idx])])]

    def _crossover(self, a: Chromosome, b: Chromosome) -> Chromosome:
        i = int(self.rng.integers(0, len(a) + 1))
        j = int(self.rng.integers(0, len(b) + 1))
        child = a[:i] + b[j:]
        return child[: self.cfg.max_waypoints]

    def _mutate(self, ch: Chromosome, start: tuple[float, float], goal: tuple[float, float]) -> Chromosome:
        ch = list(ch)
        op = self.rng.random()
        if (op < 0.5 and ch) or len(ch) >= self.cfg.max_waypoints:  # jitter one waypoint
            k = int(self.rng.integers(len(ch))) if ch else 0
            if ch:
                x, y = ch[k]
                ch[k] = (x + self.rng.normal(0, self.cfg.jitter_std), y + self.rng.normal(0, self.cfg.jitter_std))
        elif op < 0.8:  # insert a random free waypoint
            k = int(self.rng.integers(len(ch) + 1))
            ch.insert(k, self.grid.random_free_xy(self.rng, 0.3))
        elif ch:  # delete one
            ch.pop(int(self.rng.integers(len(ch))))
        return ch

    # ----- fitness -------------------------------------------------------------------- #
    def fitness(self, ch: Chromosome, start: tuple[float, float], goal: tuple[float, float]) -> float:
        self.evaluations += 1
        pts = np.asarray([start, *ch, goal], dtype=np.float64)
        seg = np.diff(pts, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        length = float(lengths.sum())
        # sample the polyline every half cell and measure how much of it is in collision
        step = self.grid.res / 2
        samples = [pts[i] + np.linspace(0, 1, max(2, int(L / step) + 1))[:, None] * seg[i]
                   for i, L in enumerate(lengths)]
        allpts = np.vstack(samples)
        free = self._free_mask(allpts, start)
        blocked = 1.0 - float(free.mean())
        # same clearance-weighted cost as A*: length x mean traversal multiplier on the free part
        clearance_cost = float(self.grid.cost_xy(allpts[free]).mean()) if free.any() else 1.0
        heading = np.arctan2(seg[:, 1], seg[:, 0])
        turns = np.abs(np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))).sum() if len(seg) > 1 else 0.0
        return (length * clearance_cost + self.cfg.collision_weight * blocked * length
                + self.cfg.turn_weight * float(turns))

    def collision_free(self, ch: Chromosome, start: tuple[float, float], goal: tuple[float, float]) -> bool:
        pts = np.asarray([start, *ch, goal], dtype=np.float64)
        for i in range(len(pts) - 1):
            n = max(2, int(np.linalg.norm(pts[i + 1] - pts[i]) / (self.grid.res / 2)) + 1)
            seg = pts[i] + np.linspace(0, 1, n)[:, None] * (pts[i + 1] - pts[i])
            if not self._free_mask(seg, start).all():
                return False
        return True

    def _free_mask(self, pts: FloatArray, start: tuple[float, float]) -> np.ndarray:
        near = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1]) < self.cfg.start_tolerance
        return self.grid.is_free_xy(pts) | near

    # ----- initialisation ---------------------------------------------------------------- #
    def _initial(self, start: tuple[float, float], goal: tuple[float, float]) -> list[Chromosome]:
        n_seed = int(round(self.cfg.seed_fraction * self.cfg.population))
        seeds = self._seeds(start, goal, n_seed) if n_seed else []
        return seeds + [self._random(start, goal) for _ in range(self.cfg.population - len(seeds))]

    def _random(self, start: tuple[float, float], goal: tuple[float, float]) -> Chromosome:
        k = int(self.rng.integers(1, self.cfg.max_waypoints // 2 + 1))
        pts = [self.grid.random_free_xy(self.rng, 0.3) for _ in range(k)]
        s, g = np.asarray(start), np.asarray(goal)
        d = g - s
        pts.sort(key=lambda p: float(np.dot(np.asarray(p) - s, d)))  # roughly ordered start -> goal
        return pts

    def _seeds(self, start: tuple[float, float], goal: tuple[float, float], n: int) -> list[Chromosome]:
        res = AStarPlanner(self.grid).plan(start, goal)
        if not res.success:
            return []
        path = res.path
        idx = np.linspace(0, len(path) - 1, min(self.cfg.max_waypoints, max(len(path) - 2, 1)) + 2).round().astype(int)[1:-1]
        base = [tuple(map(float, path[i])) for i in idx]
        out = [list(base)]
        while len(out) < n:
            out.append([(x + self.rng.normal(0, 0.15), y + self.rng.normal(0, 0.15)) for x, y in base])
        return out

    def _trim(self, ch: Chromosome, start: tuple[float, float], goal: tuple[float, float]) -> Chromosome:
        """Remove waypoints that are now behind the robot (further from the goal than the robot is)."""
        dg = math.dist(start, goal)
        return [p for p in ch if math.dist(p, goal) < dg]

    # ----- result ---------------------------------------------------------------------- #
    def _result(self, best: Chromosome, start: tuple[float, float], goal: tuple[float, float], t0: float) -> PlanResult:
        raw = np.asarray([start, *best, goal], dtype=np.float64)
        path = self.shortcut(raw)
        if not self.collision_free([tuple(p) for p in path[1:-1]], start, goal):
            path = raw  # shortcutting is conservative near the robot; fall back to the evolved route
        ok = self.collision_free([tuple(p) for p in path[1:-1]], start, goal)
        return PlanResult(ok, path, self.fitness(best, start, goal), self.evaluations,
                          1000 * (time.perf_counter() - t0), {"generations": float(len(self.history))})
