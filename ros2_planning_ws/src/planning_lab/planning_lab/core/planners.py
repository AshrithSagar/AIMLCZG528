"""
Grid planners behind one interface.

  AStarPlanner      optimal on the 8-connected cost grid — the *reference* (benchmark) for path cost
  DStarLitePlanner  Koenig & Likhachev (2002): same optimal paths, but *repairs* the previous search
                    when cells change, instead of planning from scratch -> cheap replanning

GA lives in `genetic.py` (continuous waypoints, not grid cells).
"""

from __future__ import annotations

import heapq
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from sensor_motion_lab.core.geometry import FloatArray

from .grid import Cell, PlanningGrid

INF = math.inf


@dataclass(slots=True)
class PlanResult:
    success: bool
    path: FloatArray  # (N, 2) world coordinates, start -> goal
    cost: float = INF  # grid cost (A*/D* Lite) or fitness (GA)
    expanded: int = 0  # nodes expanded / fitness evaluations
    ms: float = 0.0
    info: dict[str, float] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))) if len(self.path) > 1 else 0.0


class Planner(ABC):
    name = "planner"

    def __init__(self, grid: PlanningGrid) -> None:
        self.grid = grid
        # Footprint clearing (as Nav2 does): near the robot, cells that are lethal only because of a
        # *moving* obstacle's inflation stay passable (at a high cost) — otherwise a person walking
        # close by would make every plan "start in collision". Static walls are never cleared.
        self.escape: Cell | None = None
        self.escape_cells: frozenset[Cell] = frozenset()

    @abstractmethod
    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanResult: ...

    def replan(self, start: tuple[float, float], goal: tuple[float, float], changed: set[Cell]) -> PlanResult:
        """After obstacles moved. Default: plan from scratch (A*, GA override with smarter behaviour)."""
        return self.plan(start, goal)

    def shortcut(self, path: FloatArray) -> FloatArray:
        """Any-angle post-processing: drop a waypoint whenever its neighbours see each other."""
        if len(path) < 3:
            return path
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.grid.segment_free(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return np.asarray(out)

    # helpers ------------------------------------------------------------------ #
    ESCAPE_RADIUS = 0.5  # m
    ESCAPE_PENALTY = 10.0  # cost multiplier inside the cleared zone: get out, preferably away from the person

    def _set_escape(self, start: Cell) -> set[Cell]:
        """Update the cleared zone around the robot; returns every cell whose passability changed."""
        g = self.grid
        n = int(math.ceil(self.ESCAPE_RADIUS / g.res))
        new = {(r, c) for r in range(start[0] - n, start[0] + n + 1) for c in range(start[1] - n, start[1] + n + 1)
               if g.in_bounds((r, c)) and math.hypot(r - start[0], c - start[1]) * g.res <= self.ESCAPE_RADIUS
               and not g.static_lethal[r, c]} | {start}
        changed = set(self.escape_cells) ^ new
        self.escape, self.escape_cells = start, frozenset(new)
        return changed | {start}

    def _free(self, c: Cell) -> bool:
        return c in self.escape_cells or self.grid.is_free(c)

    def _cell_cost(self, c: Cell) -> float:
        g = self.grid
        return g.cost[c] * (self.ESCAPE_PENALTY if (g.lethal[c] and c != self.escape) else 1.0)

    def _edge(self, a: Cell, b: Cell, step: float) -> float:
        g = self.grid
        if not (g.in_bounds(a) and g.in_bounds(b) and self._free(a) and self._free(b)):
            return INF
        return step * g.res * 0.5 * (self._cell_cost(a) + self._cell_cost(b))

    def _cells_to_path(self, cells: list[Cell], start: tuple[float, float], goal: tuple[float, float]) -> FloatArray:
        pts = [start] + [self.grid.to_world(c) for c in cells[1:-1]] + [goal]
        return np.asarray(pts, dtype=np.float64)

    def _octile(self, a: Cell, b: Cell) -> float:
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return self.grid.res * ((dr + dc) + (math.sqrt(2) - 2) * min(dr, dc))


class AStarPlanner(Planner):
    name = "astar"

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanResult:
        t0 = time.perf_counter()
        g = self.grid
        s, t = g.to_cell(*start), g.to_cell(*goal)
        if not (g.in_bounds(s) and g.is_free(t)):
            return PlanResult(False, np.zeros((0, 2)), ms=1000 * (time.perf_counter() - t0))
        self._set_escape(s)
        cost = {s: 0.0}
        parent: dict[Cell, Cell] = {}
        heap = [(self._octile(s, t), 0.0, s)]
        closed: set[Cell] = set()
        while heap:
            _f, gc, u = heapq.heappop(heap)
            if u in closed:
                continue
            closed.add(u)
            if u == t:
                break
            for dr, dc, step in PlanningGrid.NEIGHBOURS:
                v = (u[0] + dr, u[1] + dc)
                if v in closed or not g.in_bounds(v):
                    continue
                w = self._edge(u, v, step)
                if w == INF:
                    continue
                if gc + w < cost.get(v, INF):
                    cost[v], parent[v] = gc + w, u
                    heapq.heappush(heap, (gc + w + self._octile(v, t), gc + w, v))
        ms = 1000 * (time.perf_counter() - t0)
        if t not in closed:
            return PlanResult(False, np.zeros((0, 2)), expanded=len(closed), ms=ms)
        cells = [t]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return PlanResult(True, self._cells_to_path(cells, start, goal), cost[t], len(closed), ms)


class DStarLitePlanner(Planner):
    """
    D* Lite (optimised version). Searches *backwards* from the goal, so when the robot moves
    and a few cells change, only the affected part of the search is repaired.
    """

    name = "dstar_lite"

    def __init__(self, grid: PlanningGrid) -> None:
        super().__init__(grid)
        self._goal: Cell | None = None
        self.reset()

    def reset(self) -> None:
        self.gv: dict[Cell, float] = {}
        self.rhs: dict[Cell, float] = {}
        self.heap: list[tuple[tuple[float, float], int, Cell]] = []
        self.inq: dict[Cell, tuple[float, float]] = {}
        self.km = 0.0
        self.start: Cell | None = None
        self.last: Cell | None = None
        self.counter = 0
        self.expanded = 0
        self.escape_cells = frozenset()

    # ----- public API ------------------------------------------------------------ #
    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanResult:
        t0 = time.perf_counter()
        g = self.grid
        s, t = g.to_cell(*start), g.to_cell(*goal)
        if not (g.in_bounds(s) and g.is_free(t)):
            return PlanResult(False, np.zeros((0, 2)), ms=1000 * (time.perf_counter() - t0))
        self.reset()
        self._set_escape(s)
        self._goal, self.start, self.last = t, s, s
        self.rhs[t] = 0.0
        self._push(t)
        return self._solve(start, goal, t0)

    def replan(self, start: tuple[float, float], goal: tuple[float, float], changed: set[Cell]) -> PlanResult:
        t0 = time.perf_counter()
        g = self.grid
        s, t = g.to_cell(*start), g.to_cell(*goal)
        if self._goal != t or self.start is None:
            return self.plan(start, goal)
        if not g.in_bounds(s):
            return PlanResult(False, np.zeros((0, 2)), ms=1000 * (time.perf_counter() - t0))
        changed = set(changed) | self._set_escape(s)  # moving the cleared zone changes edge costs too
        self.start = s
        self.km += self._h(self.last, s)  # type: ignore[arg-type]
        self.last = s
        self.expanded = 0
        affected: set[Cell] = set()
        for c in changed:
            if c is None:
                continue
            affected.add(c)
            for dr, dc, _ in PlanningGrid.NEIGHBOURS:
                n = (c[0] + dr, c[1] + dc)
                if g.in_bounds(n):
                    affected.add(n)
        for u in affected:
            self._update_vertex(u)
        return self._solve(start, goal, t0)

    # ----- internals ---------------------------------------------------------------- #
    def _h(self, a: Cell, b: Cell) -> float:
        return self._octile(a, b)

    KEY_DECIMALS = 9  # keys are compared lexicographically: float noise in the 1st element must not
    #                   override the tie-break in the 2nd (that stops the repair too early)

    def _key(self, u: Cell) -> tuple[float, float]:
        m = min(self.gv.get(u, INF), self.rhs.get(u, INF))
        k1 = m + self._h(self.start, u) + self.km  # type: ignore[arg-type]
        return (round(k1, self.KEY_DECIMALS) if k1 != INF else INF, round(m, self.KEY_DECIMALS) if m != INF else INF)

    def _push(self, u: Cell) -> None:
        k = self._key(u)
        self.inq[u] = k
        self.counter += 1
        heapq.heappush(self.heap, (k, self.counter, u))

    def _top(self) -> tuple[tuple[float, float], Cell | None]:
        while self.heap:
            k, _, u = self.heap[0]
            if self.inq.get(u) == k:
                return k, u
            heapq.heappop(self.heap)  # stale entry (lazy deletion)
        return (INF, INF), None

    def _succ(self, u: Cell):  # noqa: ANN202 - generator of (cell, edge cost)
        for dr, dc, step in PlanningGrid.NEIGHBOURS:
            v = (u[0] + dr, u[1] + dc)
            if self.grid.in_bounds(v):
                yield v, self._edge(u, v, step)

    def _update_vertex(self, u: Cell) -> None:
        if u != self._goal:
            self.rhs[u] = min((c + self.gv.get(v, INF) for v, c in self._succ(u)), default=INF)
        self.inq.pop(u, None)
        if self.gv.get(u, INF) != self.rhs.get(u, INF):
            self._push(u)

    def _compute(self) -> None:
        s = self.start
        while True:
            k_old, u = self._top()
            if u is None:
                return
            if not (k_old < self._key(s) or self.rhs.get(s, INF) != self.gv.get(s, INF)):  # type: ignore[arg-type]
                return
            heapq.heappop(self.heap)
            del self.inq[u]
            self.expanded += 1
            k_new = self._key(u)
            if k_old < k_new:
                self._push(u)
            elif self.gv.get(u, INF) > self.rhs.get(u, INF):
                self.gv[u] = self.rhs[u]
                for v, _ in self._succ(u):  # predecessors == successors on an undirected grid
                    self._update_vertex(v)
            else:
                self.gv[u] = INF
                self._update_vertex(u)
                for v, _ in self._succ(u):
                    self._update_vertex(v)

    def _solve(self, start: tuple[float, float], goal: tuple[float, float], t0: float) -> PlanResult:
        self._compute()
        s = self.start
        cost = self.gv.get(s, INF)  # type: ignore[arg-type]
        if cost == INF:
            return PlanResult(False, np.zeros((0, 2)), expanded=self.expanded, ms=1000 * (time.perf_counter() - t0))
        cells, seen = [s], {s}
        while cells[-1] != self._goal and len(cells) < self.grid.rows * self.grid.cols:
            nxt = min(self._succ(cells[-1]), key=lambda vc: vc[1] + self.gv.get(vc[0], INF))[0]
            if nxt in seen:
                break
            cells.append(nxt)
            seen.add(nxt)
        ms = 1000 * (time.perf_counter() - t0)
        return PlanResult(True, self._cells_to_path(cells, start, goal), cost, self.expanded, ms)  # type: ignore[arg-type]
