"""
The four measurement models on one scan: the CS-3 warehouse-aisle example.

    map wall at x = 5.05 m, robot at (0, 0, 0°), three beams at -10°, 0°, +10°, sigma = 0.10 m
    Case A  clear aisle:              all three beams return 5.00 m
    Case B  a person blocks B2:       B2 returns 2.00 m, B1 and B3 still see the wall

Every number comes from the same classes the ROS nodes use (BeamRangeFinderModel,
LikelihoodFieldModel, LineExtractor). Only the geometry is analytic here (a single
straight wall), so the results match the slide's hand calculation digit for digit.

Two kinds of score are reported, as on the slides:
  kernel   exp(-e² / 2σ²) per beam, multiplied   (Case A slide: "product of 3 kernels")
  density  the full mixture p(z | x, m) per beam (Case B slide: hit + short + rand terms)
Scores of different models have different meanings; compare each model with its own clear-scan score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .features import LineExtractor, LineExtractorParams, LineFeature
from .geometry import FloatArray
from .sensing import BeamModelParams, BeamRangeFinderModel, LikelihoodFieldModel, LikelihoodFieldParams
from .world import Box, GridRasterizer, WorldSpec


@dataclass(frozen=True, slots=True)
class AisleScenario:
    wall_x: float = 5.05
    angles_deg: tuple[float, ...] = (-10.0, 0.0, 10.0)
    ranges: tuple[float, ...] = (5.0, 5.0, 5.0)
    sigma: float = 0.10
    cell: float = 0.10  # correlation grid
    z_max: float = 10.0
    person_range: float | None = None  # Case B: where the person intercepts B2

    @classmethod
    def case(cls, name: str) -> AisleScenario:
        if name.upper() == "A":
            return cls()
        if name.upper() == "B":
            return cls(ranges=(5.0, 2.0, 5.0), person_range=2.0)
        raise ValueError(f"unknown case '{name}' (A | B)")

    @property
    def label(self) -> str:
        return "B: person blocks the centre beam" if self.person_range else "A: clear aisle"

    @property
    def angles(self) -> FloatArray:
        return np.radians(self.angles_deg)

    @property
    def z(self) -> FloatArray:
        return np.asarray(self.ranges, dtype=np.float64)

    @property
    def endpoints(self) -> FloatArray:
        return np.column_stack((self.z * np.cos(self.angles), self.z * np.sin(self.angles)))

    @property
    def z_star(self) -> FloatArray:
        """Ray-cast ranges to the wall: what the beam model expects."""
        return self.wall_x / np.cos(self.angles)

    @property
    def endpoint_distances(self) -> FloatArray:
        """Distance from each endpoint to the wall: what the likelihood field looks up."""
        return np.abs(self.wall_x - self.endpoints[:, 0])

    def world(self) -> WorldSpec:
        """The scenario as a world.yaml-style spec (for /map and RViz): the wall plus two storage racks."""
        t = 0.1
        return WorldSpec("cs3_aisle", 12.0, 8.0, 0.1, 0.5, (
            Box("aisle_wall", self.wall_x + t / 2, 0.0, t, 4.0, 1.0, (0.3, 0.3, 0.3, 1.0)),
            Box("rack_3", 2.5, 1.6, 3.0, 0.4, 1.0, (0.85, 0.55, 0.2, 1.0)),
            Box("rack_4", 2.5, -1.6, 3.0, 0.4, 1.0, (0.85, 0.55, 0.2, 1.0)),
        ))


@dataclass
class ModelScore:
    model: str
    question: str
    per_beam_kernel: list[float] = field(default_factory=list)
    kernel: float = math.nan
    per_beam_density: list[float] = field(default_factory=list)
    density: float = math.nan
    note: str = ""


class FourModelComparison:
    # Mixture weights used on the CS-3 Case B slide (recovered from its numbers; they reproduce every digit).
    BEAM = BeamModelParams(z_hit=0.85, z_short=0.10, z_max=0.0, z_rand=0.05, sigma_hit=0.10, lambda_short=1.0)
    FIELD = LikelihoodFieldParams(z_hit=0.95, z_rand=0.05, sigma_hit=0.10, max_dist=10.0)

    def __init__(self, scenario: AisleScenario) -> None:
        self.s = scenario
        grid = GridRasterizer(0.05).rasterize(scenario.world())
        self.beam = BeamRangeFinderModel(grid, replace(self.BEAM, sigma_hit=scenario.sigma), scenario.z_max)
        self.field = LikelihoodFieldModel(grid, replace(self.FIELD, sigma_hit=scenario.sigma), scenario.z_max)
        self.extractor = LineExtractor(LineExtractorParams(dist_tol=scenario.sigma, min_points=2, max_lines=1))

    def kernel(self, e: FloatArray) -> FloatArray:
        return np.exp(-np.asarray(e) ** 2 / (2 * self.s.sigma**2))

    def all(self) -> list[ModelScore]:
        return [self.beam_model(), self.likelihood_field(), self.correlation(), self.feature()]

    # ----- 1. beam: ray-cast, then compare lengths --------------------------------
    def beam_model(self) -> ModelScore:
        e = self.s.z - self.s.z_star
        k = self.kernel(e)
        dens = self.beam.density(self.s.z, self.s.z_star)
        return ModelScore("beam", "how long should this ray be?", list(k), float(np.prod(k)), list(dens),
                          float(np.prod(dens)), "a short return is explained by p_short (people in front of walls)")

    # ----- 2. likelihood field: project, then measure distance --------------------
    def likelihood_field(self) -> ModelScore:
        d = self.s.endpoint_distances
        k = self.kernel(d)
        dens = self.field.endpoint_density(d)
        return ModelScore("likelihood_field", "how far is this endpoint from the wall?", list(k), float(np.prod(k)),
                          list(dens), float(np.prod(dens)), "no notion of 'short': a blocked beam only hits the z_rand floor")

    # ----- 3. correlation: overlap of scan cells with map cells -------------------
    def correlation(self) -> ModelScore:
        overlap = self.overlap(0.0)
        shifts = {dx: self.overlap(dx) for dx in (-self.s.cell, 0.0, self.s.cell)}
        best = max(shifts, key=shifts.get)
        return ModelScore("correlation", "how many scan cells land on wall cells?", kernel=overlap, density=shifts[best],
                          note=f"overlap {overlap:.1%} at the pose; best {shifts[best]:.1%} at dx = {best:+.2f} m")

    EPS = 1e-9  # 5.1 / 0.1 is 50.999... in floating point; the cell is 51

    def scan_cells(self, dx: float = 0.0) -> FloatArray:
        return np.floor((self.s.endpoints[:, 0] + dx) / self.s.cell + self.EPS).astype(int)

    @property
    def wall_cell(self) -> int:
        return int(math.floor(self.s.wall_x / self.s.cell + self.EPS))

    def overlap(self, dx: float) -> float:
        return float(np.mean(self.scan_cells(dx) == self.wall_cell))

    # ----- 4. feature: fit one line, compare it with the map wall -----------------
    def blind_line(self) -> LineFeature:
        """Least squares through every endpoint: what a fit does if nobody rejects outliers."""
        return LineFeature.fit(self.s.endpoints)

    def fitted_line(self) -> LineFeature:
        """
        Outlier rejection first: among the line hypotheses with the most inliers (they can tie, as in
        Case B), keep the one that associates best with the map wall, then refit it.
        """
        pts = self.s.endpoints
        masks = self.extractor.hypotheses(pts)
        most = max(int(m.sum()) for m in masks)
        lines = [LineFeature.fit(pts[m]) for m in masks if int(m.sum()) == most]
        return min(lines, key=lambda ln: abs(self.s.wall_x - ln.rho))

    def feature(self) -> ModelScore:
        line, blind = self.fitted_line(), self.blind_line()
        d_rho = abs(self.s.wall_x - line.rho)
        k = float(self.kernel(d_rho))
        blind_k = float(self.kernel(self.s.wall_x - blind.rho))
        return ModelScore("feature", "does the fitted wall match the map wall?", [k], k, note=(
            f"rho_hat = {line.rho:.3f} m from {line.support} of {len(self.s.ranges)} endpoints, d_rho = {d_rho:.3f} m"
            f" (blind fit through all: rho = {blind.rho:.3f} m, score {blind_k:.3g})"))

    # ----- report -----------------------------------------------------------------
    def scores(self) -> dict[str, float]:
        """The headline score per model (density for beam / likelihood field, overlap, feature kernel)."""
        return {m.model: (m.density if m.model in ("beam", "likelihood_field") else m.kernel) for m in self.all()}

    def short_lines(self, clear: FourModelComparison | None = None) -> list[str]:
        ref = clear.scores() if clear else {}
        return [f"{k:<17}{v:9.4g}" + (f"  ({100 * v / ref[k]:5.2f} % of clear)" if ref.get(k) else "")
                for k, v in self.scores().items()]

    def lines(self, clear: FourModelComparison | None = None) -> list[str]:
        """One line per model; with `clear`, each score is also compared with its own clear-aisle score."""
        out = []
        for m, ref in zip(self.all(), clear.all() if clear else [None] * 4):
            main = m.density if m.model in ("beam", "likelihood_field") else m.kernel
            text = f"{m.model:<17} {main:9.4g}"
            if ref is not None:
                ref_main = ref.density if m.model in ("beam", "likelihood_field") else ref.kernel
                text += f"   ({100 * main / ref_main:5.2f} % of clear)" if ref_main else ""
            out.append(f"{text}   {m.note}")
        return out

    def table(self, clear: FourModelComparison | None = None) -> str:
        head = [f"Case {self.s.label}: ranges {self.s.ranges} m, sigma {self.s.sigma} m",
                "score = mixture density product (beam, likelihood field), overlap (correlation), kernel (feature)"]
        return "\n".join(head + self.lines(clear))
