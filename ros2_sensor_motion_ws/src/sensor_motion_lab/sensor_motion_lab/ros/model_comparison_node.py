"""
model_comparison_node — the CS-3 "four models, the same three returns" example, in RViz.

  param  case := A | B        A: clear aisle, B: a person blocks the centre beam
                              change it live:  ros2 param set /model_comparison_node case B
  pub    /map                         nav_msgs/OccupancyGrid   aisle wall + two racks (latched)
         /comparison/scan             sensor_msgs/LaserScan    the three returns (frame base_link)
         /comparison/markers          visualization_msgs/MarkerArray
              beam/        expected rays z* (grey) vs measured rays z (red)
              field/       endpoint -> wall distance d (magenta)
              correlation/ scan cells (blue) vs wall cells (grey), 0.10 m grid
              feature/     fitted wall line (green); Case B: the blind least-squares fit (red)
              scores       the score table, floating above the scene
         /diagnostics                 every model's score
  TF     map -> base_link (static, identity: the robot pose of the example)
"""

from __future__ import annotations

import math

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from ..core import Pose2D
from ..core.comparison import AisleScenario, FourModelComparison
from ..core.world import GridRasterizer
from .common import LabNode, QoS, RosConv


class MarkerFactory:
    """Small helpers so the node reads as 'what to draw', not message plumbing."""

    def __init__(self, frame: str, stamp) -> None:  # noqa: ANN001 - builtin_interfaces/Time
        self.frame, self.stamp = frame, stamp

    def _base(self, ns: str, mid: int, kind: int, rgba: tuple[float, float, float, float], scale: float) -> Marker:
        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, self.stamp
        m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def lines(self, ns: str, segments: list[tuple[tuple[float, float], tuple[float, float]]],
              rgba: tuple[float, float, float, float], width: float = 0.03, z: float = 0.05) -> Marker:
        m = self._base(ns, 0, Marker.LINE_LIST, rgba, width)
        for a, b in segments:
            m.points += [Point(x=float(a[0]), y=float(a[1]), z=z), Point(x=float(b[0]), y=float(b[1]), z=z)]
        return m

    def points(self, ns: str, pts: list[tuple[float, float]], rgba: tuple[float, float, float, float],
               size: float = 0.12, kind: int = Marker.SPHERE_LIST, z: float = 0.05) -> Marker:
        m = self._base(ns, 0, kind, rgba, size)
        m.scale.z = 0.02 if kind == Marker.CUBE_LIST else size
        m.points = [Point(x=float(x), y=float(y), z=z) for x, y in pts]
        return m

    def cylinder(self, ns: str, xy: tuple[float, float], radius: float, height: float,
                 rgba: tuple[float, float, float, float]) -> Marker:
        m = self._base(ns, 0, Marker.CYLINDER, rgba, 2 * radius)
        m.scale.z = height
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(xy[0]), float(xy[1]), height / 2
        return m

    def text(self, ns: str, xy: tuple[float, float], text: str, size: float = 0.22) -> Marker:
        m = self._base(ns, 0, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0), size)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(xy[0]), float(xy[1]), 0.5
        m.text = text
        return m

    def delete(self, ns: str) -> Marker:
        m = self._base(ns, 0, Marker.CUBE, (0.0, 0.0, 0.0, 0.0), 0.01)
        m.action = Marker.DELETE
        return m


class ModelComparisonNode(LabNode):
    PERSON_RADIUS = 0.2

    def __init__(self) -> None:
        super().__init__("model_comparison_node")
        self.case = str(self.param("case", "A")).upper()
        self.clear = FourModelComparison(AisleScenario.case("A"))
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", QoS.latched())
        self.scan_pub = self.create_publisher(LaserScan, "/comparison/scan", QoS.latched())
        self.marker_pub = self.create_publisher(MarkerArray, "/comparison/markers", QoS.latched())
        self.static_tf = StaticTransformBroadcaster(self)
        self.add_on_set_parameters_callback(self._on_params)
        self.create_timer(self.param("republish_period", 2.0), self._publish)  # also late-joining RViz
        self._publish(log=True)

    # ----- parameters ----------------------------------------------------------------
    def _on_params(self, params) -> SetParametersResult:  # noqa: ANN001 - list[rclpy.Parameter]
        for p in params:
            if p.name == "case":
                value = str(p.value).upper()
                if value not in ("A", "B"):
                    return SetParametersResult(successful=False, reason="case must be A or B")
                self.case = value
                self._publish(log=True)
        return SetParametersResult(successful=True)

    # ----- what to publish ----------------------------------------------------------
    def _publish(self, log: bool = False) -> None:
        comp = FourModelComparison(AisleScenario.case(self.case))
        s = comp.s
        stamp = self.now_msg()
        grid = GridRasterizer(0.05).rasterize(s.world())
        self.map_pub.publish(RosConv.grid_map(grid, self.MAP_FRAME, stamp))
        self.static_tf.sendTransform(RosConv.transform(Pose2D(), self.MAP_FRAME, self.BASE_FRAME, stamp))
        self.scan_pub.publish(self._scan(s, stamp))
        self.marker_pub.publish(self._markers(comp, stamp))
        self.diagnostics.publish({"case": self.case} | comp.scores(), message=s.label, log=False)
        if log:
            self.get_logger().info("\n" + comp.table(self.clear if self.case == "B" else None))

    def _scan(self, s: AisleScenario, stamp) -> LaserScan:  # noqa: ANN001
        msg = LaserScan()
        msg.header.frame_id, msg.header.stamp = self.BASE_FRAME, stamp
        a = s.angles
        msg.angle_min, msg.angle_max, msg.angle_increment = float(a[0]), float(a[-1]), float(a[1] - a[0])
        msg.range_min, msg.range_max = 0.1, float(s.z_max)
        msg.ranges = [float(r) for r in s.z]
        return msg

    def _markers(self, comp: FourModelComparison, stamp) -> MarkerArray:  # noqa: ANN001
        s, f = comp.s, MarkerFactory(self.MAP_FRAME, stamp)
        origin = (0.0, 0.0)
        ends = [tuple(e) for e in s.endpoints]
        expected = [(float(zs * math.cos(t)), float(zs * math.sin(t))) for zs, t in zip(s.z_star, s.angles)]
        cell = s.cell
        scan_cells = [((c + 0.5) * cell, round(e[1] / cell) * cell) for c, e in zip(comp.scan_cells(), ends)]
        wall_cells = [((comp.wall_cell + 0.5) * cell, k * cell) for k in range(-12, 13)]
        line, blind = comp.fitted_line(), comp.blind_line()
        out = [
            f.lines("beam/expected", [(origin, e) for e in expected], (0.6, 0.6, 0.6, 0.8), 0.02, 0.02),
            f.lines("beam/measured", [(origin, e) for e in ends], (0.9, 0.2, 0.2, 1.0), 0.03, 0.06),
            f.points("field/endpoints", ends, (1.0, 0.9, 0.1, 1.0), 0.12),
            f.lines("field/distance", [(e, (s.wall_x, e[1])) for e in ends], (0.9, 0.2, 0.9, 1.0), 0.03, 0.08),
            f.points("correlation/wall_cells", wall_cells, (0.5, 0.5, 0.5, 0.6), cell * 0.95, Marker.CUBE_LIST, 0.01),
            f.points("correlation/scan_cells", scan_cells, (0.2, 0.5, 1.0, 0.9), cell * 0.95, Marker.CUBE_LIST, 0.03),
            f.lines("feature/fitted", [(line.p0, line.p1)], (0.2, 0.9, 0.3, 1.0), 0.05, 0.1),
            f.text("scores", (2.5, 3.0), "\n".join([f"Case {s.label}"] + comp.short_lines(
                self.clear if self.case == "B" else None))),
        ]
        if s.person_range is not None:
            out += [f.cylinder("person", (s.person_range + self.PERSON_RADIUS, 0.0), self.PERSON_RADIUS, 1.7,
                               (0.9, 0.3, 0.2, 0.8)),
                    f.lines("feature/blind_fit", [(blind.p0, blind.p1)], (1.0, 0.2, 0.2, 0.8), 0.03, 0.12)]
        else:
            out += [f.delete("person"), f.delete("feature/blind_fit")]
        return MarkerArray(markers=out)

