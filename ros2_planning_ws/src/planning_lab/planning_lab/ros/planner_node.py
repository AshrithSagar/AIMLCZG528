"""
planner_node — global planner that replans around moving obstacles.

  planner := astar | dstar_lite | ga | ga_seeded

  sub  /map           nav_msgs/OccupancyGrid       (latched; from W1's world_map_node — or any map server)
       /goal_pose     geometry_msgs/PoseStamped    (RViz "2D Goal Pose")
       /obstacles     geometry_msgs/PoseArray      (moving obstacles; on a robot: a tracker's output)
       TF map -> base_link                          (tf2: wherever the robot *thinks* it is)
  pub  /plan          nav_msgs/Path
       /planner/costmap   nav_msgs/OccupancyGrid  (inflation + moving obstacles + predictions, 0..100)
       /planner/population visualization_msgs/MarkerArray (GA: the current population of routes)
       /diagnostics   planning time, path length / clearance / turning, replans
"""

from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from sensor_motion_lab.ros.common import LabNode, QoS, RosConv

from ..core.genetic import GaConfig, GeneticPlanner
from ..core.grid import Cell, PlanningGrid
from ..core.metrics import PathQuality
from ..core.navigation import ObstaclePredictor
from ..core.planners import AStarPlanner, DStarLitePlanner, Planner, PlanResult
from .conv import PlanningConv


class PlannerFactory:
    @staticmethod
    def create(kind: str, grid: PlanningGrid) -> Planner:
        if kind == "astar":
            return AStarPlanner(grid)
        if kind == "dstar_lite":
            return DStarLitePlanner(grid)
        if kind in ("ga", "ga_seeded"):
            return GeneticPlanner(grid, GaConfig(seed_fraction=0.25 if kind == "ga_seeded" else 0.0))
        raise ValueError(f"unknown planner '{kind}' (astar | dstar_lite | ga | ga_seeded)")


class PlannerNode(LabNode):
    def __init__(self) -> None:
        super().__init__("planner_node")
        self.kind = self.param("planner", "astar")
        self.replan_enabled = self.param("replan", True)
        self.horizon = self.param("horizon_m", 2.0)
        self.min_interval = self.param("min_replan_interval_s", 0.5)
        self.base_frame = self.param("base_frame", self.BASE_FRAME)
        self.goal_tolerance = self.param("goal_tolerance", 0.15)
        self.predictor = ObstaclePredictor(self.param("prediction_s", 1.5), 3)

        self.grid: PlanningGrid | None = None
        self.planner: Planner | None = None
        self.goal: tuple[float, float] | None = None
        self.path = np.zeros((0, 2))
        self.pending: set[Cell] = set()
        self.last_replan = -math.inf
        self.replans = 0
        self.last_result: PlanResult | None = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(OccupancyGrid, "/map", self._on_map, QoS.latched())
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(PoseArray, "/obstacles", self._on_obstacles, 10)
        self.plan_pub = self.create_publisher(Path, "/plan", QoS.latched())
        self.costmap_pub = self.create_publisher(OccupancyGrid, "/planner/costmap", 10)
        self.population_pub = self.create_publisher(MarkerArray, "/planner/population", 10)
        self.create_timer(1.0 / self.param("rate_hz", 10.0), self._tick)
        self.create_timer(self.param("costmap_period", 1.0), self._publish_costmap)
        goal = self.param("goal", [0.0])  # optional [x, y] for unattended demos
        if len(goal) == 2:
            self.goal = (float(goal[0]), float(goal[1]))
        self.get_logger().info(f"Planner '{self.kind}' (replan={self.replan_enabled}); waiting for /map and a goal")

    # ----- inputs ------------------------------------------------------------- #
    def _on_map(self, msg: OccupancyGrid) -> None:
        self.grid = PlanningConv.planning_grid(msg)
        self.planner = PlannerFactory.create(self.kind, self.grid)
        self.get_logger().info(f"Map received ({msg.info.width}x{msg.info.height}); planning grid {self.grid.cols}x{self.grid.rows}")

    def _on_goal(self, msg: PoseStamped) -> None:
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.path = np.zeros((0, 2))  # forces a fresh plan
        self.replans = 0
        self.get_logger().info(f"New goal {self.goal}")

    def _on_obstacles(self, msg: PoseArray) -> None:
        if self.grid is None:
            return
        pts = PlanningConv.points(msg)
        now = self._now()
        self.pending |= self.grid.set_obstacles(self.predictor.predict(pts, now))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _robot_xy(self) -> tuple[float, float] | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.MAP_FRAME, self.base_frame, Time())
        except TransformException:
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    # ----- planning loop ---------------------------------------------------------- #
    def _tick(self) -> None:
        if self.grid is None or self.planner is None or self.goal is None:
            return
        robot = self._robot_xy()
        if robot is None:
            return
        if math.dist(robot, self.goal) < self.goal_tolerance:
            self.get_logger().info(f"Goal {self.goal} reached after {self.replans} replans")
            self.goal = None
            return
        now = self._now()
        if not len(self.path):
            self._plan(robot, fresh=True)
            return
        if not self.replan_enabled or now - self.last_replan < self.min_interval:
            return
        progress = int(np.argmin(np.linalg.norm(self.path - np.asarray(robot), axis=1)))
        if not self.grid.path_free(self.path, progress, self.horizon):
            self._plan(robot, fresh=False)

    def _plan(self, robot: tuple[float, float], fresh: bool) -> None:
        assert self.planner is not None and self.goal is not None and self.grid is not None
        res = self.planner.plan(robot, self.goal) if fresh else self.planner.replan(robot, self.goal, self.pending)
        self.pending = set()
        self.last_replan = self._now()
        self.replans += 0 if fresh else 1
        self.last_result = res
        if not res.success:
            self.get_logger().warn(f"{self.kind}: no path ({res.ms:.0f} ms)")
            return
        self.path = res.path
        stamp = self.now_msg()
        msg = Path()
        msg.header.frame_id, msg.header.stamp = self.MAP_FRAME, stamp
        for x, y in res.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.plan_pub.publish(msg)
        if isinstance(self.planner, GeneticPlanner):
            self.population_pub.publish(self._population_markers(robot, stamp))
        q = PathQuality.of(res.path, self.grid)
        self.diagnostics.publish(
            {"planner": self.kind, "event": "plan" if fresh else "replan", "ms": res.ms, "work": res.expanded,
             "length_m": q.length, "min_clearance_m": q.min_clearance, "turning_deg": q.turning_deg,
             "replans": self.replans},
            message="work = nodes expanded (A*/D* Lite) or fitness evaluations (GA)")

    # ----- visualisation ------------------------------------------------------------- #
    def _publish_costmap(self) -> None:
        if self.grid is None:
            return
        g = self.grid
        values = np.where(g.lethal, 100, np.round((g.cost - 1.0) / max(g.cfg.soft_weight - 1.0, 1e-9) * 60))
        self.costmap_pub.publish(RosConv.occupancy_grid(values, g.res, (g.x0, g.y0), self.MAP_FRAME, self.now_msg()))

    def _population_markers(self, robot: tuple[float, float], stamp) -> MarkerArray:  # noqa: ANN001 - builtin Time
        """The GA's current population as faint orange polylines (robot -> waypoints -> goal)."""
        assert isinstance(self.planner, GeneticPlanner) and self.goal is not None
        m = Marker()
        m.header.frame_id, m.header.stamp = self.MAP_FRAME, stamp
        m.ns, m.id, m.type, m.action = "ga_population", 0, Marker.LINE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.01
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.0, 0.35
        for chromosome in self.planner.population:
            pts = [robot, *map(tuple, chromosome), self.goal]
            for a, b in zip(pts[:-1], pts[1:]):
                m.points += [Point(x=float(a[0]), y=float(a[1]), z=0.02), Point(x=float(b[0]), y=float(b[1]), z=0.02)]
        return MarkerArray(markers=[m])
