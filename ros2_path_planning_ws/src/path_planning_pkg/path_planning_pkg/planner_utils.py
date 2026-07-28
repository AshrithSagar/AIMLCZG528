"""
Shared utilities for the Webinar 4 path-planning demo (AIMLCZG528).

Holds:
- OccupancyGridMap: thin wrapper around nav_msgs/OccupancyGrid with
  world<->grid coordinate transforms and collision checking.
- make_demo_map(): builds a static maze-like map so RRT / PRM / A*
  visibly behave differently (dead ends, narrow corridor, open area).
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

OCC_THRESH = 50  # cell value >= this is considered an obstacle


class OccupancyGridMap:
    """Wraps a nav_msgs/OccupancyGrid for planner-side queries."""

    def __init__(self, grid_msg: OccupancyGrid):
        self.info = grid_msg.info
        self.resolution = grid_msg.info.resolution
        self.width = grid_msg.info.width
        self.height = grid_msg.info.height
        self.origin_x = grid_msg.info.origin.position.x
        self.origin_y = grid_msg.info.origin.position.y
        self.data = np.array(grid_msg.data, dtype=np.int16).reshape(
            (self.height, self.width)
        )

    def world_to_grid(self, x: float, y: float):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int):
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_free_cell(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        return self.data[gy, gx] < OCC_THRESH

    def is_free_world(self, x: float, y: float) -> bool:
        return self.is_free_cell(*self.world_to_grid(x, y))

    def line_free(self, x1, y1, x2, y2, step=None) -> bool:
        """Sample a straight line in world coords; True if fully collision-free."""
        if step is None:
            step = self.resolution * 0.5
        dist = np.hypot(x2 - x1, y2 - y1)
        n = max(2, int(dist / step))
        for i in range(n + 1):
            t = i / n
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self.is_free_world(x, y):
                return False
        return True


def make_demo_map() -> OccupancyGrid:
    """
    Builds a fixed 10m x 10m demo map (0.1m resolution) with:
    - outer walls
    - a narrow S-corridor through the middle (good for showing RRT struggle
      vs PRM roadmap vs A* optimality)
    - a couple of freestanding rectangular obstacles in the open area
    """
    resolution = 0.1
    width_m, height_m = 10.0, 10.0
    width = int(width_m / resolution)
    height = int(height_m / resolution)

    grid = np.zeros((height, width), dtype=np.int8)  # 0 = free

    def fill_rect(x0, y0, x1, y1, val=100):
        gx0, gy0 = int(x0 / resolution), int(y0 / resolution)
        gx1, gy1 = int(x1 / resolution), int(y1 / resolution)
        grid[gy0:gy1, gx0:gx1] = val

    # Outer walls (1 cell thick)
    fill_rect(0, 0, width_m, 0.1)
    fill_rect(0, height_m - 0.1, width_m, height_m)
    fill_rect(0, 0, 0.1, height_m)
    fill_rect(width_m - 0.1, 0, width_m, height_m)

    # Dividing wall with a narrow gap -> forces a corridor / bottleneck
    fill_rect(4.5, 0.0, 5.0, 6.0)  # vertical wall, lower part
    fill_rect(4.5, 7.0, 5.0, 10.0)  # vertical wall, upper part
    # -> gap between y=6.0 and y=7.0 is the only crossing point

    # A second wall creating an S-shaped detour on the right side
    fill_rect(7.0, 3.0, 7.5, 10.0)
    fill_rect(6.5, 3.0, 7.5, 3.5)

    # Freestanding obstacles in the open (left) area
    fill_rect(1.5, 1.5, 2.5, 2.5)
    fill_rect(2.0, 5.0, 3.0, 6.5)
    fill_rect(1.0, 7.5, 2.2, 8.5)

    msg = OccupancyGrid()
    msg.info.resolution = resolution
    msg.info.width = width
    msg.info.height = height
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.flatten().tolist()
    return msg


def path_to_ros_points(path_xy):
    """list[(x,y)] -> list[geometry_msgs/Point] for MarkerArray line strips."""
    pts = []
    for x, y in path_xy:
        p = Point()
        p.x, p.y, p.z = float(x), float(y), 0.02
        pts.append(p)
    return pts


class PlannerNodeBase(Node):
    """
    Common scaffolding shared by the A*, RRT, and PRM planner nodes:
      - subscribes to /map (transient-local) to get the OccupancyGridMap
      - subscribes to /initialpose (RViz "2D Pose Estimate") for the start
      - subscribes to /goal_pose (RViz "2D Goal Pose") for the goal, and
        triggers planning as soon as both are available
      - publishes the resulting path on /path (nav_msgs/Path)
      - publishes exploration/tree/roadmap visuals on /planner_markers

    Subclasses implement `plan(start_xy, goal_xy) -> (path_xy, extra_markers)`.
    """

    def __init__(self, node_name: str, marker_color=(0.1, 0.6, 1.0)):
        super().__init__(node_name)
        self.grid_map = None
        self.start_xy = None
        self.goal_xy = None
        self.marker_color = marker_color

        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._on_start, 10
        )
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)

        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/planner_markers", 10)

        self.get_logger().info(
            f"{node_name} ready. Waiting for /map, then use RViz2 "
            '"2D Pose Estimate" for start and "2D Goal Pose" for goal.'
        )

    def _on_map(self, msg: OccupancyGrid):
        self.grid_map = OccupancyGridMap(msg)

    def _on_start(self, msg: PoseWithCovarianceStamped):
        self.start_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.get_logger().info(f"Start set: {self.start_xy}")
        self._maybe_plan()

    def _on_goal(self, msg: PoseStamped):
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"Goal set: {self.goal_xy}")
        self._maybe_plan()

    def _maybe_plan(self):
        if self.grid_map is None:
            self.get_logger().warn("No map received yet.")
            return
        if self.start_xy is None or self.goal_xy is None:
            return  # wait until both are set

        if not self.grid_map.is_free_world(*self.start_xy):
            self.get_logger().warn("Start is in collision, ignoring.")
            return
        if not self.grid_map.is_free_world(*self.goal_xy):
            self.get_logger().warn("Goal is in collision, ignoring.")
            return

        t0 = time.time()
        path_xy, extra_markers = self.plan(self.start_xy, self.goal_xy)
        dt = time.time() - t0

        if path_xy is None:
            self.get_logger().warn(f"No path found (took {dt:.3f}s).")
            self.publish_markers(extra_markers or [])
            return

        self.get_logger().info(
            f"Path found: {len(path_xy)} waypoints, "
            f"length={self._path_length(path_xy):.2f} m, in {dt:.3f}s"
        )
        self.publish_path(path_xy)
        self.publish_markers((extra_markers or []) + [self._path_marker(path_xy)])

    @staticmethod
    def _path_length(path_xy):
        p = np.array(path_xy)
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    def publish_path(self, path_xy):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path_xy:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _path_marker(self, path_xy) -> Marker:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "final_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
        m.points = path_to_ros_points(path_xy)
        return m

    def publish_markers(self, markers):
        arr = MarkerArray()
        arr.markers = markers
        self.marker_pub.publish(arr)

    def make_line_list_marker(self, ns, segments, rgba=None, width=0.01, id_=1):
        """segments: list of ((x1,y1),(x2,y2)) -> LINE_LIST marker (tree/roadmap edges)."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = id_
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = width
        r, g, b = rgba[:3] if rgba else self.marker_color
        a = rgba[3] if rgba and len(rgba) > 3 else 0.6
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        pts = []
        for (x1, y1), (x2, y2) in segments:
            pts.append(Point(x=float(x1), y=float(y1), z=0.01))
            pts.append(Point(x=float(x2), y=float(y2), z=0.01))
        m.points = pts
        return m

    def plan(self, start_xy, goal_xy):
        raise NotImplementedError
