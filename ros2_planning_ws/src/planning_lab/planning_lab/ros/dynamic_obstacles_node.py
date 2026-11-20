"""
dynamic_obstacles_node — people walking across the W1 arena.

  pub  /obstacles          geometry_msgs/PoseArray           (what a people tracker would publish)
       /obstacles/markers  visualization_msgs/MarkerArray    (cylinders + predicted positions, for RViz)

W1's kinematic_sim_node subscribes to /obstacles too, so the walkers appear in /scan:
the LiDAR sees them, the static /map does not — exactly the situation a replanner is for.
`speed_scale` slows everyone down (0 freezes them) for a step-by-step demo.
"""

from __future__ import annotations

from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray

from sensor_motion_lab.ros.common import LabNode

from ..core.navigation import Crowd, Walker


class DynamicObstaclesNode(LabNode):
    HEIGHT = 1.7

    def __init__(self) -> None:
        super().__init__("dynamic_obstacles_node")
        scale = self.param("speed_scale", 1.0)
        self.crowd = Crowd(tuple(Walker(w.a, w.b, w.speed * scale, w.phase) for w in Crowd.w1_arena().walkers))
        self.radius = self.param("obstacle_radius", 0.25)
        self.t0 = self._now()
        self.pub = self.create_publisher(PoseArray, "/obstacles", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/obstacles/markers", 10)
        self.create_timer(1.0 / self.param("rate_hz", 10.0), self._publish)
        self.get_logger().info(f"{len(self.crowd.walkers)} people walking (speed x{scale})")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish(self) -> None:
        stamp = self.now_msg()
        msg = PoseArray()
        msg.header.frame_id, msg.header.stamp = self.MAP_FRAME, stamp
        markers = MarkerArray()
        for i, (x, y) in enumerate(self.crowd.positions(self._now() - self.t0)):
            p = Pose()
            p.position.x, p.position.y = float(x), float(y)
            p.orientation.w = 1.0
            msg.poses.append(p)
            markers.markers.append(self._cylinder(i, p, stamp))
        self.pub.publish(msg)
        self.marker_pub.publish(markers)

    def _cylinder(self, i: int, p: Pose, stamp) -> Marker:  # noqa: ANN001 - builtin Time
        m = Marker()
        m.header.frame_id, m.header.stamp = self.MAP_FRAME, stamp
        m.ns, m.id, m.type, m.action = "people", i, Marker.CYLINDER, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = p.position.x, p.position.y, self.HEIGHT / 2
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 2 * self.radius
        m.scale.z = self.HEIGHT
        m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.2, 0.2, 0.8
        return m
