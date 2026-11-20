"""
ROS 2 adapter layer: conversions, QoS presets, diagnostics, base node.

Teaching points that live here:
  * REP-103 — camera *optical* frame (z fwd, x right, y down) vs *body* frame (x fwd, y left, z up)
  * REP-118 — depth images: 32FC1 in metres or 16UC1 in millimetres
  * QoS     — images are best-effort/shallow (drop stale frames), maps are transient-local
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

from ..core.se3 import SE3

T = TypeVar("T")


class Frames:
    """Frame names + the fixed body->optical rotation (REP-103)."""

    MAP = "map"
    CAMERA = "camera_link"
    OPTICAL = "camera_color_optical_frame"
    R_BODY_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    T_BODY_OPTICAL = SE3(R_BODY_OPTICAL, np.zeros(3))


class MapConvention:
    """
    `map` = the camera's *body* frame (x forward, y left, z up) at the first frame, so RViz shows the
    scene upright. SLAM works internally in the first *optical* frame (z forward), and datasets give
    optical-frame poses in their own world; both are converted here, once, for every publisher.
    """

    T_BO = Frames.T_BODY_OPTICAL

    @classmethod
    def pose(cls, T_world0_optical: SE3) -> SE3:
        """Optical-frame pose relative to the first optical frame -> camera_link pose in `map`."""
        return cls.T_BO @ T_world0_optical @ cls.T_BO.inverse()

    @classmethod
    def points(cls, points_world0_optical: np.ndarray) -> np.ndarray:
        return cls.T_BO.apply(points_world0_optical) if len(points_world0_optical) else points_world0_optical

    @classmethod
    def ground_truth(cls, first: SE3, gt: SE3) -> SE3:
        """A dataset pose (optical frame, dataset world), re-based on the first frame the player published."""
        return cls.pose(first.inverse() @ gt)


class QoS:
    @staticmethod
    def images() -> QoSProfile:
        """Best effort, depth 2: if processing is slow we want the *latest* frame, not a backlog."""
        q = QoSProfile(depth=2)
        q.reliability = QoSReliabilityPolicy.BEST_EFFORT
        return q

    @staticmethod
    def latched() -> QoSProfile:
        q = QoSProfile(depth=1)
        q.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        q.reliability = QoSReliabilityPolicy.RELIABLE
        return q


class RosConv:
    @staticmethod
    def stamp_to_sec(t: TimeMsg) -> float:
        return t.sec + 1e-9 * t.nanosec

    @staticmethod
    def pose(p: SE3) -> Pose:
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = (float(v) for v in p.translation)
        q = p.quaternion()
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (float(v) for v in q)
        return msg

    @staticmethod
    def se3(msg: Pose) -> SE3:
        q = msg.orientation
        return SE3.from_quaternion(np.array([q.x, q.y, q.z, q.w]),
                                   np.array([msg.position.x, msg.position.y, msg.position.z]))

    @staticmethod
    def transform(p: SE3, parent: str, child: str, stamp: TimeMsg) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, parent, child
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = \
            (float(v) for v in p.translation)
        q = p.quaternion()
        r = tf.transform.rotation
        r.x, r.y, r.z, r.w = (float(v) for v in q)
        return tf

    @classmethod
    def pose_stamped(cls, p: SE3, frame: str, stamp: TimeMsg) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id, ps.header.stamp = frame, stamp
        ps.pose = cls.pose(p)
        return ps

    @classmethod
    def path(cls, poses: list[SE3], stamps: list[TimeMsg], frame: str) -> Path:
        msg = Path()
        msg.header.frame_id = frame
        if stamps:
            msg.header.stamp = stamps[-1]
        msg.poses = [cls.pose_stamped(p, frame, s) for p, s in zip(poses, stamps)]
        return msg

    @staticmethod
    def point_cloud(points: np.ndarray, frame: str, stamp: TimeMsg, colors: np.ndarray | None = None) -> PointCloud2:
        """(N, 3) float points (+ optional (N, 3) uint8 RGB) -> PointCloud2 (x, y, z[, rgb])."""
        n = int(points.shape[0])
        fields = [PointField(name=c, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, c in enumerate("xyz")]
        if colors is None:
            data = points.astype(np.float32)
        else:
            fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
            c = colors.astype(np.uint32)
            packed = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)
            data = np.column_stack((points.astype(np.float32), packed))
        msg = PointCloud2()
        msg.header.frame_id, msg.header.stamp = frame, stamp
        msg.height, msg.width = 1, n
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 4 * len(fields)
        msg.row_step = msg.point_step * n
        msg.is_dense = True
        msg.data = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        return msg


class DiagnosticsReporter:
    """Metrics on /diagnostics (view: `ros2 topic echo /diagnostics` or rqt_robot_monitor)."""

    def __init__(self, node: Node, name: str) -> None:
        self.node, self.name = node, name
        self.pub = node.create_publisher(DiagnosticArray, "/diagnostics", 10)

    def publish(self, metrics: dict[str, Any], message: str = "ok", level: int = 0, log: bool = True) -> None:
        st = DiagnosticStatus(level=bytes([level]) if isinstance(DiagnosticStatus.OK, bytes) else level,
                              name=self.name, message=message, hardware_id=self.node.get_name())
        st.values = [KeyValue(key=k, value=self._fmt(v)) for k, v in metrics.items()]
        arr = DiagnosticArray()
        arr.header.stamp = self.node.get_clock().now().to_msg()
        arr.status = [st]
        self.pub.publish(arr)
        if log:
            self.node.get_logger().info(f"[{self.name}] " + ", ".join(f"{k}={self._fmt(v)}" for k, v in metrics.items()))

    @staticmethod
    def _fmt(v: Any) -> str:
        return f"{v:.4g}" if isinstance(v, float) else str(v)


class VslamNodeBase(Node):
    """Typed parameters + diagnostics + a classmethod entry point."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.diagnostics = DiagnosticsReporter(self, name)

    def param(self, name: str, default: T) -> T:
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value  # type: ignore[no-any-return]

    @classmethod
    def main(cls) -> None:
        """console_scripts entry point: `pkg.module:NodeClass.main`."""
        import rclpy  # noqa: PLC0415
        from rclpy.executors import ExternalShutdownException  # noqa: PLC0415

        rclpy.init()
        node = cls()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            hook = getattr(node, "on_shutdown", None)
            if callable(hook):
                hook()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
