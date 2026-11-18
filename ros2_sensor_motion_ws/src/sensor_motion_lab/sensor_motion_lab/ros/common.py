"""
Thin ROS 2 adapter layer: message conversions, QoS presets, diagnostics
reporting, and a `LabNode` base class with the plumbing every lab node shares.

Keeping this separate from `core/` means the algorithms never import rclpy
(Dependency Inversion: nodes depend on the core, not the other way round).
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time as TimeMsg
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from ..core import GridMap, Pose2D, WorldSpec
from ..core.config import RobotConfig

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# QoS                                                                         #
# --------------------------------------------------------------------------- #
class QoS:
    """Named QoS presets (avoid magic numbers scattered across nodes)."""

    @staticmethod
    def latched() -> QoSProfile:
        """Transient-local: late joiners (e.g. RViz) still receive the last message (maps)."""
        q = QoSProfile(depth=1)
        q.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        q.reliability = QoSReliabilityPolicy.RELIABLE
        return q

    @staticmethod
    def sensor() -> QoSProfile:
        """Best-effort, shallow: the usual profile for high-rate sensor streams."""
        q = QoSProfile(depth=5)
        q.reliability = QoSReliabilityPolicy.BEST_EFFORT
        return q


# --------------------------------------------------------------------------- #
# Conversions                                                                 #
# --------------------------------------------------------------------------- #
class RosConv:
    @staticmethod
    def yaw_to_quat(yaw: float) -> Quaternion:
        return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))

    @staticmethod
    def quat_to_yaw(q: Quaternion) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @classmethod
    def pose_from_msg(cls, p: Pose) -> Pose2D:
        return Pose2D(p.position.x, p.position.y, cls.quat_to_yaw(p.orientation))

    @classmethod
    def pose_to_msg(cls, p: Pose2D) -> Pose:
        msg = Pose()
        msg.position.x, msg.position.y = float(p.x), float(p.y)
        msg.orientation = cls.yaw_to_quat(p.theta)
        return msg

    @staticmethod
    def stamp_to_sec(t: TimeMsg) -> float:
        return t.sec + 1e-9 * t.nanosec

    @classmethod
    def transform(cls, p: Pose2D, parent: str, child: str, stamp: TimeMsg) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, parent, child
        tf.transform.translation.x, tf.transform.translation.y = float(p.x), float(p.y)
        tf.transform.rotation = cls.yaw_to_quat(p.theta)
        return tf

    @staticmethod
    def occupancy_grid(values: np.ndarray, resolution: float, origin_xy: tuple[float, float],
                       frame: str, stamp: TimeMsg) -> OccupancyGrid:
        """values: (H, W) int array in [-1, 100], row 0 = lowest y."""
        msg = OccupancyGrid()
        msg.header.frame_id, msg.header.stamp = frame, stamp
        msg.info.resolution = float(resolution)
        msg.info.height, msg.info.width = int(values.shape[0]), int(values.shape[1])
        msg.info.origin.position.x, msg.info.origin.position.y = float(origin_xy[0]), float(origin_xy[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = values.astype(np.int8).ravel().tolist()
        return msg

    @classmethod
    def grid_map(cls, grid: GridMap, frame: str, stamp: TimeMsg) -> OccupancyGrid:
        return cls.occupancy_grid(np.where(grid.occupied, 100, 0), grid.resolution,
                                  (grid.origin_x, grid.origin_y), frame, stamp)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
class PoseBuffer:
    """Short history of stamped poses; returns the one closest in time (sensor sync)."""

    def __init__(self, maxlen: int = 500) -> None:
        self._t: deque[float] = deque(maxlen=maxlen)
        self._p: deque[Pose2D] = deque(maxlen=maxlen)

    def add(self, t: float, pose: Pose2D) -> None:
        self._t.append(t)
        self._p.append(pose)

    def latest(self) -> Pose2D | None:
        return self._p[-1] if self._p else None

    def closest(self, t: float) -> Pose2D | None:
        if not self._t:
            return None
        ts = list(self._t)
        i = bisect.bisect_left(ts, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(ts)]
        return self._p[min(cands, key=lambda j: abs(ts[j] - t))]


class PathAccumulator:
    """Builds a nav_msgs/Path incrementally, skipping near-duplicate poses."""

    def __init__(self, frame: str, min_step: float = 0.02, max_len: int = 5000) -> None:
        self.msg = PathMsg()
        self.msg.header.frame_id = frame
        self.min_step = min_step
        self.max_len = max_len
        self._last: Pose2D | None = None

    def add(self, pose: Pose2D, stamp: TimeMsg) -> None:
        if self._last is not None and self._last.distance_to(pose) < self.min_step:
            return
        ps = PoseStamped()
        ps.header.frame_id, ps.header.stamp = self.msg.header.frame_id, stamp
        ps.pose = RosConv.pose_to_msg(pose)
        self.msg.poses.append(ps)
        if len(self.msg.poses) > self.max_len:
            self.msg.poses.pop(0)
        self.msg.header.stamp = stamp
        self._last = pose

    def clear(self) -> None:
        self.msg.poses.clear()
        self._last = None


class DiagnosticsReporter:
    """
    Publishes metrics on /diagnostics (standard ROS channel; view with
    `ros2 topic echo /diagnostics` or rqt_robot_monitor) and logs them.
    """

    def __init__(self, node: Node, name: str) -> None:
        self.node = node
        self.name = name
        self.pub = node.create_publisher(DiagnosticArray, "/diagnostics", 10)

    def publish(self, metrics: dict[str, Any], message: str = "ok", log: bool = True) -> None:
        status = DiagnosticStatus(level=DiagnosticStatus.OK, name=self.name, message=message,
                                  hardware_id=self.node.get_name())
        status.values = [KeyValue(key=k, value=self._fmt(v)) for k, v in metrics.items()]
        arr = DiagnosticArray()
        arr.header.stamp = self.node.get_clock().now().to_msg()
        arr.status = [status]
        self.pub.publish(arr)
        if log:
            body = ", ".join(f"{k}={self._fmt(v)}" for k, v in metrics.items())
            self.node.get_logger().info(f"[{self.name}] {body}")

    @staticmethod
    def _fmt(v: Any) -> str:
        return f"{v:.4g}" if isinstance(v, float) else str(v)


class PackageShare:
    """Locations of the shared config files (installed share/ directory)."""

    @staticmethod
    def description_config(name: str) -> str:
        return str(Path(get_package_share_directory("mobile_bot_description")) / "config" / name)


# --------------------------------------------------------------------------- #
# Base node                                                                   #
# --------------------------------------------------------------------------- #
class LabNode(Node):
    """
    Base class for all lab nodes:
      * typed parameter declaration in one line,
      * access to the shared world / robot configuration,
      * a diagnostics reporter.
    """

    MAP_FRAME = "map"
    ODOM_FRAME = "odom"
    BASE_FRAME = "base_link"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.world_file: str = self.param("world_file", PackageShare.description_config("world.yaml"))
        self.robot_file: str = self.param("robot_file", PackageShare.description_config("robot.yaml"))
        self.diagnostics = DiagnosticsReporter(self, name)
        self._world: WorldSpec | None = None
        self._robot: RobotConfig | None = None

    def param(self, name: str, default: T) -> T:
        """Declare-and-get, keeping the type of `default`."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value  # type: ignore[no-any-return]

    @property
    def world(self) -> WorldSpec:
        if self._world is None:
            self._world = WorldSpec.from_yaml(self.world_file)
        return self._world

    @property
    def robot(self) -> RobotConfig:
        if self._robot is None:
            self._robot = RobotConfig.from_yaml(self.robot_file)
        return self._robot

    def now_msg(self) -> TimeMsg:
        return self.get_clock().now().to_msg()

    @staticmethod
    def odom_pose(msg: Odometry) -> Pose2D:
        return RosConv.pose_from_msg(msg.pose.pose)

    @classmethod
    def main(cls) -> None:
        """console_scripts entry point: `pkg.module:NodeClass.main`."""
        NodeRunner.run(cls)


class NodeRunner:
    """Standard rclpy lifecycle for a single node (used by every `main`)."""

    @staticmethod
    def run(node_cls: type[Node]) -> None:
        import rclpy
        from rclpy.executors import ExternalShutdownException

        rclpy.init()
        node = node_cls()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            on_shutdown = getattr(node, "on_shutdown", None)
            if callable(on_shutdown):
                on_shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
