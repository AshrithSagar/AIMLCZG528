"""
path_follower_node — pure pursuit along /plan, with a collision-monitor style safety stop.

  sub  /plan        nav_msgs/Path               (from planner_node, or Nav2's planner_server)
       /obstacles   geometry_msgs/PoseArray     (people near the robot -> stop)
       TF map -> base_link
  pub  /cmd_vel     geometry_msgs/Twist          (consumed by W1's kinematic sim or the Gazebo diff-drive)
       /follower/lookahead  geometry_msgs/PointStamped (the carrot, for RViz)

The controller is deliberately the same class the offline benchmark drives, so a
planner behaves the same in `expected_output/` and in RViz.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PointStamped, PoseArray, Twist
from nav_msgs.msg import Path
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from sensor_motion_lab.core import Pose2D
from sensor_motion_lab.ros.common import LabNode, RosConv

from ..core.navigation import FollowerConfig, PurePursuit
from .conv import PlanningConv


class PathFollowerNode(LabNode):
    def __init__(self) -> None:
        super().__init__("path_follower_node")
        d = FollowerConfig()
        self.follower = PurePursuit(FollowerConfig(
            lookahead=self.param("lookahead", d.lookahead),
            max_speed=self.param("max_speed", d.max_speed),
            max_turn=self.param("max_turn", d.max_turn),
            goal_tolerance=self.param("goal_tolerance", d.goal_tolerance),
            stop_distance=self.param("stop_distance", d.stop_distance),
            stop_cone_deg=self.param("stop_cone_deg", d.stop_cone_deg),
        ))
        self.base_frame = self.param("base_frame", self.BASE_FRAME)
        self.obstacles = np.zeros((0, 2))
        self.stopped = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Path, "/plan", self._on_plan, 10)
        self.create_subscription(PoseArray, "/obstacles", self._on_obstacles, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.carrot_pub = self.create_publisher(PointStamped, "/follower/lookahead", 10)
        self.create_timer(1.0 / self.param("rate_hz", 20.0), self._control)

    def _on_plan(self, msg: Path) -> None:
        self.follower.set_path(np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses]).reshape(-1, 2))

    def _on_obstacles(self, msg: PoseArray) -> None:
        self.obstacles = PlanningConv.points(msg)

    def _pose(self) -> Pose2D | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.MAP_FRAME, self.base_frame, Time())
        except TransformException:
            return None
        t = tf.transform
        return Pose2D(t.translation.x, t.translation.y, RosConv.quat_to_yaw(t.rotation))

    def _control(self) -> None:
        pose = self._pose()
        if pose is None or len(self.follower.path) < 2:
            return
        v, w, stopped = self.follower.command(pose, self.obstacles)
        if stopped != self.stopped:
            self.stopped = stopped
            (self.get_logger().warn if stopped else self.get_logger().info)(
                "safety stop: person ahead" if stopped else "path clear, resuming")
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.cmd_pub.publish(cmd)
        if self.follower.done(pose):
            self.follower.set_path(np.zeros((0, 2)))  # stop publishing once there
            self.cmd_pub.publish(Twist())
            return
        if self.follower.carrot is None:
            return
        carrot = PointStamped()
        carrot.header.frame_id, carrot.header.stamp = self.MAP_FRAME, self.now_msg()
        carrot.point.x, carrot.point.y = (float(c) for c in self.follower.carrot)
        self.carrot_pub.publish(carrot)
