"""
kinematic_sim_node — a lightweight stand-in for Gazebo.

Publishes *the same interface* as the Gazebo + ros_gz_bridge setup, so every
downstream node works unchanged with either simulator (Liskov, in ROS form):

  sub  /cmd_vel             geometry_msgs/Twist
  sub  /sim/teleport        geometry_msgs/PoseStamped  move the true pose only ("kidnapped robot")
  pub  /odom                nav_msgs/Odometry      wheel odometry (nominal geometry)
  pub  /ground_truth/odom   nav_msgs/Odometry      true pose (frame: map)
  pub  /scan                sensor_msgs/LaserScan  ray-cast + Gaussian noise
  pub  /joint_states        sensor_msgs/JointState wheel angles (for robot_state_publisher)
  TF   odom -> base_link

Use it when the lab VM has no GPU / Gazebo is too slow; the physics is
kinematic (no inertia), but wheel slip and calibration errors are modelled.
"""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from tf2_ros import TransformBroadcaster

from ..core import DifferentialDrive, DriveCalibrationError, GridRasterizer, KinematicRobot, Pose2D, RayCaster
from ..core import SimulatedLidar, SlipNoise
from .common import LabNode, QoS, RosConv


class KinematicSimNode(LabNode):
    def __init__(self) -> None:
        super().__init__("kinematic_sim_node")
        cfg = self.robot
        calib = DriveCalibrationError(
            self.param("diameter_ratio", cfg.calibration.diameter_ratio),
            self.param("baseline_ratio", cfg.calibration.baseline_ratio),
        )
        slip = SlipNoise(self.param("wheel_slip_std", 0.02))
        self.cmd_timeout = self.param("cmd_timeout", 0.5)
        rng = np.random.default_rng(self.param("seed", 0))

        self.sim = KinematicRobot(DifferentialDrive(cfg.drive, calib), slip, rng)
        grid = GridRasterizer(0.02).rasterize(self.world)
        self.lidar = SimulatedLidar(RayCaster(grid, cfg.lidar.range_max), cfg.lidar, rng)

        self.cmd = Twist()
        self.last_cmd_time = self.get_clock().now()
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        # "Kidnap" the robot: move the true pose, leave odometry untouched (Webinar 2).
        self.create_subscription(PoseStamped, "/sim/teleport", self._on_teleport, 10)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.gt_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", QoS.sensor())
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(self)

        self.dt = 1.0 / cfg.odom_rate_hz
        self.create_timer(self.dt, self._step)
        self.create_timer(1.0 / cfg.lidar.rate_hz, self._publish_scan)
        self.get_logger().info(
            f"Kinematic sim up: E_d={calib.diameter_ratio}, E_b={calib.baseline_ratio}, slip σ={slip.wheel_speed_std}"
        )

    # ------------------------------------------------------------------ #
    def _on_cmd(self, msg: Twist) -> None:
        self.cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def _on_teleport(self, msg: PoseStamped) -> None:
        self.sim.true_pose = RosConv.pose_from_msg(msg.pose)
        self.get_logger().warn(f"Teleported (kidnapped) to {self.sim.true_pose}")

    def _step(self) -> None:
        stale = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout
        v, w = (0.0, 0.0) if stale else (self.cmd.linear.x, self.cmd.angular.z)
        self.sim.step(v, w, self.dt)
        stamp = self.now_msg()
        self.odom_pub.publish(self._odom(self.sim.odom_pose, self.ODOM_FRAME, stamp, *self.sim.last_twist))
        self.gt_pub.publish(self._odom(self.sim.true_pose, self.MAP_FRAME, stamp, v, w))
        self.tf.sendTransform(RosConv.transform(self.sim.odom_pose, self.ODOM_FRAME, self.BASE_FRAME, stamp))
        js = JointState()
        js.header.stamp = stamp
        js.name = ["left_wheel_joint", "right_wheel_joint"]
        js.position = [float(a) for a in self.sim.wheel_angles]
        self.joint_pub.publish(js)

    def _publish_scan(self) -> None:
        li = self.robot.lidar
        msg = LaserScan()
        msg.header.stamp, msg.header.frame_id = self.now_msg(), "lidar_link"
        msg.angle_min, msg.angle_max = li.angle_min, li.angle_max - li.angle_increment
        msg.angle_increment = li.angle_increment
        msg.scan_time = 1.0 / li.rate_hz
        msg.time_increment = msg.scan_time / li.samples
        msg.range_min, msg.range_max = li.range_min, li.range_max
        msg.ranges = [float(r) for r in self.lidar.measure(self.sim.true_pose)]
        self.scan_pub.publish(msg)

    def _odom(self, pose: Pose2D, frame: str, stamp: TimeMsg, v: float, w: float) -> Odometry:
        msg = Odometry()
        msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, frame, self.BASE_FRAME
        msg.pose.pose = RosConv.pose_to_msg(pose)
        msg.twist.twist.linear.x, msg.twist.twist.angular.z = float(v), float(w)
        return msg
