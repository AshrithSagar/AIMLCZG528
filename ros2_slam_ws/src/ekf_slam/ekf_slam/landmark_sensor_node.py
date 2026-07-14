#!/usr/bin/env python3
"""
Standalone Mock Landmark Sensor Node
===================================
Simulates perfect kinematics internally at 10Hz without needing Gazebo.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from slam_sim_msgs.msg import LandmarkObservation, LandmarkObservationArray
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

LANDMARK_CONFIGS = [
    (0, [3.0, 3.0, 0.3], [1.0, 0.0, 0.0, 1.0], "Red"),
    (1, [3.0, -3.0, 0.3], [0.0, 1.0, 0.0, 1.0], "Green"),
    (2, [-3.0, 3.0, 0.3], [0.0, 0.0, 1.0, 1.0], "Blue"),
    (3, [-3.0, -3.0, 0.3], [1.0, 1.0, 0.0, 1.0], "Yellow"),
    (4, [0.0, 4.0, 0.3], [1.0, 0.0, 1.0, 1.0], "Magenta"),
    (5, [4.0, 0.0, 0.3], [0.0, 1.0, 1.0, 1.0], "Cyan"),
]

SENSOR_RANGE_MAX = 6.0
SENSOR_FOV = np.pi


class StandaloneLandmarkSensorNode(Node):
    def __init__(self):
        super().__init__("landmark_sensor_node")

        self.declare_parameter("odom_v_noise_std", 0.03)
        self.declare_parameter("odom_w_noise_std", 0.02)
        self.declare_parameter("range_noise_std", 0.15)
        self.declare_parameter("bearing_noise_std", 0.05)

        self.odom_v_noise_std = self.get_parameter("odom_v_noise_std").value
        self.odom_w_noise_std = self.get_parameter("odom_w_noise_std").value
        self.range_noise_std = self.get_parameter("range_noise_std").value
        self.bearing_noise_std = self.get_parameter("bearing_noise_std").value

        self.landmark_positions = {
            lm_id: np.array(pos[:2]) for lm_id, pos, _, _ in LANDMARK_CONFIGS
        }

        # Internal State Tracker (Replacing Gazebo Physics)
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.last_cmd_v, self.last_cmd_w = 0.0, 0.0
        self.rng = np.random.default_rng(42)

        self.gt_path = Path()
        self.gt_path.header.frame_id = "world"

        # Subscribers
        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.obs_pub = self.create_publisher(
            LandmarkObservationArray, "/landmark_observations", 10
        )
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth/path", 10)
        self.gt_landmarks_pub = self.create_publisher(
            MarkerArray, "/ground_truth/landmarks", 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        # 10Hz Sim Loop Execution Timer
        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.simulation_loop)

        self.get_logger().info(
            "Standalone Pure Kinematic Mock Simulator Active. Gazebo Not Required."
        )

    def on_cmd_vel(self, msg: Twist):
        self.last_cmd_v = msg.linear.x
        self.last_cmd_w = msg.angular.z

    def simulation_loop(self):
        stamp = self.get_clock().now().to_msg()

        # 1. Update ideal robot pose (unicycle kinematics model)
        if abs(self.last_cmd_w) < 1e-5:
            self.x += self.last_cmd_v * np.cos(self.theta) * self.dt
            self.y += self.last_cmd_v * np.sin(self.theta) * self.dt
        else:
            self.x += (self.last_cmd_v / self.last_cmd_w) * (
                np.sin(self.theta + self.last_cmd_w * self.dt) - np.sin(self.theta)
            )
            self.y -= (self.last_cmd_v / self.last_cmd_w) * (
                np.cos(self.theta + self.last_cmd_w * self.dt) - np.cos(self.theta)
            )
        self.theta += self.last_cmd_w * self.dt

        # Compute quaternion orientation
        q = quaternion_from_euler(0, 0, self.theta)
        from geometry_msgs.msg import Quaternion

        orient_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # 2. Publish Everything
        self._publish_noisy_odom(stamp)
        self._publish_ground_truth(self.x, self.y, orient_msg, stamp)
        self._publish_observations(self.x, self.y, self.theta, stamp)
        self._publish_landmark_markers()

    def _publish_noisy_odom(self, stamp):
        v_meas = self.last_cmd_v + self.rng.normal(0, self.odom_v_noise_std)
        w_meas = self.last_cmd_w + self.rng.normal(0, self.odom_w_noise_std)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = float(v_meas)
        odom.twist.twist.angular.z = float(w_meas)
        self.odom_pub.publish(odom)

    def _publish_ground_truth(self, x, y, orientation, stamp):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = "world"
        pose_stamped.pose.position.x = float(x)
        pose_stamped.pose.position.y = float(y)
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation = orientation

        self.gt_path.header.stamp = stamp
        self.gt_path.poses.append(pose_stamped)
        self.gt_path_pub.publish(self.gt_path)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link_gt"
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = orientation
        self.tf_broadcaster.sendTransform(tf_msg)

    def _publish_observations(self, x, y, theta, stamp):
        arr = LandmarkObservationArray()
        arr.header.stamp = stamp
        arr.header.frame_id = "base_link_gt"

        for lm_id, lm_pos in self.landmark_positions.items():
            dx, dy = lm_pos[0] - x, lm_pos[1] - y
            true_range = np.hypot(dx, dy)
            raw_bearing = np.arctan2(dy, dx) - theta
            true_bearing = np.arctan2(np.sin(raw_bearing), np.cos(raw_bearing))

            if true_range > SENSOR_RANGE_MAX or abs(true_bearing) > SENSOR_FOV / 2:
                continue

            obs = LandmarkObservation()
            obs.id = int(lm_id)
            obs.range = float(true_range + self.rng.normal(0, self.range_noise_std))
            obs.bearing = float(
                true_bearing + self.rng.normal(0, self.bearing_noise_std)
            )
            arr.observations.append(obs)

        self.obs_pub.publish(arr)

    def _publish_landmark_markers(self):
        markers = MarkerArray()
        for lm_id, pos, color, name in LANDMARK_CONFIGS:
            m = Marker()
            m.header.frame_id = "world"
            m.ns = "ground_truth_landmarks"
            m.id = lm_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = color
            markers.markers.append(m)
        self.gt_landmarks_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = StandaloneLandmarkSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
