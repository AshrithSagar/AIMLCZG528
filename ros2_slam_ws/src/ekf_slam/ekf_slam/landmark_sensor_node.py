#!/usr/bin/env python3
"""
landmark_sensor_node
=====================
Gazebo-backed replacement for the sensing half of the old
``pybullet_landmark_sim_node``. The robot's *real* physics/motion now live
in Gazebo; this node only adds the two things Gazebo doesn't model for us:

  1. A noisy wheel-odometry estimate of the commanded (v, w), for the
     EKF's prediction step -- deliberately corrupted, so the EKF has
     something real to correct for. (Gazebo's own bridged odometry is
     treated as ground truth, not fed to the EKF directly.)
  2. A noisy range-bearing sensor against six known landmark positions
     (must match the spheres in ``worlds/landmarks.sdf``), standing in
     for a landmark classifier (colour-based here, could be a CNN as in
     the ``cnn_visual_slam`` package) that has already resolved landmark
     identity.

Subscribes:
  * ``/ground_truth/odom_raw`` (nav_msgs/Odometry) -- exact pose, from Gazebo
  * ``/cmd_vel`` (geometry_msgs/Twist)             -- commanded velocity

Publishes:
  * ``/odom``                    (nav_msgs/Odometry)   -- noisy (v, w) for the EKF
  * ``/landmark_observations``   (slam_sim_msgs/LandmarkObservationArray)
  * ``/ground_truth/path``       (nav_msgs/Path)
  * ``/ground_truth/landmarks``  (visualization_msgs/MarkerArray)
  * ``tf: world -> base_link_gt``
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from slam_sim_msgs.msg import LandmarkObservation, LandmarkObservationArray
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

# Must match worlds/landmarks.sdf landmark poses.
LANDMARK_CONFIGS = [
    (0, [3, 3, 0.3], [1.0, 0.0, 0.0, 1.0], "Red"),
    (1, [3, -3, 0.3], [0.0, 1.0, 0.0, 1.0], "Green"),
    (2, [-3, 3, 0.3], [0.0, 0.0, 1.0, 1.0], "Blue"),
    (3, [-3, -3, 0.3], [1.0, 1.0, 0.0, 1.0], "Yellow"),
    (4, [0, 4, 0.3], [1.0, 0.0, 1.0, 1.0], "Magenta"),
    (5, [4, 0, 0.3], [0.0, 1.0, 1.0, 1.0], "Cyan"),
]

SENSOR_RANGE_MAX = 6.0
SENSOR_FOV = np.pi  # +-90 deg either side of heading


class LandmarkSensorNode(Node):
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
        self.last_cmd_v, self.last_cmd_w = 0.0, 0.0
        self.rng = np.random.default_rng(42)

        self.gt_path = Path()
        self.gt_path.header.frame_id = "world"

        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.gt_odom_sub = self.create_subscription(
            Odometry, "/ground_truth/odom_raw", self.on_ground_truth_odom, 10
        )

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.obs_pub = self.create_publisher(
            LandmarkObservationArray, "/landmark_observations", 10
        )
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth/path", 10)
        self.gt_landmarks_pub = self.create_publisher(
            MarkerArray, "/ground_truth/landmarks", 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        self._publish_landmark_markers()
        self.get_logger().info(
            f"Landmark sensor model ready with {len(self.landmark_positions)} landmarks, "
            "waiting for Gazebo odometry..."
        )

    def on_cmd_vel(self, msg: Twist):
        self.last_cmd_v = msg.linear.x
        self.last_cmd_w = msg.angular.z

    def on_ground_truth_odom(self, msg: Odometry):
        stamp = msg.header.stamp
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y

        self._publish_noisy_odom(stamp)
        self._publish_ground_truth(x, y, msg.pose.pose.orientation, stamp)
        self._publish_observations(x, y, theta, stamp)

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
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.orientation = orientation
        self.gt_path.header.stamp = stamp
        self.gt_path.poses.append(pose_stamped)
        self.gt_path_pub.publish(self.gt_path)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link_gt"
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
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
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = color
            markers.markers.append(m)
        self.gt_landmarks_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
