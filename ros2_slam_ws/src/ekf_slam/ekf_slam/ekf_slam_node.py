#!/usr/bin/env python3
"""
ekf_slam_node
=============
ROS 2 port of the notebook's ``EKF_SLAM`` class -- a classical full-state
Extended Kalman Filter that jointly estimates the robot pose and a set of
landmark positions:

    state = [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]

Prediction runs on noisy wheel odometry received on ``/odom``
(v, w twist -- the same "control" signal the ground-truth simulator used
to move the robot, but corrupted with realistic noise). Correction runs
on range-bearing landmark observations received on
``/landmark_observations`` -- landmark *identity* is assumed resolved by
an upstream classifier (see ``ekf_slam`` package docstring / README),
which is standard for range-bearing EKF-SLAM with a fiducial/colour or
CNN-based landmark recognizer.

Publishes:
  * ``/ekf/pose``       (geometry_msgs/PoseWithCovarianceStamped)
  * ``/ekf/path``        (nav_msgs/Path)
  * ``/ekf/landmarks``   (visualization_msgs/MarkerArray) -- estimated
    landmark positions plus 95% covariance ellipses
  * ``tf: map -> base_link``
"""

import numpy as np
import rclpy
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from slam_sim_msgs.msg import LandmarkObservationArray
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray


class EKF_SLAM:
    """Extended Kalman Filter for SLAM (unchanged from the Lab 4 notebook)."""

    def __init__(self, q_diag=(0.1, 0.1, 0.05), r_diag=(0.2, 0.1)):
        self.state = np.array([0.0, 0.0, 0.0])
        self.P = np.eye(3) * 0.1
        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)
        self.landmark_ids = {}
        self.num_landmarks = 0

    def predict(self, v, w, dt):
        x, y, theta = self.state[0], self.state[1], self.state[2]

        if abs(w) < 1e-6:
            x_new = x + v * dt * np.cos(theta)
            y_new = y + v * dt * np.sin(theta)
            theta_new = theta
        else:
            x_new = x + (v / w) * (np.sin(theta + w * dt) - np.sin(theta))
            y_new = y + (v / w) * (-np.cos(theta + w * dt) + np.cos(theta))
            theta_new = theta + w * dt

        self.state[0] = x_new
        self.state[1] = y_new
        self.state[2] = self.normalize_angle(theta_new)

        G = np.eye(len(self.state))
        if abs(w) < 1e-6:
            G[0, 2] = -v * dt * np.sin(theta)
            G[1, 2] = v * dt * np.cos(theta)
        else:
            G[0, 2] = (v / w) * (np.cos(theta + w * dt) - np.cos(theta))
            G[1, 2] = (v / w) * (np.sin(theta + w * dt) - np.sin(theta))

        Q_full = np.zeros((len(self.state), len(self.state)))
        Q_full[0:3, 0:3] = self.Q
        self.P = G @ self.P @ G.T + Q_full

    def update(self, observations):
        for landmark_id, measured_range, measured_bearing in observations:
            if landmark_id not in self.landmark_ids:
                self.add_new_landmark(landmark_id, measured_range, measured_bearing)
            else:
                self.update_landmark(landmark_id, measured_range, measured_bearing)

    def add_new_landmark(self, landmark_id, r, phi):
        x, y, theta = self.state[0], self.state[1], self.state[2]
        lm_x = x + r * np.cos(theta + phi)
        lm_y = y + r * np.sin(theta + phi)

        self.state = np.append(self.state, [lm_x, lm_y])
        self.landmark_ids[landmark_id] = self.num_landmarks
        self.num_landmarks += 1

        n = len(self.state)
        P_new = np.zeros((n, n))
        P_new[:-2, :-2] = self.P
        P_new[-2, -2] = 1000
        P_new[-1, -1] = 1000
        self.P = P_new

    def update_landmark(self, landmark_id, measured_range, measured_bearing):
        lm_idx = self.landmark_ids[landmark_id]
        lm_state_idx = 3 + lm_idx * 2

        x, y, theta = self.state[0], self.state[1], self.state[2]
        lm_x = self.state[lm_state_idx]
        lm_y = self.state[lm_state_idx + 1]

        delta_x = lm_x - x
        delta_y = lm_y - y
        q = delta_x**2 + delta_y**2
        predicted_range = np.sqrt(q)
        predicted_bearing = self.normalize_angle(np.arctan2(delta_y, delta_x) - theta)

        z = np.array([measured_range, measured_bearing])
        z_hat = np.array([predicted_range, predicted_bearing])
        innovation = z - z_hat
        innovation[1] = self.normalize_angle(innovation[1])

        H = np.zeros((2, len(self.state)))
        H[0, 0] = -delta_x / predicted_range
        H[0, 1] = -delta_y / predicted_range
        H[0, lm_state_idx] = delta_x / predicted_range
        H[0, lm_state_idx + 1] = delta_y / predicted_range
        H[1, 0] = delta_y / q
        H[1, 1] = -delta_x / q
        H[1, 2] = -1
        H[1, lm_state_idx] = -delta_y / q
        H[1, lm_state_idx + 1] = delta_x / q

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[2] = self.normalize_angle(self.state[2])
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

    @staticmethod
    def normalize_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def get_landmarks(self):
        landmarks = []
        for i in range(self.num_landmarks):
            idx = 3 + i * 2
            landmarks.append([self.state[idx], self.state[idx + 1]])
        return np.array(landmarks)

    def get_robot_pose(self):
        return self.state[0], self.state[1], self.state[2]

    def get_landmark_covariance(self, landmark_id):
        idx = 3 + self.landmark_ids[landmark_id] * 2
        return self.P[idx : idx + 2, idx : idx + 2]


class EkfSlamNode(Node):
    def __init__(self):
        super().__init__("ekf_slam_node")

        self.declare_parameter("q_diag", [0.1, 0.1, 0.05])
        self.declare_parameter("r_diag", [0.2, 0.1])
        q_diag = tuple(self.get_parameter("q_diag").value)
        r_diag = tuple(self.get_parameter("r_diag").value)

        self.ekf = EKF_SLAM(q_diag=q_diag, r_diag=r_diag)
        self.last_v, self.last_w = 0.0, 0.0
        self.last_stamp = None

        self.path = Path()
        self.path.header.frame_id = "map"

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.obs_sub = self.create_subscription(
            LandmarkObservationArray, "/landmark_observations", self.on_observations, 10
        )

        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/ekf/pose", 10
        )
        self.path_pub = self.create_publisher(Path, "/ekf/path", 10)
        self.landmarks_pub = self.create_publisher(MarkerArray, "/ekf/landmarks", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"EKF-SLAM node ready (Q={q_diag}, R={r_diag}). "
            "Waiting for /odom and /landmark_observations..."
        )

    def on_odom(self, msg: Odometry):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        if self.last_stamp is not None:
            dt = t - self.last_stamp
            if dt > 0:
                self.ekf.predict(self.last_v, self.last_w, dt)
                self._publish_state(stamp)
        self.last_stamp = t
        self.last_v = msg.twist.twist.linear.x
        self.last_w = msg.twist.twist.angular.z

    def on_observations(self, msg: LandmarkObservationArray):
        observations = [(o.id, o.range, o.bearing) for o in msg.observations]
        if observations:
            self.ekf.update(observations)
            self._publish_state(msg.header.stamp)

    def _publish_state(self, stamp):
        x, y, theta = self.ekf.get_robot_pose()
        q = quaternion_from_euler(0, 0, theta)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = float(x)
        pose_msg.pose.pose.position.y = float(y)
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]
        cov = np.zeros((6, 6))
        cov[0:2, 0:2] = self.ekf.P[0:2, 0:2]
        cov[5, 5] = self.ekf.P[2, 2]
        pose_msg.pose.covariance = cov.flatten().tolist()
        self.pose_pub.publish(pose_msg)

        pose_stamped = PoseStamped()
        pose_stamped.header = pose_msg.header
        pose_stamped.pose = pose_msg.pose.pose
        self.path.header.stamp = stamp
        self.path.poses.append(pose_stamped)
        self.path_pub.publish(self.path)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(tf_msg)

        self._publish_landmark_markers(stamp)

    def _publish_landmark_markers(self, stamp):
        markers = MarkerArray()
        landmarks = self.ekf.get_landmarks()

        for landmark_id, idx in self.ekf.landmark_ids.items():
            lx, ly = landmarks[idx]
            cov2x2 = self.ekf.get_landmark_covariance(landmark_id)

            point_marker = Marker()
            point_marker.header.frame_id = "map"
            point_marker.header.stamp = stamp
            point_marker.ns = "ekf_landmarks"
            point_marker.id = int(landmark_id)
            point_marker.type = Marker.SPHERE
            point_marker.action = Marker.ADD
            point_marker.pose.position.x = float(lx)
            point_marker.pose.position.y = float(ly)
            point_marker.pose.position.z = 0.3
            point_marker.pose.orientation.w = 1.0
            point_marker.scale.x = point_marker.scale.y = point_marker.scale.z = 0.25
            (
                point_marker.color.r,
                point_marker.color.g,
                point_marker.color.b,
                point_marker.color.a,
            ) = 1.0, 0.3, 0.0, 0.9
            markers.markers.append(point_marker)

            ellipse_marker = self._covariance_ellipse_marker(
                landmark_id, lx, ly, cov2x2, stamp
            )
            if ellipse_marker is not None:
                markers.markers.append(ellipse_marker)

        self.landmarks_pub.publish(markers)

    @staticmethod
    def _covariance_ellipse_marker(landmark_id, cx, cy, cov2x2, stamp, n_sigma=2.45):
        """95% confidence ellipse (n_sigma ~= sqrt(chi2(2 dof, 0.95)) = 2.45)."""
        try:
            eigvals, eigvecs = np.linalg.eigh(cov2x2)
        except np.linalg.LinAlgError:
            return None
        eigvals = np.clip(eigvals, 1e-6, None)
        a, b = n_sigma * np.sqrt(eigvals[1]), n_sigma * np.sqrt(eigvals[0])
        angle = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "ekf_covariance_ellipses"
        marker.id = int(landmark_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
            1.0,
            1.0,
            0.0,
            0.7,
        )
        marker.pose.orientation.w = 1.0

        thetas = np.linspace(0, 2 * np.pi, 32)
        c, s = np.cos(angle), np.sin(angle)
        for th in thetas:
            ex, ey = a * np.cos(th), b * np.sin(th)
            px = cx + ex * c - ey * s
            py = cy + ex * s + ey * c
            pt = Point()
            pt.x, pt.y, pt.z = float(px), float(py), 0.3
            marker.points.append(pt)

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = EkfSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
