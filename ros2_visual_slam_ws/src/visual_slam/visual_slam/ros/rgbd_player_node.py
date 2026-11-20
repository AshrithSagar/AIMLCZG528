"""
rgbd_player_node — turns an RGB-D sequence into a live ROS 2 camera.

  source := synthetic   CPU ray-cast room (no download)
  source := tum         TUM RGB-D folder (or the synthetic loop exported with `export_synthetic`)

Publishes exactly what a real RGB-D driver (RealSense, Kinect) would:

  /camera/color/image_raw          sensor_msgs/Image       rgb8
  /camera/color/camera_info        sensor_msgs/CameraInfo  (K, D, P)
  /camera/depth/image_rect_raw     sensor_msgs/Image       32FC1 metres (REP-118), registered to colour
  /camera/depth/camera_info        sensor_msgs/CameraInfo
  /ground_truth/pose, /ground_truth/path                   (frame map) — only because it is a dataset!
  TF (static) camera_link -> camera_color_optical_frame

Record it once and replay it anywhere:
    ros2 bag record -o synthetic_loop /camera/color/image_raw /camera/color/camera_info \
        /camera/depth/image_rect_raw /ground_truth/pose /tf_static
"""

from __future__ import annotations

import time

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from ..core.camera import PinholeCamera
from ..core.datasets import SequenceFactory
from ..core.se3 import SE3
from .common import Frames, QoS, RosConv, VslamNodeBase


class RgbdPlayerNode(VslamNodeBase):
    def __init__(self) -> None:
        super().__init__("rgbd_player_node")
        source = self.param("source", "synthetic")
        self.seq = SequenceFactory.create(source, self.param("path", ""))
        self.loop = self.param("loop", False)
        self.start_delay = self.param("start_delay", 2.0)
        rate = self.param("rate", 0.0)  # 0 = native dataset rate
        self.rate = rate if rate > 0 else self._native_rate()
        self.bridge = CvBridge()
        self.index = 0
        self.t_start = time.monotonic()

        self.color_pub = self.create_publisher(Image, "/camera/color/image_raw", QoS.images())
        self.color_info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", QoS.images())
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", QoS.images())
        self.depth_info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info", QoS.images())
        self.gt_pub = self.create_publisher(PoseStamped, "/ground_truth/pose", 10)
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth/path", QoS.latched())
        self.gt_path = Path()
        self.gt_path.header.frame_id = Frames.MAP

        StaticTransformBroadcaster(self).sendTransform(
            RosConv.transform(Frames.T_BODY_OPTICAL, Frames.CAMERA, Frames.OPTICAL, self.get_clock().now().to_msg()))
        self.create_timer(1.0 / self.rate, self._tick)
        c = self.seq.camera
        self.get_logger().info(f"Playing '{self.seq.name}': {len(self.seq)} frames, {c.width}x{c.height} "
                               f"@ {self.rate:.1f} Hz (source={source})")

    def _native_rate(self) -> float:
        n = min(len(self.seq), 30)
        return (n - 1) / (self.seq.stamp(n - 1) - self.seq.stamp(0)) if n > 1 else 15.0

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if time.monotonic() - self.t_start < self.start_delay:
            return
        if self.index >= len(self.seq):
            if not self.loop:
                return
            self.index = 0
        frame = self.seq.frame(self.index)
        stamp = self.get_clock().now().to_msg()  # live stamp; the dataset time is irrelevant downstream

        color = self.bridge.cv2_to_imgmsg(frame.rgb, encoding="rgb8")
        depth = self.bridge.cv2_to_imgmsg(frame.depth, encoding="32FC1")
        info = self._camera_info(self.seq.camera)
        for msg in (color, depth, info):
            msg.header.stamp, msg.header.frame_id = stamp, Frames.OPTICAL
        self.color_pub.publish(color)
        self.color_info_pub.publish(info)
        self.depth_pub.publish(depth)
        self.depth_info_pub.publish(info)

        gt = self.seq.ground_truth(self.index)
        if gt is not None:
            self._publish_ground_truth(gt, stamp)
        self.index += 1
        if self.index == len(self.seq):
            self.get_logger().info("End of sequence" + (" — looping" if self.loop else ""))

    def _publish_ground_truth(self, gt: SE3, stamp) -> None:  # noqa: ANN001 - builtin_interfaces/Time
        # datasets give the *optical* frame pose; publish camera_link (body) like the SLAM estimate does
        ps = RosConv.pose_stamped(gt @ Frames.T_BODY_OPTICAL.inverse(), Frames.MAP, stamp)
        self.gt_pub.publish(ps)
        if self.index % 3 == 0:  # thin the path a little
            self.gt_path.poses.append(ps)
            self.gt_path.header.stamp = stamp
            self.gt_path_pub.publish(self.gt_path)

    @staticmethod
    def _camera_info(c: PinholeCamera) -> CameraInfo:
        info = CameraInfo()
        info.width, info.height = c.width, c.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5  # TUM/synthetic images are used as already rectified
        info.k = [c.fx, 0.0, c.cx, 0.0, c.fy, c.cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [c.fx, 0.0, c.cx, 0.0, 0.0, c.fy, c.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info
