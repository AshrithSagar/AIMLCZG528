"""
depth_estimator_node — turns an RGB camera into an RGB-D camera with a learned depth network.

  sub  /camera/color/image_raw              sensor_msgs/Image (rgb8)
       /camera/depth_sensor/image_rect_raw  sensor_msgs/Image (32FC1)  optional: the real depth, only to score ours
  pub  /camera/depth/image_rect_raw         sensor_msgs/Image (32FC1 metres, REP-118), *same header as the colour
                                            image*, so visual_slam_node's time synchroniser pairs them exactly
       /depth_estimator/debug               sensor_msgs/Image (bgr8): colour | predicted | sensor depth
       /diagnostics                         ms per frame, fps, AbsRel / RMSE / δ<1.25 vs the sensor depth

The SLAM node does not know (or care) that its depth is now predicted: same topic,
same message type. Only the measured accuracy changes.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ..core.depth import DepthEstimatorFactory
from ..core.metrics import DepthEvaluator
from .common import QoS, RosConv, VslamNodeBase


class DepthColorizer:
    """Depth (m) -> false-colour BGR image with a fixed range, so frames are comparable."""

    def __init__(self, max_depth: float = 5.0) -> None:
        self.max_depth = max_depth

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        d = np.nan_to_num(depth, nan=0.0, posinf=0.0)
        u8 = np.clip(255.0 * d / self.max_depth, 0, 255).astype(np.uint8)
        out = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        out[d <= 0] = 0
        return out


class DepthEstimatorNode(VslamNodeBase):
    def __init__(self) -> None:
        super().__init__("depth_estimator_node")
        self.estimator = DepthEstimatorFactory.create(
            self.param("model", "depth_anything"), self.param("model_id", ""), self.param("device", "cpu"))
        self.evaluator = DepthEvaluator(max_depth=self.param("max_depth", 10.0))
        self.colorize = DepthColorizer(self.param("debug_max_depth", 5.0))
        self.eval_every = self.param("eval_every", 5)
        self.bridge = CvBridge()
        # colour and sensor depth arrive on separate topics, in either order: keep a few of each, keyed by stamp
        self.sensor: dict[float, np.ndarray] = {}
        self.predicted: dict[float, tuple[np.ndarray, np.ndarray]] = {}  # stamp -> (rgb, predicted depth)
        self.seen_sensor = False
        self.ms: list[float] = []
        self.scores: list[dict[str, float]] = []
        self.processed = 0
        self.t0 = time.monotonic()

        self.create_subscription(Image, "/camera/color/image_raw", self._on_color, QoS.images())
        self.create_subscription(Image, "/camera/depth_sensor/image_rect_raw", self._on_sensor, QoS.images())
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", QoS.images())
        self.debug_pub = self.create_publisher(Image, "/depth_estimator/debug", QoS.images())
        self.create_timer(self.param("report_period", 5.0), self._report)
        self.get_logger().info(f"Learned depth: {self.estimator.name} on {self.estimator.device}")

    def _on_sensor(self, msg: Image) -> None:
        stamp = RosConv.stamp_to_sec(msg.header.stamp)
        self.seen_sensor = True
        self._remember(self.sensor, stamp, self.bridge.imgmsg_to_cv2(msg, "passthrough"))
        if stamp in self.predicted:
            self._compare(msg, stamp)

    def _on_color(self, msg: Image) -> None:
        stamp = RosConv.stamp_to_sec(msg.header.stamp)
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        depth = self.estimator.estimate(rgb)
        self.ms.append(self.estimator.last_ms)
        self.processed += 1
        out = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        out.header = msg.header  # identical stamp + frame: exact pairing downstream
        self.depth_pub.publish(out)
        self._remember(self.predicted, stamp, (rgb, depth))
        if stamp in self.sensor:
            self._compare(msg, stamp)
        elif not self.seen_sensor:  # a real RGB-only camera: nothing to compare with
            self._publish_debug(msg, [rgb, depth])

    def _compare(self, msg: Image, stamp: float) -> None:
        rgb, depth = self.predicted.pop(stamp)
        sensor = self.sensor.pop(stamp)
        if self.processed % self.eval_every == 0 or not self.scores:
            self.scores.append(self.evaluator.evaluate(depth, sensor).as_dict())
        self._publish_debug(msg, [rgb, depth, sensor])

    @staticmethod
    def _remember(store: dict, stamp: float, value: object, keep: int = 10) -> None:
        store[stamp] = value
        while len(store) > keep:
            store.pop(next(iter(store)))

    def _publish_debug(self, msg: Image, panels: list[np.ndarray]) -> None:
        rgb, *depths = panels
        img = self.bridge.cv2_to_imgmsg(
            np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), *(self.colorize(d) for d in depths)]), encoding="bgr8")
        img.header = msg.header
        self.debug_pub.publish(img)

    def _report(self) -> None:
        if not self.processed:
            return
        elapsed = max(time.monotonic() - self.t0, 1e-6)
        metrics: dict[str, float | str] = {
            "model": self.estimator.name, "device": self.estimator.device,
            "ms_per_frame": float(np.mean(self.ms[-50:])), "processed_fps": self.processed / elapsed}
        if self.scores:
            recent = self.scores[-20:]
            metrics |= {k: float(np.nanmean([s[k] for s in recent])) for k in ("abs_rel", "rmse_m", "delta_1",
                                                                                 "median_scale")}
        self.diagnostics.publish(metrics, message="depth metrics vs the sensor depth (NYU/KITTI protocol)")
