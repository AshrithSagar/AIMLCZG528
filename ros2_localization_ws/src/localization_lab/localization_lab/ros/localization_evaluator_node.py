"""
localization_evaluator_node — scores any pose estimate against ground truth, live.

  sub  /ground_truth/odom                 nav_msgs/Odometry (W1 sim / Gazebo)
       <estimate_topic>                   geometry_msgs/PoseWithCovarianceStamped
                                          /localization/pose (ours), /amcl_pose (Nav2 AMCL), /ml/pose (ML only)
       /sim/teleport                      marks a kidnapping event
  pub  /diagnostics                       error now, convergence / recovery time, tracking RMSE
  srv  ~/report                           std_srvs/Trigger

The estimate is just a topic, so Nav2 AMCL is benchmarked by the very same node.
"""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger

from sensor_motion_lab.ros.common import LabNode, PoseBuffer, RosConv

from ..core.evaluation import LocalizationEvaluator, LocalizationLog


class LocalizationEvaluatorNode(LabNode):
    def __init__(self) -> None:
        super().__init__("localization_evaluator_node")
        self.label = self.param("label", "localization")
        self.truth = PoseBuffer(maxlen=2000)
        self.log = LocalizationLog()
        self.t0: float | None = None
        self.kidnap_t: float | None = None
        self.evaluator = LocalizationEvaluator()
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_truth, 50)
        self.create_subscription(PoseWithCovarianceStamped, self.param("estimate_topic", "/localization/pose"),
                                 self._on_estimate, 10)
        self.create_subscription(PoseStamped, "/sim/teleport", self._on_teleport, 10)
        self.create_service(Trigger, "~/report", self._on_report)
        self.create_timer(self.param("report_period", 5.0), self._report)

    def _on_truth(self, msg: Odometry) -> None:
        self.truth.add(RosConv.stamp_to_sec(msg.header.stamp), self.odom_pose(msg))

    def _on_teleport(self, _msg: PoseStamped) -> None:
        self.kidnap_t = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().warn(f"[{self.label}] kidnapping registered at t={self.kidnap_t:.1f}")

    def _on_estimate(self, msg: PoseWithCovarianceStamped) -> None:
        t = RosConv.stamp_to_sec(msg.header.stamp)
        truth = self.truth.closest(t)
        if truth is None:
            return
        self.t0 = self.t0 if self.t0 is not None else t
        self.log.add(t, RosConv.pose_from_msg(msg.pose.pose), truth)

    def _summary(self) -> dict[str, float]:
        rep = self.evaluator.report(self.log, self.t0 or 0.0, self.kidnap_t)
        return {"error_now_m": self.log.pos_err[-1], "heading_error_now_deg": self.log.head_err_deg[-1]} | {
            k: v for k, v in rep.as_dict().items() if not k.startswith("update_ms")}

    def _report(self) -> None:
        if self.log.stamps:
            self.diagnostics.publish(self._summary(), message=f"{self.label} vs ground truth")

    def _on_report(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if not self.log.stamps:
            resp.success, resp.message = False, "no estimates yet"
        else:
            resp.success, resp.message = True, ", ".join(f"{k}={v:.3g}" for k, v in self._summary().items())
        return resp
