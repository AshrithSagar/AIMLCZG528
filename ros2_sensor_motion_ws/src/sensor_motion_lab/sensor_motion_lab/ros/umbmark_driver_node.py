"""
umbmark_driver_node — runs the UMBmark odometry benchmark (Borenstein & Feng, 1996).

The robot drives a square using *only wheel odometry* as feedback, first in one
direction then the other. Ground truth tells us where it really stopped:

    error_i = (true end pose) − (true start pose), expressed in the start frame
    r_cw, r_ccw = |centroid of the CW / CCW errors|
    E_max,syst  = max(r_cw, r_ccw)          ← the benchmark number

  sub  /odom, /ground_truth/odom
  pub  /cmd_vel, /diagnostics
"""

from __future__ import annotations

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from ..core import Direction, Pose2D, SquarePathConfig, SquarePathController, UMBmarkReport, UMBmarkRun
from .common import LabNode


class UMBmarkDriverNode(LabNode):
    def __init__(self) -> None:
        super().__init__("umbmark_driver_node")
        self.side = self.param("side", 2.0)
        self.v = self.param("linear_speed", 0.2)
        self.w = self.param("angular_speed", 0.5)
        self.runs_per_direction = self.param("runs_per_direction", 1)
        order = [Direction(d) for d in self.param("directions", ["ccw", "cw"])]
        self.schedule: list[Direction] = [d for d in order for _ in range(self.runs_per_direction)]
        self.start_delay = self.param("start_delay", 2.0)

        self.odom: Pose2D | None = None
        self.truth: Pose2D | None = None
        self.controller: SquarePathController | None = None
        self.run_start_truth: Pose2D | None = None
        self.results: list[UMBmarkRun] = []
        self.started_at = self.get_clock().now()

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_truth, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / self.param("rate_hz", 50.0), self._tick)
        self.get_logger().info(f"UMBmark: {self.side} m square, schedule = {[d.value for d in self.schedule]}")

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = self.odom_pose(msg)

    def _on_truth(self, msg: Odometry) -> None:
        self.truth = self.odom_pose(msg)

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self.odom is None or self.truth is None:
            return
        if (self.get_clock().now() - self.started_at).nanoseconds * 1e-9 < self.start_delay:
            return
        if self.controller is None:
            if not self.schedule:
                return  # all runs finished
            self._begin_run(self.schedule.pop(0))
        assert self.controller is not None
        v, w = self.controller.command(self.odom)
        self._publish(v, w)
        if self.controller.done:
            self._finish_run()

    def _begin_run(self, direction: Direction) -> None:
        self.controller = SquarePathController(
            SquarePathConfig(side=self.side, direction=direction, linear_speed=self.v, angular_speed=self.w)
        )
        self.run_start_truth = self.truth
        self.get_logger().info(f"UMBmark run #{len(self.results) + 1}: {direction.value}")

    def _finish_run(self) -> None:
        assert self.controller is not None and self.run_start_truth is not None and self.truth is not None
        err = self.run_start_truth.between(self.truth)
        run = UMBmarkRun(self.controller.config.direction, err.x, err.y)
        self.results.append(run)
        self.diagnostics.publish(
            {"run": len(self.results), "direction": run.direction.value,
             "return_error_x_m": run.error_x, "return_error_y_m": run.error_y,
             "return_error_m": (run.error_x**2 + run.error_y**2) ** 0.5},
            message="run finished",
        )
        self.controller = None
        if not self.schedule:
            rep = UMBmarkReport.from_runs(self.results)
            self.diagnostics.publish(
                {"E_max_syst_m": rep.e_max_syst, "r_cw_m": rep.r_cw, "r_ccw_m": rep.r_ccw,
                 "runs": len(self.results)},
                message="UMBmark complete",
            )
        else:
            self.started_at = self.get_clock().now()  # short pause between runs

    def _publish(self, v: float, w: float) -> None:
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.cmd_pub.publish(cmd)
