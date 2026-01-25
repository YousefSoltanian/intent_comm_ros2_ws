#!/usr/bin/env python3
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class EKFEvalLoggerNode(Node):
    def __init__(self):
        super().__init__("ekf_eval_logger")

        self.declare_parameter("run_seconds", 10.0)
        self.declare_parameter("out_dir", "~/.ros/ekf_eval")

        self.declare_parameter("human_cmd_topic", "/human/cmd_vel")
        self.declare_parameter("robot_cmd_topic", "/robot/cmd_vel")
        self.declare_parameter("human_est_topic", "/ekf/human/u_est")
        self.declare_parameter("robot_est_topic", "/ekf/robot/u_est")

        self.run_seconds = float(self.get_parameter("run_seconds").value)

        out_dir = self.get_parameter("out_dir").value
        # IMPORTANT FIX: if launch passes "", don’t crash
        if not out_dir:
            out_dir = "~/.ros/ekf_eval"
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.h_cmd_topic = self.get_parameter("human_cmd_topic").value
        self.r_cmd_topic = self.get_parameter("robot_cmd_topic").value
        self.h_est_topic = self.get_parameter("human_est_topic").value
        self.r_est_topic = self.get_parameter("robot_est_topic").value

        self.t0 = time.time()

        self._t_h_cmd = []
        self._h_cmd = []  # (v,w)

        self._t_r_cmd = []
        self._r_cmd = []

        self._t_h_est = []
        self._h_est = []

        self._t_r_est = []
        self._r_est = []

        self.create_subscription(Twist, self.h_cmd_topic, self._cb_h_cmd, 50)
        self.create_subscription(Twist, self.r_cmd_topic, self._cb_r_cmd, 50)
        self.create_subscription(Float32MultiArray, self.h_est_topic, self._cb_h_est, 50)
        self.create_subscription(Float32MultiArray, self.r_est_topic, self._cb_r_est, 50)

        self.get_logger().info(f"EKF eval logger running for {self.run_seconds:.2f}s")
        self.get_logger().info(f"  cmd topics: {self.h_cmd_topic}, {self.r_cmd_topic}")
        self.get_logger().info(f"  est topics: {self.h_est_topic}, {self.r_est_topic}")
        self.get_logger().info(f"  out_dir:    {self.out_dir}")

        self.timer = self.create_timer(self.run_seconds, self._finish_once)
        self._done = False

    def _now(self):
        return time.time() - self.t0

    def _cb_h_cmd(self, msg: Twist):
        self._t_h_cmd.append(self._now())
        self._h_cmd.append([float(msg.linear.x), float(msg.angular.z)])

    def _cb_r_cmd(self, msg: Twist):
        self._t_r_cmd.append(self._now())
        self._r_cmd.append([float(msg.linear.x), float(msg.angular.z)])

    def _cb_h_est(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self._t_h_est.append(self._now())
            self._h_est.append([float(msg.data[0]), float(msg.data[1])])

    def _cb_r_est(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self._t_r_est.append(self._now())
            self._r_est.append([float(msg.data[0]), float(msg.data[1])])

    def _finish_once(self):
        if self._done:
            return
        self._done = True
        self._save()

    def _save(self):
        def arr(t, x):
            return (np.asarray(t, dtype=np.float64), np.asarray(x, dtype=np.float64))

        t_h_cmd, h_cmd = arr(self._t_h_cmd, self._h_cmd)
        t_r_cmd, r_cmd = arr(self._t_r_cmd, self._r_cmd)
        t_h_est, h_est = arr(self._t_h_est, self._h_est)
        t_r_est, r_est = arr(self._t_r_est, self._r_est)

        ts = time.strftime("%Y%m%d_%H%M%S")
        png = os.path.join(self.out_dir, f"ekf_eval_{ts}.png")
        npz = os.path.join(self.out_dir, f"ekf_eval_{ts}.npz")

        np.savez(
            npz,
            t_h_cmd=t_h_cmd, h_cmd=h_cmd,
            t_r_cmd=t_r_cmd, r_cmd=r_cmd,
            t_h_est=t_h_est, h_est=h_est,
            t_r_est=t_r_est, r_est=r_est,
        )

        fig, axs = plt.subplots(2, 2, figsize=(12, 6), dpi=150)

        # Human v
        axs[0, 0].plot(t_h_cmd, h_cmd[:, 0] if h_cmd.size else [], label="human cmd v")
        axs[0, 0].plot(t_h_est, h_est[:, 0] if h_est.size else [], label="human EKF v")
        axs[0, 0].set_title("Human linear velocity v")
        axs[0, 0].grid(True); axs[0, 0].legend()

        # Robot v
        axs[0, 1].plot(t_r_cmd, r_cmd[:, 0] if r_cmd.size else [], label="robot cmd v")
        axs[0, 1].plot(t_r_est, r_est[:, 0] if r_est.size else [], label="robot EKF v")
        axs[0, 1].set_title("Robot linear velocity v")
        axs[0, 1].grid(True); axs[0, 1].legend()

        # Human w
        axs[1, 0].plot(t_h_cmd, h_cmd[:, 1] if h_cmd.size else [], label="human cmd w")
        axs[1, 0].plot(t_h_est, h_est[:, 1] if h_est.size else [], label="human EKF w")
        axs[1, 0].set_title("Human angular velocity w")
        axs[1, 0].grid(True); axs[1, 0].legend()

        # Robot w
        axs[1, 1].plot(t_r_cmd, r_cmd[:, 1] if r_cmd.size else [], label="robot cmd w")
        axs[1, 1].plot(t_r_est, r_est[:, 1] if r_est.size else [], label="robot EKF w")
        axs[1, 1].set_title("Robot angular velocity w")
        axs[1, 1].grid(True); axs[1, 1].legend()

        fig.tight_layout()
        fig.savefig(png)
        plt.close(fig)

        self.get_logger().info("Saved:")
        self.get_logger().info(f"  {png}")
        self.get_logger().info(f"  {npz}")
        self.get_logger().info("Done. (Launch keeps running; Ctrl+C when you’re finished.)")


def main():
    rclpy.init()
    node = EKFEvalLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

