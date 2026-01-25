#!/usr/bin/env python3
"""high_level_bridge_node.py

Bridges high-rate EKF outputs to low-rate (dt_hl) messages for high-level control.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


def _u_to_twist(u: np.ndarray) -> Twist:
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    msg = Twist()
    msg.linear.x = float(u[0]) if u.size > 0 else 0.0
    msg.angular.z = float(u[1]) if u.size > 1 else 0.0
    return msg


def _pick_topic(preferred: str, preferred_default: str, legacy: str, legacy_default: str, node: Node) -> str:
    pref = str(node.declare_parameter(preferred, preferred_default).value)
    leg  = str(node.declare_parameter(legacy, legacy_default).value)
    return pref if pref else leg


class HighLevelBridgeNode(Node):
    def __init__(self):
        super().__init__("high_level_bridge")

        self.dt_hl = float(self.declare_parameter("dt_hl", 0.5).value)

        self.topic_x_hat_in = _pick_topic(
            preferred="x_hat_in_topic", preferred_default="",
            legacy="topic_x_hat_in", legacy_default="/ekf/stacked_state",
            node=self,
        )
        self.topic_uh_in = _pick_topic(
            preferred="human_u_est_topic", preferred_default="",
            legacy="topic_u_h_in", legacy_default="/ekf/human/u_est",
            node=self,
        )
        self.topic_ur_in = _pick_topic(
            preferred="robot_u_est_topic", preferred_default="",
            legacy="topic_u_r_in", legacy_default="/ekf/robot/u_est",
            node=self,
        )

        self.topic_x_hat_out = _pick_topic(
            preferred="x_hat_topic", preferred_default="",
            legacy="topic_x_hat_out", legacy_default="/hl/x_hat",
            node=self,
        )
        self.topic_x_hat_prev_out = _pick_topic(
            preferred="x_hat_prev_topic", preferred_default="",
            legacy="topic_x_hat_prev_out", legacy_default="/hl/x_hat_prev",
            node=self,
        )
        self.topic_uh_obs_out = _pick_topic(
            preferred="human_u_obs_topic", preferred_default="",
            legacy="topic_h_u_obs_out", legacy_default="/hl/human/u_obs",
            node=self,
        )
        self.topic_ur_obs_out = _pick_topic(
            preferred="robot_u_obs_topic", preferred_default="",
            legacy="topic_r_u_obs_out", legacy_default="/hl/robot/u_obs",
            node=self,
        )

        self.sub_x = self.create_subscription(Float32MultiArray, self.topic_x_hat_in, self._on_x, 20)
        self.sub_uh = self.create_subscription(Float32MultiArray, self.topic_uh_in, self._on_uh, 50)
        self.sub_ur = self.create_subscription(Float32MultiArray, self.topic_ur_in, self._on_ur, 50)

        self.pub_x = self.create_publisher(Float32MultiArray, self.topic_x_hat_out, 10)
        self.pub_x_prev = self.create_publisher(Float32MultiArray, self.topic_x_hat_prev_out, 10)
        self.pub_uh_obs = self.create_publisher(Twist, self.topic_uh_obs_out, 10)
        self.pub_ur_obs = self.create_publisher(Twist, self.topic_ur_obs_out, 10)

        self._x_latest: np.ndarray | None = None
        self._x_prev: np.ndarray | None = None

        self._uh_latest = np.zeros(2, dtype=np.float32)
        self._ur_latest = np.zeros(2, dtype=np.float32)

        self._uh_sum = np.zeros(2, dtype=np.float64)
        self._ur_sum = np.zeros(2, dtype=np.float64)
        self._uh_count = 0
        self._ur_count = 0

        self._uh_prev_win: np.ndarray | None = None
        self._ur_prev_win: np.ndarray | None = None

        self.timer = self.create_timer(self.dt_hl, self._tick)

        self.get_logger().info(
            f"[HL Bridge] dt_hl={self.dt_hl:.3f} x_hat_in={self.topic_x_hat_in} "
            f"u_h_in={self.topic_uh_in} u_r_in={self.topic_ur_in} -> "
            f"x_hat={self.topic_x_hat_out} x_hat_prev={self.topic_x_hat_prev_out} "
            f"u_obs_h={self.topic_uh_obs_out} u_obs_r={self.topic_ur_obs_out}"
        )

    def _on_x(self, msg: Float32MultiArray) -> None:
        self._x_latest = np.asarray(msg.data, dtype=np.float32).reshape(-1)

    def _on_uh(self, msg: Float32MultiArray) -> None:
        u = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if u.size >= 2:
            u2 = u[:2].copy()
        elif u.size == 1:
            u2 = np.array([u[0], 0.0], dtype=np.float32)
        else:
            u2 = np.zeros(2, dtype=np.float32)
        self._uh_latest = u2
        self._uh_sum += u2.astype(np.float64)
        self._uh_count += 1

    def _on_ur(self, msg: Float32MultiArray) -> None:
        u = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if u.size >= 2:
            u2 = u[:2].copy()
        elif u.size == 1:
            u2 = np.array([u[0], 0.0], dtype=np.float32)
        else:
            u2 = np.zeros(2, dtype=np.float32)
        self._ur_latest = u2
        self._ur_sum += u2.astype(np.float64)
        self._ur_count += 1

    def _tick(self) -> None:
        if self._x_latest is None:
            return

        x = self._x_latest.copy()
        if self._x_prev is None:
            self._x_prev = x.copy()

        msg_x = Float32MultiArray()
        msg_x.data = x.astype(np.float32).tolist()
        self.pub_x.publish(msg_x)

        msg_prev = Float32MultiArray()
        msg_prev.data = self._x_prev.astype(np.float32).tolist()
        self.pub_x_prev.publish(msg_prev)

        uh_win = (self._uh_sum / max(self._uh_count, 1)).astype(np.float32) if self._uh_count > 0 else self._uh_latest
        ur_win = (self._ur_sum / max(self._ur_count, 1)).astype(np.float32) if self._ur_count > 0 else self._ur_latest

        if self._uh_prev_win is None:
            self._uh_prev_win = uh_win.copy()
            #self._uh_prev_win = self._uh_latest.copy()
        if self._ur_prev_win is None:
            self._ur_prev_win = ur_win.copy()
            #self._ur_prev_win = self._ur_latest.copy()
		
        self.pub_uh_obs.publish(_u_to_twist(self._uh_prev_win))
        self.pub_ur_obs.publish(_u_to_twist(self._ur_prev_win))

        self._uh_prev_win = uh_win.copy()
        self._ur_prev_win = ur_win.copy()

        self._uh_sum[:] = 0.0
        self._ur_sum[:] = 0.0
        self._uh_count = 0
        self._ur_count = 0

        self._x_prev = x.copy()


def main(args=None):
    rclpy.init(args=args)
    node = HighLevelBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

