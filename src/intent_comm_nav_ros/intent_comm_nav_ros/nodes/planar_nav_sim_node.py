#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist


def yaw_to_quat(yaw: float):
    # planar yaw-only quaternion (x,y,z,w)
    h = 0.5 * yaw
    return (0.0, 0.0, math.sin(h), math.cos(h))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PlanarNavSimNode(Node):
    """
    Two-agent planar simulator (unicycle model):
      x_dot = v cos(theta)
      y_dot = v sin(theta)
      theta_dot = w

    Publishes PoseStamped truth and consumes Twist commands.
    """

    def __init__(self):
        super().__init__("planar_nav_sim")

        # timing
        self.dt_sim = float(self.declare_parameter("dt_sim", 0.01).value)
        self.pose_pub_rate = float(self.declare_parameter("pose_pub_rate", 120.0).value)

        # Topics (support both your launch params and safe defaults)
        self.topic_truth_h = str(self.declare_parameter("topic_truth_h", "/truth/human/pose").value)
        self.topic_truth_r = str(self.declare_parameter("topic_truth_r", "/truth/robot/pose").value)

        # Your launch uses cmd_h_topic/cmd_r_topic; keep compatibility
        self.cmd_h_topic = str(self.declare_parameter("cmd_h_topic", "/human/cmd_vel").value)
        self.cmd_r_topic = str(self.declare_parameter("cmd_r_topic", "/robot/cmd_vel").value)

        # Initial states
        self.xh = float(self.declare_parameter("x0_h", -4.0).value)
        self.yh = float(self.declare_parameter("y0_h",  0.0).value)
        self.th_h = float(self.declare_parameter("th0_h", 0.0).value)

        self.xr = float(self.declare_parameter("x0_r",  4.0).value)
        self.yr = float(self.declare_parameter("y0_r",  0.0).value)
        self.th_r = float(self.declare_parameter("th0_r", math.pi).value)

        # Command limits (optional)
        self.v_max = float(self.declare_parameter("v_max", 2.0).value)
        self.w_max = float(self.declare_parameter("w_max", 2.0).value)

        # Current commands
        self.uh_v = 0.0
        self.uh_w = 0.0
        self.ur_v = 0.0
        self.ur_w = 0.0

        # Debug counters
        self._truth_pub_count = 0
        self._cmd_h_count = 0
        self._cmd_r_count = 0

        # pubs/subs
        self.pub_h = self.create_publisher(PoseStamped, self.topic_truth_h, 10)
        self.pub_r = self.create_publisher(PoseStamped, self.topic_truth_r, 10)

        self.sub_h = self.create_subscription(Twist, self.cmd_h_topic, self._on_cmd_h, 50)
        self.sub_r = self.create_subscription(Twist, self.cmd_r_topic, self._on_cmd_r, 50)

        # timers
        self.sim_timer = self.create_timer(self.dt_sim, self._step_sim)

        pub_dt = 1.0 / max(1e-6, self.pose_pub_rate)
        self.pub_timer = self.create_timer(pub_dt, self._publish_truth)

        self.dbg_timer = self.create_timer(1.0, self._debug_heartbeat)

        self.get_logger().info(
            f"[planar_nav_sim] dt_sim={self.dt_sim:.4f} pose_pub_rate={self.pose_pub_rate:.1f}\n"
            f"  truth_h={self.topic_truth_h}\n"
            f"  truth_r={self.topic_truth_r}\n"
            f"  cmd_h={self.cmd_h_topic}\n"
            f"  cmd_r={self.cmd_r_topic}\n"
            f"  init_h=({self.xh:.2f},{self.yh:.2f},{self.th_h:.2f}) init_r=({self.xr:.2f},{self.yr:.2f},{self.th_r:.2f})"
        )

    def _on_cmd_h(self, msg: Twist):
        self._cmd_h_count += 1
        self.uh_v = float(max(-self.v_max, min(self.v_max, msg.linear.x)))
        self.uh_w = float(max(-self.w_max, min(self.w_max, msg.angular.z)))

    def _on_cmd_r(self, msg: Twist):
        self._cmd_r_count += 1
        self.ur_v = float(max(-self.v_max, min(self.v_max, msg.linear.x)))
        self.ur_w = float(max(-self.w_max, min(self.w_max, msg.angular.z)))

    def _step_sim(self):
        dt = self.dt_sim

        # human
        self.xh += dt * self.uh_v * math.cos(self.th_h)
        self.yh += dt * self.uh_v * math.sin(self.th_h)
        self.th_h = wrap_pi(self.th_h + dt * self.uh_w)

        # robot
        self.xr += dt * self.ur_v * math.cos(self.th_r)
        self.yr += dt * self.ur_v * math.sin(self.th_r)
        self.th_r = wrap_pi(self.th_r + dt * self.ur_w)

    def _publish_truth(self):
        now = self.get_clock().now().to_msg()

        mh = PoseStamped()
        mh.header.stamp = now
        mh.header.frame_id = "map"
        mh.pose.position.x = float(self.xh)
        mh.pose.position.y = float(self.yh)
        qx, qy, qz, qw = yaw_to_quat(self.th_h)
        mh.pose.orientation.x = qx
        mh.pose.orientation.y = qy
        mh.pose.orientation.z = qz
        mh.pose.orientation.w = qw

        mr = PoseStamped()
        mr.header.stamp = now
        mr.header.frame_id = "map"
        mr.pose.position.x = float(self.xr)
        mr.pose.position.y = float(self.yr)
        qx, qy, qz, qw = yaw_to_quat(self.th_r)
        mr.pose.orientation.x = qx
        mr.pose.orientation.y = qy
        mr.pose.orientation.z = qz
        mr.pose.orientation.w = qw

        self.pub_h.publish(mh)
        self.pub_r.publish(mr)
        self._truth_pub_count += 1

    def _debug_heartbeat(self):
        self.get_logger().info(
            f"[planar_nav_sim] truth_pub={self._truth_pub_count}/s | "
            f"cmd_h_rx={self._cmd_h_count}/s cmd_r_rx={self._cmd_r_count}/s | "
            f"u_h=[{self.uh_v:+.2f},{self.uh_w:+.2f}] u_r=[{self.ur_v:+.2f},{self.ur_w:+.2f}] | "
            f"h=({self.xh:+.2f},{self.yh:+.2f},{self.th_h:+.2f}) r=({self.xr:+.2f},{self.yr:+.2f},{self.th_r:+.2f})"
        )
        # reset per-second counters
        self._truth_pub_count = 0
        self._cmd_h_count = 0
        self._cmd_r_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = PlanarNavSimNode()
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

