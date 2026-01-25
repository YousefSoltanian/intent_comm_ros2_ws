#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


def _quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _yaw_to_quat(yaw: float):
    # planar yaw-only quaternion
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))  # x,y,z,w


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class MocapNoiserAgentNode(Node):
    def __init__(self):
        super().__init__("mocap_noiser_agent")

        self.input_topic = str(self.declare_parameter("input_topic", "/truth/human/pose").value)
        self.output_topic = str(self.declare_parameter("output_topic", "/mocap/human/pose").value)

        self.sigma_xy = float(self.declare_parameter("sigma_xy", 0.005).value)
        self.sigma_theta = float(self.declare_parameter("sigma_theta", 0.002).value)
        self.seed = int(self.declare_parameter("seed", 1).value)

        self.rng = np.random.default_rng(self.seed)

        self.pub = self.create_publisher(PoseStamped, self.output_topic, 20)
        self.sub = self.create_subscription(PoseStamped, self.input_topic, self._on_pose, 50)

        self.get_logger().info("MocapNoiserAgentNode started.")
        self.get_logger().info(f"  input:  {self.input_topic}")
        self.get_logger().info(f"  output: {self.output_topic}")
        self.get_logger().info(f"  sigma_xy={self.sigma_xy}, sigma_theta={self.sigma_theta}, seed={self.seed}")

    def _on_pose(self, msg: PoseStamped) -> None:
        out = PoseStamped()
        out.header = msg.header

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = float(_quat_to_yaw(msg.pose.orientation))

        nx = self.rng.normal(0.0, self.sigma_xy)
        ny = self.rng.normal(0.0, self.sigma_xy)
        nth = self.rng.normal(0.0, self.sigma_theta)

        x2 = x + nx
        y2 = y + ny
        yaw2 = _wrap_pi(yaw + nth)

        out.pose.position.x = x2
        out.pose.position.y = y2
        out.pose.position.z = msg.pose.position.z

        qx, qy, qz, qw = _yaw_to_quat(yaw2)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MocapNoiserAgentNode()
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

