import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


def wrap_angle(th: float) -> float:
    return (th + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    # yaw-only expected, but use general formula
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quat_from_yaw(yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, float(sy), float(cy))


class MocapNoiserNode(Node):
    """
    Takes clean truth PoseStamped and outputs noisy mocap PoseStamped.

    Subscribes:
      /truth/human/pose
      /truth/robot/pose

    Publishes:
      /mocap/human/pose
      /mocap/robot/pose

    Noise:
      x,y += N(0, sigma_xy)
      theta += N(0, sigma_theta)
    """

    def __init__(self):
        super().__init__("mocap_noiser")

        # Parameters
        self.declare_parameter("sigma_xy", 0.01)       # meters
        self.declare_parameter("sigma_theta", 0.005)   # radians
        self.declare_parameter("seed", 0)

        self.declare_parameter("truth_human_topic", "/truth/human/pose")
        self.declare_parameter("truth_robot_topic", "/truth/robot/pose")
        self.declare_parameter("mocap_human_topic", "/mocap/human/pose")
        self.declare_parameter("mocap_robot_topic", "/mocap/robot/pose")

        self.sigma_xy = float(self.get_parameter("sigma_xy").value)
        self.sigma_theta = float(self.get_parameter("sigma_theta").value)
        seed = int(self.get_parameter("seed").value)

        self.truth_human_topic = str(self.get_parameter("truth_human_topic").value)
        self.truth_robot_topic = str(self.get_parameter("truth_robot_topic").value)
        self.mocap_human_topic = str(self.get_parameter("mocap_human_topic").value)
        self.mocap_robot_topic = str(self.get_parameter("mocap_robot_topic").value)

        self.rng = np.random.default_rng(seed)

        # Publishers
        self.pub_h = self.create_publisher(PoseStamped, self.mocap_human_topic, 10)
        self.pub_r = self.create_publisher(PoseStamped, self.mocap_robot_topic, 10)

        # Subscribers
        self.sub_h = self.create_subscription(PoseStamped, self.truth_human_topic, self._on_truth_h, 10)
        self.sub_r = self.create_subscription(PoseStamped, self.truth_robot_topic, self._on_truth_r, 10)

        self.get_logger().info(
            f"MocapNoiserNode started.\n"
            f"  sigma_xy={self.sigma_xy}, sigma_theta={self.sigma_theta}, seed={seed}\n"
            f"  truth topics: {self.truth_human_topic}, {self.truth_robot_topic}\n"
            f"  mocap topics: {self.mocap_human_topic}, {self.mocap_robot_topic}"
        )

    def _noisify_pose(self, msg: PoseStamped) -> PoseStamped:
        out = PoseStamped()
        out.header = msg.header  # keep same stamp/frame

        x = msg.pose.position.x
        y = msg.pose.position.y
        th = yaw_from_quat(msg.pose.orientation)

        nx = x + self.rng.normal(0.0, self.sigma_xy)
        ny = y + self.rng.normal(0.0, self.sigma_xy)
        nth = wrap_angle(th + self.rng.normal(0.0, self.sigma_theta))

        out.pose.position.x = float(nx)
        out.pose.position.y = float(ny)
        out.pose.position.z = msg.pose.position.z

        qx, qy, qz, qw = quat_from_yaw(nth)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw

        return out

    def _on_truth_h(self, msg: PoseStamped):
        self.pub_h.publish(self._noisify_pose(msg))

    def _on_truth_r(self, msg: PoseStamped):
        self.pub_r.publish(self._noisify_pose(msg))


def main():
    rclpy.init()
    node = MocapNoiserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

