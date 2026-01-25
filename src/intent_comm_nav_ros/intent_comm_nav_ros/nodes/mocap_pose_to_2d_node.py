#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose2D
from std_msgs.msg import Float32MultiArray


def _quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MocapPoseTo2DNode(Node):
    def __init__(self):
        super().__init__("mocap_pose_to_2d")

        self.in_topic = str(self.declare_parameter("in_topic", "/mocap/human/pose").value)
        self.out_pose2d = str(self.declare_parameter("output_pose2d_topic", "/mocap/human/pose2d").value)
        self.out_rpy = str(self.declare_parameter("output_rpy_topic", "/mocap/human/rpy").value)

        self.pub_pose2d = self.create_publisher(Pose2D, self.out_pose2d, 20)
        self.pub_rpy = self.create_publisher(Float32MultiArray, self.out_rpy, 20)

        self.sub = self.create_subscription(PoseStamped, self.in_topic, self._on_pose, 50)

        self.get_logger().info(f"Pose->2D started. in={self.in_topic}")
        self.get_logger().info(f"  out Pose2D: {self.out_pose2d}")
        self.get_logger().info(f"  out RPY:    {self.out_rpy}")

    def _on_pose(self, msg: PoseStamped) -> None:
        yaw = float(_quat_to_yaw(msg.pose.orientation))

        p2d = Pose2D()
        p2d.x = float(msg.pose.position.x)
        p2d.y = float(msg.pose.position.y)
        p2d.theta = yaw
        self.pub_pose2d.publish(p2d)

        rpy = Float32MultiArray()
        rpy.data = [0.0, 0.0, yaw]
        self.pub_rpy.publish(rpy)


def main(args=None):
    rclpy.init(args=args)
    node = MocapPoseTo2DNode()
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

