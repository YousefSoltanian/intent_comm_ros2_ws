import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelSineNode(Node):
    """
    Publishes sinusoidal / scripted cmd_vel to both agents.

    Publishes:
      /human/cmd_vel
      /robot/cmd_vel
    """

    def __init__(self):
        super().__init__("cmd_vel_sine")

        self.declare_parameter("rate_hz", 20.0)

        self.declare_parameter("human_topic", "/human/cmd_vel")
        self.declare_parameter("robot_topic", "/robot/cmd_vel")

        self.declare_parameter("v0", 0.6)
        self.declare_parameter("v_amp", 0.2)
        self.declare_parameter("v_freq_hz", 0.15)

        self.declare_parameter("w0", 0.0)
        self.declare_parameter("w_amp", 0.25)
        self.declare_parameter("w_freq_hz", 0.10)

        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.human_topic = str(self.get_parameter("human_topic").value)
        self.robot_topic = str(self.get_parameter("robot_topic").value)

        self.pub_h = self.create_publisher(Twist, self.human_topic, 10)
        self.pub_r = self.create_publisher(Twist, self.robot_topic, 10)

        self.t0 = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(f"CmdVelSineNode publishing at {self.rate_hz} Hz")

    def _tick(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9

        v0 = float(self.get_parameter("v0").value)
        vA = float(self.get_parameter("v_amp").value)
        vf = float(self.get_parameter("v_freq_hz").value)

        w0 = float(self.get_parameter("w0").value)
        wA = float(self.get_parameter("w_amp").value)
        wf = float(self.get_parameter("w_freq_hz").value)

        v = v0 + vA * math.sin(2.0 * math.pi * vf * t)
        w = w0 + wA * math.sin(2.0 * math.pi * wf * t + 0.7)

        # human
        mh = Twist()
        mh.linear.x = float(v)
        mh.angular.z = float(w)
        self.pub_h.publish(mh)

        # robot (slightly phase-shifted to make it different)
        mr = Twist()
        mr.linear.x = float(v0 + vA * math.sin(2.0 * math.pi * vf * t + 1.2))
        mr.angular.z = float(w0 + wA * math.sin(2.0 * math.pi * wf * t + 2.0))
        self.pub_r.publish(mr)


def main():
    rclpy.init()
    node = CmdVelSineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

