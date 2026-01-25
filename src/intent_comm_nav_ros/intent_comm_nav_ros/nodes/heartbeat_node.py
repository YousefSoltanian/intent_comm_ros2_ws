import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HeartbeatNode(Node):
    def __init__(self):
        super().__init__("intent_comm_heartbeat")
        self.pub = self.create_publisher(String, "/intent_comm/heartbeat", 10)
        self.timer = self.create_timer(1.0, self._tick)
        self.count = 0
        self.get_logger().info("Heartbeat node started. Publishing on /intent_comm/heartbeat")

    def _tick(self):
        msg = String()
        msg.data = f"alive: {self.count}"
        self.pub.publish(msg)
        self.count += 1


def main():
    rclpy.init()
    node = HeartbeatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

