# intent_comm_nav_ros/nodes/hl_downsample_node.py
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class HLDownsampleNode(Node):
    def __init__(self):
        super().__init__("hl_downsample")

        self.declare_parameter("dt_hl", 0.5)
        self.declare_parameter("state_in", "/ekf/stacked_state")
        self.declare_parameter("human_u_in", "/ekf/human/u_est")
        self.declare_parameter("robot_u_in", "/ekf/robot/u_est")

        self.declare_parameter("state_out", "/hl/stacked_state")
        self.declare_parameter("human_u_out", "/hl/human/u_obs")
        self.declare_parameter("robot_u_out", "/hl/robot/u_obs")

        self.dt_hl = float(self.get_parameter("dt_hl").value)

        self.state_in = str(self.get_parameter("state_in").value)
        self.hu_in = str(self.get_parameter("human_u_in").value)
        self.ru_in = str(self.get_parameter("robot_u_in").value)

        self.state_out = str(self.get_parameter("state_out").value)
        self.hu_out = str(self.get_parameter("human_u_out").value)
        self.ru_out = str(self.get_parameter("robot_u_out").value)

        self._x_latest = None
        self._hu_buf = deque()
        self._ru_buf = deque()

        self.pub_x = self.create_publisher(Float32MultiArray, self.state_out, 10)
        self.pub_hu = self.create_publisher(Float32MultiArray, self.hu_out, 10)
        self.pub_ru = self.create_publisher(Float32MultiArray, self.ru_out, 10)

        self.sub_x = self.create_subscription(Float32MultiArray, self.state_in, self._on_x, 10)
        self.sub_hu = self.create_subscription(Float32MultiArray, self.hu_in, self._on_hu, 10)
        self.sub_ru = self.create_subscription(Float32MultiArray, self.ru_in, self._on_ru, 10)

        self.timer = self.create_timer(self.dt_hl, self._tick)

        self.get_logger().info(f"HLDownsampleNode: dt_hl={self.dt_hl}s")
        self.get_logger().info(f"  in:  {self.state_in}, {self.hu_in}, {self.ru_in}")
        self.get_logger().info(f"  out: {self.state_out}, {self.hu_out}, {self.ru_out}")

    def _now(self) -> float:
        return time.time()

    def _on_x(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float64)
        if arr.size >= 6:
            self._x_latest = arr[:6]

    def _on_hu(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float64)
        if arr.size >= 2:
            self._hu_buf.append((self._now(), arr[:2]))

    def _on_ru(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float64)
        if arr.size >= 2:
            self._ru_buf.append((self._now(), arr[:2]))

    def _mean_in_window(self, buf: deque) -> np.ndarray:
        t_now = self._now()
        t_min = t_now - self.dt_hl

        while buf and buf[0][0] < t_min:
            buf.popleft()

        if not buf:
            return np.zeros(2, dtype=np.float64)

        vals = np.stack([v for (_, v) in buf], axis=0)
        return vals.mean(axis=0)

    def _tick(self):
        if self._x_latest is None:
            return

        # publish latest state
        xmsg = Float32MultiArray()
        xmsg.data = [float(v) for v in self._x_latest.tolist()]
        self.pub_x.publish(xmsg)

        # publish window-mean observed actions
        hu = self._mean_in_window(self._hu_buf)
        rum = self._mean_in_window(self._ru_buf)

        humsg = Float32MultiArray()
        humsg.data = [float(hu[0]), float(hu[1])]
        self.pub_hu.publish(humsg)

        rumsg = Float32MultiArray()
        rumsg.data = [float(rum[0]), float(rum[1])]
        self.pub_ru.publish(rumsg)

def main():
    rclpy.init()
    node = HLDownsampleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

