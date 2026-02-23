#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray


# def wrap_angle(a: float) -> float:
#     return (a + math.pi) % (2 * math.pi) - math.pi

def wrap_angle(a: float) -> float:
    return a


def quat_to_yaw(q) -> float:
    
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    
    return yaw + math.pi/2


class EKF5:
    """
    State: x = [px, py, theta, v, w]
    Meas : z = [px, py, theta]
    Model (discrete):
      px'    = px + dt * v * cos(theta)
      py'    = py + dt * v * sin(theta)
      th'    = th + dt * w
      v'     = v
      w'     = w
    """

    def __init__(self, P0: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.P0 = P0.astype(np.float64)
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.x = None  # np.ndarray (5,)
        self.P = None  # np.ndarray (5,5)

    def initialize_from_meas(self, z: np.ndarray):
        self.x = np.array([z[0], z[1], z[2], 0.0, 0.0], dtype=np.float64)
        self.P = self.P0.copy()

    def predict(self, dt: float):
        if self.x is None:
            return
        px, py, th, v, w = self.x
        c = math.cos(th)
        s = math.sin(th)

        px2 = px + dt * v * c
        py2 = py + dt * v * s
        th2 = wrap_angle(th + dt * w)
        v2 = v
        w2 = w
        self.x = np.array([px2, py2, th2, v2, w2], dtype=np.float64)

        F = np.eye(5, dtype=np.float64)
        F[0, 2] = -dt * v * s
        F[0, 3] = dt * c
        F[1, 2] = dt * v * c
        F[1, 3] = dt * s
        F[2, 4] = dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray):
        if self.x is None:
            self.initialize_from_meas(z)
            return

        H = np.zeros((3, 5), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        y = z.astype(np.float64) - self.x[:3]
        y[2] = wrap_angle(y[2])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])

        I = np.eye(5, dtype=np.float64)
        self.P = (I - K @ H) @ self.P


class EKFStackNode(Node):
    def __init__(self):
        super().__init__("ekf_stack")

        self.declare_parameter("ekf_rate", 60.0)
        self.get_logger().info("*********************** EKF Rate")
        self.get_logger().info(str(self.get_parameter("ekf_rate").value))
                
        # Covariances (variances, not std)
        self.declare_parameter("p0_xy", 0.5)
        self.declare_parameter("p0_th", 0.5)
        self.declare_parameter("p0_v", 1.0)
        self.declare_parameter("p0_w", 1.0)

        self.declare_parameter("q_xy", 1e-4)
        self.declare_parameter("q_th", 1e-4)
        self.declare_parameter("q_v", 1e-2)
        self.declare_parameter("q_w", 1e-2)

        self.declare_parameter("r_xy", 1e-4)
        self.declare_parameter("r_th", 1e-3)

        self.declare_parameter("freeze_vel_s", 0.5)
        self.declare_parameter("freeze_vel_cov", 1e-4)
        self.declare_parameter("max_v", 3.0)
        self.declare_parameter("max_w", 3.0)

        # Topics (these are the ONLY names your launch should set)
        self.topic_h_pose = str(self.declare_parameter("topic_h_pose", "/mocap/human/pose").value)
        self.topic_r_pose = str(self.declare_parameter("topic_r_pose", "/mocap/robot/pose").value)

        self.topic_x_hat = str(self.declare_parameter("topic_x_hat", "/ekf/stacked_state").value)
        self.topic_h_u_est = str(self.declare_parameter("topic_h_u_est", "/ekf/human/u_est").value)
        self.topic_r_u_est = str(self.declare_parameter("topic_r_u_est", "/ekf/robot/u_est").value)

        rate = float(self.get_parameter("ekf_rate").value)
        self._dt = 1.0 / max(1e-6, rate)

        p0_xy = float(self.get_parameter("p0_xy").value)
        p0_th = float(self.get_parameter("p0_th").value)
        p0_v = float(self.get_parameter("p0_v").value)
        p0_w = float(self.get_parameter("p0_w").value)

        q_xy = float(self.get_parameter("q_xy").value)
        q_th = float(self.get_parameter("q_th").value)
        q_v = float(self.get_parameter("q_v").value)
        q_w = float(self.get_parameter("q_w").value)

        r_xy = float(self.get_parameter("r_xy").value)
        r_th = float(self.get_parameter("r_th").value)

        P0 = np.diag([p0_xy, p0_xy, p0_th, p0_v, p0_w])
        Q = np.diag([q_xy, q_xy, q_th, q_v, q_w])
        R = np.diag([r_xy, r_xy, r_th])
        
        Qh = np.diag([q_xy, q_xy, q_th*1e-2, q_v, q_w*1e-4])

        self.freeze_vel_s = float(self.get_parameter("freeze_vel_s").value)
        self.freeze_vel_cov = float(self.get_parameter("freeze_vel_cov").value)
        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)

        self.ekf_h = EKF5(P0=P0, Q=Qh, R=R)
        self.ekf_r = EKF5(P0=P0, Q=Q, R=R)

        self._z_h = None
        self._z_r = None
        self._new_h = False
        self._new_r = False

        self._init_time_h = None
        self._init_time_r = None

        self.sub_h = self.create_subscription(PoseStamped, self.topic_h_pose, self._on_h_pose, 50)
        self.sub_r = self.create_subscription(PoseStamped, self.topic_r_pose, self._on_r_pose, 50)

        self.pub_x = self.create_publisher(Float32MultiArray, self.topic_x_hat, 10)
        self.pub_uh = self.create_publisher(Float32MultiArray, self.topic_h_u_est, 10)
        self.pub_ur = self.create_publisher(Float32MultiArray, self.topic_r_u_est, 10)

        self.timer = self.create_timer(self._dt, self._tick_ekf)

        self.get_logger().info(
            f"[EKF] rate={rate:.1f}Hz dt={self._dt:.4f}s freeze_vel_s={self.freeze_vel_s:.3f} "
            f"max_v={self.max_v:.2f} max_w={self.max_w:.2f} | x_hat={self.topic_x_hat} "
            f"| h_pose={self.topic_h_pose} r_pose={self.topic_r_pose}"
        )

    def _on_h_pose(self, msg: PoseStamped):
        new_x = -msg.pose.position.y
        new_y = msg.pose.position.x
        p = msg.pose.position
        p.x=new_x
        p.y=new_y
        yaw = quat_to_yaw(msg.pose.orientation)
        self._z_h = np.array([p.x, p.y, yaw], dtype=np.float64)
        self._new_h = True

    def _on_r_pose(self, msg: PoseStamped):
        new_x = -msg.pose.position.y
        new_y = msg.pose.position.x
        p = msg.pose.position
        p.x=new_x
        p.y = new_y
        yaw = quat_to_yaw(msg.pose.orientation)
        self._z_r = np.array([p.x, p.y, yaw], dtype=np.float64)
        self._new_r = True

    def _clamp_and_freeze(self, ekf: EKF5, init_time: float, now: float):
        if ekf.x is None:
            return

        ekf.x[3] = float(np.clip(ekf.x[3], -self.max_v, self.max_v))
        ekf.x[4] = float(np.clip(ekf.x[4], -self.max_w, self.max_w))

        if self.freeze_vel_s > 0.0 and (now - init_time) <= self.freeze_vel_s:
            ekf.x[3] = 0.0
            ekf.x[4] = 0.0
            if ekf.P is not None:
                ekf.P[3, :] = 0.0
                ekf.P[:, 3] = 0.0
                ekf.P[4, :] = 0.0
                ekf.P[:, 4] = 0.0
                ekf.P[3, 3] = min(float(ekf.P[3, 3]), self.freeze_vel_cov)
                ekf.P[4, 4] = min(float(ekf.P[4, 4]), self.freeze_vel_cov)

    def _tick_ekf(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        self.ekf_h.predict(self._dt)
        self.ekf_r.predict(self._dt)

        if self._new_h and self._z_h is not None:
            was_none = self.ekf_h.x is None
            self.ekf_h.update(self._z_h)
            if was_none and self.ekf_h.x is not None and self._init_time_h is None:
                self._init_time_h = now
            self._new_h = False

        if self._new_r and self._z_r is not None:
            was_none = self.ekf_r.x is None
            self.ekf_r.update(self._z_r)
            if was_none and self.ekf_r.x is not None and self._init_time_r is None:
                self._init_time_r = now
            self._new_r = False

        if self.ekf_h.x is not None and self._init_time_h is not None:
            self._clamp_and_freeze(self.ekf_h, self._init_time_h, now)
        if self.ekf_r.x is not None and self._init_time_r is not None:
            self._clamp_and_freeze(self.ekf_r, self._init_time_r, now)

        if self.ekf_h.x is None or self.ekf_r.x is None:
            return

        x_hat = np.array(
            [self.ekf_h.x[0], self.ekf_h.x[1], self.ekf_h.x[2],
             self.ekf_r.x[0], self.ekf_r.x[1], self.ekf_r.x[2]],
            dtype=np.float32,
        )
        msg_x = Float32MultiArray()
        msg_x.data = x_hat.tolist()
        self.pub_x.publish(msg_x)

        uh = Float32MultiArray()
        ur = Float32MultiArray()
        uh.data = [float(self.ekf_h.x[3]), float(self.ekf_h.x[4])]
        ur.data = [float(self.ekf_r.x[3]), float(self.ekf_r.x[4])]
        self.pub_uh.publish(uh)
        self.pub_ur.publish(ur)


def main(args=None):
    rclpy.init(args=args)
    node = EKFStackNode()
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

