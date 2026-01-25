import math
import numpy as np


def wrap_angle(th: float) -> float:
    return (th + math.pi) % (2.0 * math.pi) - math.pi


class UnicycleAugmentedEKF:
    """
    EKF with augmented input-as-state for unicycle model.

    State:
      x = [px, py, yaw, v, w]^T

    Measurement (from mocap):
      z = [px, py, yaw]^T
    """

    def __init__(
        self,
        *,
        q_diag: np.ndarray,     # shape (5,) process noise variances
        r_diag: np.ndarray,     # shape (3,) meas noise variances
        p0_diag: np.ndarray,    # shape (5,) initial covariance diag
    ):
        self.q_diag = np.asarray(q_diag, dtype=float).reshape(5,)
        self.r_diag = np.asarray(r_diag, dtype=float).reshape(3,)
        self.Q = np.diag(self.q_diag)
        self.R = np.diag(self.r_diag)

        self.x = np.zeros((5,), dtype=float)
        self.P = np.diag(np.asarray(p0_diag, dtype=float).reshape(5,))
        self.initialized = False

    def reset_from_measurement(self, px: float, py: float, yaw: float, v0: float = 0.0, w0: float = 0.0):
        self.x[:] = [float(px), float(py), wrap_angle(float(yaw)), float(v0), float(w0)]
        self.initialized = True

    def predict(self, dt: float):
        dt = float(dt)
        px, py, yaw, v, w = self.x

        # Nonlinear motion model
        px2  = px + dt * v * math.cos(yaw)
        py2  = py + dt * v * math.sin(yaw)
        yaw2 = wrap_angle(yaw + dt * w)
        v2, w2 = v, w

        self.x[:] = [px2, py2, yaw2, v2, w2]

        # Jacobian F = df/dx
        F = np.eye(5, dtype=float)
        F[0, 2] = -dt * v * math.sin(yaw)
        F[0, 3] =  dt * math.cos(yaw)
        F[1, 2] =  dt * v * math.cos(yaw)
        F[1, 3] =  dt * math.sin(yaw)
        F[2, 4] =  dt

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update(self, px_meas: float, py_meas: float, yaw_meas: float):
        z = np.array([float(px_meas), float(py_meas), wrap_angle(float(yaw_meas))], dtype=float)

        # h(x) = [px, py, yaw]
        hx = self.x[:3].copy()
        y = z - hx
        y[2] = wrap_angle(y[2])  # yaw innovation wrapped

        # H = dh/dx
        H = np.zeros((3, 5), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])

        # Joseph form covariance update (more numerically stable)
        I = np.eye(5, dtype=float)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

    def step(self, *, px: float, py: float, yaw: float, dt: float):
        if not self.initialized:
            self.reset_from_measurement(px, py, yaw)
            return

        self.predict(dt)
        self.update(px, py, yaw)

    @property
    def pose2d(self):
        return self.x[0], self.x[1], self.x[2]

    @property
    def action(self):
        return self.x[3], self.x[4]

