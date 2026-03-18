#!/usr/bin/env python3
import os
import math
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray


def _quat_to_yaw(q) -> float:
    
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    
    return yaw + math.pi/2




def _twist_to_u(msg: Twist) -> np.ndarray:
    return np.array([msg.linear.x, msg.angular.z], dtype=np.float32)


def _u_array_msg_to_u(msg: Float32MultiArray) -> np.ndarray:
    u = np.asarray(msg.data, dtype=np.float32).reshape(-1)
    if u.size < 2:
        u = np.pad(u, (0, 2 - u.size))
    else:
        u = u[:2]
    return u

def _u_array_msg_to_vec(msg: Float32MultiArray, n: int) -> np.ndarray:
    v = np.asarray(msg.data, dtype=np.float32).reshape(-1)
    if v.size < n:
        v = np.pad(v, (0, n - v.size))
    else:
        v = v[:n]
    return v


@dataclass
class Series:
    t: list
    y: list

    def add(self, t: float, y: np.ndarray) -> None:
        self.t.append(float(t))
        self.y.append(np.asarray(y, dtype=np.float32))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.t) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
        t = np.asarray(self.t, dtype=np.float32)
        y = np.stack(self.y, axis=0).astype(np.float32)
        return t, y


def _zoh_resample(
    t: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    y0: np.ndarray,
    *,
    nan_before_first: bool = False,
) -> np.ndarray:
    """Zero-order hold resampling. If nan_before_first=True, fill with NaN prior to first sample."""
    if t.size == 0:
        out = np.tile(y0[None, :], (ts.size, 1))
        if nan_before_first:
            out[:] = np.nan
        return out

    idx = np.searchsorted(t, ts, side="right") - 1
    out = np.empty((ts.size, y.shape[1]), dtype=np.float32)
    for i, k in enumerate(idx):
        if k < 0:
            out[i] = np.nan if nan_before_first else y0
        else:
            out[i] = y[k]
    return out


def _safe_intent_index(intents: List[int], theta_true: int) -> int:
    """Map a theta id to its column index in the published belief vector."""
    try:
        return int(intents.index(int(theta_true)))
    except ValueError:
        idx = int(theta_true)
        return int(np.clip(idx, 0, max(0, len(intents) - 1)))


class NavRolloutRecorderNode(Node):
    def __init__(self):
        super().__init__("nav_rollout_recorder")

        # NEW: controller + trial_id (ONLY identifiers used for saving)
        self.controller = str(self.declare_parameter("controller", "npace").value).strip().lower()
        self.trial_id = int(self.declare_parameter("trial_id", 0).value)

        self.duration_s = float(self.declare_parameter("duration_s", 30.0).value)
        self.plot_dt = float(self.declare_parameter("plot_dt", 0.02).value)
        self.gif_stride = int(self.declare_parameter("gif_stride", 10).value)
        self.max_gif_frames = int(self.declare_parameter("max_gif_frames", 400).value)

        self.ptrue_default = float(self.declare_parameter("ptrue_default", 0.5).value)

        self.intents_h = list(self.declare_parameter("intents_h", [0, 1]).value)
        self.intents_r = list(self.declare_parameter("intents_r", [0, 1]).value)
        self.theta_h_true = int(self.declare_parameter("theta_h_true", 0).value)
        self.theta_r_true = int(self.declare_parameter("theta_r_true", 0).value)

        out_dir = str(self.declare_parameter("out_dir", "").value).strip()
        if out_dir == "":
            out_dir = os.path.join(os.getcwd(), "rollouts")

        # Put each controller into its own folder (still "based only on controller + trial_id")
        out_dir = os.path.join(out_dir, self.controller)
        os.makedirs(out_dir, exist_ok=True)

        self.topic_truth_h = str(self.declare_parameter("topic_truth_h", "/truth/human/pose").value)
        self.topic_truth_r = str(self.declare_parameter("topic_truth_r", "/truth/robot/pose").value)
        self.topic_cmd_h = str(self.declare_parameter("topic_cmd_h", "/human/cmd_vel").value)
        self.topic_cmd_r = str(self.declare_parameter("topic_cmd_r", "/robot/cmd_vel").value)

        self.topic_u_est_h = str(self.declare_parameter("topic_u_est_h", "/ekf/human/u_est").value)
        self.topic_u_est_r = str(self.declare_parameter("topic_u_est_r", "/ekf/robot/u_est").value)
        self.topic_p_est = str(self.declare_parameter("topic_p_est", "/ekf/stacked_state").value)

        self.topic_u_obs_h = str(self.declare_parameter("topic_u_obs_h", "/hl/human/u_obs").value)
        self.topic_u_obs_r = str(self.declare_parameter("topic_u_obs_r", "/hl/robot/u_obs").value)

        self.topic_p_true_h = str(self.declare_parameter("topic_p_true_h", "/hl/human/p_true").value)
        self.topic_p_true_r = str(self.declare_parameter("topic_p_true_r", "/hl/robot/p_true").value)

        self.topic_belief_h = str(self.declare_parameter("topic_belief_h", "/hl/beliefs/human_about_robot").value)
        self.topic_belief_r = str(self.declare_parameter("topic_belief_r", "/hl/beliefs/robot_about_human").value)
        self.topic_ready   = str(self.declare_parameter("topic_ready", "/hl/ready").value)
        self.topic_intent_recognized = str(
            self.declare_parameter("topic_intent_recognized", "/hl/intent_recognized").value
        )
        
        self.subject_name = str(self.declare_parameter("subject_name", "subject_0").value)

        self.goals_h = np.asarray(
            self.declare_parameter("goals_h_flat", [4.0, 1.6, 4.0, -1.6]).value,
            dtype=np.float32
        ).reshape(-1, 2)
        self.goals_r = np.asarray(
            self.declare_parameter("goals_r_flat", [-4.0, 1.6, -4.0, -1.6]).value,
            dtype=np.float32
        ).reshape(-1, 2)

        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self._armed = False
        self._recognition_pressed = False
        self._recognition_t_press = float("nan")
        self._recognition_wall_time_unix = float("nan")
        self._recognition_source = "web_button"

        self.first_seen = {}

        self.truth_h = Series([], [])
        self.truth_r = Series([], [])
        self.cmd_h = Series([], [])
        self.cmd_r = Series([], [])
        self.u_est_h = Series([], [])
        self.u_est_r = Series([], [])
        self.p_est = Series([], [])
        self.u_obs_h = Series([], [])
        self.u_obs_r = Series([], [])
        self.p_true_h = Series([], [])
        self.p_true_r = Series([], [])
        self.bel_h = Series([], [])
        self.bel_r = Series([], [])

        self.create_subscription(PoseStamped, self.topic_truth_h, self._on_truth_h, 50)
        self.create_subscription(PoseStamped, self.topic_truth_r, self._on_truth_r, 50)
        self.create_subscription(Twist, self.topic_cmd_h, self._on_cmd_h, 50)
        self.create_subscription(Twist, self.topic_cmd_r, self._on_cmd_r, 50)

        self.create_subscription(Float32MultiArray, self.topic_u_est_h, self._on_u_est_h, 50)
        self.create_subscription(Float32MultiArray, self.topic_u_est_r, self._on_u_est_r, 50)
        self.create_subscription(Float32MultiArray, self.topic_p_est, self._on_p_est, 50)

        self.create_subscription(Twist, self.topic_u_obs_h, self._on_u_obs_h, 50)
        self.create_subscription(Twist, self.topic_u_obs_r, self._on_u_obs_r, 50)

        self.create_subscription(Float32, self.topic_p_true_h, self._on_p_true_h, 50)
        self.create_subscription(Float32, self.topic_p_true_r, self._on_p_true_r, 50)

        self.create_subscription(Float32MultiArray, self.topic_belief_h, self._on_bel_h, 50)
        self.create_subscription(Float32MultiArray, self.topic_belief_r, self._on_bel_r, 50)
        self.create_subscription(Bool, self.topic_ready, self._on_ready, 10)
        self.create_subscription(Bool, self.topic_intent_recognized, self._on_intent_recognized, 10)

        self.out_dir = out_dir
        
        subject = str(self.get_parameter('subject_name').value)
        out_dir = os.path.join(str(self.get_parameter('out_dir').value), subject, self.controller)
        os.makedirs(out_dir, exist_ok=True)
        self.base = os.path.join(out_dir, f"{self.controller}_trial_{self.trial_id}")

        # NEW: filenames depend ONLY on controller + trial_id
        # self.base = os.path.join(out_dir, f"{self.controller}_trial_{self.trial_id}")

        self._finished = False
        self.done_timer = None
        self.get_logger().info(
            f"[Recorder] controller={self.controller} trial_id={self.trial_id} duration={self.duration_s}s plot_dt={self.plot_dt:.3f}s "
            f"gif_stride={self.gif_stride} out={out_dir}"
        )

    def _t(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9 - self.t0

    def _mark_first(self, key: str) -> None:
        if key not in self.first_seen:
            self.first_seen[key] = self._t()

    def _on_ready(self, msg: Bool) -> None:
        if self._armed:
            return
        if not bool(getattr(msg, "data", False)):
            return

        self._armed = True
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        self.first_seen.clear()
        for s in [
            self.truth_h, self.truth_r, self.cmd_h, self.cmd_r,
            self.u_est_h, self.u_est_r, self.u_obs_h, self.u_obs_r,
            self.p_true_h, self.p_true_r, self.bel_h, self.bel_r, self.p_est
        ]:
            s.t.clear()
            s.y.clear()

        if len(self.intents_r) > 0:
            self.bel_h.add(0.0, np.ones(len(self.intents_r), dtype=np.float32) / float(len(self.intents_r)))
        if len(self.intents_h) > 0:
            self.bel_r.add(0.0, np.ones(len(self.intents_h), dtype=np.float32) / float(len(self.intents_h)))

        if self.done_timer is None:
            self.done_timer = self.create_timer(self.duration_s, self._finish_once)

        self._recognition_pressed = False
        self._recognition_t_press = float("nan")
        self._recognition_wall_time_unix = float("nan")

        self.get_logger().info("[Recorder] armed: /hl/ready=True, starting rollout timer.")

    def _on_intent_recognized(self, msg: Bool) -> None:
        if not self._armed or self._finished:
            return
        if self._recognition_pressed:
            return
        if not bool(getattr(msg, "data", False)):
            return

        self._recognition_pressed = True
        self._recognition_t_press = float(max(0.0, self._t()))
        self._recognition_wall_time_unix = float(time.time())
        self._mark_first("intent_recognized")
        self.get_logger().info(
            f"[Recorder] intent recognized at t={self._recognition_t_press:.3f}s from arm."
        )

    def _on_truth_h(self, msg: PoseStamped) -> None:
        self._mark_first("truth_h")
        new_x = -msg.pose.position.y
        new_y = msg.pose.position.x
        p = msg.pose.position
        p.x=new_x
        p.y = new_y
        
        yaw = _quat_to_yaw(msg.pose.orientation)
        self.truth_h.add(self._t(), np.array([p.x, p.y, yaw], dtype=np.float32))

    def _on_truth_r(self, msg: PoseStamped) -> None:
        self._mark_first("truth_r")
        new_x = -msg.pose.position.y
        new_y = msg.pose.position.x
        p = msg.pose.position
        p.x=new_x
        p.y=new_y
        yaw = _quat_to_yaw(msg.pose.orientation)
        self.truth_r.add(self._t(), np.array([p.x, p.y, yaw], dtype=np.float32))

    def _on_cmd_h(self, msg: Twist) -> None:
        self._mark_first("cmd_h")
        self.cmd_h.add(self._t(), _twist_to_u(msg))

    def _on_cmd_r(self, msg: Twist) -> None:
        self._mark_first("cmd_r")
        self.cmd_r.add(self._t(), _twist_to_u(msg))

    def _on_u_est_h(self, msg: Float32MultiArray) -> None:
        self._mark_first("u_est_h")
        self.u_est_h.add(self._t(), _u_array_msg_to_u(msg))

    def _on_u_est_r(self, msg: Float32MultiArray) -> None:
        self._mark_first("u_est_r")
        self.u_est_r.add(self._t(), _u_array_msg_to_u(msg))

    def _on_p_est(self, msg: Float32MultiArray) -> None:
        self._mark_first("p_est")
        self.p_est.add(self._t(), _u_array_msg_to_vec(msg, 6))

    def _on_u_obs_h(self, msg: Twist) -> None:
        self._mark_first("u_obs_h")
        self.u_obs_h.add(self._t(), _twist_to_u(msg))

    def _on_u_obs_r(self, msg: Twist) -> None:
        self._mark_first("u_obs_r")
        self.u_obs_r.add(self._t(), _twist_to_u(msg))

    def _on_p_true_h(self, msg: Float32) -> None:
        self._mark_first("p_true_h")
        self.p_true_h.add(self._t(), np.array([msg.data], dtype=np.float32))

    def _on_p_true_r(self, msg: Float32) -> None:
        self._mark_first("p_true_r")
        self.p_true_r.add(self._t(), np.array([msg.data], dtype=np.float32))

    def _on_bel_h(self, msg: Float32MultiArray) -> None:
        self._mark_first("bel_h")
        self.bel_h.add(self._t(), np.asarray(msg.data, dtype=np.float32))

    def _on_bel_r(self, msg: Float32MultiArray) -> None:
        self._mark_first("bel_r")
        self.bel_r.add(self._t(), np.asarray(msg.data, dtype=np.float32))

    def _finish_once(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.get_logger().info("[Recorder] finishing, saving artifacts...")

        t_h, y_h = self.truth_h.as_arrays()
        t_r, y_r = self.truth_r.as_arrays()
        t_ch, y_ch = self.cmd_h.as_arrays()
        t_cr, y_cr = self.cmd_r.as_arrays()
        t_ueh, y_ueh = self.u_est_h.as_arrays()
        t_uer, y_uer = self.u_est_r.as_arrays()
        t_uoh, y_uoh = self.u_obs_h.as_arrays()
        t_uor, y_uor = self.u_obs_r.as_arrays()
        t_ph, y_ph = self.p_true_h.as_arrays()
        t_pr, y_pr = self.p_true_r.as_arrays()
        t_bh, y_bh = self.bel_h.as_arrays()
        t_br, y_br = self.bel_r.as_arrays()

        ts = np.arange(0.0, self.duration_s + 1e-9, self.plot_dt, dtype=np.float32)

        H = _zoh_resample(t_h, y_h, ts, y0=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        R = _zoh_resample(t_r, y_r, ts, y0=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        CmdH = _zoh_resample(t_ch, y_ch, ts, y0=np.array([0.0, 0.0], dtype=np.float32))
        CmdR = _zoh_resample(t_cr, y_cr, ts, y0=np.array([0.0, 0.0], dtype=np.float32))
        UestH = _zoh_resample(t_ueh, y_ueh, ts, y0=np.array([0.0, 0.0], dtype=np.float32))
        UestR = _zoh_resample(t_uer, y_uer, ts, y0=np.array([0.0, 0.0], dtype=np.float32))
        UobsH = _zoh_resample(t_uoh, y_uoh, ts, y0=np.array([0.0, 0.0], dtype=np.float32))
        UobsR = _zoh_resample(t_uor, y_uor, ts, y0=np.array([0.0, 0.0], dtype=np.float32))

        idx_r_true = _safe_intent_index(self.intents_r, self.theta_r_true)
        idx_h_true = _safe_intent_index(self.intents_h, self.theta_h_true)

        t_pe, y_pe = self.p_est.as_arrays()

        if y_bh.size > 0 and y_bh.ndim == 2 and y_bh.shape[1] > idx_r_true:
            t_pth, y_pth = t_bh, y_bh[:, idx_r_true:idx_r_true + 1]
        else:
            t_pth, y_pth = np.zeros((0,), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

        if y_br.size > 0 and y_br.ndim == 2 and y_br.shape[1] > idx_h_true:
            t_ptr, y_ptr = t_br, y_br[:, idx_h_true:idx_h_true + 1]
        else:
            t_ptr, y_ptr = np.zeros((0,), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

        PtrueH = _zoh_resample(
            t_pth, y_pth, ts,
            y0=np.array([self.ptrue_default], dtype=np.float32),
            nan_before_first=False
        )
        PtrueR = _zoh_resample(
            t_ptr, y_ptr, ts,
            y0=np.array([self.ptrue_default], dtype=np.float32),
            nan_before_first=False
        )

        Pest = _zoh_resample(
            t_pe, y_pe, ts,
            y0=np.zeros((6,), dtype=np.float32),
            nan_before_first=True
        )

        np.savez(
            f"{self.base}_raw.npz",
            controller=np.array(self.controller),
            trial_id=np.int32(self.trial_id),
            ts=ts,
            H=H, R=R,
            CmdH=CmdH, CmdR=CmdR,
            UestH=UestH, UestR=UestR,
            UobsH=UobsH, UobsR=UobsR,
            t_ph=t_ph, y_ph=y_ph, t_pr=t_pr, y_pr=y_pr,
            PtrueH=PtrueH, PtrueR=PtrueR,
            t_bh=t_bh, y_bh=y_bh, t_br=t_br, y_br=y_br,
            intents_h=np.asarray(self.intents_h, dtype=np.int32),
            intents_r=np.asarray(self.intents_r, dtype=np.int32),
            theta_h_true=np.int32(self.theta_h_true),
            theta_r_true=np.int32(self.theta_r_true),
            goals_h=self.goals_h, goals_r=self.goals_r,
            recognition_pressed=np.bool_(self._recognition_pressed),
            recognition_t_press=np.float32(self._recognition_t_press),
            recognition_wall_time_unix=np.float64(self._recognition_wall_time_unix),
            recognition_source=np.array(self._recognition_source),
            first_seen=self.first_seen,
        )

        self._plot_xy(H, R)
        self._plot_controls(ts, CmdH, CmdR, UestH, UestR, UobsH, UobsR)
        self._plot_beliefs(ts, PtrueH, PtrueR)
        self._plot_estimated_and_observed_positions(ts, H, R, Pest)
        self._plot_estimated_and_observed_yaw(ts, H, R, Pest)
        self._make_gif(ts, H, R)


        self.get_logger().info(f"[Recorder] first_seen={self.first_seen}")
        self.get_logger().info(
            f"[Recorder] saved: {self.base}_raw.npz, {self.base}_xy.png, {self.base}_controls.png, {self.base}_beliefs.png, {self.base}_demo.gif"
        )

        try:
            self.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        
        

    def _plot_xy(self, H: np.ndarray, R: np.ndarray) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(H[:, 0], H[:, 1], "-", label="human (truth)")
        ax.plot(R[:, 0], R[:, 1], "-", label="robot (truth)")
        ax.scatter(self.goals_h[:, 0], self.goals_h[:, 1], marker="x", label="human goals")
        ax.scatter(self.goals_r[:, 0], self.goals_r[:, 1], marker="x", label="robot goals")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{self.base}_xy.png")
        plt.close(fig)

    def _plot_estimated_and_observed_positions(self, ts, H, R, p_est) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 7), dpi=160)

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(ts, H[:, 0], color="red", label="human mocap x")
        ax1.plot(ts, R[:, 0], color="blue", label="robot mocap x")
        ax1.plot(ts, p_est[:, 0], "--", color="red", label="human estimated x")
        ax1.plot(ts, p_est[:, 3], "--", color="blue", label="robot estimated x")
        ax1.grid(alpha=0.3)
        ax1.set_xlabel("t [s]")
        ax1.set_ylabel("x [m]")
        ax1.legend()

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(ts, H[:, 1], color="red", label="human mocap y")
        ax2.plot(ts, R[:, 1], color="blue", label="robot mocap y")
        ax2.plot(ts, p_est[:, 1], "--", color="red", label="human estimated y")
        ax2.plot(ts, p_est[:, 4], "--", color="blue", label="robot estimated y")
        ax2.grid(alpha=0.3)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("y [m]")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(f"{self.base}_estimated_and_observed_positions.png")
        plt.close(fig)
        
    def _plot_estimated_and_observed_yaw(self, ts, H, R, p_est) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 7), dpi=160)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(ts, H[:, 2], color="red", label="human mocap yaw")
        ax1.plot(ts, R[:, 2], color="blue", label="robot mocap yaw")
        ax1.plot(ts, p_est[:, 2], "--", color="red", label="human estimated yaw")
        ax1.plot(ts, p_est[:, 5], "--", color="blue", label="robot estimated yaw")
        ax1.grid(alpha=0.3)
        ax1.set_xlabel("t [s]")
        ax1.set_ylabel("yaw [radians]")
        ax1.legend()

        fig.tight_layout()
        fig.savefig(f"{self.base}_estimated_and_observed_yaw.png")
        plt.close(fig)

    def _plot_controls(self, ts, CmdH, CmdR, UestH, UestR, UobsH, UobsR) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 7), dpi=160)

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(ts, CmdH[:, 0], label="human cmd v")
        ax1.plot(ts, CmdR[:, 0], label="robot cmd v")
        ax1.plot(ts, UestH[:, 0], "--", label="human u_est v")
        ax1.plot(ts, UestR[:, 0], "--", label="robot u_est v")
        ax1.grid(alpha=0.3)
        ax1.set_ylabel("v")
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(ts, CmdH[:, 1], label="human cmd w")
        ax2.plot(ts, CmdR[:, 1], label="robot cmd w")
        ax2.plot(ts, UestH[:, 1], "--", label="human u_est w")
        ax2.plot(ts, UestR[:, 1], "--", label="robot u_est w")
        ax2.grid(alpha=0.3)
        ax2.set_ylabel("w")
        ax2.legend()

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(ts, UobsH[:, 0], label="human u_obs v (HL)")
        ax3.plot(ts, UobsR[:, 0], label="robot u_obs v (HL)")
        ax3.grid(alpha=0.3)
        ax3.set_xlabel("t [s]")
        ax3.set_ylabel("v_obs")
        ax3.legend()

        fig.tight_layout()
        fig.savefig(f"{self.base}_controls.png")
        plt.close(fig)

    def _plot_beliefs(self, ts, PtrueH, PtrueR) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 4), dpi=160)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(ts, PtrueH[:, 0], label=f"human belief in robot θ={self.theta_r_true}")
        ax.plot(ts, PtrueR[:, 0], label=f"robot belief in human θ={self.theta_h_true}")

        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("p(true intent)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{self.base}_beliefs.png")
        plt.close(fig)

    def _make_gif(self, ts: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import imageio.v2 as imageio

        frames = []
        stride = max(1, int(self.gif_stride))

        xs = np.concatenate([H[:, 0], R[:, 0]])
        ys = np.concatenate([H[:, 1], R[:, 1]])

        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_rng = max(1e-6, x_max - x_min)
        y_rng = max(1e-6, y_max - y_min)
        rng = max(x_rng, y_rng)

        min_window = 4.0
        window = max(min_window, rng + 2.0)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        xmin, xmax = cx - 0.5 * window, cx + 0.5 * window
        ymin, ymax = cy - 0.5 * window, cy + 0.5 * window

        fig = plt.figure(figsize=(6, 6), dpi=160)
        ax = fig.add_subplot(1, 1, 1)

        max_i = ts.size
        if self.max_gif_frames > 0:
            max_i = min(max_i, self.max_gif_frames * stride)

        def _add_triangle(ax, x: float, y: float, yaw: float, facecolor: str):
            L = 0.25
            W = 0.14
            pts_b = np.array([[L, 0.0], [-0.6 * L, W], [-0.6 * L, -W]], dtype=np.float64)
            c = math.cos(yaw)
            s = math.sin(yaw)
            Rm = np.array([[c, -s], [s, c]], dtype=np.float64)
            pts_w = pts_b @ Rm.T + np.array([x, y], dtype=np.float64)
            tri = patches.Polygon(
                pts_w, closed=True, facecolor=facecolor, edgecolor="k",
                linewidth=0.5, alpha=0.9, label="_nolegend_"
            )
            ax.add_patch(tri)

        for i in range(0, max_i, stride):
            ax.clear()
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid(alpha=0.3)

            ax.scatter(self.goals_h[:, 0], self.goals_h[:, 1], marker="x", label="human goals")
            ax.scatter(self.goals_r[:, 0], self.goals_r[:, 1], marker="x", label="robot goals")
            ax.plot(H[: i + 1, 0], H[: i + 1, 1], "-", label="human")
            ax.plot(R[: i + 1, 0], R[: i + 1, 1], "-", label="robot")

            if i >= 1:
                dh = H[i, :2] - H[i - 1, :2]
                dr = R[i, :2] - R[i - 1, :2]
                yaw_h = math.atan2(float(dh[1]), float(dh[0])) if float(np.hypot(dh[0], dh[1])) > 1e-3 else float(H[i, 2])
                yaw_r = math.atan2(float(dr[1]), float(dr[0])) if float(np.hypot(dr[0], dr[1])) > 1e-3 else float(R[i, 2])
            else:
                yaw_h = float(H[i, 2])
                yaw_r = float(R[i, 2])

            _add_triangle(ax, float(H[i, 0]), float(H[i, 1]), yaw_h, facecolor="C0")
            _add_triangle(ax, float(R[i, 0]), float(R[i, 1]), yaw_r, facecolor="C1")

            ax.set_title(f"t = {ts[i]:.2f}s")
            ax.legend(loc="upper right", fontsize=8)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            frames.append(img)

        plt.close(fig)
        imageio.mimsave(f"{self.base}_demo.gif", frames, duration=float(self.plot_dt * stride))


def main(args=None):
    rclpy.init(args=args)
    node = NavRolloutRecorderNode()
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

