#!/usr/bin/env python3
from typing import Optional
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist

from intent_comm_nav_vendor.controllers.navigation_npace_influence import NavigationNPACEInfluence

# Blame-Me controller (optional)
try:
    from intent_comm_nav_vendor.controllers.navigation_blame_me_controller import NavigationBlameMeController
except Exception:
    NavigationBlameMeController = None


def _twist_to_u(msg: Twist) -> np.ndarray:
    return np.array([msg.linear.x, msg.angular.z], dtype=np.float32)


def _u_to_twist(u: np.ndarray) -> Twist:
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    m = Twist()
    m.linear.x = float(u[0]) if u.size > 0 else 0.0
    m.angular.z = float(u[1]) if u.size > 1 else 0.0
    return m


class HighLevelRobotRunnerNode(Node):
    def __init__(self):
        super().__init__("high_level_robot_runner")

        self.dt_hl = float(self.declare_parameter("dt_hl", 0.5).value)
        self.max_runtime_s = float(self.declare_parameter("max_runtime_s", 0.0).value)  # 0 => run forever

        self.topic_x_hat   = str(self.declare_parameter("topic_x_hat", "/hl/x_hat").value)
        self.topic_h_u_obs = str(self.declare_parameter("topic_h_u_obs", "/hl/human/u_obs").value)
        self.topic_r_cmd   = str(self.declare_parameter("topic_r_cmd", "/cmd_vel").value)
        self.topic_ready   = str(self.declare_parameter("topic_ready", "/hl/ready").value)

        self.swap_agents = bool(self.declare_parameter("swap_agents", False).value)

        # controller selector (same convention as sim)
        self.controller = str(self.declare_parameter("controller", "npace").value).strip().lower()

        self.intents_h = list(self.declare_parameter("intents_h", [0, 1]).value)
        self.intents_r = list(self.declare_parameter("intents_r", [0, 1]).value)
        self.theta_h_true = int(self.declare_parameter("theta_h_true", 0).value)
        self.theta_r_true = int(self.declare_parameter("theta_r_true", 0).value)

        self.goals_h = np.asarray(
            self.declare_parameter("goals_h_flat", [4.0, 1.6, 4.0, -1.6]).value,
            dtype=np.float32
        ).reshape(-1, 2)
        self.goals_r = np.asarray(
            self.declare_parameter("goals_r_flat", [-4.0, 1.6, -4.0, -1.6]).value,
            dtype=np.float32
        ).reshape(-1, 2)

        self.w_goal_pos = float(self.declare_parameter("w_goal_pos", 60.0).value)
        self.w_head     = float(self.declare_parameter("w_head", 1.108).value)
        self.w_speed    = float(self.declare_parameter("w_speed", 0.0).value)
        self.w_eff      = float(self.declare_parameter("w_eff", 50.08).value)
        self.v_nom      = float(self.declare_parameter("v_nom", 0.9).value)

        self.w_lat      = float(self.declare_parameter("w_lat", 0.2).value)
        self.w_wall     = float(self.declare_parameter("w_wall", 100.0).value)
        self.w_coll     = float(self.declare_parameter("w_coll", 20.0).value)
        self.r_safe_coll = float(self.declare_parameter("r_safe_coll", 2.0).value)

        self.hall_y0         = float(self.declare_parameter("hall_y0", 0.0).value)
        self.hall_half_width = float(self.declare_parameter("hall_half_width", 2.41).value)

        self.v_lo = float(self.declare_parameter("v_lo", 0.0).value)
        self.v_hi = float(self.declare_parameter("v_hi", 1.2).value)
        self.w_lo = float(self.declare_parameter("w_lo", -0.6).value)
        self.w_hi = float(self.declare_parameter("w_hi", 0.6).value)

        self.max_iter = int(self.declare_parameter("max_iter", 25).value)
        self.beta_r = float(self.declare_parameter("beta_r", 1.0).value)

        self.beta_state = float(self.declare_parameter("beta_state", 1.0).value)
        self.rho_forget = float(self.declare_parameter("rho_forget", 0.0).value)

        sigma2_r = list(self.declare_parameter(
            "sigma2_state_robot_flat",
            [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]
        ).value)
        self.sigma2_state_r = tuple(float(x) for x in sigma2_r)

        # NPACE params
        self.gamma_teach = float(self.declare_parameter("gamma_teach", 0.0).value)
        self.effort_w_qmdp = float(self.declare_parameter("effort_w_qmdp", 0.0).value)
        self.obs_smoothing_alpha = float(self.declare_parameter("obs_smoothing_alpha", 1.0).value)

        # Blame-Me params
        self.beta_action_like = float(self.declare_parameter("beta_action_like", 0.1).value)
        sigma2_act = list(self.declare_parameter("sigma2_action_obs_flat", [0.01, 0.01]).value)
        if len(sigma2_act) < 2:
            sigma2_act = (sigma2_act + [0.01, 0.01])[:2]
        self.sigma2_action_obs = (float(sigma2_act[0]), float(sigma2_act[1]))

        self.horizon = int(self.declare_parameter("horizon", 10).value)

        # ----------------------
        # Robot controller (switch)
        # ----------------------
        self.robot_kind = "npace"
        if self.controller in ("blame_me", "blameme", "blame-me"):
            if NavigationBlameMeController is None:
                self.get_logger().error("controller=blame_me requested but import failed. Falling back to NPACE.")
            else:
                self.robot_kind = "blame_me"
                self.robot = NavigationBlameMeController(
                    theta_self=self.theta_r_true,
                    goals_self=self.goals_r,
                    goals_opp=self.goals_h,
                    ctrl_index_self=1,
                    intents=tuple(self.intents_h),
                    dt=self.dt_hl, horizon=self.horizon,
                    w_goal_xy=(self.w_goal_pos, self.w_goal_pos),
                    w_head=self.w_head, w_speed=self.w_speed, w_effort=self.w_eff,
                    w_lat=self.w_lat, w_wall=self.w_wall, w_coll=self.w_coll,
                    v_nom=self.v_nom, hall_y0=self.hall_y0, hall_half_width=self.hall_half_width,
                    r_safe_coll=self.r_safe_coll,
                    v_lo=self.v_lo, v_hi=self.v_hi, w_lo=self.w_lo, w_hi=self.w_hi,
                    max_iter=self.max_iter, verbose=False,
                    effort_w=self.effort_w_qmdp,
                    beta_action_like=self.beta_action_like,
                    sigma2_action_obs=self.sigma2_action_obs,
                )
        if self.robot_kind != "blame_me":
            self.robot_kind = "npace"
            self.robot = NavigationNPACEInfluence(
                theta_robot_true=self.theta_r_true,
                goals_human=self.goals_h,
                goals_robot=self.goals_r,
                intents=tuple(self.intents_h),
                dt=self.dt_hl, horizon=self.horizon,
                w_goal_xy=(self.w_goal_pos, self.w_goal_pos),
                w_head=self.w_head, w_speed=self.w_speed, w_effort=self.w_eff,
                w_lat=self.w_lat, w_wall=self.w_wall, w_coll=self.w_coll,
                v_nom=self.v_nom, hall_y0=self.hall_y0, hall_half_width=self.hall_half_width,
                r_safe_coll=self.r_safe_coll,
                v_lo=self.v_lo, v_hi=self.v_hi, w_lo=self.w_lo, w_hi=self.w_hi,
                beta_state=self.beta_state, rho_forget=self.rho_forget,
                sigma2_state=self.sigma2_state_r,
                max_iter=self.max_iter, verbose=False,
                gamma_teach=self.gamma_teach,
                effort_w_qmdp=self.effort_w_qmdp,
                sigma2_action_obs=self.sigma2_action_obs,
                obs_smoothing_alpha=self.obs_smoothing_alpha,
                beta=self.beta_r,
            )

        self.sub_x_hat = self.create_subscription(Float32MultiArray, self.topic_x_hat, self._on_x_hat, 20)
        self.sub_h_u_obs = self.create_subscription(Twist, self.topic_h_u_obs, self._on_h_u_obs, 20)

        self.pub_r_cmd = self.create_publisher(Twist, self.topic_r_cmd, 10)
        self.pub_ready = self.create_publisher(Bool, self.topic_ready, 1)

        # Keep belief topics consistent with recorder expectations
        self.pub_b_human = self.create_publisher(Float32MultiArray, "/hl/beliefs/human_about_robot", 10)
        self.pub_b_robot = self.create_publisher(Float32MultiArray, "/hl/beliefs/robot_about_human", 10)

        self._x_hat: Optional[np.ndarray] = None
        self._h_u_obs: Optional[np.ndarray] = None

        self._ready = False
        self._last_u_r = np.zeros(2, dtype=np.float32)
        self._tick_i = 0

        self._t0 = time.perf_counter()

        self.timer = self.create_timer(self.dt_hl, self._tick)

        self.get_logger().info("Warmup: compiling robot controller in background...")
        threading.Thread(target=self._warmup_worker, daemon=True).start()

        self.get_logger().info(
            f"HighLevelRobotRunner controller={self.robot_kind} dt_hl={self.dt_hl:.3f}s horizon={self.horizon} "
            f"max_runtime_s={self.max_runtime_s:.1f} swap_agents={self.swap_agents} | "
            f"x_hat={self.topic_x_hat} | h_u_obs={self.topic_h_u_obs} | cmd_r={self.topic_r_cmd}"
        )

    def _warmup_worker(self) -> None:
        x0 = np.array([-4.0, 0.0, 0.0,  4.0, 0.0, np.pi], dtype=np.float32)
        a1 = np.array([0.0, 0.0], dtype=np.float32)
        try:
            ur = self._call_robot(obs=x0, a1=a1)
            try:
                import jax
                jax.block_until_ready(ur)
            except Exception:
                pass
            self._ready = True
            self.get_logger().info("Warmup: robot compute_action compiled OK.")
        except Exception as e:
            self.get_logger().error(f"Warmup failed: {repr(e)}")

    def _on_x_hat(self, msg: Float32MultiArray) -> None:
        x = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if x.size >= 6 and self.swap_agents:
            x = np.array([x[3], x[4], x[5], x[0], x[1], x[2]], dtype=np.float32)
        self._x_hat = x

    def _on_h_u_obs(self, msg: Twist) -> None:
        self._h_u_obs = _twist_to_u(msg)

    def _call_robot(self, obs: np.ndarray, a1: np.ndarray) -> np.ndarray:
        fn = self.robot.compute_action
        try:
            return fn(obs=obs, a_opponent_observed=a1)
        except TypeError:
            try:
                return fn(obs=obs, a1_observed=a1)
            except TypeError:
                return fn(obs, a1)

    def stop_robot(self) -> None:
        try:
            self.pub_r_cmd.publish(_u_to_twist(np.zeros(2, dtype=np.float32)))
        except Exception:
            pass

    def _tick(self) -> None:
        self._tick_i += 1

        # hard stop by runtime (safety)
        if self.max_runtime_s > 0.0 and (time.perf_counter() - self._t0) >= self.max_runtime_s:
            self.get_logger().warn("Max runtime reached -> stopping robot + shutting down.")
            self.stop_robot()
            try:
                rclpy.shutdown()
            except Exception:
                pass
            return

        if self._x_hat is None:
            return

        h_u_obs = self._h_u_obs if self._h_u_obs is not None else np.zeros(2, dtype=np.float32)
        x = self._x_hat.copy()

        if not self._ready:
            u_r = np.zeros(2, dtype=np.float32)
        else:
            try:
                _t0 = time.perf_counter()
                u_r = self._call_robot(obs=x, a1=h_u_obs.copy())
                _dt = time.perf_counter() - _t0
                if _dt > 0.9 * self.dt_hl:
                    self.get_logger().warn(f"tick compute_action took {_dt:.3f}s > 0.9*dt_hl ({self.dt_hl:.3f}s)")
            except Exception as e:
                self.get_logger().error(f"compute_action error: {repr(e)}")
                u_r = self._last_u_r

        u_r = np.array([np.clip(u_r[0], self.v_lo, self.v_hi), np.clip(u_r[1], self.w_lo, self.w_hi)], dtype=np.float32)
        self._last_u_r = u_r

        self.pub_r_cmd.publish(_u_to_twist(u_r))
        try:
            self.pub_ready.publish(Bool(data=bool(self._ready)))
        except Exception:
            pass

        # publish robot belief if available
        try:
            b_r = None
            if hasattr(self.robot, "robot_belief_over_human"):
                b_r = getattr(self.robot, "robot_belief_over_human")
            elif hasattr(self.robot, "belief_over_theta"):
                b_r = getattr(self.robot, "belief_over_theta")
            if b_r is not None:
                msg_br = Float32MultiArray()
                msg_br.data = [float(b_r.get(k, 0.0)) for k in self.intents_h]
                self.pub_b_robot.publish(msg_br)
        except Exception:
            pass

        # dummy "human belief" (uniform), keeps recorder happy
        try:
            msg_bh = Float32MultiArray()
            if len(self.intents_r) > 0:
                msg_bh.data = [1.0 / float(len(self.intents_r)) for _ in self.intents_r]
            else:
                msg_bh.data = []
            self.pub_b_human.publish(msg_bh)
        except Exception:
            pass

        if self._tick_i % max(1, int(round(1.0 / self.dt_hl))) == 0:
            self.get_logger().info(
                f"tick ready={self._ready} x=[{x[0]:+.2f},{x[1]:+.2f},{x[2]:+.2f} | {x[3]:+.2f},{x[4]:+.2f},{x[5]:+.2f}] "
                f"h_u_obs=[{h_u_obs[0]:+.2f},{h_u_obs[1]:+.2f}] u_r=[{u_r[0]:+.2f},{u_r[1]:+.2f}]"
            )


def main(args=None):
    rclpy.init(args=args)
    node = HighLevelRobotRunnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
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

