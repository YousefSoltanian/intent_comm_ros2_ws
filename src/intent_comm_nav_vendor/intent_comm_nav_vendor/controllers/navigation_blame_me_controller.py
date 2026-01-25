#!/usr/bin/env python
"""
Blame-Me velocity controller – exact Q-MDP (opponent θ = goal index)
Now time-aligned like your human/influence controllers:

• At each call:
  1) If a cached prediction of opponent action exists (from previous call),
     update belief with log-softmax using that cache and the *last observed*
     opponent action.
  2) Run all per-θ ILQ solvers at the current state.
  3) Compute our Q-MDP control.
  4) Build & cache the opponent action prediction (μ, Σ) for the *next* call.

This avoids the t=0 belief jump without changing your API or parameter names.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import jax.numpy as jnp

from iLQGame.cost        import SocialNavigationPlayerCost
from iLQGame.player_cost import PlayerCost
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import PlanarNavigation2PlayerSystem


# ────────────────────────── small numeric helpers ──────────────────────────
def _spd_guard(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Symmetrize + tiny ridge; returns SPD-ish matrix."""
    M = 0.5 * (M + M.T)
    return M + eps * np.eye(M.shape[0])


# ╭────────────────────────────────────────────────────────────────╮
# │                Internal warm-startable ILQ wrapper             │
# ╰────────────────────────────────────────────────────────────────╯
class _Solver2P:
    """
    One ILQ solver instance for a *fixed* intent pair (θ_self, θ_opp).
    Keeps its own warm-start (Ps, alphas). Navigation version (2-D control).
    """

    def __init__(
        self,
        *,
        # intents/goals
        theta_self: int,
        theta_opp: int,
        goals_self: np.ndarray,
        goals_opp:  np.ndarray,
        # horizon / time
        horizon: int,
        dt: float,
        # cost weights / corridor
        w_goal_xy: Tuple[float, float],
        w_head: float,
        w_speed: float,
        w_effort: float,
        w_lat: float,
        w_wall: float,
        w_coll: float,
        v_nom: float,
        hall_y0: float,
        hall_half_width: float,
        r_safe_coll: float,
        # control limits
        v_lo: float, v_hi: float, w_lo: float, w_hi: float,
        # which player we control in the outer controller (0 or 1)
        ctrl_index_self: int,
        max_iter: int = 25,
        verbose: bool = False,
    ):
        self.theta_self = int(theta_self)
        self.theta_opp  = int(theta_opp)
        self.horizon    = int(horizon)
        self.ctrl_i     = int(ctrl_index_self)

        dyn = PlanarNavigation2PlayerSystem(T=float(dt))

        pc_1 = PlayerCost()
        pc_2 = PlayerCost()

        # Map θs to players depending on who "self" is:
        th_p0_self = self.theta_self if self.ctrl_i == 0 else self.theta_opp
        th_p1_self = self.theta_opp  if self.ctrl_i == 0 else self.theta_self
        goals_p0_self = goals_self    if self.ctrl_i == 0 else goals_opp
        goals_p0_opp  = goals_opp     if self.ctrl_i == 0 else goals_self
        goals_p1_self = goals_opp     if self.ctrl_i == 0 else goals_self
        goals_p1_opp  = goals_self    if self.ctrl_i == 0 else goals_opp

        pc_1.add_cost(
            SocialNavigationPlayerCost(
                player_index=0,
                goals_self=goals_p0_self, goals_other=goals_p0_opp,
                theta_self=th_p0_self, theta_other=0,
                horizon=horizon, name="NavCost_P0",
                w_goal_xy=w_goal_xy, w_head=w_head, w_speed=w_speed,
                w_effort=w_effort, w_lat=w_lat, w_wall=w_wall, w_coll=w_coll,
                hall_y0=hall_y0, hall_half_width=hall_half_width,
                r_safe_coll=r_safe_coll, v_nom=v_nom
            ),
            arg="x", weight=1.0
        )
        pc_2.add_cost(
            SocialNavigationPlayerCost(
                player_index=1,
                goals_self=goals_p1_self, goals_other=goals_p1_opp,
                theta_self=th_p1_self, theta_other=0,
                horizon=horizon, name="NavCost_P1",
                w_goal_xy=w_goal_xy, w_head=w_head, w_speed=w_speed,
                w_effort=w_effort, w_lat=w_lat, w_wall=w_wall, w_coll=w_coll,
                hall_y0=hall_y0, hall_half_width=hall_half_width,
                r_safe_coll=r_safe_coll, v_nom=v_nom
            ),
            arg="x", weight=1.0
        )

        u_limit_1 = BoxConstraint(jnp.array([v_lo, w_lo]), jnp.array([v_hi, w_hi]))
        u_limit_2 = BoxConstraint(jnp.array([v_lo, w_lo]), jnp.array([v_hi, w_hi]))

        zeros_P = jnp.zeros((2, 6, horizon), dtype=jnp.float32)
        zeros_a = jnp.zeros((2,    horizon), dtype=jnp.float32)

        self.solver = ILQSolver(
            dyn,
            [pc_1, pc_2],
            [zeros_P, zeros_P],
            [zeros_a, zeros_a],
            max_iter=max_iter,
            u_constraints=[u_limit_1, u_limit_2],
            verbose=verbose
        )

        self._Ps = None
        self._a  = None

    def run(self, x: jnp.ndarray):
        self.solver.run(x, Ps_warm=self._Ps, alphas_warm=self._a)
        self._Ps, self._a = self.solver._Ps, self.solver._alphas

    def nom(self, pl: int) -> np.ndarray:
        u = self.solver._best_operating_point[1][pl][:, 0]   # (2,)
        return np.array([float(u[0]), float(u[1])], dtype=np.float64)

    def B_R(self, pl: int):
        xs, us = self.solver._best_operating_point
        _, Bs  = self.solver._linearize_dynamics(xs, us)
        _, _, _, H = self.solver._quadraticize_costs(xs, us)
        return Bs[pl][:, :, 0], H[pl][:, :, 0]  # B: (nx,2), R0: (2,2)

    def Zz_next(self, pl: int):
        Zs, zs = self.solver._best_op[4], self.solver._best_op[5]
        return Zs[pl][:, :, 1], zs[pl][:, 1]


# ╭────────────────────────────────────────────────────────────────╮
# │                     Public controller class                    │
# ╰────────────────────────────────────────────────────────────────╯
class NavigationBlameMeController:
    """
    Blame-Me/Q-MDP controller with action-observation noise + tempering.
    Time-aligned Bayes using an internal prediction cache to avoid the t=0 jump.

    Posterior uses Σ_tot(θ) = (β_like*S(θ))^{-1} + Σ_obs, where S = R0 + Bᵀ Z1 B.
    """

    def __init__(
        self,
        *,
        # intents & goals
        theta_self: int,
        goals_self: np.ndarray,
        goals_opp:  np.ndarray,
        ctrl_index_self: int = 0,
        intents: Tuple[int, ...] = (0, 1),
        # time / horizon
        dt: float = 0.25,
        horizon: int = 10,
        # cost params (aligned with your sim)
        w_goal_xy: Tuple[float, float] = (6.0, 6.0),
        w_head: float = 1.108,
        w_speed: float = 0.0,
        w_effort: float = 10.08,
        w_lat: float = 0.2,
        w_wall: float = 60.0,
        w_coll: float = 15.0,
        v_nom: float = 0.9,
        hall_y0: float = 0.0,
        hall_half_width: float = 2.41,
        r_safe_coll: float = 2.0,
        # constraints
        v_lo: float = 0.0, v_hi: float = 1.0,
        w_lo: float = -0.4, w_hi: float = 0.4,
        # ILQ
        max_iter: int = 25,
        verbose: bool = False,
        # optional extra quadratic effort (Q-MDP denom)
        effort_w: float = 0.0,
        # likelihood shaping for action Bayes
        beta_action_like: float = 1.0,                 # scales policy precision
        sigma2_action_obs: Tuple[float, float] = (0.0, 0.0),  # diag(σ_v^2, σ_ω^2)
        like_power: float = 1.0,                       # tempering (η): log-like *= η
    ):
        self._intents = tuple(int(i) for i in intents)
        self._ctrl_i  = int(ctrl_index_self)
        self._theta_self = int(theta_self)

        self._goals_self = np.asarray(goals_self, dtype=np.float32).reshape((-1, 2))
        self._goals_opp  = np.asarray(goals_opp,  dtype=np.float32).reshape((-1, 2))

        self._belief = np.ones(len(self._intents), dtype=np.float64)
        self._belief /= self._belief.sum()

        self._params = dict(
            dt=float(dt), horizon=int(horizon),
            w_goal_xy=tuple(float(x) for x in w_goal_xy),
            w_head=float(w_head), w_speed=float(w_speed), w_effort=float(w_effort),
            w_lat=float(w_lat), w_wall=float(w_wall), w_coll=float(w_coll),
            v_nom=float(v_nom), hall_y0=float(hall_y0),
            hall_half_width=float(hall_half_width), r_safe_coll=float(r_safe_coll),
            v_lo=float(v_lo), v_hi=float(v_hi), w_lo=float(w_lo), w_hi=float(w_hi),
            max_iter=int(max_iter), verbose=bool(verbose),
            effort_w=float(effort_w),
        )

        # action-likelihood knobs
        self._beta_like = float(beta_action_like)
        self._Sigma_obs = np.diag(np.asarray(sigma2_action_obs, dtype=np.float64))
        self._eta_like  = float(like_power)

        self._solvers: List[_Solver2P] = []
        self._build_solvers(theta_self=self._theta_self)

        # prediction cache for opponent action stats (set on *exit* of compute_action)
        self._pred_ctrl = None  # dict with keys: "mu_u" (n,2), "Prec_tot" (n,2,2)

        self._log_2pi = math.log(2.0 * math.pi)

    # internal: rebuild all per-θ solvers if θ_self changes
    def _build_solvers(self, *, theta_self: int):
        p = self._params
        self._solvers = [
            _Solver2P(
                theta_self=theta_self,
                theta_opp=th,
                goals_self=self._goals_self,
                goals_opp=self._goals_opp,
                ctrl_index_self=self._ctrl_i,
                horizon=p["horizon"], dt=p["dt"],
                w_goal_xy=p["w_goal_xy"], w_head=p["w_head"], w_speed=p["w_speed"],
                w_effort=p["w_effort"], w_lat=p["w_lat"], w_wall=p["w_wall"], w_coll=p["w_coll"],
                v_nom=p["v_nom"], hall_y0=p["hall_y0"], hall_half_width=p["hall_half_width"],
                r_safe_coll=p["r_safe_coll"],
                v_lo=p["v_lo"], v_hi=p["v_hi"], w_lo=p["w_lo"], w_hi=p["w_hi"],
                max_iter=p["max_iter"], verbose=p["verbose"]
            )
            for th in self._intents
        ]

    # ───── utils ─────
    @staticmethod
    def _mv_loglike_prec(u: np.ndarray, mu: np.ndarray, Prec: np.ndarray) -> float:
        """
        log N(u | mu, Σ) with Σ^{-1} = Prec (2×2):
            0.5 log det(Prec) - 0.5 (u-mu)^T Prec (u-mu)
        """
        d = u - mu
        Prec = _spd_guard(Prec, 1e-9)
        q = float(d.T @ Prec @ d)
        sign, logdet = np.linalg.slogdet(Prec)
        if sign <= 0:
            Prec = Prec + 1e-9 * np.eye(2)
            sign, logdet = np.linalg.slogdet(Prec)
        return 0.5 * logdet - 0.5 * q

    @staticmethod
    def _softmax_logw(logw: np.ndarray) -> np.ndarray:
        m = np.max(logw)
        w = np.exp(logw - m)
        s = np.sum(w)
        if not np.isfinite(s) or s <= 0.0:
            return np.ones_like(logw) / float(len(logw))
        return w / s

    # ───── main API ─────
    def compute_action(
        self,
        *,
        obs: np.ndarray,
        a_opponent_observed: np.ndarray,   # vector [v, ω] from last step
        theta_self: Optional[int] = None
    ) -> np.ndarray:
        """
        Returns our control u = [v, ω] using Q-MDP over opponent θ.
        Time-aligned belief update: uses cached prediction from previous call.
        """
        if theta_self is not None and int(theta_self) != self._theta_self:
            self._theta_self = int(theta_self)
            self._build_solvers(theta_self=self._theta_self)

        x = jnp.asarray(obs, dtype=jnp.float32)

        # (A) Bayes over opponent θ using cached prediction from previous call
        if self._pred_ctrl is not None:
            u_opp_obs = np.asarray(a_opponent_observed, dtype=np.float64).reshape(2)
            mu_arr    = np.asarray(self._pred_ctrl["mu_u"], dtype=np.float64)       # (n,2)
            Prec_arr  = np.asarray(self._pred_ctrl["Prec_tot"], dtype=np.float64)   # (n,2,2)

            priors = np.asarray(self._belief, dtype=np.float64)
            logw   = np.empty_like(priors)

            # --- key change: post-scale precision like in Influence (Prec_use = β * Prec_tot)
            for i in range(len(self._intents)):
                Prec_use = self._beta_like * Prec_arr[i]          # soften the likelihood
                log_like = self._mv_loglike_prec(u_opp_obs, mu_arr[i], Prec_use)
                logw[i]  = math.log(priors[i] + 1e-30) + self._eta_like * log_like

            self._belief = self._softmax_logw(logw)


        # (B) Run per-θ solvers (warm-started) at the CURRENT state
        for s in self._solvers:
            s.run(x)

        # (C) Q-MDP closed-form (vector case) for OUR control
        pl_self = self._ctrl_i
        eff_lambda = self._params["effort_w"]
        v_lo, v_hi = self._params["v_lo"], self._params["v_hi"]
        w_lo, w_hi = self._params["w_lo"], self._params["w_hi"]

        S_bar = np.zeros((2, 2), dtype=np.float64)
        num   = np.zeros((2,),   dtype=np.float64)

        for w, s in zip(self._belief, self._solvers):
            B, R0 = s.B_R(pl_self)
            Z1, z1 = s.Zz_next(pl_self)
            H  = np.asarray(B.T @ Z1 @ B, dtype=np.float64)         # 2×2
            S  = np.asarray(R0 + H,       dtype=np.float64)         # 2×2
            l  = np.asarray(B.T @ z1,      dtype=np.float64).reshape(2)
            u0 = s.nom(pl_self)                                     # 2,

            S_bar += float(w) * S
            num   += float(w) * (H @ u0 - l)

        if eff_lambda > 0.0:
            S_bar = S_bar + 2.0 * eff_lambda * np.eye(2)

        S_bar_reg = _spd_guard(S_bar, 1e-9)
        try:
            u_cmd = np.linalg.solve(S_bar_reg, num)
        except np.linalg.LinAlgError:
            u_cmd = np.linalg.pinv(S_bar_reg) @ num

        # box clip
        v = float(np.minimum(np.maximum(u_cmd[0], v_lo), v_hi))
        w = float(np.minimum(np.maximum(u_cmd[1], w_lo), w_hi))
        u_cmd = np.array([v, w], dtype=np.float64)

        # (D) Build prediction cache for NEXT call (opponent action stats per θ)
        pl_opp = 1 - pl_self
        n = len(self._intents)
        mu_u   = np.zeros((n, 2), dtype=np.float64)
        Prec_t = np.zeros((n, 2, 2), dtype=np.float64)

        for i, s in enumerate(self._solvers):
            mu_u[i, :] = s.nom(pl_opp)

            B, R0 = s.B_R(pl_opp)
            Z1, _ = s.Zz_next(pl_opp)
            S = np.asarray(R0 + B.T @ Z1 @ B, dtype=np.float64)     # 2×2

            # policy precision and total precision (add observation noise)
            Prec_pol  = _spd_guard(self._beta_like * S, 1e-9)
            try:
                Sigma_pol = np.linalg.inv(Prec_pol)
            except np.linalg.LinAlgError:
                Sigma_pol = np.linalg.pinv(Prec_pol)

            Sigma_tot = Sigma_pol + self._Sigma_obs
            try:
                Prec_tot = np.linalg.inv(_spd_guard(Sigma_tot, 1e-12))
            except np.linalg.LinAlgError:
                Prec_tot = np.linalg.pinv(_spd_guard(Sigma_tot, 1e-12))

            Prec_t[i, :, :] = Prec_tot

        self._pred_ctrl = {"mu_u": mu_u, "Prec_tot": Prec_t}

        return u_cmd  # np.array([v, ω])

    # expose beliefs and intent order
    @property
    def belief_over_theta(self) -> Dict[int, float]:
        return {int_th: float(p) for int_th, p in zip(self._intents, self._belief)}

    @property
    def belief_vector(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def intent_order(self) -> Tuple[int, ...]:
        return self._intents
