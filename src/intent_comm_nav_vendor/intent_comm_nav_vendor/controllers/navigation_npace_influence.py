#!/usr/bin/env python
"""
Navigation NPACE-Influence (corrected, intersection-aligned)
------------------------------------------------------------
Two-player planar navigation:
  state x = [x1, y1, th1,  x2, y2, th2]
  u1 = [v1, w1], u2 = [v2, w2]

This mirrors your Intersection NPACE-Influence logic:
- Human & robot mixers use S = R0 + Bᵀ Z B
- Bayes updates are numerically stable (softmax in log-space)
- Vector Gaussian log-like uses SPD-guarded precision
- Teaching term uses capped action-gap and PSD-projected sensitivity
- Solves use symmetrize + ridge before linear solves
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

# ---- iLQGame imports ----
from iLQGame.cost        import SocialNavigationPlayerCost
from iLQGame.player_cost import PlayerCost
from iLQGame.ilq_solver  import ILQSolver
from iLQGame.constraint  import BoxConstraint
from iLQGame.multiplayer_dynamical_system import PlanarNavigation2PlayerSystem

# Global dyn helper (used for one-step predictions)
_ddt = 0.25
Dyn = PlanarNavigation2PlayerSystem(T=float(_ddt))

# ---------- numeric helpers ----------
def _spd_guard(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    M = 0.5 * (M + M.T)
    # small ridge
    return M + eps * np.eye(M.shape[0])

@jax.jit
def _mv_loglike_prec(u: jnp.ndarray, mu: jnp.ndarray, Prec: jnp.ndarray) -> jnp.ndarray:
    """log N(u | mu, Σ) with Σ^{-1} = Prec. Ensures SPD-ish precision."""
    Prec = 0.5 * (Prec + Prec.T) + 1e-8 * jnp.eye(2)
    d = u - mu
    q = jnp.dot(d, Prec @ d)
    sign, logdet = jnp.linalg.slogdet(Prec)
    logdet = jnp.where(sign > 0, logdet, jnp.linalg.slogdet(Prec + 1e-6 * jnp.eye(2))[1])
    return 0.5 * logdet - 0.5 * q

# ---------- warm-startable ILQ solver bank ----------
class _Solver2P_NPACE_Nav:
    def __init__(
        self,
        *,
        theta_h: int,
        theta_r: int,
        # goals
        goals_h: np.ndarray,
        goals_r: np.ndarray,
        # horizon/time
        dt: float,
        horizon: int,
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
        # ctrl limits
        v_lo: float, v_hi: float, w_lo: float, w_hi: float,
        max_iter: int = 25,
        verbose: bool = False,
    ):
        self.theta_h = int(theta_h)
        self.theta_r = int(theta_r)
        dyn = PlanarNavigation2PlayerSystem(T=float(dt))

        pc_h = PlayerCost()
        pc_r = PlayerCost()

        pc_h.add_cost(
            SocialNavigationPlayerCost(
                player_index=0,
                goals_self=jnp.asarray(goals_h, dtype=jnp.float32),
                goals_other=jnp.asarray(goals_r, dtype=jnp.float32),
                theta_self=self.theta_h, theta_other=0,  # opp index unused in stage
                horizon=horizon, name="NavCost_Human",
                w_goal_xy=w_goal_xy, w_head=w_head, w_speed=w_speed,
                w_effort=w_effort, w_lat=w_lat, w_wall=w_wall, w_coll=w_coll,
                hall_y0=hall_y0, hall_half_width=hall_half_width,
                r_safe_coll=r_safe_coll, v_nom=v_nom
            ),
            arg="x", weight=1.0
        )
        pc_r.add_cost(
            SocialNavigationPlayerCost(
                player_index=1,
                goals_self=jnp.asarray(goals_r, dtype=jnp.float32),
                goals_other=jnp.asarray(goals_h, dtype=jnp.float32),
                theta_self=self.theta_r, theta_other=0,
                horizon=horizon, name="NavCost_Robot",
                w_goal_xy=w_goal_xy, w_head=w_head, w_speed=w_speed,
                w_effort=w_effort, w_lat=w_lat, w_wall=w_wall, w_coll=w_coll,
                hall_y0=hall_y0, hall_half_width=hall_half_width,
                r_safe_coll=r_safe_coll, v_nom=v_nom
            ),
            arg="x", weight=1.0
        )

        zeros_P = jnp.zeros((2, 6, horizon), dtype=jnp.float32)
        zeros_a = jnp.zeros((2,    horizon), dtype=jnp.float32)

        self.solver = ILQSolver(
            dyn,
            [pc_h, pc_r],
            [zeros_P, zeros_P],
            [zeros_a, zeros_a],
            max_iter=max_iter,
            u_constraints=[
                BoxConstraint(jnp.array([v_lo, w_lo]), jnp.array([v_hi, w_hi])),
                BoxConstraint(jnp.array([v_lo, w_lo]), jnp.array([v_hi, w_hi])),
            ],
            verbose=verbose
        )

        self._Ps = None
        self._a  = None

    def run(self, x: jnp.ndarray):
        self.solver.run(x, Ps_warm=self._Ps, alphas_warm=self._a)
        self._Ps, self._a = self.solver._Ps, self.solver._alphas

    def xs_nom(self):
        return self.solver._best_operating_point[0]  # (nx,N)

    def nom(self, pl: int) -> np.ndarray:
        u = self.solver._best_operating_point[1][pl][:, 0]   # (2,)
        return np.array([float(u[0]), float(u[1])], dtype=float)

    def B_R(self, pl: int):
        xs, us = self.solver._best_operating_point
        _, Bs  = self.solver._linearize_dynamics(xs, us)
        _, _, _, H = self.solver._quadraticize_costs(xs, us)
        return Bs[pl][:, :, 0], H[pl][:, :, 0]  # B: (nx,2), R0: (2,2)

    def Zz_next(self, pl: int):
        Zs, zs = self.solver._best_op[4], self.solver._best_op[5]
        return Zs[pl][:, :, 1], zs[pl][:, 1]   # at k=1

# ---------- Navigation NPACE-Influence ----------
class NavigationNPACEInfluence:
    def __init__(
        self,
        *,
        # robot intent & goals
        theta_robot_true: int,
        goals_human: np.ndarray,
        goals_robot: np.ndarray,
        intents: Tuple[int, ...] = (0, 1),   # goal indices
        # time / horizon
        dt: float = 0.25,
        horizon: int = 10,
        # cost params (aligned with your nav sim)
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
        # ctrl limits
        v_lo: float = 0.0, v_hi: float = 1.0,
        w_lo: float = -0.4, w_hi: float = 0.4,
        # Bayes tuning (same logic)
        beta_state: float = 1.0,
        rho_forget: float = 0.00,
        # state noise diag for human Bayes (x=[x1,y1,th1,x2,y2,th2])
        sigma2_state: Tuple[float, float, float, float, float, float] = (
            625.0, 625.0, 0.0305, 625.0, 625.0, 0.0305
        ),
        # teaching strength
        gamma_teach: float = 0.0,
        # optional extra Q-MDP effort ridge (2λ I) like intersection’s effort_w (set 0 by default)
        effort_w_qmdp: float = 0.0,
        max_iter: int = 25,
        verbose: bool = False,
        beta: float = 1.0,
    ):
        
        self.beta    = float(beta)
        self._sigma2_action_obs = np.array([1e-2, 1e-2], dtype=float)
        # intents & indices
        self._intents = tuple(int(i) for i in intents)
        self._idx_of  = {th: i for i, th in enumerate(self._intents)}
        self._th_of   = {i: th for i, th in enumerate(self._intents)}
        self._ir_true = self._idx_of[int(theta_robot_true)]

        # goals
        self._goals_h = np.asarray(goals_human, dtype=np.float32).reshape((-1, 2))
        self._goals_r = np.asarray(goals_robot, dtype=np.float32).reshape((-1, 2))

        # constants
        self._dt = float(dt)
        self._gamma_teach = float(gamma_teach)
        self._effort_w_qmdp = float(effort_w_qmdp)

        # store control bounds for clipping
        self._v_lo, self._v_hi = float(v_lo), float(v_hi)
        self._w_lo, self._w_hi = float(w_lo), float(w_hi)

        # beliefs
        nH = len(self._intents)
        nR = len(self._intents)
        self._b_h = np.ones(nH, dtype=np.float64) / nH  # robot's belief over human θ_h
        uni = np.ones(nR, dtype=np.float64) / nR
        self._q_r: Dict[int, np.ndarray] = {ih: uni.copy() for ih in range(nH)}  # modeled human beliefs over robot θ_r, per θ_h

        # state-Bayes constants (6-D)
        self._inv_sigma2 = jnp.asarray([1.0 / s for s in sigma2_state], dtype=jnp.float32)
        self._lgv_const  = float(-0.5 * (np.sum(np.log(sigma2_state)) + 6 * math.log(2.0 * math.pi)))
        self._beta_state = float(beta_state)
        self._rho_forget = float(rho_forget)

        # solver params & bank
        self._params = dict(
            dt=self._dt, horizon=int(horizon),
            w_goal_xy=tuple(float(x) for x in w_goal_xy),
            w_head=float(w_head), w_speed=float(w_speed), w_effort=float(w_effort),
            w_lat=float(w_lat), w_wall=float(w_wall), w_coll=float(w_coll),
            v_nom=float(v_nom), hall_y0=float(hall_y0),
            hall_half_width=float(hall_half_width), r_safe_coll=float(r_safe_coll),
            v_lo=float(v_lo), v_hi=float(v_hi), w_lo=float(w_lo), w_hi=float(w_hi),
            max_iter=int(max_iter), verbose=bool(verbose),
        )
        self._solv: Dict[Tuple[int, int], _Solver2P_NPACE_Nav] = {}
        for ih, th_h in enumerate(self._intents):
            for ir, th_r in enumerate(self._intents):
                self._solv[(ih, ir)] = _Solver2P_NPACE_Nav(
                    theta_h=th_h, theta_r=th_r,
                    goals_h=self._goals_h, goals_r=self._goals_r,
                    **self._params
                )

        # caches
        self._pred_ctrl = None   # {"mu_u1": (nH,2), "Prec_u1": (nH,2,2)}
        self._x_cache   = None   # (nH, nR, nx)

        self._log_2pi = math.log(2.0 * math.pi)

    # ---- helper: box clip for 2D actions ----
    @staticmethod
    def _clip2(u: np.ndarray, v_lo: float, v_hi: float, w_lo: float, w_hi: float) -> np.ndarray:
        v = float(np.minimum(np.maximum(u[0], v_lo), v_hi))
        w = float(np.minimum(np.maximum(u[1], w_lo), w_hi))
        return np.array([v, w], dtype=np.float64)

    # inspectors
    @property
    def robot_belief_over_human(self) -> Dict[int, float]:
        return {self._th_of[ih]: float(self._b_h[ih]) for ih in range(len(self._intents))}

    @property
    def modeled_human_beliefs(self) -> Dict[int, Dict[int, float]]:
        return { self._th_of[ih]: { self._th_of[ir]: float(self._q_r[ih][ir])
                                    for ir in range(len(self._intents)) }
                 for ih in range(len(self._intents)) }

    # main step
    def compute_action(self, *, obs: np.ndarray, a1_observed: np.ndarray) -> np.ndarray:
        x = jnp.asarray(obs, dtype=jnp.float32)
        nH = len(self._intents); nR = len(self._intents)
        ir_true = self._ir_true

        # (A) Human's Bayes over robot θ_r based on current state (state-based)
        if self._x_cache is not None:
            x_cache = jnp.asarray(self._x_cache)         # (nH, nR, nx)
            errs    = x[None, None, :] - x_cache         # (nH, nR, nx)
            logp    = -0.5 * jnp.sum(errs * errs * self._inv_sigma2, axis=2)
            logp    = self._beta_state * logp + self._lgv_const
            for ih in range(nH):
                prior = jnp.asarray(self._q_r[ih])        # (nR,)
                log_post = jnp.log(prior) + logp[ih]
                log_post = log_post - jnp.max(log_post)   # numerical stability
                q_new = jax.nn.softmax(log_post)
                self._q_r[ih] = (
                    (1.0 - self._rho_forget) * np.array(q_new, dtype=np.float64)
                    + self._rho_forget * (np.ones(nR) / nR)
                )

        # (B) Robot's Bayes over human θ_h using observed u1 (vector likelihood)
        if self._pred_ctrl is not None:
            mu_arr   = jnp.asarray(self._pred_ctrl["mu_u1"])    # (nH,2)
            Prec_arr = jnp.asarray(self._pred_ctrl["Prec_u1"])  # (nH,2,2)
            u_obs = jnp.asarray(a1_observed, dtype=jnp.float32).reshape(2,)
            log_like = jnp.array([_mv_loglike_prec(u_obs, mu_arr[ih], Prec_arr[ih]) for ih in range(nH)])
            prior_j  = jnp.asarray(self._b_h)
            post = jnp.log(prior_j) + log_like
            post = post - jnp.max(post)                         # numerical stability
            self._b_h = np.array(jax.nn.softmax(post), dtype=np.float64)

        # (C) Solve all (θ_h, θ_r)
        for ih in range(nH):
            for ir in range(nR):
                self._solv[(ih, ir)].run(x)

        # (D) Human Q-MDP over θ_r (vector), also collect x_next for cache
        H1_mat = [[None for _ in range(nR)] for _ in range(nH)]
        S1_mat = [[None for _ in range(nR)] for _ in range(nH)]
        l1_mat = [[None for _ in range(nR)] for _ in range(nH)]
        u1_mat = [[None for _ in range(nR)] for _ in range(nH)]
        nx = 6
        x1   = np.zeros((nH, nR, nx), dtype=np.float32)

        for ih in range(nH):
            for ir in range(nR):
                s = self._solv[(ih, ir)]
                B1, R01  = s.B_R(pl=0)
                Z1, z1   = s.Zz_next(pl=0)
                H1 = np.asarray(B1.T @ Z1 @ B1, dtype=np.float64)      # (2,2)
                S1 = np.asarray(R01 + H1,        dtype=np.float64)      # (2,2)
                l1 = np.asarray(B1.T @ z1,       dtype=np.float64).reshape(2,)
                u1 = s.nom(pl=0)                                        # (2,)

                H1_mat[ih][ir] = H1
                S1_mat[ih][ir] = S1
                l1_mat[ih][ir] = l1
                u1_mat[ih][ir] = u1
                x1[ih, ir, :]  = np.asarray(s.xs_nom()[:, 1], dtype=np.float32)

        # mix over θ_r with q_r[ih]
        a1_Q = np.zeros((nH, 2), dtype=np.float64)
        for ih in range(nH):
            q = np.asarray(self._q_r[ih], dtype=np.float64)  # (nR,)
            S_bar = np.zeros((2,2), dtype=np.float64)
            num   = np.zeros((2,),   dtype=np.float64)
            for ir in range(nR):
                H1 = H1_mat[ih][ir]; S1 = S1_mat[ih][ir]; l1 = l1_mat[ih][ir]; u0 = u1_mat[ih][ir]
                S_bar += q[ir] * S1
                num   += q[ir] * (H1 @ u0 - l1)
            if self._effort_w_qmdp > 0.0:
                S_bar = S_bar + 2.0 * self._effort_w_qmdp * np.eye(2)
            S_bar = _spd_guard(S_bar, 1e-9)
            try:
                a1_Q[ih, :] = np.linalg.solve(S_bar, num)
            except np.linalg.LinAlgError:
                a1_Q[ih, :] = np.linalg.pinv(S_bar) @ num
            a1_Q[ih, :] = self._clip2(a1_Q[ih, :], self._v_lo, self._v_hi, self._w_lo, self._w_hi)

        a1_nom_true = np.array([u1_mat[ih][ir_true] for ih in range(nH)], dtype=float)  # (nH,2)

        # (E) Robot Q-MDP across θ_h with coupling to Δu1  +  TEACHING (vector)
        S2_arr   = [None for _ in range(nH)]       # each (2,2)
        bias_arr = [None for _ in range(nH)]       # each (2,)
        for ih in range(nH):
            s      = self._solv[(ih, ir_true)]
            B2, R02  = s.B_R(pl=1)
            Z2, z2   = s.Zz_next(pl=1)
            B1_true, _ = s.B_R(pl=0)

            # human deviation
            delta_u1 = (a1_Q[ih, :] - a1_nom_true[ih, :]).reshape(2,)

            # H2 and S2 (denominator)
            H2 = np.asarray(B2.T @ Z2 @ B2, dtype=np.float64)          # (2,2)
            S2 = np.asarray(R02 + H2,        dtype=np.float64)          # (2,2)

            # linear part: ℓ + C δu1  (C symmetric cross-term)
            l2_lin  = np.asarray(B2.T @ z2, dtype=np.float64).reshape(2,)   # ℓ = B2^T z2
            C_cross = 0.5 * (np.asarray(B2.T @ Z2 @ B1_true, dtype=np.float64) +
                              np.asarray(B1_true.T @ Z2 @ B2, dtype=np.float64).T)  # (2,2)
            l2_adj  = l2_lin + C_cross @ delta_u1

            # teaching term (bounded)
            u2_all  = np.array([self._solv[(ih, ir)].nom(pl=1) for ir in range(nR)], dtype=float)  # (nR,2)
            u2_true = u2_all[ir_true, :]
            q_vec   = np.asarray(self._q_r[ih], dtype=float)                                       # (nR,)
            gap_vec = (q_vec @ u2_all) - u2_true                                                  # (2,)
            gap_cap = np.array([self._v_hi - self._v_lo, self._w_hi - self._w_lo], dtype=float)
            #gap_vec = np.clip(gap_vec, -gap_cap, gap_cap)

            # sensitivity κ ≈ PSD(B2ᵀ Σ^{-1} B2), use diag as per-intervention weight
            inv_sig = np.asarray(self._beta_state * self._inv_sigma2, dtype=float)  # (nx,)
            B2_np   = np.asarray(B2, dtype=float)  # (nx,2)
            K = (B2_np.T * inv_sig) @ B2_np                    # (2,2)
            K = 0.5 * (K + K.T)
            w, V = np.linalg.eigh(K); w = np.clip(w, 0.0, None)
            K_psd = (V * w) @ V.T 
            kappa_diag = np.diag(K_psd) + 1e-12                # strictly positive
            #g_teach_vec = kappa_diag * gap_vec *self._q_r[ir_true]     #
            g_teach_vec = K_psd @ gap_vec *(1-self._q_r[ir_true])              # (2,)

            # numerator bias for robot (affine):
            u2_nom_true = s.nom(pl=1)  # (2,)
            bias_arr[ih] = H2 @ u2_nom_true - (l2_adj + self._gamma_teach * g_teach_vec)
            S2_arr[ih]   = S2

        # final mixing over θ_h
        b_h = np.asarray(self._b_h, dtype=np.float64)   # (nH,)
        S_bar = np.zeros((2,2), dtype=np.float64)
        num   = np.zeros((2,),   dtype=np.float64)
        for ih in range(nH):
            S_bar += b_h[ih] * S2_arr[ih]
            num   += b_h[ih] * bias_arr[ih]
        if self._effort_w_qmdp > 0.0:
            S_bar = S_bar + 2.0 * self._effort_w_qmdp * np.eye(2)
        S_bar = _spd_guard(S_bar, 1e-9)
        try:
            a2_cmd = np.linalg.solve(S_bar, num)
        except np.linalg.LinAlgError:
            a2_cmd = np.linalg.pinv(S_bar) @ num
        a2_cmd = self._clip2(a2_cmd, self._v_lo, self._v_hi, self._w_lo, self._w_hi)

        # (F) caches: predicted control stats for Bayes, and x-cache
        mu_u1  = a1_Q.copy()                       # (nH,2)
        Prec_u1= np.zeros((nH, 2, 2), dtype=np.float64)
        for ih in range(nH):
            S1_true = np.zeros((2,2), dtype=np.float64)
            for ir in range(nR):
                S1_true += float(self._q_r[ih][ir]) * S1_mat[ih][ir]
            if self._effort_w_qmdp > 0.0:
                S1_true = S1_true + 2.0 * self._effort_w_qmdp * np.eye(2)
            Prec_policy  = self.beta * _spd_guard(S1_true, 1e-9)
            Sigma_sens  = np.diag(self._sigma2_action_obs)
            Sigma_policy = np.linalg.inv(Prec_policy)  # or pinv]
            Sigma_total  = Sigma_policy + Sigma_sens
            Prec_total   = np.linalg.inv(Sigma_total)  # or pinv
            #Prec_u1[ih, :, :] = self.beta*_spd_guard(S1_true, 1e-9)   # Σ^{-1}
            Prec_u1[ih, :, :] = self.beta*_spd_guard(Prec_total, 1e-9)   # Σ^{-1}

        self._pred_ctrl = {"mu_u1": mu_u1, "Prec_u1": Prec_u1}

        # Cache predicted next states using human QMDP action (per θ_h)
        nx = int(x.shape[0])  # 6
        x_cache = np.zeros((nH, nR, nx), dtype=np.float32)
        for ih in range(nH):
            u1_pred = a1_Q[ih, :].reshape(2,)
            for ir in range(nR):
                u2_pred = self._solv[(ih, ir)].nom(pl=1).reshape(2,)
                x_next = Dyn.disc_time_dyn(
                    x,
                    [jnp.asarray(u1_pred, dtype=jnp.float32),
                     jnp.asarray(u2_pred, dtype=jnp.float32)]
                )
                x_cache[ih, ir, :] = np.asarray(x_next, dtype=np.float32)
        self._x_cache = x_cache

        return a2_cmd  # np.array([v2, w2])
