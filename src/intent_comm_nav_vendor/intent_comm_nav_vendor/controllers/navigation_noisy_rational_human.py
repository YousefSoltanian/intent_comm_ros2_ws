#!/usr/bin/env python
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

# NOTE: same symbol as your intersection code, but now from navigation controller.
# Expect _Solver2P to expose: run(x), B_R(pl), Zz_next(pl), nom(pl)
from .navigation_blame_me_controller import _Solver2P # required import
from iLQGame.multiplayer_dynamical_system import PlanarNavigation2PlayerSystem


class NavigationNoisyRationalHuman:
    """
    Noisy-rational human controller (navigation, 2-D control [v, ω]).

    Human knows its own intent θ_self (index into its goal set) and is uncertain
    only over the robot's intent θ_r ∈ intents (indices into robot's goal set).
    Belief over θ_r is updated via state-based Bayes using predicted next states x_{t+1}.

    Q-MDP mixture over θ_r branches (EFFORT-ONLY discussion kept; here we use S_i = R0_i + Bᵀ Z1 B):
        For each branch i:
            H_i = B_i^T Z1_i B_i      (2×2)
            l_i = B_i^T z1_i          (2,)
            u0_i = nominal control    (2,)  from branch i

        μ_u  solves:
            [ Σ_i b_i * S_i ] μ_u  =  Σ_i b_i * ( H_i u0_i − l_i )

        (Optionally you can add + 2*effort_w * I to the LHS; see commented code.)

    Noisy-rational sampling:
        u ~ N( μ_u,  (β * S̄)^(-1) ),   where S̄ = Σ_i b_i * S_i

    Notes
    -----
    • Intent indices refer to rows in the provided goal sets (goals_self/goals_opp).
    • Controls are 2-D (v, ω); clipping uses [v_lo, v_hi] × [w_lo, w_hi].
    """

    def __init__(
        self,
        *,
        goals_self: np.ndarray,
        goals_opp: np.ndarray,
        theta_self: int,
        ctrl_index_self: int = 0,            # which player's control this human outputs (0 or 1)
        intents: Tuple[int, ...] = (0, 1),   # opponent intent set = goal indices into goals_opp
        # rollout / ILQ
        dt: float = 0.25,
        horizon: int = 10,
        max_iter: int = 25,
        verbose: bool = False,
        # effort/weights (effort kept for comments; solver exposes R0 anyway)
        effort_w: float = 10.0,
        # corridor / safety (passed through to solver; names kept compact)
        w_goal_xy: Tuple[float, float] = (6.0, 6.0),
        w_head: float = 1.1,
        w_speed: float = 0.0,
        w_effort: float = 10.0,
        w_lat: float = 0.2,
        w_wall: float = 60.0,
        w_coll: float = 15.0,
        v_nom: float = 0.9,
        hall_y0: float = 0.0,
        hall_half_width: float = 2.41,
        d_safe_wall: float = 0.25,
        r_safe_coll: float = 2.0,
        # control bounds
        v_lo: float = 0.0, v_hi: float = 1.0,
        w_lo: float = -0.4, w_hi: float = 0.4,
        # stochasticity / Bayes
        beta: float = 1.0,              # inverse temperature for action noise
        stochastic: bool = True,
        seed: Optional[int] = None,
        beta_state: float = 1.0,        # state-based Bayes scale
        rho_forget: float = 0.00,       # forgetting toward uniform
        sigma2_state: Tuple[float, float, float, float, float, float] = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
    ):
        # intent set (opponent goal indices)
        self._intents = tuple(int(i) for i in intents)
        self._ctrl_i  = int(ctrl_index_self)
        self._theta_self = int(theta_self)

        # goal sets (rows = candidate goals)
        self._goals_self = np.asarray(goals_self, dtype=np.float32).reshape((-1, 2))
        self._goals_opp  = np.asarray(goals_opp,  dtype=np.float32).reshape((-1, 2))

        # initial uniform belief over opponent intents
        self._belief = np.ones(len(self._intents), dtype=np.float64)
        self._belief /= self._belief.sum()

        # parameters for solvers (kept bundled as in your intersection code)
        self._params = dict(
            dt=float(dt), horizon=int(horizon),
            # pass-through weights for the navigation cost (solver will use these)
            w_goal_xy=tuple(float(x) for x in w_goal_xy),
            w_head=float(w_head), w_speed=float(w_speed), w_effort=float(w_effort),
            w_lat=float(w_lat), w_wall=float(w_wall), w_coll=float(w_coll),
            v_nom=float(v_nom), hall_y0=float(hall_y0), hall_half_width=float(hall_half_width),
            d_safe_wall=float(d_safe_wall),
            r_safe_coll=float(r_safe_coll),
            # legacy effort scalar kept for the optional +2*effort_w*I note
            effort_w=float(effort_w),
            # solver controls / verbosity
            max_iter=int(max_iter), verbose=bool(verbose),
            # bounds
            v_lo=float(v_lo), v_hi=float(v_hi), w_lo=float(w_lo), w_hi=float(w_hi),
        )

        # inverse temperature (action noise) and state-Bayes scales
        self._beta = float(beta)
        self._beta_state = float(beta_state)
        self._stochastic = bool(stochastic)
        self._rng = np.random.default_rng(seed)

        self._rho_forget = float(rho_forget)
        self._inv_sigma2 = jnp.asarray([1.0 / s for s in sigma2_state], dtype=jnp.float32)
        self._lgv_const  = float(-0.5 * (np.sum(np.log(sigma2_state)) + 6 * math.log(2.0 * math.pi)))

        # cache predicted next states x_{t+1} per robot intent
        self._state_cache = None  # shape (nIntents, nx)

        # build ILQ solvers for each robot-intent branch (human intent fixed)
        self._solvers: List[_Solver2P] = []
        self._build_solvers(theta_self=self._theta_self)
        self._dyn = PlanarNavigation2PlayerSystem(T=dt)

        # optional: external code may set self._ir_true for sampling variance branch
        self._log_2pi = math.log(2.0 * math.pi)

    def _build_solvers(self, *, theta_self: int):
        p = self._params
        # Each branch i uses a different robot intent (goal index into goals_opp)
        self._solvers = [
            _Solver2P(
                # intents / goals
                theta_self=theta_self,              # human known intent (index into goals_self)
                theta_opp=int(th),                  # robot intent branch (index into goals_opp)
                goals_self=self._goals_self,        # full goal sets
                goals_opp=self._goals_opp,
                ctrl_index_self=self._ctrl_i,
                # horizon / dt
                horizon=p["horizon"], dt=p["dt"],
                # cost weights
                w_goal_xy=p["w_goal_xy"], w_head=p["w_head"], w_speed=p["w_speed"],
                w_effort=p["w_effort"], w_lat=p["w_lat"], w_wall=p["w_wall"], w_coll=p["w_coll"],
                v_nom=p["v_nom"], hall_y0=p["hall_y0"], hall_half_width=p["hall_half_width"],
                d_safe_wall=p["d_safe_wall"],
                r_safe_coll=p["r_safe_coll"],
                # bounds
                v_lo=p["v_lo"], v_hi=p["v_hi"], w_lo=p["w_lo"], w_hi=p["w_hi"],
                # ILQ params
                max_iter=p["max_iter"], verbose=p["verbose"]
            )
            for th in self._intents
        ]

    @staticmethod
    def _clip2(u: np.ndarray, v_lo: float, v_hi: float, w_lo: float, w_hi: float) -> np.ndarray:
        v = float(np.minimum(np.maximum(u[0], v_lo), v_hi))
        w = float(np.minimum(np.maximum(u[1], w_lo), w_hi))
        return np.array([v, w], dtype=np.float64)

    def set_seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def compute_action(
        self,
        *,
        obs: np.ndarray,
        a_opponent_observed: float,  # unused (state-based update, kept for API symmetry)
        theta_self: Optional[int] = None
    ) -> np.ndarray:
        # update our own intent if changed (still known perfectly)
        if theta_self is not None and int(theta_self) != self._theta_self:
            self._theta_self = int(theta_self)
            self._build_solvers(theta_self=self._theta_self)

        x = jnp.asarray(obs, dtype=jnp.float32)

        # Run ILQ for each robot-intent branch to get current linearizations
        for s in self._solvers:
            s.run(x)

        # 1) State-based Bayes over robot intents using predicted x_{t+1}
        if self._state_cache is not None:
            # err shape: (nIntents, nx)
            err  = x[None, :] - jnp.asarray(self._state_cache)
            logp = -0.5 * jnp.sum(err * err * self._inv_sigma2, axis=1)  # ∝ log N(x | x̂, Σ)
            logp = 1.0 * self._beta_state * logp + self._lgv_const       # scale + const
            logw = jnp.log(self._belief) + logp
            logw = logw - jnp.max(logw)                                  # numerical stability
            new_b = jax.nn.softmax(logw)
            # forgetting toward uniform (optional; keeps numbers aligned with NPACE)
            uniform = jnp.ones_like(new_b) / new_b.shape[0]
            new_b = (1.0 - self._rho_forget) * new_b + self._rho_forget * uniform
            self._belief = np.array(new_b, dtype=np.float64)

        # 2) Q-MDP mixture for the human control (vector case)
        pl_self = self._ctrl_i
        nInt = len(self._solvers)

        # Accumulate per-branch curvature/linearization terms
        # H_i = B^T Z1 B (2×2), S_i = R0 + B^T Z1 B (2×2), l_i = B^T z1 (2,), u0_i ∈ R^2
        H_list: List[np.ndarray] = []
        S_list: List[np.ndarray] = []
        l_list: List[np.ndarray] = []
        u0_list: List[np.ndarray] = []

        for s in self._solvers:
            B_i, R0_i   = s.B_R(pl_self)          # B: (nx×2), R0: (2×2)
            Z1_i, z1_i  = s.Zz_next(pl_self)      # Z1: (nx×nx), z1: (nx,)
            H_i = (B_i.T @ Z1_i @ B_i)
            S_i = (R0_i + H_i)                    # keep R0 in S (more stable than "effort-only")
            l_i = (B_i.T @ z1_i).reshape(2)
            u0_i = np.asarray(s.nom(pl_self), dtype=np.float64).reshape(2)

            H_list.append(np.asarray(H_i, dtype=np.float64))
            S_list.append(np.asarray(S_i, dtype=np.float64))
            l_list.append(np.asarray(l_i, dtype=np.float64))
            u0_list.append(u0_i)

        # Weighted sums
        b = self._belief
        S_bar = np.zeros((2, 2), dtype=np.float64)
        num   = np.zeros((2,),   dtype=np.float64)

        for i in range(nInt):
            S_bar += b[i] * S_list[i]
            num   += b[i] * (H_list[i] @ u0_list[i] - l_list[i])

        # Optionally add + 2*effort_w * I (kept from your intersection notes)
        # S_bar = S_bar + 2.0 * self._params["effort_w"] * np.eye(2)

        # Solve for μ_u (with tiny ridge for numerical hygiene)
        S_bar_reg = S_bar + 1e-9 * np.eye(2)
        try:
            mu_u = np.linalg.solve(S_bar_reg, num)
        except np.linalg.LinAlgError:
            mu_u = np.linalg.pinv(S_bar_reg) @ num

        # 3) Noisy-rational sampling: cov = (β * S̄)^(-1)
        if self._stochastic:
            Prec = self._beta * (S_bar + 1e-9 * np.eye(2))
            try:
                Cov = np.linalg.inv(Prec)
            except np.linalg.LinAlgError:
                Cov = np.linalg.pinv(Prec)
            # sample z ~ N(0, I), then u = μ + L z with L L^T = Cov
            try:
                L = np.linalg.cholesky(Cov)
            except np.linalg.LinAlgError:
                # fall back to symmetric sqrt via eig
                w, V = np.linalg.eigh(Cov)
                w = np.clip(w, 0.0, None)
                L = V @ np.diag(np.sqrt(w)) @ V.T
            z = self._rng.standard_normal(size=(2,))
            u_cmd = (mu_u + L @ z)
        else:
            u_cmd = mu_u

        # clip to box constraints
        u_cmd = self._clip2(u_cmd, self._params["v_lo"], self._params["v_hi"],
                            self._params["w_lo"], self._params["w_hi"])

        # 4) Cache predicted next states for the next Bayes step
        xs_next = np.zeros((nInt, x.shape[0]), dtype=np.float32)
        for i, s in enumerate(self._solvers):
            u2_pred = np.asarray(s.nom(pl=1 - pl_self), dtype=np.float64).reshape(2)  # predicted opponent action
            x_next = self._dyn.disc_time_dyn(
                jnp.asarray(x, dtype=jnp.float32),
                [jnp.array(u_cmd,  dtype=jnp.float32),
                 jnp.array(u2_pred, dtype=jnp.float32)]
            )
            xs_next[i, :] = np.asarray(x_next, dtype=np.float32)
        self._state_cache = xs_next

        return u_cmd  # shape (2,) → [v, ω]

    @property
    def belief_over_theta(self) -> Dict[int, float]:
        return {int_th: float(p) for int_th, p in zip(self._intents, self._belief)}

    @property
    def intent_order(self) -> Tuple[int, ...]:
        return self._intents
