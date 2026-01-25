# ──────────────────────────────────────────────────────────────────────────────
#  src/iLQGame/ilq_solver.py
#  Fully-JAX iterative-LQ (Nash) solver — single solve reuse + tighter assembly
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import time
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, tree_map
from functools import partial
from jaxlib.xla_extension import ArrayImpl  # only for typing

# your project modules ---------------------------------------------------------
from .player_cost       import PlayerCost
from .constraint        import BoxConstraint
#  the dynamics class (e.g. LunarLander2PlayerSystem) is passed by the caller
# ──────────────────────────────────────────────────────────────────────────────


class ILQSolver:
    """Iterative LQ Nash solver – everything runs inside JAX."""

    # ───────────────────────────────────── constructor / reset
    def __init__(
        self,
        dynamics,
        player_costs: List[PlayerCost],
        Ps:      List[jnp.ndarray],
        alphas:  List[jnp.ndarray],
        *,
        alpha_scaling=jnp.linspace(0.1, 2.0, 4),
        max_iter: int = 100,
        u_constraints: List[BoxConstraint] | None = None,
        verbose: bool = False,
        name: str | None = None,
    ):
        self._dyn           = dynamics
        self._costs         = player_costs
        self._Ps_init       = Ps
        self._alphas_init   = alphas
        self._horizon       = Ps[0].shape[2]
        self._num_players   = len(Ps)
        self._alpha_scaling = jnp.asarray(alpha_scaling)
        self._max_iter      = max_iter
        self._u_bounds      = u_constraints
        self._verbose       = verbose
        self._name          = name
        self.reset()

    # -------------------------------------------------------------------------
    def reset(self, Ps_warm=None, alphas_warm=None):
        self._Ps     = Ps_warm     if Ps_warm     is not None else self._Ps_init
        self._alphas = alphas_warm if alphas_warm is not None else self._alphas_init

        x_dim, H   = self._dyn._x_dim, self._horizon
        u_dims     = self._dyn._u_dims
        zero_xs    = jnp.zeros((x_dim, H))
        zero_us    = [jnp.zeros((d, H)) for d in u_dims]

        # operating-point tuple: (xs, us_list, Ps, alphas, Zs, zetas, costs)
        self._current_op = (
            zero_xs, zero_us,
            [jnp.zeros_like(p) for p in self._Ps],
            [jnp.zeros_like(a) for a in self._alphas],
            [jnp.zeros((x_dim, x_dim, H+1)) for _ in range(self._num_players)],
            [jnp.zeros((x_dim, H+1))        for _ in range(self._num_players)],
            jnp.full((self._num_players,), jnp.inf),
        )
        self._best_op   = self._current_op
        self._best_cost = jnp.inf

        # Ensure callers can read a valid operating point immediately
        self._best_operating_point = self._current_op[:2]   # (xs, us_list)

    # ────────────────────────────────────────── public interface
    def run(self, x0: jnp.ndarray, Ps_warm=None, alphas_warm=None):
        self.reset(Ps_warm, alphas_warm)
        carry0 = (self._current_op, self._best_op, self._best_cost, x0)
        carry_final, _ = self._run_jitted(carry0)
        self._current_op, self._best_op, self._best_cost, _ = carry_final
        # expose the best operating-point just like the original code
        self._best_operating_point = self._best_op[:2]   # (xs, us_list)
        self._Ps, self._alphas     = self._best_op[2], self._best_op[3]

    # ────────────────────────────────────────── main jit-compiled loop
    @partial(jit, static_argnums=0)
    def _run_jitted(self, carry0):
        """JIT-compiled outer iLQ iterations."""

        def one_iter(carry, _):
            current_op, best_op, best_cost, x0 = carry
            xs_hist, us_hist, Ps, alphas, *_ = current_op

            # ═════════════ line search over alpha_scaling ════════════════
            def ls_body(ls_carry, a_scale):
                best_xs, best_us, best_al, best_c = ls_carry

                al_scaled = [a_scale * a for a in alphas]
                xs_ls, us_ls, cost_ls   = self._compute_operating_point(
                    xs_hist, us_hist, Ps, al_scaled, x0
                )
                total_ls = jnp.sum(jnp.stack([jnp.sum(c) for c in cost_ls]))

                better = total_ls < best_c
                best_xs = lax.select(better, xs_ls, best_xs)
                best_us = tree_map(lambda n, o: lax.select(better, n, o),
                                   us_ls, best_us)
                best_al = tree_map(lambda n, o: lax.select(better, n, o),
                                   al_scaled, best_al)
                best_c  = lax.select(better, total_ls, best_c)
                return (best_xs, best_us, best_al, best_c), None

            init_ls = (xs_hist, us_hist, alphas, jnp.inf)
            (xs, us_list, al_star, cost_ls), _ = lax.scan(
                ls_body, init_ls, self._alpha_scaling
            )

            # ═════════════ linearization & quadraticization ══════════════
            As, Bs_list            = self._linearize_dynamics(xs, us_list)
            costs, lxs, Hxxs, Huus = self._quadraticize_costs(xs, us_list)

            Ps_new, al_new, Zs, zetas = self._solve_lq_game(
                jax.lax.stop_gradient(As),
                [jax.lax.stop_gradient(b) for b in Bs_list],
                [jax.lax.stop_gradient(hx) for hx in Hxxs],
                [jax.lax.stop_gradient(lx) for lx in lxs],
                [jax.lax.stop_gradient(hu) for hu in Huus],
            )

            new_cost = jnp.sum(jnp.stack([jnp.sum(c) for c in costs]))
            new_op   = (xs, us_list, Ps_new, al_new, Zs, zetas,
                        jnp.stack([jnp.sum(c) for c in costs]))

            better_global = new_cost < best_cost
            best_op   = tree_map(lambda n, o: lax.select(better_global, n, o),
                                 new_op, best_op)
            best_cost = lax.select(better_global, new_cost, best_cost)

            return (new_op, best_op, best_cost, x0), ()

        return lax.scan(one_iter, carry0, None, length=self._max_iter)

    # ───────────────────────────── helper – compute operating point (JIT)
    @partial(jit, static_argnums=0)
    def _compute_operating_point(
        self,
        xs_hist, us_hist, Ps, alphas,
        x0: jnp.ndarray,
    ):
        H, nP = self._horizon, self._num_players
        xs = xs_hist.at[:, 0].set(x0)
        us = us_hist

        def step(carry, k):
            xs_c, us_c = carry
            x_ref = xs_hist[:, k]
            for i in range(nP):
                u = us_c[i][:, k]
                u = u - Ps[i][:, :, k] @ (xs_c[:, k] - x_ref) - alphas[i][:, k]
                if self._u_bounds:
                    u = self._u_bounds[i].clip(u)
                us_c[i] = us_c[i].at[:, k].set(u)
            x_next = self._dyn.disc_time_dyn(xs_c[:, k], [us_c[i][:, k] for i in range(nP)])
            xs_c   = xs_c.at[:, k+1].set(x_next)
            return (xs_c, us_c), None

        (xs_out, us_out), _ = lax.scan(step, (xs, us), jnp.arange(H-1))
        xs_out = xs_out[:, :H]
        costs  = [self._costs[i].get_cost(xs_out, us_out[i]) for i in range(nP)]
        return xs_out, us_out, costs

    # ───────────────────────────── helper – linearize dynamics (JIT)
    @partial(jit, static_argnums=0)
    def _linearize_dynamics(self, xs, us_list):
        n, H = self._dyn._x_dim, self._horizon
        u_dims = self._dyn._u_dims
        As = jnp.zeros((n, n, H))
        Bs = [jnp.zeros((n, u_dims[i], H)) for i in range(self._num_players)]

        def body(k, vals):
            A_k, B_k = self._dyn.linearize_discrete_jitted(
                xs[:, k], [u[:, k] for u in us_list], k
            )
            As_, Bs_ = vals
            As_ = As_.at[:, :, k].set(A_k)
            for i in range(self._num_players):
                Bs_[i] = Bs_[i].at[:, :, k].set(B_k[i])
            return As_, Bs_

        As, Bs = lax.fori_loop(0, H, body, (As, Bs))
        return As, Bs

    # ───────────────────────────── helper – quadraticize costs (JIT)
    @partial(jit, static_argnums=0)
    def _quadraticize_costs(self, xs, us_list):
        costs, lxs, Hxxs, Huus = [], [], [], []
        for i in range(self._num_players):
            c, lx, _, Hxx, Huu = self._costs[i].quadraticize_jitted(xs, us_list[i])
            costs.append(c); lxs.append(lx); Hxxs.append(Hxx); Huus.append(Huu)
        return costs, lxs, Hxxs, Huus

    # ───────────────────────────── helper – solve LQ game (JIT)
    @partial(jit, static_argnums=0)
    def _solve_lq_game(self, A, B_list, Q_list, l_list, R_list):
        """
        Backward sweep (unchanged math), optimized:
          • Single linear solve on (S + εI) reused for both P_big and a_big.
          • Assemble S, Y, Y2 with preallocated blocks (no list-concats).
          • Use a stacked B to form F_k and beta with one matmul each.
          • Tiny Tikhonov ε for numerical stability at small DT.
        """
        H, n = self._horizon, self._dyn._x_dim
        m    = self._dyn._u_dims
        kP   = self._num_players
        m_tot = sum(m)
        eps  = 1e-8  # numeric regularizer (does not change the algorithm)

        P   = [jnp.zeros((m[i], n, H)) for i in range(kP)]
        alp = [jnp.zeros((m[i], H))    for i in range(kP)]
        Z   = [jnp.zeros((n, n, H+1))  for _ in range(kP)]
        ze  = [jnp.zeros((n, H+1))     for _ in range(kP)]

        I_n    = jnp.eye(n)
        I_big  = jnp.eye(m_tot)

        for i in range(kP):
            # regularize terminal Riccati a touch for stability
            Z[i]  = Z[i].at[:, :, -1].set(Q_list[i][:, :, -1] + eps * I_n)
            ze[i] = ze[i].at[:, -1].set(l_list[i][:, -1])

        # Player block offsets (static ints)
        offs = []
        o = 0
        for i in range(kP):
            offs.append(o)
            o += m[i]

        def back_body(idx, carry):
            # idx = 0..H-1 → k = H-1-idx
            k = H - 1 - idx
            P_, alp_, Z_, ze_ = carry

            A_k = A[:, :, k]
            B_k = [B_list[i][:, :, k] for i in range(kP)]
            Q_k = [Q_list[i][:, :, k] for i in range(kP)]
            l_k = [l_list[i][:, k]    for i in range(kP)]
            R_k = [R_list[i][:, :, k] for i in range(kP)]
            Z_n = [Z_[i][:, :, k+1]   for i in range(kP)]
            z_n = [ze_[i][:, k+1]     for i in range(kP)]

            # Preallocate blocks
            S   = jnp.zeros((m_tot, m_tot), dtype=A_k.dtype)
            Y   = jnp.zeros((m_tot, n),     dtype=A_k.dtype)
            Y2  = jnp.zeros((m_tot,),       dtype=A_k.dtype)
            Bstk= jnp.zeros((n, m_tot),     dtype=A_k.dtype)

            # Fill blocks
            for i in range(kP):
                oi, mi = offs[i], m[i]
                Y    = Y.at[oi:oi+mi, :].set(B_k[i].T @ Z_n[i] @ A_k)
                Y2   = Y2.at[oi:oi+mi].set(B_k[i].T @ z_n[i])
                Bstk = Bstk.at[:, oi:oi+mi].set(B_k[i])
                for j in range(kP):
                    oj, mj = offs[j], m[j]
                    blk = B_k[i].T @ Z_n[i] @ B_k[j]
                    blk = blk + (R_k[i] if i == j else 0.0)
                    S   = S.at[oi:oi+mi, oj:oj+mj].set(blk)

            # One linear solve on (S + eps*I)
            S_reg = S + eps * I_big
            P_big = jnp.linalg.solve(S_reg, Y)   # (m_tot, n)
            a_big = jnp.linalg.solve(S_reg, Y2)  # (m_tot,)

            # Scatter to per-player blocks
            for i in range(kP):
                oi, mi = offs[i], m[i]
                P_[i]   = P_[i].at[:, :, k].set(P_big[oi:oi+mi, :])
                alp_[i] = alp_[i].at[:, k].set(a_big[oi:oi+mi])

            # F_k and beta using stacked B
            F_k  = A_k - Bstk @ P_big
            beta = -(Bstk @ a_big)

            # Update Riccati terms
            for i in range(kP):
                Zi_next = Z_n[i]
                Pi_k    = P_[i][:, :, k]
                Z_[i] = Z_[i].at[:, :, k].set(
                    F_k.T @ Zi_next @ F_k +
                    Q_k[i] +
                    Pi_k.T @ R_k[i] @ Pi_k
                )
                ze_[i] = ze_[i].at[:, k].set(
                    F_k.T @ (z_n[i] + Zi_next @ beta) +
                    l_k[i] +
                    Pi_k.T @ R_k[i] @ alp_[i][:, k]
                )

            return (P_, alp_, Z_, ze_)

        P, alp, Z, ze = lax.fori_loop(0, H, back_body, (P, alp, Z, ze))
        return P, alp, Z, ze
