# ──────────────────────────────────────────────────────────────────────────────
#  src/iLQGame/util.py
#  Minimal JAX TV‑LQ Nash solver (used only to get α₀ gradients)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial

@partial(jit, static_argnums=(0,))
def solve_lq_game_jax(A, B_list, Q_list, l_list, R_list):
    """
    Pure‑JAX backward pass identical to the one inside your ILQSolver, but
    **stateless** and therefore differentiable.

    Inputs:
      A          : (n, n, H)
      B_list[i]  : (n, m_i, H)
      Q_list[i]  : (n, n, H)
      l_list[i]  : (n,     H)
      R_list[i]  : (m_i, m_i, H)

    Returns lists (Ps, alphas, Zs, zetas) with the same shapes as in the solver.
    """
    H         = A.shape[2]
    n         = A.shape[0]
    kP        = len(B_list)
    m         = [B.shape[1] for B in B_list]

    Ps   = [jnp.zeros((m[i], n, H)) for i in range(kP)]
    alp  = [jnp.zeros((m[i],   H)) for i in range(kP)]
    Z    = [jnp.zeros((n, n, H+1)) for _ in range(kP)]
    zeta = [jnp.zeros((n,   H+1)) for _ in range(kP)]

    for i in range(kP):
        Z[i]   = Z[i].at[:,:, -1].set(Q_list[i][:,:,-1])
        zeta[i]= zeta[i].at[:, -1].set(l_list[i][:,-1])

    def one_step(carry, k):
        Ps, alp, Z, zeta = carry
        k_ = H-1-k                          # reverse loop index

        A_k   = A[:,:,k_]
        B_k   = [B[:,:,k_] for B in B_list]
        Q_k   = [Q[:,:,k_] for Q in Q_list]
        l_k   = [l[:,k_]   for l in l_list]
        R_k   = [R[:,:,k_] for R in R_list]
        Z_n   = [Z[i][:,:,k_+1] for i in range(kP)]
        z_n   = [zeta[i][:,k_+1] for i in range(kP)]

        # build S
        rows = []
        for i in range(kP):
            row = []
            for j in range(kP):
                blk = B_k[i].T @ Z_n[i] @ B_k[j]
                blk += R_k[i] if i == j else 0.0
                row.append(blk)
            rows.append(jnp.concatenate(row,1))
        S = jnp.concatenate(rows,0)
        S_inv = jnp.linalg.pinv(S)

        Y = jnp.concatenate([B_k[i].T @ Z_n[i] @ A_k for i in range(kP)],0)
        P_big = S_inv @ Y

        off = 0
        for i in range(kP):
            Ps[i]  = Ps[i].at[:,:,k_].set(P_big[off:off+m[i], :])
            off   += m[i]

        F_k = A_k - sum(B_k[i] @ Ps[i][:,:,k_] for i in range(kP))

        for i in range(kP):
            Z[i] = Z[i].at[:,:,k_].set(
                F_k.T @ Z_n[i] @ F_k +
                Q_k[i] +
                Ps[i][:,:,k_].T @ R_k[i] @ Ps[i][:,:,k_])

        Y2    = jnp.concatenate([B_k[i].T @ z_n[i] for i in range(kP)],0)
        a_big = S_inv @ Y2
        off = 0
        for i in range(kP):
            alp[i] = alp[i].at[:,k_].set(a_big[off:off+m[i]])
            off   += m[i]

        beta = -sum(B_k[i] @ alp[i][:,k_] for i in range(kP))
        for i in range(kP):
            zeta[i] = zeta[i].at[:,k_].set(
                F_k.T @ (z_n[i] + Z_n[i] @ beta) +
                l_k[i] +
                Ps[i][:,:,k_].T @ R_k[i] @ alp[i][:,k_])

        return (Ps, alp, Z, zeta), None

    (Ps, alp, Z, zeta), _ = lax.scan(one_step,
                                     (Ps, alp, Z, zeta),
                                     jnp.arange(H))
    return Ps, alp, Z, zeta
