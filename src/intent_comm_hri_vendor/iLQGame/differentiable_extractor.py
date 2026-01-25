# File: src/iLQGame/differentiable_extractor.py
import jax
import jax.numpy as jnp
from jax import lax

from .multiplayer_dynamical_system import LunarLander2PlayerSystem
from .ilq_solver                   import ILQSolver
from .player_cost                  import PlayerCost
from .cost                         import ThrustPlayerParamCost, TorquePlayerParamCost
from .constraint                   import BoxConstraint

_DT        = 1 / 60
_MAX_ITER  = 15

# cache:  H -> (solver, raw_thrust_cost, raw_torque_cost)
_SOLVER_CACHE: dict[int, tuple[ILQSolver,
                               ThrustPlayerParamCost,
                               TorquePlayerParamCost]] = {}


# ──────────────────────────── SOLVER FACTORY ─────────────────────────
def _make_solver(H: int):
    """
    Return an ILQ solver (plus raw cost objects) of horizon H.
    The *weights* inside the cost objects are overwritten by the caller
    before every solve, so we can keep a single warm-cached instance.
    """
    if H not in _SOLVER_CACHE:
        # 1) stub param-cost objects (real weights/goals set later)
        cT = ThrustPlayerParamCost([1, 1, 1, 1, 1], (0.0, 0.0), horizon=H)
        cQ = TorquePlayerParamCost([1, 1, 1, 1, 1], (0.0, 0.0), horizon=H)

        # 2) wrap each in a PlayerCost
        pcT = PlayerCost(); pcT.add_cost(cT, "x", 1.0)
        pcQ = PlayerCost(); pcQ.add_cost(cQ, "x", 1.0)

        # 3) warm-start arrays
        P0 = jnp.zeros((1, 6, H));  P1 = jnp.zeros((1, 6, H))
        a0 = jnp.zeros((1,   H));   a1 = jnp.zeros((1,   H))

        # 4) solver
        solver = ILQSolver(
            LunarLander2PlayerSystem(T=_DT),
            [pcT, pcQ],
            [P0,  P1],
            [a0,  a1],
            max_iter=_MAX_ITER,
            u_constraints=[
                BoxConstraint(0.0,    200000.0),   # thrust
                BoxConstraint(-100000.0, 100000.0)   # torque
            ],
            verbose=False
        )
        _SOLVER_CACHE[H] = (solver, cT, cQ)

    return _SOLVER_CACHE[H]


# ────────────────────────── EXTRACTOR (JIT) ──────────────────────────
def get_extractor(H: int):
    """
    Return a jit-compiled function

        u_mod, alpha0 = extractor(xs, us_list, w, q, gx, gy)

    where
        xs       : (6,H)   stop-grad trajectory
        us_list  : [uT, uτ] each (1,H)   controls along xs
        w, q     : length-5 vectors (last component = effort weight)
        gx, gy   : goal coordinates

    The function linearises dynamics, quadraticises the game costs
    **with the supplied tracking & effort weights**, solves the one-shot
    LQ Nash game, and returns the immediate model control u_mod and the
    affine term α₀ needed for the surrogate gradient.
    """
    solver, _, _ = _make_solver(H)

    @jax.jit
    def _extractor(xs, us_list, w, q, gx, gy):
        # ---------------- dynamics linearisation -------------------
        A, B_list = solver._linearize_dynamics(
            lax.stop_gradient(xs),
            [lax.stop_gradient(u) for u in us_list]
        )

        # ---------------- cost quadraticisation --------------------
        px, py, vx, vy, th, om = lax.stop_gradient(xs)

        # split weights
        w_trk, λT  = w[:4], w[4]
        q_trk, λτ  = q[:4], q[4]

        # ---- thrust player ---------------------------------------
        lx_T = jnp.stack([
            2*w_trk[0]*(px-gx),
            2*w_trk[1]*(py-gy),
            2*w_trk[2]*vx,
            2*w_trk[3]*vy,
            jnp.zeros_like(px),
            jnp.zeros_like(px)
        ], axis=0)
        diag_T = jnp.array([2*w_trk[0], 2*w_trk[1],
                            2*w_trk[2], 2*w_trk[3], 0., 0.])
        Hxx_T  = jnp.zeros((6, 6, H)).at[:, :, :].set(
                     jnp.diag(diag_T)[:, :, None])
        Huu_T  = jnp.full((1, 1, H), 2*λT)

        # ---- torque player ---------------------------------------
        lx_Q = jnp.stack([
            2*q_trk[0]*(px-gx),
            2*q_trk[1]*(py-gy),
            jnp.zeros_like(px),
            jnp.zeros_like(px),
            2*q_trk[2]*th,
            2*q_trk[3]*om
        ], axis=0)
        diag_Q = jnp.array([2*q_trk[0], 2*q_trk[1],
                            0., 0., 2*q_trk[2], 2*q_trk[3]])
        Hxx_Q  = jnp.zeros((6, 6, H)).at[:, :, :].set(
                     jnp.diag(diag_Q)[:, :, None])
        Huu_Q  = jnp.full((1, 1, H), 2*λτ)

        # ---------------- LQ Nash solve ----------------------------
        P, alphas, _, _ = solver._solve_lq_game(
            A, B_list,
            [Hxx_T, Hxx_Q],
            [lx_T,  lx_Q],
            [Huu_T, Huu_Q]
        )

        # immediate model control & affine term
        uT   = us_list[0][0, 0]
        tau  = us_list[1][0, 0]
        u_mod   = jnp.array([uT, tau])
        alpha0  = -jnp.array([alphas[0][0, 0], alphas[1][0, 0]])
        return u_mod, alpha0

    return _extractor
