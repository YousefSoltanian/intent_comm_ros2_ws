# src/iLQGame/ilq_alpha0.py
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# import your existing fully‑JAX solver
from iLQGame.ilq_solver import ILQSolver

def make_ilq_alpha0_fn(solver: ILQSolver):
    """
    Given a configured ILQSolver instance, returns a function
        ilq_alpha0(x0, net_p, xf, b) -> u0
    whose backward pass uses d alpha0 / d theta for all theta.

    - solver._Ps_init, solver._alphas_init, solver._costs, solver._dyn, etc.
      must already be set up before calling this.
    - solver._horizon must also be set.
    """
    # pull out the lower‐level routines for reuse
    run_jitted        = solver._run_jitted
    linearize        = solver._linearize_dynamics
    quadratize       = solver._quadraticize_costs
    solve_lq_game    = solver._solve_lq_game

    @jax.custom_vjp
    def ilq_alpha0(x0, net_p, xf, b):
        # 1) run full ILQ to convergence (no stop_gradient)
        solver.reset()
        # assume solver._costs entries capture net_p, xf, b by closure
        carry0 = (solver._current_op, solver._best_op, solver._best_cost, x0)
        (curr, best, cost_final, _), _ = run_jitted(carry0)

        xs_star, us_list_star = best[0], best[1]    # xs*: (n,H), [u1*,u2*]
        # forward actual control at t=0
        u0 = us_list_star[0][0, 0]

        # 2) re‑linearize & re‑quadraticize around xs_star, us_list_star
        As, Bs    = linearize(xs_star, us_list_star)
        costs, lxs, Hxxs, Huus = quadratize(xs_star, us_list_star)

        # 3) solve LQ‐game backward once (no line‐search)
        P_star, alpha_star, _, _ = solve_lq_game(
            As, Bs, Hxxs, lxs, Huus
        )

        # α₀ for player 1 at time 0:
        a0 = alpha_star[0][0, 0]
        return u0, a0

    def fwd(x0, net_p, xf, b):
        (u0, a0) = ilq_alpha0(x0, net_p, xf, b)
        # stash inputs for backward
        return (u0, a0), (x0, net_p, xf, b)

    def bwd(res, g):
        # g = (dL/du0, dL/da0)
        _, (x0, net_p, xf, b) = (None, res)
        g_u0, g_a0 = g

        # we *ignore* dL/du0 and only route through α₀:
        g_comb = g_a0

        # now compute dolpha0/d(x0, net_p, xf, b)
        # we re‑call ilq_alpha0, but pull out only the α₀ part
        # and autodiff that.
        def alpha_only(x, p, xf_, b_):
            return ilq_alpha0(x, p, xf_, b_)[1]

        # get jacobians
        jac_x0, jac_p, jac_xf, jac_b = jax.jacrev(alpha_only, (0,1,2,3))(x0, net_p, xf, b)

        # multiply through
        gx0  = g_comb * jac_x0
        gp   = jax.tree_map(lambda J: g_comb * J, jac_p)
        gxf  = g_comb * jac_xf
        gb   = g_comb * jac_b

        return gx0, gp, gxf, gb

    ilq_alpha0.defvjp(fwd, bwd)
    return ilq_alpha0
