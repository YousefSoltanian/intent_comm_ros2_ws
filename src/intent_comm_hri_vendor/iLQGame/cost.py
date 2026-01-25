"""
Game costs  – Huu guaranteed to be 2-D and the quadratic effort weight
is stored as the 5-th entry of w_vec / q_vec.

Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (Fridovich-Keil & Ratner)
"""

from functools import partial
from jax import jit, lax, jacfwd, hessian
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp

# ────────────────────────────────────────────────────────────────────
#  BASE CLASS (unchanged)
# ────────────────────────────────────────────────────────────────────
class Cost(object):
    """Abstract cost-function parent class."""

    def __init__(self, name="", horizon=None, x_dim=None, ui_dim=None):
        self._name, self._horizon = name, horizon
        self._x_dim, self._ui_dim = x_dim, ui_dim

    # ---------------- one-step primitives ---------------------------
    @partial(jit, static_argnums=(0,))
    def get_cost(self, x: ArrayImpl, ui: ArrayImpl, k: int = 0):
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def get_grad(self, x, ui, k=0):
        return jacfwd(self.get_cost, argnums=[0, 1])(x, ui, k)

    @partial(jit, static_argnums=(0,))
    def get_hess(self, x, ui, k=0):
        Hxx = hessian(self.get_cost, argnums=0)(x, ui, k)
        Huu = hessian(self.get_cost, argnums=1)(x, ui, k)
        # guarantee Huu is at least (1,1)
        if Huu.ndim == 0:
            Huu = Huu.reshape(1, 1)
        elif Huu.ndim == 1:               # (m,) → (m,1)
            Huu = Huu[:, None]
        return Hxx, Huu

    # ---------------- trajectory helpers ---------------------------
    @partial(jit, static_argnums=(0,))
    def get_traj_cost(self, x, ui):
        def body(k, costs):
            return costs.at[k].set(self.get_cost(x[:, k], ui[:, k], k))
        costs = jnp.zeros(self._horizon)
        return lax.fori_loop(0, self._horizon, body, costs)

    @partial(jit, static_argnums=(0,))
    def get_traj_grad(self, x, ui):
        def body(k, carry):
            lxs, lus = carry
            lx, lu = self.get_grad(x[:, k], ui[:, k], k)
            lxs = lxs.at[:, k].set(lx)
            lus = lus.at[:, k].set(lu)
            return lxs, lus
        lxs = jnp.zeros((self._x_dim,  self._horizon))
        lus = jnp.zeros((self._ui_dim, self._horizon))
        return lax.fori_loop(0, self._horizon, body, (lxs, lus))

    @partial(jit, static_argnums=(0,))
    def get_traj_hess(self, x, ui):
        def body(k, carry):
            Hxxs, Huus = carry
            Hxx, Huu = self.get_hess(x[:, k], ui[:, k], k)
            Hxxs = Hxxs.at[:, :, k].set(Hxx)
            Huus = Huus.at[:, :, k].set(Huu)
            return Hxxs, Huus
        Hxxs = jnp.zeros((self._x_dim,  self._x_dim,  self._horizon))
        Huus = jnp.zeros((self._ui_dim, self._ui_dim, self._horizon))
        return lax.fori_loop(0, self._horizon, body, (Hxxs, Huus))


# ────────────────────────────────────────────────────────────────────
#  PARAMETRIC  LUNAR-LANDER COSTS  (vector-length-5 format)
# ────────────────────────────────────────────────────────────────────
PX, PY, THETA, VX, VY, OMEGA = 0, 1, 2, 3, 4, 5


class ThrustPlayerParamCost(Cost):
    """
    Player-0 cost:

        w1(px−gx)² + w2(py−gy)² + w3 vx² + w4 vy² + w5 · T²

    where w5 is the **effort weight** (default 1.0 if you pass `[1,1,1,1,1]`).
    """
    def __init__(self, w_vec, goal_xy, name="ThrustCost", horizon=None):
        super().__init__(name, horizon, x_dim=6, ui_dim=1)
        assert len(w_vec) == 5, "w_vec must have 5 entries (last = effort weight)"
        self.w   = jnp.asarray(w_vec)
        self.gx, self.gy = goal_xy

    @partial(jit, static_argnums=(0,))
    def get_cost(self, x, ui, k=0):
        T = ui[0]
        return (
            self.w[0]*(600*(x[PX]-self.gx)/self.gx)**2 +
            self.w[1]*(450*(x[PY]-self.gy)/self.gy)**2 +
            self.w[2]*x[VX]**2           +
            self.w[3]*x[VY]**2           +
            self.w[4]*T**2               # effort term
        )


class TorquePlayerParamCost(Cost):
    """
    Player-1 cost:

        q1(px−gx)² + q2(py−gy)² + q3 θ² + q4 ω² + q5 · τ²

    where q5 is the **effort weight** (default 1.0 if you pass `[1,1,1,1,1]`).
    """
    def __init__(self, q_vec, goal_xy, name="TorqueCost", horizon=None):
        super().__init__(name, horizon, x_dim=6, ui_dim=1)
        assert len(q_vec) == 5, "q_vec must have 5 entries (last = effort weight)"
        self.q   = jnp.asarray(q_vec)
        self.gx, self.gy = goal_xy

    @partial(jit, static_argnums=(0,))
    def get_cost(self, x, ui, k=0):
        tau = ui[0]
        return (
            self.q[0]*(600*(x[PX]-self.gx)/self.gx)**2 +
            self.q[1]*(450*(x[PY]-self.gy)/self.gy)**2 +
            self.q[2]*x[THETA]**2        +
            self.q[3]*x[OMEGA]**2        +
            self.q[4]*tau**2             # effort term
        )


# ────────────────────────────────────────────────────────────────────
#  NEW: UNCONTROLLED-INTERSECTION COST (rewritten to mirror lander style)
# ────────────────────────────────────────────────────────────────────
# State indices for the intersection model (x = [d1, v1, d2, v2])
D1, V1, D2, V2 = 0, 1, 2, 3


class UncontrolledIntersectionPlayerCost(Cost):
    """
    Symmetric per-player cost for the 2-car uncontrolled intersection.

    Stage cost at time k:
        L_i(x,u_i) = 10·w_eff · a_i^2
                     + b · φ(co_occ(d_i, d_j))
                     − d_i
                     + 0.5 · (v_i − v_nom)^2
                     + back_pen(v_i) + slow_in_box(v_i, co_occ)

    where:
      • co_occ(d_i,d_j) is a smooth co-occupancy of the conflict zone
        via sigmoid “rectangular” windows;
      • φ(z) = (1 − exp(−α z)) / α is a saturating barrier (bounded curvature).

    Notes
    -----
    • Keeps API and parameters intact. DT=0.1 and action limits are handled elsewhere.
    • θ_other defaults to 1.0 but can be set via `theta_other`.
    """

    def __init__(self,
                 player_index: int,
                 theta_self: float,
                 name: str = "IntersectionCost",
                 horizon: int = None,
                 effort_weight: float = 1.0,
                 b: float = 1e4,
                 gamma: float = 5.0,
                 mu: float = 1e-6,
                 v_nom: float = 18.0,
                 R: float = 20.0,
                 W: float = 3.5,
                 L: float = 4.5,
                 theta_other: float = 1.0):
        super().__init__(name, horizon, x_dim=4, ui_dim=1)
        assert player_index in (0, 1), "player_index must be 0 or 1"

        # Indices/intents (store as arrays to keep everything in jnp)
        self.i          = int(player_index)
        self.theta_self = jnp.asarray(theta_self)
        self.theta_oth  = jnp.asarray(theta_other)

        # Weights / geometry / smoothing stored as JAX scalars
        self.w_eff = jnp.asarray(effort_weight)  # effort (quadratic in a_i)
        self.b     = jnp.asarray(b)              # collision-window penalty
        self.gamma = jnp.asarray(gamma)          # logistic sharpness
        self.mu    = jnp.asarray(mu)             # terminal progress reward
        self.v_nom = jnp.asarray(v_nom)          # nominal terminal speed
        self.R     = jnp.asarray(R)              # road geometry
        self.W     = jnp.asarray(W)
        self.L     = jnp.asarray(L)

    @partial(jit, static_argnums=(0,))
    def get_cost(self, x, ui, k=0):
        # pick (d_i, v_i) and opponent (d_j, v_j)
        d1, v1, d2, v2 = x[0], x[1], x[2], x[3]
        d_i, v_i = (d1, v1) if self.i == 0 else (d2, v2)
        d_j, v_j = (d2, v2) if self.i == 0 else (d1, v1)

        a_i = ui[0]

        # ---- Smooth rectangular windows along each approach (bounded curvature) ----
        # Intersection center on each road
        D_CROSS = self.R / 2.0

        # Use gamma as the logistic slope; smaller gamma -> softer edges
        kappa = self.gamma

        def sigmoid(z):
            #return 1.0 / (1.0 + jnp.exp(-kappa * z))
            return jnp.exp(-kappa * z + 0.0)

        # Smooth "inside segment" window: w(d) ~ 1 inside [left,right], ~0 outside
        def rect_window(d, left, right):
            #return sigmoid(d - left) * sigmoid(right - d)
            return sigmoid((d - left)**2) 

        # Vehicle i window (shifted by its lane/intent)
        #left_i  = D_CROSS - 2*0.5 * self.theta_self * self.W
        #right_i = D_CROSS + self.L + 2*0.5 * self.W * 1

        # Opponent window: use provided theta_other (defaults to 1.0)
        #left_j  = D_CROSS - 2*0.5 * self.theta_oth * self.W
        #left_j  = D_CROSS - 2*0.5 * 1 * self.W * 1
        #right_j = D_CROSS + self.L + 2*0.5 * self.W* 1

        #occ_i = rect_window(d_i, left_i, right_i)
        #occ_j = rect_window(d_j, left_j, right_j)
        dist = ((d_i - D_CROSS)**2 + (d_j - D_CROSS)**2)**0.5
        d_safe = (self.theta_self*self.W + self.L)
        # Co-occupancy measure in [0,1]; highest when both are inside the box
        #co_occ = occ_i * occ_j
        co_occ = sigmoid(abs(dist - d_safe))
        # ---- Saturating collision barrier (bounded gradient & Hessian) ----
        alpha = 0.1
        #coll_pen = self.b * (1.0 - jnp.exp(-alpha * co_occ)) / alpha  # <= self.b/alpha
        coll_pen = self.b * co_occ  # linearized variant (simpler)
        # ---- Task terms: progress forward + sensible speed, discourage reversing ----
        # Keep original scaling: progress = -1 * d_i  (mu kept as a field but unused here)
        progress = 1.0 * (d_i-100)**2
        # Track nominal speed generally; keep coefficient modest to avoid stiffness
        speed_track = 0.00 * (v_i - 1.0 * self.v_nom) ** 2 + 1.00*v_i**2

        # Strongly discourage going backward; smooth (bounded curvature) variant
        #back_pen = 1*10000.0 * (jnp.maximum(-v_i, 0.0) ** 2)
        back_pen = 000.0*(-v_i)
        # ---- Effort and context-aware slowdown inside conflict region ----
        # Preserve prior scaling on effort (10 * w_eff)
        effort = 1* self.w_eff * (a_i ** 2)

        # When co-occupancy is high, gently penalize excessive speed (brake, don't reverse)
        # Threshold at ~60% of v_nom; linear in co_occ to avoid harsh curvature
        slow_in_box = 0.0 * co_occ * (jnp.maximum(v_i - 1 * self.v_nom, 0.0) ** 2)

        # ---- Total stage cost ----
        stage = effort + coll_pen + progress + speed_track + back_pen + slow_in_box

        # terminal add-on only at the last step (if horizon is known)
        # if self._horizon is not None:
        #     term_T = (-self.mu * d_i) + (v_i - self.v_nom) ** 2
        #     stage  = stage + jnp.where(k == (self._horizon - 1), term_T, 0.0)

        return stage
    


    # ────────────────────────────────────────────────────────────────────
#  SOCIAL NAVIGATION COST (2×2 hidden goals, unicycle controls)
#  x = [x1, y1, th1, x2, y2, th2],  u_i = [v_i, omega_i]
# ────────────────────────────────────────────────────────────────────
X1, Y1, TH1, X2, Y2, TH2 = 0, 1, 2, 3, 4, 5

class SocialNavigationPlayerCost(Cost):
    """
    Per-player stage cost for hallway navigation with hidden goals.

    Args
    ----
    player_index : int
        0 for agent-1 (x1,y1,th1), 1 for agent-2 (x2,y2,th2).
    goals_self : array-like, shape (Ns, 2)
        Candidate goals (x,y) for this player.
    goals_other : array-like, shape (No, 2)
        Candidate goals (x,y) for the other player (kept for symmetry; not required in cost aside from future extensions).
    theta_self : int
        Index into goals_self for this player's current intent hypothesis.
    theta_other : int
        Index into goals_other for the opponent's current intent hypothesis (not directly penalized here; collision uses actual state).
    horizon : int or None
        If provided, you can optionally add a terminal cost at k == horizon-1 (see commented section below).

    Weights (all scalars)
    ---------------------
    w_goal_xy : (w_gx, w_gy) – position error weights
    w_head    : heading alignment weight
    w_speed   : nominal speed tracking weight
    w_effort  : control effort weight (applies to both v and ω)
    w_lat     : lateral comfort weight (penalizes v·ω)
    w_wall    : corridor wall soft-barrier weight
    w_coll    : collision soft-barrier weight

    Corridor / safety
    -----------------
    hall_y0   : corridor centerline (y)
    hall_half_width : half-width of corridor
    d_safe_wall     : keep inside walls by ≥ this margin (soft)
    r_safe_coll     : target min separation (soft)
    kappa           : softness for softplus barriers

    Notes
    -----
    • Smooth everywhere (uses softplus and smooth |·|).
    • Uses control (v, ω) directly for speed/effort terms (fits unicycle).
    • Symmetric and ILQ-friendly (no discrete switches).
    """
    def __init__(self,
                 player_index: int,
                 goals_self,
                 goals_other,
                 theta_self: int,
                 theta_other: int,
                 name: str = "NavCost",
                 horizon: int = None,
                 # weights
                 w_goal_xy=(0.1, 0.1),
                 w_head=0.0,
                 w_speed=100.0,
                 w_effort=0.005,
                 w_lat=10.0,
                 w_wall=60.0,
                 w_coll=150.0,
                 # corridor / safety
                 hall_y0=0.0,
                 hall_half_width=5.0,
                 d_safe_wall=0.25,
                 r_safe_coll=1.6,
                 kappa=10.0,
                 v_nom=1.0):
        super().__init__(name, horizon, x_dim=6, ui_dim=2)
        assert player_index in (0, 1), "player_index must be 0 or 1"
        self.i = int(player_index)

        # Store goal sets and chosen indices
        self.goals_self  = jnp.asarray(goals_self, dtype=jnp.float32).reshape((-1, 2))
        self.goals_other = jnp.asarray(goals_other, dtype=jnp.float32).reshape((-1, 2))
        self.theta_self  = int(theta_self)
        self.theta_other = int(theta_other)

        # Weights
        self.w_gx, self.w_gy = (jnp.asarray(w_goal_xy[0]), jnp.asarray(w_goal_xy[1]))
        self.w_head  = jnp.asarray(w_head)
        self.w_speed = jnp.asarray(w_speed)
        self.w_eff   = jnp.asarray(w_effort)
        self.w_lat   = jnp.asarray(w_lat)
        self.w_wall  = jnp.asarray(w_wall)
        self.w_coll  = jnp.asarray(w_coll)

        # Corridor / safety params
        self.hall_y0        = jnp.asarray(hall_y0)
        self.half_w         = jnp.asarray(hall_half_width)
        self.d_safe_wall    = jnp.asarray(d_safe_wall)
        self.r_safe         = jnp.asarray(r_safe_coll)
        self.kappa          = jnp.asarray(kappa)
        self.v_nom          = jnp.asarray(v_nom)

    # ---- helpers (JAX-friendly) ----
    @partial(jit, static_argnums=(0,))
    def _softplus(self, z):
        # stable softplus with tunable sharpness
        return jnp.logaddexp(0.0, self.kappa * z) / self.kappa

    @partial(jit, static_argnums=(0,))
    def _wrap_angle(self, a):
        # Wrap to (-pi, pi]
        return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @partial(jit, static_argnums=(0,))
    def get_cost(self, x, ui, k=0):
        # Parse own/opponent states
        if self.i == 0:
            x_self, y_self, th_self = x[X1], x[Y1], x[TH1]
            x_oth,  y_oth,  th_oth  = x[X2], x[Y2], x[TH2]
        else:
            x_self, y_self, th_self = x[X2], x[Y2], x[TH2]
            x_oth,  y_oth,  th_oth  = x[X1], x[Y1], x[TH1]

        v, omega = ui[0], ui[1]

        # Select goals via intent indices
        gx_self, gy_self = self.goals_self[self.theta_self]
        # gx_oth, gy_oth = self.goals_other[self.theta_other]  # (kept for extensions)

        # 1) Goal tracking (position)
        dx = x_self - gx_self
        dy = y_self - gy_self
        goal_pos = self.w_gx * (dx * dx) + self.w_gy * (dy * dy)

        # 2) Heading alignment to goal
        goal_bear = jnp.arctan2(gy_self - y_self, gx_self - x_self)
        dth = self._wrap_angle(th_self - goal_bear)
        head_align = self.w_head * (dth * dth)

        # 3) Nominal speed tracking (use control v as commanded speed)
        speed_track = self.w_speed * ((v - self.v_nom) ** 2)

        # 4) Effort (quadratic in controls)
        effort = self.w_eff * (v * v + 10*omega * omega)

        # 5) Lateral comfort (discourage sharp turning at speed)
        lat_comfort = self.w_lat * ((v * omega) ** 2)

        # 6) Corridor wall soft-barrier (smooth |·|)
        dy_c   = y_self - self.hall_y0
        abs_dy = jnp.sqrt(dy_c * dy_c + 1e-9)  # smooth abs
        d_wall = self.half_w - abs_dy          # positive inside corridor
        wall_barrier = self.w_wall * (self._softplus(self.d_safe_wall - d_wall) ** 2)

        # 7) Collision avoidance (soft barrier on separation)
        dx12 = x_self - x_oth
        dy12 = y_self - y_oth
        dist = jnp.sqrt(dx12 * dx12 + dy12 * dy12 + 1e-9)
        coll_barrier = self.w_coll * (self._softplus(self.r_safe - dist) ** 2)

        stage = goal_pos + head_align + speed_track + effort + lat_comfort + wall_barrier + coll_barrier

        # Optional terminal add-on (uncomment if desired)
        #if self._horizon is not None:
         #   term_T = 5.0 * (dx * dx + dy * dy) + 1.0 * (self._wrap_angle(dth) ** 2)
          #  stage  = stage + jnp.where(k == (self._horizon - 1), term_T, 0.0)

        return stage

