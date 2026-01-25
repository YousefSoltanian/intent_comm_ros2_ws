"""
Multiplayer dynamical systems.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)
"""
import numpy as np
from typing import Tuple
from functools import partial
from jax import jit, jacfwd
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────
#  Generic base classes  (unchanged)
# ─────────────────────────────────────────────────────────────────────
class MultiPlayerDynamicalSystem(object):
  """
  Base class for all multiplayer continuous-time dynamical systems.
  Supports numerical integration and linearization.
  """

  def __init__(self, x_dim, u_dims, T=0.1):
    """
    Parameters
    ----------
    x_dim  : int      – joint-state dimension
    u_dims : [int]    – list (one per player) of control dimensions
    T      : float    – integration step (s)
    """
    self._x_dim        = x_dim
    self._u_dims       = u_dims
    self._T            = T
    self._num_players  = len(u_dims)
    self.jac_f         = jit(jacfwd(self.disc_time_dyn, argnums=[0, 1]))

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self,
                    x: ArrayImpl,
                    u_list: list,
                    k: int = 0,
                    args=()) -> ArrayImpl:
    raise NotImplementedError

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def disc_time_dyn(self,
                    x0: ArrayImpl,
                    u0_list: list,
                    k: int = 0,
                    args=()) -> ArrayImpl:
    """Forward-Euler step:  x⁺ = x + T·f(x,u)."""
    return x0 + self._T * self.cont_time_dyn(x0, u0_list, k, args)

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def linearize_discrete_jitted(self,
                                x0: ArrayImpl,
                                u0_list: list,
                                k: int = 0,
                                args=()) -> Tuple[ArrayImpl, list]:
    """Jacobian of the discrete-time dynamics about (x₀,u₀)."""
    return self.jac_f(x0, u0_list, k, args)


# ─────────────────────────────────────────────────────────────────────
class ProductMultiPlayerDynamicalSystem(MultiPlayerDynamicalSystem):
  """Cartesian product of independent subsystems (unchanged)."""

  def __init__(self, subsystems, T=0.1):
    self._subsystems = subsystems
    self._x_dims     = [sys._x_dim for sys in subsystems]
    super().__init__(sum(self._x_dims), [sys._u_dim for sys in subsystems], T)
    self.update_lifting_matrices()
    self._num_opn_dyn = 0

  # (helper methods unchanged …)
  # ------------------------------------------------------------------
  def update_lifting_matrices(self):
    _split = np.hstack((0, np.cumsum(np.asarray(self._x_dims))))
    self._LMx = [jnp.asarray(np.eye(d, self._x_dim, k=_split[i]))
                 for i, d in enumerate(self._x_dims)]

    u_dim    = sum(self._u_dims)
    _split_u = np.hstack((0, np.cumsum(np.asarray(self._u_dims))))
    self._LMu = [jnp.asarray(np.eye(d, u_dim, k=_split_u[i]))
                 for i, d in enumerate(self._u_dims)]

  # ------------------------------------------------------------------
  def add_opinion_dyn(self, opn_dyns):
    opn_dyns._start_index = self._x_dim
    self._subsystems.append(opn_dyns)
    self._num_opn_dyn += 1
    self._x_dim += opn_dyns._x_dim
    self._x_dims.append(opn_dyns._x_dim)
    self.update_lifting_matrices()
    self._LMx += [jnp.eye(self._x_dim)] * self._num_opn_dyn

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self,
                    x: ArrayImpl,
                    u_list: list,
                    k: int = 0,
                    args=()) -> ArrayImpl:
    u_list += [None] * self._num_opn_dyn
    parts = [sys.cont_time_dyn(LMx @ x, u_i, k, args)
             for sys, LMx, u_i in zip(self._subsystems, self._LMx, u_list)]
    return jnp.concatenate(parts, axis=0)


# ─────────────────────────────────────────────────────────────────────
#  Original 2-player planar Lunar-Lander (kept for backward-compat)
# ─────────────────────────────────────────────────────────────────────
class LunarLander2PlayerSystem(MultiPlayerDynamicalSystem):
  """
  6-state, 2-input lunar-lander model (world frame +y upward).

      x = [px, py, vx, vy, θ, ω]
      u₀ = thrust  T  ∈ [0 , 900]   (N)   (body-axis +y)
      u₁ = torque  τ  ∈ [-300, 300] (N·m) (CCW positive)
  """

  def __init__(self,
               T: float        = 1/60,
               mass: float     = 1.0,
               inertia: float  = 500.0,
               lin_drag: float = 5.0,
               ang_drag: float = 100.0,
               g: float        = -800.0):
    super().__init__(x_dim=6, u_dims=[1, 1], T=T)

    # physical parameters
    self.mass, self.inertia = mass, inertia
    self.lin_drag, self.ang_drag = lin_drag, ang_drag
    self.g = g                         # negative ⇒ gravity acts downward

    # state-lifting
    self._LMx = [jnp.eye(self._x_dim)]

    # joint-u = [T, τ]ᵀ
    self._LMu = [
        jnp.array([[1.0, 0.0]]),   # player-0 sees thrust
        jnp.array([[0.0, 1.0]])    # player-1 sees torque
    ]

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self,
                    x: ArrayImpl,
                    u_list: list,
                    k: int = 0,
                    args=()) -> ArrayImpl:
    px, py, vx, vy, theta, omega = x
    T   = u_list[0][0]
    tau = u_list[1][0]

    sinθ, cosθ = jnp.sin(theta), jnp.cos(theta)

    px_dot = vx
    py_dot = vy
    vx_dot = (  -T * sinθ - self.lin_drag * vx          ) / self.mass
    vy_dot = (   T * cosθ - self.lin_drag * vy + self.g ) / self.mass

    theta_dot = omega
    omega_dot = (tau - self.ang_drag * omega) / self.inertia

    return jnp.array([px_dot, py_dot, vx_dot, vy_dot, theta_dot, omega_dot])


# ─────────────────────────────────────────────────────────────────────
#  NEW 3-player planar Lunar-Lander
# ─────────────────────────────────────────────────────────────────────
class LunarLander3PlayerSystem(MultiPlayerDynamicalSystem):
  """
  6-state, **3-input** lunar-lander model.

      x = [px, py, vx, vy, θ, ω]

      Player-0 : u₀ = F_h   (human thrust)  ∈ [-1000 , 1000]  N
      Player-1 : u₁ = F_r   (robot Δ-thrust)∈ [-1000 , 1000]  N
      Player-2 : u₂ = τ     (torque)        ∈ [-1000 , 1000] N·m

      Total thrust F_tot = clip(F_h + F_r , 0 , max_thrust)
  """

  def __init__(self,
               T: float        = 1/60,
               mass: float     = 1.0,
               inertia: float  = 500.0,
               lin_drag: float = 5.0,
               ang_drag: float = 100.0,
               g: float        = -800.0,
               max_thrust: float = 2000.0):
    super().__init__(x_dim=6, u_dims=[1, 1, 1], T=T)

    # physical parameters
    self.mass, self.inertia = mass, inertia
    self.lin_drag, self.ang_drag = lin_drag, ang_drag
    self.g = g
    self.max_thrust = max_thrust

    # state-lifting
    self._LMx = [jnp.eye(self._x_dim)]

    # joint-u = [F_h, F_r, τ]ᵀ
    self._LMu = [
        jnp.array([[1.0, 0.0, 0.0]]),   # human thrust
        jnp.array([[0.0, 1.0, 0.0]]),   # robot Δ-thrust
        jnp.array([[0.0, 0.0, 1.0]])    # torque
    ]

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self,
                    x: ArrayImpl,
                    u_list: list,
                    k: int = 0,
                    args=()) -> ArrayImpl:

    px, py, vx, vy, theta, omega = x
    F_h   = u_list[0][0]
    F_r   = u_list[1][0]
    tau   = u_list[2][0]

    # Compose thrust, but never let it go negative
    F_tot = jnp.clip(F_h + F_r, 0.0, self.max_thrust)

    sinθ, cosθ = jnp.sin(theta), jnp.cos(theta)

    # translational dynamics
    px_dot = vx
    py_dot = vy
    vx_dot = (  -F_tot * sinθ - self.lin_drag * vx          ) / self.mass
    vy_dot = (   F_tot * cosθ - self.lin_drag * vy + self.g ) / self.mass

    # rotational dynamics
    theta_dot = omega
    omega_dot = (tau - self.ang_drag * omega) / self.inertia

    return jnp.array([px_dot, py_dot, vx_dot, vy_dot, theta_dot, omega_dot])


# ─────────────────────────────────────────────────────────────────────
#  NEW 2-player Uncontrolled-Intersection (double-integrator lanes)
# ─────────────────────────────────────────────────────────────────────
class UncontrolledIntersection2PlayerSystem(MultiPlayerDynamicalSystem):
  """
  4-state, 2-input model for an uncontrolled intersection with two vehicles.
  Each vehicle moves along its lane with double-integrator longitudinal dynamics.

      x = [d1, v1, d2, v2]          # distances-to-intersection and speeds
      u₀ = a1  (veh-1 acceleration)  ∈ ℝ
      u₁ = a2  (veh-2 acceleration)  ∈ ℝ

  Note: Collision-avoidance and "intent" (e.g., cautious vs. aggressive)
        are handled in the *costs/constraints*, not the dynamics.
  """

  def __init__(self,
               T: float = 0.05):
    super().__init__(x_dim=4, u_dims=[1, 1], T=T)

    # state-lifting
    self._LMx = [jnp.eye(self._x_dim)]

    # joint-u = [a1, a2]ᵀ
    self._LMu = [
        jnp.array([[1.0, 0.0]]),   # player-0 acceleration
        jnp.array([[0.0, 1.0]])    # player-1 acceleration
    ]

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self,
                    x: ArrayImpl,
                    u_list: list,
                    k: int = 0,
                    args=()) -> ArrayImpl:
    d1, v1, d2, v2 = x
    a1 = u_list[0][0]
    a2 = u_list[1][0]

    d1_dot = v1
    v1_dot = a1
    d2_dot = v2
    v2_dot = a2

    return jnp.array([d1_dot, v1_dot, d2_dot, v2_dot])


# ─────────────────────────────────────────────────────────────────────
#  NEW 2-player Planar Navigation (unicycle-heading, v & ω controls)
# ─────────────────────────────────────────────────────────────────────
class PlanarNavigation2PlayerSystem(MultiPlayerDynamicalSystem):
  """
  6-state, 2-input-per-player planar navigation model (heading + speed).

      State (joint):
        x = [x1, y1, θ1,  x2, y2, θ2]

      Controls:
        Player-0: u0 = [v1, ω1]  (speed magnitude, angular rate)
        Player-1: u1 = [v2, ω2]

      Continuous-time dynamics:
        ẋ1 = v1 * cos(θ1)         ẏ1 = v1 * sin(θ1)         θ̇1 = ω1
        ẋ2 = v2 * cos(θ2)         ẏ2 = v2 * sin(θ2)         θ̇2 = ω2

      Notes:
        • Units: v in m/s, ω in rad/s, θ in rad.
        • Use constraints/costs to bound v, ω if needed (no clipping here).
  """

  def __init__(self, T: float = 0.1):
    super().__init__(x_dim=6, u_dims=[2, 2], T=T)

    # state-lifting (identity since this is a single joint state block)
    self._LMx = [jnp.eye(self._x_dim)]

    # joint-u = [v1, ω1, v2, ω2]ᵀ  → split to each player's 2D control
    self._LMu = [
      jnp.array([[1., 0., 0., 0.],
                 [0., 1., 0., 0.]]),   # player-0: [v1, ω1]
      jnp.array([[0., 0., 1., 0.],
                 [0., 0., 0., 1.]])    # player-1: [v2, ω2]
    ]

  # ------------------------------------------------------------------
  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self,
                    x: ArrayImpl,
                    u_list: list,
                    k: int = 0,
                    args=()) -> ArrayImpl:
    x1, y1, th1, x2, y2, th2 = x

    # Each player's control is a length-2 vector: [v, ω]
    v1, w1 = u_list[0][0], u_list[0][1]
    v2, w2 = u_list[1][0], u_list[1][1]

    c1, s1 = jnp.cos(th1), jnp.sin(th1)
    c2, s2 = jnp.cos(th2), jnp.sin(th2)

    x1_dot  = v1 * c1
    y1_dot  = v1 * s1
    th1_dot = w1

    x2_dot  = v2 * c2
    y2_dot  = v2 * s2
    th2_dot = w2

    return jnp.array([x1_dot, y1_dot, th1_dot, x2_dot, y2_dot, th2_dot])
