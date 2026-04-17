"""
MPC oracle controllers for target_gym environments.

Two implementations are provided:

GradientMPC  (JAX — Car, Plane)
    Single-shooting gradient MPC: differentiates through a JAX scan rollout
    and runs gradient descent on the action sequence.  Requires a long horizon
    to be accurate, and suffers from vanishing gradients for stiff systems.

CasadiMPC  (CasADi / IPOPT — CSTR, FirstOrder, Nonsmooth, FourTank)
    Proper NLP-based receding-horizon MPC solved by IPOPT via do_mpc.  The
    solver receives analytic Jacobians and finds the exact optimum in each
    window — matching the PC-gym oracle approach.  Requires::

        pip install casadi do-mpc

    Even a short horizon (N=5, matching PC-gym's default) produces near-oracle
    performance because the NLP is solved exactly at each step.

Common API (both classes)::

    mpc = make_<env>_mpc(env, params)
    obs, state = env.reset_env(key, params)
    for _ in range(T):
        action = mpc.step(obs, state)   # obs ignored by MPC, kept for symmetry
        obs, state, *_ = env.step_env(key, state, action, params)
    mpc.reset()
"""

import jax
import jax.numpy as jnp
import numpy as np

try:
    import casadi
    import do_mpc

    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False


# ============================================================================
# Gradient MPC  — JAX-based, used for Car and Plane
# ============================================================================


class GradientMPC:
    """
    Single-shooting gradient MPC controller.

    Parameters
    ----------
    env :
        A gymnax-style environment with a JAX-traceable ``step_env`` method.
    params :
        Environment parameters dataclass.
    horizon : int
        Number of steps to optimise over.
    n_iter : int
        Number of gradient descent iterations per call to ``step``.
    lr : float
        Learning rate for gradient descent.
    action_dim : int
        Dimensionality of the action vector.
    action_lb, action_ub : float
        Lower/upper bounds applied via clip after each gradient step.
    """

    def __init__(
        self,
        env,
        params,
        horizon: int = 20,
        n_iter: int = 50,
        lr: float = 0.05,
        action_dim: int = 1,
        action_lb: float = -1.0,
        action_ub: float = 1.0,
    ):
        self.env = env
        self.params = params
        self.horizon = horizon
        self.n_iter = n_iter
        self.lr = lr
        self.action_dim = action_dim
        self.action_lb = float(action_lb)
        self.action_ub = float(action_ub)

        self._actions = jnp.zeros((horizon, action_dim))
        self._jit_optimize = jax.jit(self._optimize)

    def _env_action(self, u: jnp.ndarray):
        """Convert a per-step action vector to the format expected by step_env."""
        if self.action_dim == 1:
            return u[0]
        return u

    def _rollout(self, actions: jnp.ndarray, state) -> jnp.ndarray:
        key = jax.random.PRNGKey(0)

        def step_fn(carry, u):
            s = carry
            _, new_s, r, _, _ = self.env.step_env(
                key, s, self._env_action(u), self.params
            )
            return new_s, r

        _, rewards = jax.lax.scan(step_fn, state, actions)
        return jnp.sum(rewards)

    def _optimize(self, actions_init: jnp.ndarray, state) -> jnp.ndarray:
        cost_grad = jax.grad(lambda a: -self._rollout(a, state))
        lb, ub, lr = self.action_lb, self.action_ub, self.lr

        def body(_, actions):
            g = cost_grad(actions)
            # Replace NaN gradients with zero (can arise from numerically
            # unstable rollouts, e.g. near-stall flight dynamics)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            # Clip gradient norm
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-8)
            g = jnp.where(g_norm > 1.0, g / g_norm, g)
            return jnp.clip(actions - lr * g, lb, ub)

        return jax.lax.fori_loop(0, self.n_iter, body, actions_init)

    def step(self, _obs, state):
        """Return next action. ``_obs`` is ignored (kept for API symmetry)."""
        actions_init = jnp.concatenate([self._actions[1:], self._actions[-1:]], axis=0)
        self._actions = self._jit_optimize(actions_init, state)
        first = self._actions[0]
        if self.action_dim == 1:
            return float(first[0])
        return np.array(first)

    def reset(self):
        """Reset the internal action sequence to zeros."""
        self._actions = jnp.zeros((self.horizon, self.action_dim))


# ============================================================================
# CasADi MPC  — IPOPT-based, used for CSTR / FirstOrder / Nonsmooth / FourTank
# ============================================================================


class CasadiMPC:
    """
    Receding-horizon MPC solved exactly by IPOPT via do_mpc / CasADi.

    Equivalent to the PC-gym oracle approach (N=5 by default).  The NLP solver
    receives analytic Jacobians so even a short horizon gives near-optimal
    performance — no gradient vanishing, no learning-rate tuning.

    Subclasses implement:
      - ``_build_mpc()`` → returns a configured, set-up ``do_mpc.controller.MPC``
      - ``_extract_x0(state)`` → numpy array of physical states for IPOPT
      - ``_update_setpoint(state)`` → refreshes the mutable setpoint attribute(s)
    """

    def __init__(self, env, params, horizon: int = 5, mpc_dt: float = None):
        if not _CASADI_AVAILABLE:
            raise ImportError(
                "casadi and do_mpc are required for CasadiMPC: "
                "pip install casadi do-mpc"
            )
        self.env = env
        self.params = params
        self.horizon = horizon
        # mpc_dt is the prediction step used inside the NLP (may differ from
        # the env's delta_t to give a meaningful planning horizon).
        self.mpc_dt = float(mpc_dt) if mpc_dt is not None else float(params.delta_t)
        self._initialized = False
        self._mpc = self._build_mpc()

    # ------------------------------------------------------------------
    # Override in subclasses
    # ------------------------------------------------------------------

    def _build_mpc(self):
        raise NotImplementedError

    def _extract_x0(self, state) -> np.ndarray:
        raise NotImplementedError

    def _update_setpoint(self, state):
        pass  # override when the setpoint is read from state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, _obs, state):
        """Compute the MPC action for the current environment state."""
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        if not self._initialized:
            self._mpc.x0 = x0
            self._mpc.set_initial_guess()
            self._initialized = True
        u = np.array(self._mpc.make_step(x0)).flatten()
        u_clipped = np.clip(u, -1.0, 1.0)
        return float(u_clipped[0]) if len(u_clipped) == 1 else u_clipped

    def reset(self):
        """Reset so that the next step re-initialises the warm-start."""
        self._initialized = False

    # ------------------------------------------------------------------
    # Shared do_mpc boilerplate
    # ------------------------------------------------------------------

    @staticmethod
    def _quiet_ipopt():
        return {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}


# ---------------------------------------------------------------------------
# CSTR
# ---------------------------------------------------------------------------


class CSTRCasadiMPC(CasadiMPC):
    """
    CasADi MPC for CSTR.

    States : [C_a, T]
    Input  : u_raw ∈ [-1, 1]  →  T_c ∈ [T_c_min, T_c_max]
    ODE    :
        dC_a/dt = q/V*(Caf - C_a) - k0*exp(-EA/R/T)*C_a
        dT/dt   = q/V*(Ti - T) + (-ΔHr)*rA/(ρ·C) + UA*(T_c - T)/(ρ·C·V)
    """

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        C_a = model.set_variable("_x", "C_a")
        T = model.set_variable("_x", "T")
        u_raw = model.set_variable("_u", "u_raw")
        target_CA = model.set_variable("_p", "target_CA")

        # Action scaling
        T_c = p.T_c_min + 0.5 * (u_raw + 1.0) * (p.T_c_max - p.T_c_min)
        rA = p.k0 * casadi.exp(-p.EA_over_R / T) * C_a

        model.set_rhs("C_a", p.q / p.V * (p.Caf - C_a) - rA)
        model.set_rhs(
            "T",
            p.q / p.V * (p.Ti - T)
            + (-p.deltaHr) * rA / (p.rho * p.C)
            + p.UA * (T_c - T) / (p.rho * p.C * p.V),
        )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        lterm = (target_CA - C_a) ** 2
        mpc.set_objective(lterm=lterm, mterm=lterm)
        mpc.set_rterm(u_raw=1e-4)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        self._target_CA = float(p.target_CA_range[0])
        p_tpl = mpc.get_p_template(1)

        def p_fun(_t):
            p_tpl["_p", 0, "target_CA"] = self._target_CA
            return p_tpl

        mpc.set_p_fun(p_fun)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.C_a), float(state.T)])

    def _update_setpoint(self, state):
        self._target_CA = float(state.target_CA)


# ---------------------------------------------------------------------------
# FirstOrderSystem
# ---------------------------------------------------------------------------


class FirstOrderCasadiMPC(CasadiMPC):
    """
    CasADi MPC for FirstOrderSystem.

    State : [x]
    Input : u_raw ∈ [-1, 1]  →  u ∈ [u_min, u_max]
    ODE   : dx/dt = (K·u - x) / tau
    """

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        x = model.set_variable("_x", "x")
        u_raw = model.set_variable("_u", "u_raw")
        target = model.set_variable("_p", "target_x")

        u = p.u_min + 0.5 * (u_raw + 1.0) * (p.u_max - p.u_min)
        model.set_rhs("x", (p.K * u - x) / p.tau)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        lterm = (target - x) ** 2
        mpc.set_objective(lterm=lterm, mterm=lterm)
        mpc.set_rterm(u_raw=1e-4)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        self._target_x = float(p.target_x_range[0])
        p_tpl = mpc.get_p_template(1)

        def p_fun(_t):
            p_tpl["_p", 0, "target_x"] = self._target_x
            return p_tpl

        mpc.set_p_fun(p_fun)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.x)])

    def _update_setpoint(self, state):
        self._target_x = float(state.target_x)


# ---------------------------------------------------------------------------
# FourTank
# ---------------------------------------------------------------------------


class FourTankCasadiMPC(CasadiMPC):
    """
    CasADi MPC for FourTank.

    States : [h1, h2, h3, h4]
    Inputs : [v1_raw, v2_raw] each ∈ [-1, 1]  →  [v1, v2] ∈ [v_min, v_max]
    ODE    : four-tank gravity-drain dynamics (see env.py)
    """

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        h1 = model.set_variable("_x", "h1")
        h2 = model.set_variable("_x", "h2")
        h3 = model.set_variable("_x", "h3")
        h4 = model.set_variable("_x", "h4")
        v1_raw = model.set_variable("_u", "v1_raw")
        v2_raw = model.set_variable("_u", "v2_raw")
        target_h1 = model.set_variable("_p", "target_h1")
        target_h2 = model.set_variable("_p", "target_h2")

        # Action scaling: raw ∈ [-1,1] → physical ∈ [v_min, v_max]
        v1 = p.v_min + 0.5 * (v1_raw + 1.0) * (p.v_max - p.v_min)
        v2 = p.v_min + 0.5 * (v2_raw + 1.0) * (p.v_max - p.v_min)

        eps = 1e-6  # avoid sqrt(0)
        sq = casadi.sqrt
        g2 = casadi.sqrt(2.0 * p.g)

        dh1 = (
            -(p.a1 / p.A1) * g2 * sq(casadi.fmax(h1, eps))
            + (p.a3 / p.A1) * g2 * sq(casadi.fmax(h3, eps))
            + (p.gamma1 * p.k1 / p.A1) * v1
        )
        dh2 = (
            -(p.a2 / p.A2) * g2 * sq(casadi.fmax(h2, eps))
            + (p.a4 / p.A2) * g2 * sq(casadi.fmax(h4, eps))
            + (p.gamma2 * p.k2 / p.A2) * v2
        )
        dh3 = (
            -(p.a3 / p.A3) * g2 * sq(casadi.fmax(h3, eps))
            + ((1 - p.gamma2) * p.k2 / p.A3) * v2
        )
        dh4 = (
            -(p.a4 / p.A4) * g2 * sq(casadi.fmax(h4, eps))
            + ((1 - p.gamma1) * p.k1 / p.A4) * v1
        )

        model.set_rhs("h1", dh1)
        model.set_rhs("h2", dh2)
        model.set_rhs("h3", dh3)
        model.set_rhs("h4", dh4)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        lterm = (target_h1 - h1) ** 2 + (target_h2 - h2) ** 2
        mpc.set_objective(lterm=lterm, mterm=lterm)
        mpc.set_rterm(v1_raw=1e-4, v2_raw=1e-4)

        mpc.bounds["lower", "_u", "v1_raw"] = -1.0
        mpc.bounds["upper", "_u", "v1_raw"] = 1.0
        mpc.bounds["lower", "_u", "v2_raw"] = -1.0
        mpc.bounds["upper", "_u", "v2_raw"] = 1.0

        # Keep levels above minimum to avoid sqrt(0)
        mpc.bounds["lower", "_x", "h1"] = float(p.h_min)
        mpc.bounds["lower", "_x", "h2"] = float(p.h_min)
        mpc.bounds["lower", "_x", "h3"] = float(p.h_min)
        mpc.bounds["lower", "_x", "h4"] = float(p.h_min)

        self._target_h1 = float(p.target_h1_range[0])
        self._target_h2 = float(p.target_h2_range[0])
        p_tpl = mpc.get_p_template(1)

        def p_fun(_t):
            p_tpl["_p", 0, "target_h1"] = self._target_h1
            p_tpl["_p", 0, "target_h2"] = self._target_h2
            return p_tpl

        mpc.set_p_fun(p_fun)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array(
            [float(state.h1), float(state.h2), float(state.h3), float(state.h4)]
        )

    def _update_setpoint(self, state):
        self._target_h1 = float(state.target_h1)
        self._target_h2 = float(state.target_h2)


# ---------------------------------------------------------------------------
# GlassFurnace
# ---------------------------------------------------------------------------


class GlassFurnaceCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the GlassFurnace (3-zone lumped thermal model).

    States : [T_crown, T_melt, T_work]   (°C)
    Input  : u_raw ∈ [-1, 1]  →  m_fuel ∈ [fuel_min, fuel_max]  (kg/s)

    Mirrors the physics in target_gym.glass_furnace.env.compute_velocity:
        - combustion heat  Q_comb = m_fuel * LHV
        - exhaust loss     Q_exh  = (1 + AFR) * m_fuel * c_p_gas * (T_crown - T_amb)
        - radiation crown→melt/work (Stefan-Boltzmann, Kelvin inside)
        - convection crown→melt/work
        - wall losses for each zone
        - batch fusion (endothermic, constant pull rate)
        - glass advection between zones at pull rate

    Task-hardening hooks:
        - Setpoint schedule: the target is a time-varying parameter (TVP) so
          MPC anticipates upcoming step changes over its horizon.
        - Fuel cost: the objective includes the same normalised fuel penalty
          used by the environment reward.
        - Pull-rate disturbance: MPC predicts with the *nominal* pull rate and
          relies on receding-horizon feedback to reject the noise.
    """

    def _build_mpc(self):
        p = self.params
        from target_gym.glass_furnace.env import N_SETPOINTS

        model = do_mpc.model.Model("continuous")

        T_crown = model.set_variable("_x", "T_crown")
        T_melt = model.set_variable("_x", "T_melt")
        T_work = model.set_variable("_x", "T_work")
        u_raw = model.set_variable("_u", "u_raw")
        # Target is a TVP so it can step along the schedule within the horizon.
        model.set_variable("_tvp", "target_T_crown")

        # Action scaling: raw ∈ [-1,1] → physical [fuel_min, fuel_max]
        m_fuel = p.fuel_min + 0.5 * (u_raw + 1.0) * (p.fuel_max - p.fuel_min)

        # Combustion / exhaust
        Q_comb = m_fuel * p.LHV
        m_gas = (1.0 + p.AFR) * m_fuel
        Q_exhaust = m_gas * p.c_p_gas * (T_crown - p.T_ambient)

        # Radiation (Stefan-Boltzmann requires absolute Kelvin)
        SIGMA_SB = 5.670374419e-8
        K = 273.15
        T_crown_K = T_crown + K
        T_melt_K = T_melt + K
        T_work_K = T_work + K
        Q_rad_CM = p.eps_rad * SIGMA_SB * p.A_melt * (T_crown_K**4 - T_melt_K**4)
        Q_rad_CW = p.eps_rad * SIGMA_SB * p.A_work * (T_crown_K**4 - T_work_K**4)

        # Convection
        Q_conv_CM = p.h_conv * p.A_melt * (T_crown - T_melt)
        Q_conv_CW = p.h_conv * p.A_work * (T_crown - T_work)

        # Wall losses
        Q_wall_crown = p.U_wall * p.A_wall_crown * (T_crown - p.T_ambient)
        Q_wall_melt = p.U_wall * p.A_wall_melt * (T_melt - p.T_ambient)
        Q_wall_work = p.U_wall * p.A_wall_work * (T_work - p.T_ambient)

        # Batch fusion + glass advection (use nominal pull rate for prediction)
        Q_fusion = p.m_pull * p.dH_fusion
        Q_adv_in_melt = p.m_pull * p.c_p_glass * (p.T_batch_in - T_melt)
        Q_adv_melt_to_work = p.m_pull * p.c_p_glass * (T_melt - T_work)

        model.set_rhs(
            "T_crown",
            (
                Q_comb
                - Q_rad_CM
                - Q_rad_CW
                - Q_conv_CM
                - Q_conv_CW
                - Q_wall_crown
                - Q_exhaust
            )
            / p.C_crown,
        )
        model.set_rhs(
            "T_melt",
            (Q_rad_CM + Q_conv_CM - Q_wall_melt - Q_fusion + Q_adv_in_melt) / p.C_melt,
        )
        model.set_rhs(
            "T_work",
            (Q_rad_CW + Q_conv_CW - Q_wall_work + Q_adv_melt_to_work) / p.C_work,
        )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        # Rebuild the u-dependent expressions from the *post-setup* model handle
        # (with a TVP present, do-mpc's pre-setup u_raw symbol becomes dangling
        # for objective construction — see issue in lterm Function build).
        u_raw_post = model.u["u_raw"]
        T_crown_post = model.x["T_crown"]
        target_post = model.tvp["target_T_crown"]
        m_fuel_post = p.fuel_min + 0.5 * (u_raw_post + 1.0) * (p.fuel_max - p.fuel_min)

        # Objective mirrors the environment reward exactly so the MPC isn't
        # making a different trade-off than the one used for scoring:
        #     reward = ((scale - |err|) / scale)^2  -  w * (fuel - fuel_min)/fuel_span
        # lterm = -reward (minimise).  We use a smoothed abs(err) so IPOPT can
        # take gradients through err=0.
        scale = float(p.T_crown_max - p.T_crown_min)
        fuel_span = float(p.fuel_max - p.fuel_min)
        err = target_post - T_crown_post
        err_abs = casadi.sqrt(err * err + 1e-4)  # smooth |err|
        tracking_reward = ((scale - err_abs) / scale) ** 2  # ∈ [0, 1]
        fuel_norm = (m_fuel_post - p.fuel_min) / fuel_span  # ∈ [0, 1]
        fuel_penalty = float(p.fuel_cost_weight) * fuel_norm
        lterm = -tracking_reward + fuel_penalty
        # mterm is only a function of x/tvp/p → drop fuel term (and the constant).
        mpc.set_objective(lterm=lterm, mterm=-tracking_reward)
        mpc.set_rterm(u_raw=1e-4)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        # Mutable schedule buffer — refreshed in _update_setpoint from the env state.
        default_target = float(sum(p.target_T_crown_range) / 2.0)
        self._target_schedule = np.full(N_SETPOINTS, default_target)
        self._current_step = 0
        self._max_steps = int(p.max_steps_in_episode)
        self._n_setpoints = int(N_SETPOINTS)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            # For each horizon step k = 0..N, look up which schedule slot the
            # env will be in and use that target.
            for k in range(self.horizon + 1):
                future_step = self._current_step + k
                slot = min(
                    (future_step * self._n_setpoints) // self._max_steps,
                    self._n_setpoints - 1,
                )
                tvp_tpl["_tvp", k, "target_T_crown"] = float(
                    self._target_schedule[slot]
                )
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array(
            [float(state.T_crown), float(state.T_melt), float(state.T_work)]
        )

    def _update_setpoint(self, state):
        self._target_schedule = np.asarray(state.target_schedule, dtype=float)
        self._current_step = int(state.time)


# ---------------------------------------------------------------------------
# Reactor
# ---------------------------------------------------------------------------


class ReactorCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the nuclear reactor (point-kinetics + thermal feedback).

    States : [n, C_1..6, T_fuel, T_coolant]       (9 states)
    Input  : u_raw in [-1, 1]  →  rho_ext in [rho_ext_min, rho_ext_max]

    Mirrors ``target_gym.reactor.env.compute_velocity``:
      - PKE with Lambda_gen, 6 delayed-neutron groups
      - Doppler + moderator thermal feedback (both negative)
      - Two-node thermal hydraulics

    The target is a time-varying parameter (TVP) so MPC anticipates setpoint
    changes scheduled later in the episode. Objective mirrors the env reward:
    quadratic tracking on ``n`` minus a small rod-motion penalty.
    """

    def _build_mpc(self):
        p = self.params
        from target_gym.reactor.env import (
            BETA_I,
            BETA_TOT,
            LAMBDA_I,
            N_GROUPS,
        )

        model = do_mpc.model.Model("continuous")

        n = model.set_variable("_x", "n")
        C = [model.set_variable("_x", f"C{i}") for i in range(N_GROUPS)]
        T_fuel = model.set_variable("_x", "T_fuel")
        T_coolant = model.set_variable("_x", "T_coolant")
        u_raw = model.set_variable("_u", "u_raw")
        model.set_variable("_tvp", "target_n")

        # Action scaling: raw ∈ [-1,1] → physical [rho_ext_min, rho_ext_max]
        rho_ext = p.rho_ext_min + 0.5 * (u_raw + 1.0) * (p.rho_ext_max - p.rho_ext_min)

        # Thermal feedback on reactivity (Doppler + moderator)
        rho_feedback = p.alpha_fuel * (T_fuel - p.T_fuel_ref) + p.alpha_coolant * (
            T_coolant - p.T_coolant_ref
        )
        rho = rho_ext + rho_feedback

        # Point kinetics
        sum_lambda_C = sum(float(LAMBDA_I[i]) * C[i] for i in range(N_GROUPS))
        model.set_rhs("n", ((rho - BETA_TOT) / p.Lambda_gen) * n + sum_lambda_C)
        for i in range(N_GROUPS):
            model.set_rhs(
                f"C{i}",
                (float(BETA_I[i]) / p.Lambda_gen) * n - float(LAMBDA_I[i]) * C[i],
            )

        # Two-node thermal model
        P_thermal = p.P_thermal_ref * n
        Q_fuel_to_cool = p.UA * (T_fuel - T_coolant)
        Q_flow_out = p.m_dot_cp * (T_coolant - p.T_inlet)
        model.set_rhs("T_fuel", (P_thermal - Q_fuel_to_cool) / p.C_fuel)
        model.set_rhs("T_coolant", (Q_fuel_to_cool - Q_flow_out) / p.C_coolant)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        # Collocation is needed: PKE is stiff (|prompt eigenvalue| ~60/s at
        # rho_ext=rho_ext_max), so explicit integrators inside the NLP would
        # require tiny substeps. do-mpc's default orthogonal collocation is
        # implicit and A-stable, handling the stiffness without substepping.
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
            state_discretization="collocation",
            collocation_type="radau",
            collocation_deg=3,
            collocation_ni=1,
        )

        # Post-setup handles for the objective
        u_raw_post = model.u["u_raw"]
        n_post = model.x["n"]
        target_post = model.tvp["target_n"]

        # Objective mirrors the env reward (minimise = -reward). Tracking term
        # is quadratic with scale = n_max - n_min so it matches the env's
        # ((max_diff - |err|) / max_diff)^2. We use a smooth |err| for IPOPT.
        scale = float(p.n_max - p.n_min)
        err = target_post - n_post
        err_abs = casadi.sqrt(err * err + 1e-6)  # smooth |err|
        tracking_reward = ((scale - err_abs) / scale) ** 2

        # Rod-motion penalty (matches env.compute_reward)
        rho_scale = float(max(abs(p.rho_ext_min), abs(p.rho_ext_max)))
        rho_ext_post = p.rho_ext_min + 0.5 * (u_raw_post + 1.0) * (
            p.rho_ext_max - p.rho_ext_min
        )
        rod_penalty = (
            float(p.rod_motion_weight)
            * casadi.sqrt(rho_ext_post * rho_ext_post + 1e-12)
            / rho_scale
        )

        lterm = -tracking_reward + rod_penalty
        mpc.set_objective(lterm=lterm, mterm=-tracking_reward)
        mpc.set_rterm(u_raw=1e-4)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        # Current target — refreshed in _update_setpoint from the env state.
        # With OU demand the future is unknown; MPC uses the current target
        # for the entire horizon (no schedule lookahead).
        self._current_target = float(sum(p.target_n_range) / 2.0)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for k in range(self.horizon + 1):
                tvp_tpl["_tvp", k, "target_n"] = self._current_target
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.concatenate(
            [
                np.array([float(state.n)]),
                np.asarray(state.C, dtype=float),
                np.array([float(state.T_fuel), float(state.T_coolant)]),
            ]
        )

    def _update_setpoint(self, state):
        self._current_target = float(state.target_n)


# ============================================================================
# Factory functions
# ============================================================================


def make_plane_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 50,
    lr: float = 0.05,
):
    """Gradient MPC for Airplane2D — optimises both power and stick in [-1, 1].

    Uses gradient-based MPC because the Plane has 9 coupled nonlinear ODEs
    including aerodynamic coefficients that are not expressible in CasADi
    without a full symbolic re-implementation.  dt=1.0 s; horizon=30.
    """
    return GradientMPC(
        env,
        params,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
    )


def make_plane3d_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 50,
    lr: float = 0.05,
):
    """Gradient MPC for the 3D plane tasks — optimises [power, stick, aileron] in [-1, 1].

    Same rationale as the 2D Plane MPC: the 3D dynamics extend the 2D
    aerodynamic model with roll, so it remains differentiable JAX but not
    expressible in CasADi. Works for all three task variants (Heading,
    Circle, FigureEight) since they share step_env.
    """
    return GradientMPC(
        env,
        params,
        action_dim=3,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
    )


def make_cstr_mpc(env, params, horizon: int = 5):
    """CasADi/IPOPT MPC for CSTR — matches the PC-gym oracle (N=5).

    With delta_t=0.25 s (PC-gym standard: tsim=25s, N=100), horizon=5 gives
    1.25 s lookahead — about one residence time (V/q=1 s).
    """
    return CSTRCasadiMPC(env, params, horizon=horizon)


def make_first_order_mpc(env, params, horizon: int = 5):
    """CasADi/IPOPT MPC for FirstOrderSystem — matches the PC-gym oracle (N=5)."""
    return FirstOrderCasadiMPC(env, params, horizon=horizon)


def make_four_tank_mpc(env, params, horizon: int = 5):
    """CasADi/IPOPT MPC for FourTank — matches the PC-gym oracle (N=5)."""
    return FourTankCasadiMPC(env, params, horizon=horizon)


def make_reactor_mpc(env, params, horizon: int = 20):
    """CasADi/IPOPT MPC for the nuclear reactor (point-kinetics + thermal feedback).

    With delta_t=0.5 s, horizon=20 gives 10 s of lookahead — enough to feel
    the fastest delayed-neutron group (λ≈3.0/s → τ≈0.33 s) and several
    slow-group time constants (λ≈0.012/s → τ≈80 s) are still visible via
    the integrator-like precursor dynamics. Longer horizons make the NLP
    expensive without meaningfully improving near-term tracking.
    """
    return ReactorCasadiMPC(env, params, horizon=horizon)


def make_glass_furnace_mpc(env, params, horizon: int = 60):
    """CasADi/IPOPT MPC for the GlassFurnace (3-zone lumped thermal model).

    With delta_t=30 s, horizon=60 gives 30 min lookahead.  The crown thermal
    time constant is ~15 min, so 2×τ of lookahead is enough to see the next
    scheduled setpoint change and pre-cool / pre-heat accordingly (which PID
    cannot do — that's the whole point of the schedule).
    """
    return GlassFurnaceCasadiMPC(env, params, horizon=horizon)
