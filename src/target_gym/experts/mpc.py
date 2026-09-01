"""
MPC oracle controllers for target_gym environments.

Three implementations are provided:

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

SamplingMPC  (JAX, gradient-free — Cement Kiln)
    Cross-entropy-method shooting: samples action sequences, rolls them out,
    refits to the elite fraction.  For plants whose forward rollout is well
    behaved but whose adjoint is not -- the kiln's Arrhenius-coupled advection
    makes reverse-mode gradients overflow to NaN after about eight steps, while
    finite differences on the same objective stay clean.

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
    n_tail : int
        Extra steps simulated past the optimised horizon, holding the last
        action, whose reward is added to the objective. This is a terminal cost:
        it charges the plan for the state it *leaves the plant in*, without
        adding decision variables, so the controller stops preferring plans that
        look excellent for ``horizon`` steps and reach a terminal state just
        after it.

        It assumes holding the last action approximates continuing sensibly.
        That is true where holding is close to trim and false where the plant
        needs active stabilisation, so it is enabled per environment on
        measurement rather than by default -- see ``make_plane_mpc``.
    done_value : float
        Per-step objective charged for every step *after* the planned rollout
        reaches a terminal state. ``step_env`` reports termination and the
        rollout previously ignored it, summing rewards straight through a plant
        that had already tripped or crashed -- so a plan that destroyed the
        plant at step 21 of 60 scored the same as one that flew it to the end.
        The default of 0.0 is the right charge whenever the objective is
        positive while the plant is healthy, which makes ending early cost the
        rest of the horizon. An objective that is negative when healthy has to
        set this below its own worst per-step value, or terminating early looks
        like an improvement.
    objective_fn : callable or None
        ``f(state, params) -> scalar`` summed over the rollout in place of the
        environment's own reward. Use it where that reward has a flat or clipped
        region: the controller descends it, and a gradient of zero is not a
        statement that the state is good. The surrogate must share the reward's
        *minimiser* while being smooth -- see ``make_wind_turbine_mpc`` for the
        case that motivates it.
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
        n_tail: int = 0,
        done_value: float = 0.0,
        objective_fn=None,
    ):
        self.env = env
        self.params = params
        self.horizon = horizon
        self.n_iter = n_iter
        self.lr = lr
        self.action_dim = action_dim
        self.action_lb = float(action_lb)
        self.action_ub = float(action_ub)
        self.n_tail = int(n_tail)
        self.done_value = float(done_value)
        self.objective_fn = objective_fn

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
            s, done = carry
            _, new_s, r, terminated, _ = self.env.step_env(
                key, s, self._env_action(u), self.params
            )
            if self.objective_fn is not None:
                r = self.objective_fn(new_s, self.params)
            # Past a terminal state the plant no longer exists; charge the rest
            # of the horizon rather than pretending it kept earning.
            r = jnp.where(done, self.done_value, r)
            return (new_s, jnp.logical_or(done, terminated)), r

        init = (state, jnp.zeros((), dtype=bool))
        (final, done), rewards = jax.lax.scan(step_fn, init, actions)
        total = jnp.sum(rewards)
        if self.n_tail:
            held = jnp.broadcast_to(actions[-1], (self.n_tail,) + actions.shape[1:])
            _, tail_rewards = jax.lax.scan(step_fn, (final, done), held)
            total = total + jnp.sum(tail_rewards)
        return total

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
    CasADi MPC for the regenerative glass furnace.

    Prediction model (7 states): ``[T_crown, T_melt, T_work, m_batch,
    T_regen_hot, T_regen_mid, T_regen_cold]``, plus the flame temperature as
    an *algebraic* variable.

    Deliberately a reduced model of the 11-state plant, and the reductions are
    chosen so the controller still sees everything that sets the crown's
    response to fuel:

    * **The regenerator is kept**, collapsed from two alternating 4-node
      chambers to one 3-node stack at the cycle average. Air preheat supplies
      ~40 % of the useful heat input, so a controller blind to it mis-predicts
      the steady-state gain badly -- the same mistake the reactor MPC made by
      dropping xenon.
    * **The reversal cycle is averaged out.** Its 25 min period is well inside
      the 30 min horizon and it is a known, autonomous oscillation the
      controller cannot influence; predicting its phase buys nothing.
    * **The flame is algebraic**, matching the plant's quasi-steady treatment,
      so no stiff fast state enters the NLP.
    * **The batch blanket is kept** because its shielding sets how much
      radiation reaches the glass, which is strongly pull-rate dependent.

    The setpoint schedule enters as a time-varying parameter so the MPC
    anticipates step changes -- the advantage PID structurally cannot have.
    """

    def _build_mpc(self):
        p = self.params
        from target_gym.glass_furnace.env import N_SETPOINTS

        model = do_mpc.model.Model("continuous")
        SB = 5.670374419e-8
        K = 273.15

        T_crown = model.set_variable("_x", "T_crown")
        T_melt = model.set_variable("_x", "T_melt")
        T_work = model.set_variable("_x", "T_work")
        m_batch = model.set_variable("_x", "m_batch")
        T_rh = model.set_variable("_x", "T_rh")
        T_rm = model.set_variable("_x", "T_rm")
        T_rc = model.set_variable("_x", "T_rc")
        T_gas = model.set_variable("_z", "T_gas")  # algebraic: quasi-steady flame
        u_raw = model.set_variable("_u", "u_raw")
        model.set_variable("_tvp", "target_T_crown")

        m_fuel = p.fuel_min + 0.5 * (u_raw + 1.0) * (p.fuel_max - p.fuel_min)
        m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
        m_gas = m_fuel + m_air

        # Cycle-averaged regenerator: air climbs cold -> hot, exhaust descends.
        eps = p.eps_regen_node
        Ta1 = p.T_ambient + eps * (T_rc - p.T_ambient)
        Ta2 = Ta1 + eps * (T_rm - Ta1)
        T_air = Ta2 + eps * (T_rh - Ta2)
        Te1 = T_gas - eps * (T_gas - T_rh)
        Te2 = Te1 - eps * (Te1 - T_rm)
        T_stack = Te2 - eps * (Te2 - T_rc)

        # Half the cycle in each role -> average the two duties.
        Q_rh = 0.5 * (
            m_gas * p.c_p_gas * (T_gas - Te1) - m_air * p.c_p_air * (T_air - Ta2)
        )
        Q_rm = 0.5 * (m_gas * p.c_p_gas * (Te1 - Te2) - m_air * p.c_p_air * (Ta2 - Ta1))
        Q_rc = 0.5 * (
            m_gas * p.c_p_gas * (Te2 - T_stack)
            - m_air * p.c_p_air * (Ta1 - p.T_ambient)
        )
        UA_node = p.U_regen * p.A_regen / 3.0

        coverage = m_batch / p.m_batch_full
        melt_open = 1.0 - p.batch_shield * coverage

        T_gas_K = T_gas + K
        T_crown_K = T_crown + K
        T_melt_K = T_melt + K
        T_work_K = T_work + K

        A_eff = p.A_crown + p.A_melt * melt_open + p.A_work
        Q_in = m_fuel * p.LHV + m_air * p.c_p_air * (T_air - p.T_ambient)
        sink_K4 = (
            p.A_crown * T_crown_K**4
            + p.A_melt * melt_open * T_melt_K**4
            + p.A_work * T_work_K**4
        )
        sink_T = p.A_crown * T_crown + p.A_melt * melt_open * T_melt + p.A_work * T_work
        # Algebraic constraint: the flame's own energy balance closes.
        model.set_alg(
            "flame_balance",
            Q_in
            - p.eps_rad * SB * (A_eff * T_gas_K**4 - sink_K4)
            - p.h_conv * (A_eff * T_gas - sink_T)
            - m_gas * p.c_p_gas * (T_gas - p.T_ambient),
        )

        Q_rad_gc = p.eps_rad * SB * p.A_crown * (T_gas_K**4 - T_crown_K**4)
        Q_rad_gm = p.eps_rad * SB * p.A_melt * (T_gas_K**4 - T_melt_K**4) * melt_open
        Q_rad_gw = p.eps_rad * SB * p.A_work * (T_gas_K**4 - T_work_K**4)
        Q_rad_cm = p.eps_rad * SB * p.A_melt * (T_crown_K**4 - T_melt_K**4) * melt_open
        Q_rad_cw = p.eps_rad * SB * p.A_work * (T_crown_K**4 - T_work_K**4)
        Q_conv_gc = p.h_conv * p.A_crown * (T_gas - T_crown)
        Q_conv_gm = p.h_conv * p.A_melt * (T_gas - T_melt) * melt_open
        Q_conv_gw = p.h_conv * p.A_work * (T_gas - T_work)

        Q_wall_c = p.U_wall * p.A_wall_crown * (T_crown - p.T_ambient)
        Q_wall_m = p.U_wall * p.A_wall_melt * (T_melt - p.T_ambient)
        Q_wall_w = p.U_wall * p.A_wall_work * (T_work - p.T_ambient)
        Q_cool_w = p.UA_work_cooling * (T_work - p.T_ambient)

        Q_to_batch = (
            p.batch_shield
            * coverage
            * p.eps_rad
            * SB
            * p.A_melt
            * ((T_gas_K**4 - T_melt_K**4) + (T_crown_K**4 - T_melt_K**4))
        )
        melt_rate = Q_to_batch / p.dH_fusion

        cp_melt = p.c_p_glass_a + p.c_p_glass_b * T_melt
        cp_work = p.c_p_glass_a + p.c_p_glass_b * T_work

        model.set_rhs(
            "T_crown",
            (Q_rad_gc + Q_conv_gc - Q_rad_cm - Q_rad_cw - Q_wall_c) / p.C_crown,
        )
        model.set_rhs(
            "T_melt",
            (
                Q_rad_gm
                + Q_conv_gm
                + Q_rad_cm
                - Q_wall_m
                - melt_rate * p.dH_fusion
                + p.m_pull * cp_melt * (p.T_batch_in - T_melt)
            )
            / (p.C_melt * cp_melt / p.c_p_glass_a),
        )
        model.set_rhs(
            "T_work",
            (
                Q_rad_gw
                + Q_conv_gw
                + Q_rad_cw
                - Q_wall_w
                - Q_cool_w
                + p.m_pull * cp_work * (T_melt - T_work)
            )
            / (p.C_work * cp_work / p.c_p_glass_a),
        )
        model.set_rhs("m_batch", p.m_pull / p.batch_yield - melt_rate)
        model.set_rhs("T_rh", (Q_rh - UA_node * (T_rh - p.T_ambient)) / p.C_regen_node)
        model.set_rhs("T_rm", (Q_rm - UA_node * (T_rm - p.T_ambient)) / p.C_regen_node)
        model.set_rhs("T_rc", (Q_rc - UA_node * (T_rc - p.T_ambient)) / p.C_regen_node)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        u_post = model.u["u_raw"]
        T_crown_post = model.x["T_crown"]
        target_post = model.tvp["target_T_crown"]
        m_fuel_post = p.fuel_min + 0.5 * (u_post + 1.0) * (p.fuel_max - p.fuel_min)

        # Share the minimiser of env.compute_reward, not its shape.
        #
        # This previously normalised the error by the crown's whole 250 K
        # envelope while the environment scores it against ``tracking_scale``,
        # 40 K -- so a 40 K error, which the environment scores as zero, entered
        # the objective at 0.71, six times flatter than the reward being graded.
        # Against an unchanged fuel penalty the controller duly sold tracking for
        # fuel, and lost to the PID on 7 of 10 seeds.
        #
        # The old form was also non-monotonic: ``((scale - err)/scale)**2`` turns
        # back upward past ``err = scale``, so beyond twice it the objective
        # preferred *more* error. A plain squared normalised error is monotone,
        # smooth, and minimised in the same place as the reward.
        scale = float(p.tracking_scale)
        fuel_span = float(p.fuel_max - p.fuel_min)
        err = target_post - T_crown_post
        err_abs = casadi.sqrt(err * err + 1e-4)  # smooth |err|
        # Bounded rather than a bare quadratic. Normalising by 40 K instead of
        # 250 K is what fixes the weighting, but it also makes an unbounded
        # ``e**2`` reach 6.25 at a 100 K error where the old term gave 0.36, and
        # IPOPT does not cope: one seed of ten went from ~48 s to over 400 s.
        # ``e**2/(1+e**2)`` has the same minimiser and the same slope near zero,
        # is monotone in the error, and saturates instead of diverging -- which
        # also matches the environment's own reward, itself bounded in [0, 1].
        e_norm = err_abs / scale
        tracking_cost = e_norm**2 / (1.0 + e_norm**2)
        fuel_pen = float(p.fuel_cost_weight) * (m_fuel_post - p.fuel_min) / fuel_span
        mpc.set_objective(lterm=tracking_cost + fuel_pen, mterm=tracking_cost)
        mpc.set_rterm(u_raw=1e-3)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        default_target = float(sum(p.target_T_crown_range) / 2.0)
        self._target_schedule = np.full(N_SETPOINTS, default_target)
        self._current_step = 0
        self._max_steps = int(p.max_steps_in_episode)
        self._n_setpoints = int(N_SETPOINTS)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for k in range(self.horizon + 1):
                future = self._current_step + k
                slot = min(
                    (future * self._n_setpoints) // self._max_steps,
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
        # Collapse the plant's two 4-node chambers onto the model's 3 nodes by
        # averaging the chambers (they alternate) and resampling the profile.
        profile = 0.5 * (
            np.asarray(state.T_rA, dtype=float) + np.asarray(state.T_rB, dtype=float)
        )
        resampled = np.interp(
            np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, len(profile)), profile
        )
        return np.array(
            [
                float(state.T_crown),
                float(state.T_melt),
                float(state.T_work),
                float(state.m_batch),
                resampled[0],
                resampled[1],
                resampled[2],
            ]
        )

    def _update_setpoint(self, state):
        self._target_schedule = np.asarray(state.target_schedule, dtype=float)
        self._current_step = int(state.time)


# ---------------------------------------------------------------------------
# Reactor
# ---------------------------------------------------------------------------


class ReactorCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the nuclear reactor (point kinetics + xenon + thermal).

    States : [n, C_1..6, T_fuel, T_coolant, I_hat, Xe_hat, rho_ext]  (12)
    Input  : rho_rate -- the control-rod *speed*, not its position.

    Two modelling choices matter, and the original version got both wrong:

    **Xenon is in the model.** ``I_hat``/``Xe_hat`` and the reactivity term
    ``-rho_Xe_full * (Xe_hat - 1)`` were previously omitted entirely, so the
    "oracle" was blind to the multi-hour iodine/xenon swing that the
    environment docstring calls the dominant control challenge. Equilibrium
    xenon worth (~2500 pcm) dwarfs total rod authority (500 pcm), so an MPC
    that cannot see it is not an oracle -- the reported PID-vs-MPC gap was
    measuring model mismatch rather than controller quality.

    **Rod speed is a rate limit, not a position bound.** The environment moves
    ``rho_ext`` toward the demanded position at ``rod_speed_withdraw`` /
    ``rod_speed_insert``, asymmetrically. Treating the action as a position the
    plant reaches instantly lets the MPC plan trajectories it cannot fly. Here
    ``rho_ext`` is a *state* and the input is its rate, bounded by the two rod
    speeds -- an exact, and conveniently linear, encoding of the constraint.

    ``mpc_dt`` defaults to ``delta_t * control_period``: the environment holds
    one action across ``control_period`` physics sub-steps, so planning at the
    raw physics step would model a control authority that does not exist.
    """

    def __init__(self, env, params, horizon: int = 20, mpc_dt: float = None):
        if mpc_dt is None:
            control_period = getattr(env, "control_period", 1)
            mpc_dt = float(params.delta_t) * float(control_period)
        super().__init__(env, params, horizon=horizon, mpc_dt=mpc_dt)

    def _build_mpc(self):
        p = self.params
        from target_gym.reactor.env import (
            BETA_I,
            BETA_TOT,
            LAMBDA_I,
            LAMBDA_IODINE,
            LAMBDA_XENON,
            N_GROUPS,
        )

        model = do_mpc.model.Model("continuous")

        n = model.set_variable("_x", "n")
        C = [model.set_variable("_x", f"C{i}") for i in range(N_GROUPS)]
        T_fuel = model.set_variable("_x", "T_fuel")
        T_coolant = model.set_variable("_x", "T_coolant")
        I_hat = model.set_variable("_x", "I_hat")
        Xe_hat = model.set_variable("_x", "Xe_hat")
        rho_ext = model.set_variable("_x", "rho_ext")
        rho_rate = model.set_variable("_u", "rho_rate")
        model.set_variable("_tvp", "target_n")

        # Rod position integrates the commanded rod speed.
        model.set_rhs("rho_ext", rho_rate)

        # Reactivity: rod + thermal feedback (Doppler + moderator) + xenon.
        rho_feedback = p.alpha_fuel * (T_fuel - p.T_fuel_ref) + p.alpha_coolant * (
            T_coolant - p.T_coolant_ref
        )
        rho_xenon = -p.rho_Xe_full * (Xe_hat - 1.0)
        rho = rho_ext + rho_feedback + rho_xenon

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

        # Iodine / xenon chain (normalised; matches reactor.env.compute_velocity)
        lam_sum = LAMBDA_XENON + p.sigma_phi0
        a_coeff = p.gamma_ratio * lam_sum / (1.0 + p.gamma_ratio)
        b_coeff = lam_sum / (1.0 + p.gamma_ratio)
        model.set_rhs("I_hat", LAMBDA_IODINE * (n - I_hat))
        model.set_rhs(
            "Xe_hat",
            a_coeff * n + b_coeff * I_hat - (LAMBDA_XENON + p.sigma_phi0 * n) * Xe_hat,
        )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        # Collocation is needed: PKE is stiff (|prompt eigenvalue| ~60/s at
        # rho_ext=rho_ext_max), so explicit integrators inside the NLP would
        # require tiny substeps. do-mpc's orthogonal collocation is implicit
        # and A-stable, handling the stiffness without substepping.
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

        n_post = model.x["n"]
        rho_ext_post = model.x["rho_ext"]
        target_post = model.tvp["target_n"]

        # Objective mirrors env.compute_reward: a Gaussian tracking term and a
        # rod-position penalty. The environment's reward is
        # ``exp(-0.5*(err/band)^2) - w*|rho_ext|/rho_scale``; exp() is flat far
        # from the target and gives IPOPT almost no gradient there, so we
        # minimise the equivalent quadratic ``(err/band)^2`` instead -- same
        # minimiser, far better conditioned.
        err = target_post - n_post
        tracking_cost = (err / p.reward_band) ** 2
        rho_scale = float(max(abs(p.rho_ext_min), abs(p.rho_ext_max)))
        rod_penalty = (
            float(p.rod_motion_weight)
            * casadi.sqrt(rho_ext_post * rho_ext_post + 1e-12)
            / rho_scale
        )
        mpc.set_objective(lterm=tracking_cost + rod_penalty, mterm=tracking_cost)
        mpc.set_rterm(rho_rate=1e2)

        # Rod *speed* limits (asymmetric: insertion is faster than withdrawal).
        mpc.bounds["lower", "_u", "rho_rate"] = -float(p.rod_speed_insert)
        mpc.bounds["upper", "_u", "rho_rate"] = float(p.rod_speed_withdraw)
        # Rod travel limits.
        mpc.bounds["lower", "_x", "rho_ext"] = float(p.rho_ext_min)
        mpc.bounds["upper", "_x", "rho_ext"] = float(p.rho_ext_max)

        # With OU demand the future is unknown; hold the current target across
        # the horizon.
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
                np.array(
                    [
                        float(state.T_fuel),
                        float(state.T_coolant),
                        float(state.I_hat),
                        float(state.Xe_hat),
                        float(state.rho_ext),
                    ]
                ),
            ]
        )

    def _update_setpoint(self, state):
        self._current_target = float(state.target_n)

    def step(self, _obs, state):
        """Return the raw action in [-1, 1] the environment expects.

        The NLP optimises a rod *rate*, but ``step_env`` takes a demanded rod
        *position*. Integrating one step of the optimal rate gives the position
        to demand; because the environment rate-limits toward that demand with
        the same bounds the NLP respects, the realised motion matches the plan.
        """
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        if not self._initialized:
            self._mpc.x0 = x0
            self._mpc.set_initial_guess()
            self._initialized = True
        rho_rate = float(np.array(self._mpc.make_step(x0)).flatten()[0])

        p = self.params
        rho_next = float(
            np.clip(
                float(state.rho_ext) + rho_rate * self.mpc_dt,
                p.rho_ext_min,
                p.rho_ext_max,
            )
        )
        span = p.rho_ext_max - p.rho_ext_min
        raw = 2.0 * (rho_next - p.rho_ext_min) / span - 1.0
        return float(np.clip(raw, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Building HVAC
# ---------------------------------------------------------------------------


class HVACCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the single-zone building (ISO 13790 5R1C).

    States : [T_mass, Q_emitter]      Input: u_raw in [-1, 1] -> [0, Q_heat_max]

    Only two differential states: the 5R1C air and surface nodes have no
    capacitance, so they are substituted in closed form exactly as the plant
    does. The model is therefore an almost-exact copy of the plant rather than
    a reduction -- unusual here, and possible because the building model is
    genuinely low-order.

    Where the MPC earns its advantage is **anticipation**. Setpoint schedule,
    outdoor temperature, solar gain and occupancy gain all enter as
    time-varying parameters over the horizon, so the controller can pre-heat
    ahead of the morning setback recovery and back off ahead of a sunny
    afternoon. A PID sees none of that until it has already happened, and with
    a 43 h thermal time constant "already happened" is far too late.
    """

    def __init__(self, env, params, horizon: int = 24, mpc_dt: float = None):
        super().__init__(
            env, params, horizon=horizon, mpc_dt=mpc_dt or float(params.delta_t)
        )

    def _build_mpc(self):
        p = self.params
        from target_gym.hvac.env import zone_conductances

        c = zone_conductances(p)
        H_is, H_ms, H_w, H_em, H_ve = (
            c["H_tr_is"],
            c["H_tr_ms"],
            c["H_tr_w"],
            c["H_tr_em"],
            c["H_ve"],
        )
        A_tot, A_m, C_m = c["A_tot"], c["A_m"], c["C_m"]

        model = do_mpc.model.Model("continuous")
        T_mass = model.set_variable("_x", "T_mass")
        Q_emitter = model.set_variable("_x", "Q_emitter")
        u_raw = model.set_variable("_u", "u_raw")
        T_out = model.set_variable("_tvp", "T_out")
        phi_int = model.set_variable("_tvp", "phi_int")
        phi_sol = model.set_variable("_tvp", "phi_sol")
        model.set_variable("_tvp", "target_T")

        Q_command = 0.5 * (u_raw + 1.0) * p.Q_heat_max

        # Gain split (ISO 13790), mirroring hvac.env.split_gains.
        phi_ia = 0.5 * phi_int
        remainder = 0.5 * phi_int + phi_sol
        phi_m = (A_m / A_tot) * remainder
        phi_st = (1.0 - A_m / A_tot - H_w / (9.1 * A_tot)) * remainder

        # Algebraic air/surface nodes in closed form (see solve_air_and_surface).
        denom_air = H_is + H_ve
        a = H_is / denom_air
        b = (H_ve * T_out + phi_ia + Q_emitter) / denom_air
        T_surface = (H_ms * T_mass + H_w * T_out + phi_st + H_is * b) / (
            H_ms + H_w + H_is * (1.0 - a)
        )
        T_air = a * T_surface + b

        model.set_rhs(
            "T_mass",
            (H_ms * (T_surface - T_mass) + H_em * (T_out - T_mass) + phi_m) / C_m,
        )
        model.set_rhs("Q_emitter", (Q_command - Q_emitter) / p.emitter_tau)
        model.set_expression("T_air", T_air)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        # Objective mirrors env.compute_reward: squared comfort band minus
        # normalised energy. Minimising -reward.
        T_air_post = model.aux["T_air"]
        target_post = model.tvp["target_T"]
        err = target_post - T_air_post
        # Quadratic in the error rather than a copy of the reward. The reward
        # is clipped, and both failure modes of copying it have been observed:
        # dropping the clip makes the quadratic turn back *upward* past
        # 2*comfort_band so large errors score better (the MPC stopped heating
        # entirely), while keeping the clip makes the objective *flat* out
        # there so the solver sees no gradient at all. What matters is a shared
        # minimiser with a usable gradient everywhere, which a quadratic gives.
        comfort = -((err / (2.0 * p.comfort_band)) ** 2)
        energy = model.x["Q_emitter"] / p.Q_heat_max
        mpc.set_objective(
            lterm=-comfort + float(p.energy_weight) * energy, mterm=-comfort
        )
        mpc.set_rterm(u_raw=1e-3)
        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        self._current_step = 0
        self._setpoint_occupied = float(p.setpoint_occupied)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            from target_gym.hvac.env import (
                internal_gain,
                outdoor_temperature,
                scheduled_setpoint,
                solar_gain,
            )

            for k in range(self.horizon + 1):
                t = self._current_step + k
                # Forecast uses the deterministic daily cycle; the stochastic
                # weather deviation is unknown, so it is left at zero and
                # rejected by receding-horizon feedback.
                tvp_tpl["_tvp", k, "T_out"] = float(outdoor_temperature(t, 0.0, p))
                tvp_tpl["_tvp", k, "phi_int"] = float(internal_gain(t, p))
                tvp_tpl["_tvp", k, "phi_sol"] = float(solar_gain(t, p))
                tvp_tpl["_tvp", k, "target_T"] = float(
                    scheduled_setpoint(t, self._setpoint_occupied, p)
                )
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.T_mass), float(state.Q_emitter)])

    def _update_setpoint(self, state):
        self._current_step = int(state.time)
        self._setpoint_occupied = float(state.setpoint_occupied)


# ---------------------------------------------------------------------------
# pH neutralisation
# ---------------------------------------------------------------------------


class PHCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the pH neutralisation CSTR.

    States : [Wa, Wb]  reaction invariants     Input: u_raw -> base flow q3
    Algebraic: pH, defined implicitly by the charge balance.

    The invariants mix *linearly*, so the only nonlinearity is the titration
    curve -- and that is exactly where an MPC should beat a PID. Expressing pH
    as an algebraic variable constrained by the charge balance lets IPOPT see
    the curve directly and take its gradient, instead of a fixed-gain
    controller having to compromise across a ~45x gain variation.

    Buffer flow is *unmeasured*, so the prediction model uses the nominal value
    and leans on receding-horizon feedback to reject the drift -- the same
    treatment the glass furnace gives its pull-rate disturbance.
    """

    def __init__(self, env, params, horizon: int = 20, mpc_dt: float = None):
        super().__init__(
            env, params, horizon=horizon, mpc_dt=mpc_dt or float(params.delta_t)
        )

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        Wa = model.set_variable("_x", "Wa")
        Wb = model.set_variable("_x", "Wb")
        pH = model.set_variable("_z", "pH")
        u_raw = model.set_variable("_u", "u_raw")
        model.set_variable("_tvp", "target_pH")

        q3 = p.q3_min + 0.5 * (u_raw + 1.0) * (p.q3_max - p.q3_min)
        q2 = p.q2_nominal  # unmeasured; nominal in the model

        model.set_rhs(
            "Wa",
            (p.q1 * (p.Wa1 - Wa) + q2 * (p.Wa2 - Wa) + q3 * (p.Wa3 - Wa)) / p.V,
        )
        model.set_rhs(
            "Wb",
            (p.q1 * (p.Wb1 - Wb) + q2 * (p.Wb2 - Wb) + q3 * (p.Wb3 - Wb)) / p.V,
        )

        # Charge balance as the algebraic constraint defining pH.
        carbonate = (1.0 + 2.0 * 10.0 ** (pH - p.pK2)) / (
            1.0 + 10.0 ** (p.pK1 - pH) + 10.0 ** (pH - p.pK2)
        )
        model.set_alg(
            "charge_balance",
            Wa + 10.0 ** (pH - 14.0) - 10.0 ** (-pH) + Wb * carbonate,
        )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        # Tracking cost is a plain quadratic in the error, NOT a copy of the
        # environment's clipped reward. The two must share a *minimiser*, but
        # the reward's clip is flat once the error exceeds the tracking band,
        # and a flat objective gives IPOPT no gradient: the solver optimised
        # the only live term (reagent cost) and railed the valve shut, driving
        # pH away from the setpoint at ~3.9 pH mean error. A quadratic has the
        # same minimiser and a usable gradient everywhere.
        pH_post = model.z["pH"]
        target_post = model.tvp["target_pH"]
        u_post = model.u["u_raw"]
        err = target_post - pH_post
        tracking = -((err / p.tracking_band) ** 2)
        q3_post = p.q3_min + 0.5 * (u_post + 1.0) * (p.q3_max - p.q3_min)
        reagent = (q3_post - p.q3_min) / (p.q3_max - p.q3_min)
        # do-mpc's terminal cost may only reference differential states, and
        # pH here is algebraic -- so the tracking term lives entirely in the
        # stage cost and mterm is a symbolic zero.
        mpc.set_objective(
            lterm=-tracking + float(p.reagent_cost_weight) * reagent,
            mterm=0.0 * model.x["Wa"],
        )
        mpc.set_rterm(u_raw=1e-3)
        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0
        mpc.bounds["lower", "_z", "pH"] = 0.0
        mpc.bounds["upper", "_z", "pH"] = 14.0

        self._target = float(sum(p.target_pH_range) / 2.0)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for k in range(self.horizon + 1):
                tvp_tpl["_tvp", k, "target_pH"] = self._target
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.Wa), float(state.Wb)])

    def _update_setpoint(self, state):
        self._target = float(state.target_pH)

    def step(self, _obs, state):
        """Seed the algebraic pH before solving.

        The DAE's algebraic variable needs a starting point on the titration
        curve. Left at do-mpc's default the solver begins far off it, where the
        charge balance is nearly flat in pH, and converges somewhere useless --
        which showed up as ~3.9 pH mean error, worse than a constant valve.
        The plant's measured pH is the obvious guess.
        """
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        if not self._initialized:
            self._mpc.x0 = x0
            self._mpc.z0 = np.array([float(state.pH)])
            self._mpc.set_initial_guess()
            self._initialized = True
        u = np.array(self._mpc.make_step(x0)).flatten()
        return float(np.clip(u, -1.0, 1.0)[0])


# ============================================================================
# Factory functions
# ============================================================================


_PLANE_STALL_ONSET = 0.5  # fraction of aoa_stall at which the barrier starts
_PLANE_STALL_WEIGHT = 20.0


def _plane_objective(state, params):
    """Environment reward plus a soft stall-margin barrier.

    The aircraft's irrecoverable event is the stall, not the ground: by the time
    altitude is low enough for a boundary penalty to bite, a departed aircraft is
    already descending at 270 m/s and cannot recover. A barrier on altitude was
    measured and does nothing for that reason.

    The stall itself is invisible to the planner for the usual reason -- the
    crash penalty sits behind ``where(terminated, ...)`` on a boolean, so it
    carries a cost but no derivative. Charging the *approach* to the stall angle
    does carry one, and it converts one of the two failing seeds from a crash
    (-503) to 422, ahead of the PID's 267 there.

    It does not fix everything: one seed still commands a descent it cannot
    arrest, diving through the target at 265 m/s with the angle of attack small
    throughout, so no stall barrier applies to it. Altitude barriers, tails to
    240 steps and a matched crash charge were all measured against that case and
    none helped.
    """
    from target_gym.plane.env import compute_reward

    reward = compute_reward(state, params)
    gamma = jnp.arctan2(state.z_dot, jnp.maximum(jnp.abs(state.x_dot), 1e-3))
    aoa_deg = jnp.rad2deg(state.theta - gamma)
    margin = jnp.abs(aoa_deg) / params.aoa_stall
    barrier = jnp.maximum(margin - _PLANE_STALL_ONSET, 0.0) ** 2
    return reward - _PLANE_STALL_WEIGHT * barrier


def make_plane_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 50,
    lr: float = 0.05,
    n_tail: int = 60,
    objective_fn=_plane_objective,
):
    """Gradient MPC for Airplane2D — optimises both power and stick in [-1, 1].

    Uses gradient-based MPC because the Plane has 9 coupled nonlinear ODEs
    including aerodynamic coefficients that are not expressible in CasADi
    without a full symbolic re-implementation.  dt=1.0 s; horizon=30.

    ``n_tail=60`` is what makes this controller work. Optimising 30 s of flight
    and being charged for nothing beyond it, the plan climbed hard, ran the
    airspeed down and left the aircraft outside the altitude envelope just past
    the horizon: it settled 654x worse than the PID and crashed in one episode
    of two. Simulating 60 further seconds on the held action -- which for this
    aircraft is close to trim -- prices that ending into the objective. Measured
    over 600-step episodes, settled tracking error goes from 2949 m to 0.083 m,
    which is 55x *better* than the PID rather than 654x worse, with no
    terminations. Sixty is the knee: 120 is no better (0.092 m) and costs twice.
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
        n_tail=n_tail,
        objective_fn=objective_fn,
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


def make_four_tank_mpc(env, params, horizon: int = 10, mpc_dt: float = None):
    """CasADi/IPOPT MPC for FourTank.

    PC-gym's oracle is N=5 at the environment's own step, and that is what this
    shipped: a horizon covering 5 s of a plant whose tracking error takes ~198 s
    to close (``scripts/audit_mpc_horizons.py`` puts it at ratio 0.03, the worst
    in the suite). It settled 47x worse than the PID and drove a tank to a
    terminal state in one episode of two.

    The fix is covered *time*, not more decision variables -- measured, raising
    the horizon to 20 steps at the environment's own step is still 54x worse,
    while any configuration reaching ~200 s drives the settled error to zero
    with no terminations. So the prediction step is decoupled from the
    environment's: ten steps of 20 dt cover 200 s for a tenth of the decision
    variables that would otherwise take.
    """
    if mpc_dt is None:
        mpc_dt = 20.0 * float(params.delta_t)
    return FourTankCasadiMPC(env, params, horizon=horizon, mpc_dt=mpc_dt)


def make_reactor_mpc(env, params, horizon: int = 20):
    """CasADi/IPOPT MPC for the nuclear reactor (point-kinetics + thermal feedback).

    With delta_t=0.5 s, horizon=20 gives 10 s of lookahead — enough to feel
    the fastest delayed-neutron group (λ≈3.0/s → τ≈0.33 s) and several
    slow-group time constants (λ≈0.012/s → τ≈80 s) are still visible via
    the integrator-like precursor dynamics. Longer horizons make the NLP
    expensive without meaningfully improving near-term tracking.
    """
    return ReactorCasadiMPC(env, params, horizon=horizon)


class BoilerDrumGradientMPC(GradientMPC):
    """Gradient MPC for the drum boiler, on a quadratic objective.

    The environment's reward uses clipped tracking bands, which go flat once
    the level is more than ``level_band`` from normal -- precisely the
    situation the controller is called on to fix. Optimising it directly leaves
    no gradient exactly when one is needed, so the objective here is a
    quadratic sharing the reward's *minimiser* (level at normal, pressure on
    target, fuel low) while staying informative far from it.

    Horizon is what earns MPC its keep here. Drum level is non-minimum phase:
    the level's first move after a load change is the wrong way. A controller
    optimising over 30 steps (60 s at dt = 2 s, covering the ~35 s swell peak)
    sees the reversal coming and keeps adding feedwater through a swell, where
    a reactive loop cuts it.
    """

    def __init__(
        self,
        env,
        params,
        level_weight: float = 1.0,
        pressure_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(env, params, **kwargs)
        self.level_weight = level_weight
        self.pressure_weight = pressure_weight

    def _rollout(self, actions: jnp.ndarray, state) -> jnp.ndarray:
        key = jax.random.PRNGKey(0)
        pr = self.params

        def step_fn(carry, u):
            s = carry
            _, new_s, _, _, _ = self.env.step_env(key, s, self._env_action(u), pr)
            level_err = new_s.level / pr.level_band
            press_err = (new_s.pressure - new_s.target_pressure) / pr.pressure_band
            fuel = new_s.Q_fuel / pr.Q_max
            cost = (
                self.level_weight * level_err**2
                + self.pressure_weight * press_err**2
                + pr.fuel_weight * fuel
            )
            return new_s, -cost

        _, rewards = jax.lax.scan(step_fn, state, actions)
        return jnp.sum(rewards)


def make_boiler_drum_mpc(
    env, params, horizon: int = 30, n_iter: int = 40, lr: float = 0.05
):
    """Gradient MPC for the drum boiler.

    Gradient-based rather than CasADi: the plant is already differentiable JAX
    and its steam-property fits and voidage algebra would have to be duplicated
    symbolically for no gain. Optimises firing and feedwater jointly, which
    matters because they are coupled through pressure -- firing harder raises
    pressure, which collapses bubbles and *lowers* the level.
    """
    return BoilerDrumGradientMPC(
        env,
        params,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
    )


class SamplingMPC:
    """Cross-entropy-method MPC — a gradient-free shooting controller.

    Exists because some plants are not differentiable in practice even when
    they are differentiable in principle. On the cement kiln the *forward*
    rollout is perfectly well behaved, but the adjoint is not: free lime
    depends on temperature through an Arrhenius term with a 280 kJ/mol
    activation energy, that temperature is itself advected down the kiln, and
    the resulting tangent system grows by roughly two orders of magnitude per
    step. Reverse-mode gradients overflow to NaN after about eight steps --
    measured at 1.6e-3 over five steps and 8.3e3 over eight -- while finite
    differences on the same objective stay clean.

    So this samples instead of differentiating: draw action sequences, roll
    them out, keep the elite fraction, refit, repeat. Everything is vmapped and
    jitted, so the cost is dominated by forward rollouts, which are cheap.
    """

    def __init__(
        self,
        env,
        params,
        objective,
        action_dim: int = 1,
        action_lb: float = -1.0,
        action_ub: float = 1.0,
        horizon: int = 30,
        n_samples: int = 96,
        n_elite: int = 12,
        n_iter: int = 4,
        init_std: float = 0.5,
        min_std: float = 0.05,
        alpha: float = 0.4,
        seed: int = 0,
    ):
        self.env = env
        self.params = params
        self.objective = objective
        self.action_dim = action_dim
        self.action_lb, self.action_ub = float(action_lb), float(action_ub)
        self.horizon = horizon
        self.n_samples, self.n_elite, self.n_iter = n_samples, n_elite, n_iter
        self.init_std, self.min_std, self.alpha = init_std, min_std, alpha
        self._key = jax.random.PRNGKey(seed)
        self.reset()
        self._jit_optimize = jax.jit(self._optimize)

    def _env_action(self, u):
        return u[0] if self.action_dim == 1 else u

    def _score(self, actions, state):
        """Total objective for one action sequence."""
        key = jax.random.PRNGKey(0)

        def step_fn(carry, u):
            _, new_s, _, _, _ = self.env.step_env(
                key, carry, self._env_action(u), self.params
            )
            return new_s, self.objective(new_s, self.params)

        _, rewards = jax.lax.scan(step_fn, state, actions)
        return jnp.sum(rewards)

    def _optimize(self, mean, std, state, key):
        batch_score = jax.vmap(self._score, in_axes=(0, None))

        def body(carry, _):
            mean, std, key = carry
            key, sub = jax.random.split(key)
            noise = jax.random.normal(
                sub, (self.n_samples, self.horizon, self.action_dim)
            )
            samples = jnp.clip(
                mean[None] + std[None] * noise, self.action_lb, self.action_ub
            )
            scores = batch_score(samples, state)
            scores = jnp.where(jnp.isnan(scores), -jnp.inf, scores)
            elite_idx = jnp.argsort(scores)[-self.n_elite :]
            elite = samples[elite_idx]
            new_mean = elite.mean(axis=0)
            new_std = jnp.maximum(elite.std(axis=0), self.min_std)
            mean = self.alpha * mean + (1.0 - self.alpha) * new_mean
            std = self.alpha * std + (1.0 - self.alpha) * new_std
            return (mean, std, key), None

        (mean, std, _), _ = jax.lax.scan(
            body, (mean, std, key), None, length=self.n_iter
        )
        return mean, std

    def step(self, _obs, state):
        """Return the next action. ``_obs`` is ignored (kept for API symmetry)."""
        self._key, sub = jax.random.split(self._key)
        self._mean, self._std = self._jit_optimize(self._mean, self._std, state, sub)
        first = np.array(self._mean[0])
        # Shift the plan forward one step for the next solve.
        self._mean = jnp.concatenate([self._mean[1:], self._mean[-1:]], axis=0)
        self._std = jnp.full_like(self._std, self.init_std)
        return float(first[0]) if self.action_dim == 1 else first

    def reset(self):
        self._mean = jnp.zeros((self.horizon, self.action_dim))
        self._std = jnp.full((self.horizon, self.action_dim), self.init_std)


def _cement_kiln_objective(state, params):
    """Quadratic in the free-lime error, sharing the reward's minimiser.

    The environment's reward clips flat once free lime is more than
    ``lime_band`` from target -- exactly the situation the controller is called
    on to fix -- so a quadratic that stays informative far from target is what
    the optimiser needs.
    """
    err = (state.lime[-1] - state.target_lime) / params.lime_band
    fuel = (state.fuel - params.fuel_min) / (params.fuel_max - params.fuel_min)
    return -(err**2 + 0.02 * fuel)


def make_cement_kiln_mpc(
    env, params, horizon: int = 40, n_samples: int = 96, n_iter: int = 4, **kwargs
):
    """Sampling (CEM) MPC for the rotary kiln.

    Gradient-free by necessity, not preference -- see ``SamplingMPC`` for the
    measured reason.

    A 40-step horizon is 20 minutes at dt = 30 s, most of the ~25 minute
    transport delay. That is the point: a controller whose horizon is shorter
    than the delay is choosing fuel whose consequences it cannot see.
    """
    return SamplingMPC(
        env,
        params,
        objective=_cement_kiln_objective,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_samples=n_samples,
        n_iter=n_iter,
        **kwargs,
    )


def _battery_objective(state, params):
    """Smooth stand-in for the battery's reward, with the same minimiser.

    The environment scores dispatch tracking as ``clip(1 - err/band, 0, 1)**2``,
    the same clipped form the wind turbine uses, and it fails the same way: once
    the power error leaves the band the tracking term is flat, and the only
    gradient left belongs to the degradation and state-of-charge terms, which
    both pull toward doing nothing. Measured over ten seeds the controller
    returned 95 against the PID's 157, ahead on 1 seed.

    Replacing the clipped term with a plain quadratic in the normalised error
    keeps the minimiser and restores a gradient that grows with the error:
    164 against 157 on the mean.

    That mean is worth reading carefully. It is carried by one seed where
    lookahead pays enormously (358 against 165); on the other nine the MPC is
    still behind by 5 to 24, for a median of -11. So this is a large improvement
    and *not* an upper bound, and horizon, iterations and step size were all
    swept without closing the remainder.
    """
    from target_gym.energy.battery.env import degradation_rate

    err = (state.target_power - state.power) / params.power_band
    fade = degradation_rate(state.current, state.T_cell, params) * params.delta_t
    headroom = (state.soc - 0.5) ** 2
    # Offset so a healthy step scores ~1, matching ``done_value`` = 0.
    return (
        1.0
        - err**2
        - params.degradation_weight * fade
        - params.soc_comfort_weight * headroom
    )


def make_battery_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 40,
    lr: float = 0.08,
    objective_fn=_battery_objective,
):
    """Gradient MPC for the grid battery.

    Horizon matters more here than in most environments: the battery has a
    *finite energy budget*, so the value of discharging now depends on what the
    dispatch is likely to ask for later. 30 steps is 2.5 min at dt = 5 s --
    long enough to see the state-of-charge limits coming, which is exactly what
    a reactive controller cannot do.
    """
    return GradientMPC(
        env,
        params,
        action_dim=1,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
        objective_fn=objective_fn,
    )


_WT_BARRIER_ONSET = 0.85  # fraction of the trip speed at which the barrier starts
_WT_BARRIER_WEIGHT = 10.0


def _wind_turbine_objective(state, params):
    """Smooth stand-in for the turbine's reward, with the same minimiser.

    The environment scores power tracking as ``clip(1 - err/band, 0, 1)**2``
    minus a pitch-activity penalty. That is a fine thing to be scored on and a
    useless thing to descend: one step off the operating point puts the error at
    nearly four times the band, where the tracking term is clipped flat and the
    only surviving gradient belongs to the *penalty*. The optimiser is then
    correctly guided to stop moving the pitch, and the controller returns ~0 for
    the rest of the episode -- which is what it did, at every horizon tried.

    Replacing the clipped term with a plain quadratic in the normalised error
    keeps the minimiser (zero error, no activity) and restores a gradient that
    grows with the error instead of vanishing.
    """
    from target_gym.energy.wind_turbine.env import electrical_power, omega_rated

    power = electrical_power(state.omega, state.torque, params)
    err = (state.target_power - power) / params.power_band
    activity = jnp.abs(state.pitch_cmd - state.pitch) / params.pitch_max

    # Soft barrier on the rotor-speed trip. The environment terminates outside
    # [underspeed, overspeed] x rated, and a *hard* stop is invisible to a
    # gradient planner: ``done`` is a boolean, so masking the reward after it
    # tells the optimiser what a trip costs while giving it no derivative
    # pointing away from one. Measured, that is exactly what happened -- the
    # planned pitch command stayed at 0.000 while the predicted rotor speed ran
    # past the trip, on every horizon from 60 to 200.
    #
    # A differentiable penalty that switches on before the boundary does give a
    # gradient, and it is what makes this controller stable: over twelve seeds
    # the worst episode goes from 22 to 307 and the spread from sd 152 to 24.
    # The onset matters (0.85 beats 0.80); the weight barely does (10, 30 and
    # 100 land within 0.4 of each other), which is the signature of a term that
    # is shaping the approach rather than trading against the objective.
    w_rated = omega_rated(params)
    over = state.omega / (params.overspeed_factor * w_rated)
    under = (params.underspeed_factor * w_rated) / jnp.maximum(state.omega, 1e-6)
    barrier = (
        jnp.maximum(over - _WT_BARRIER_ONSET, 0.0) ** 2
        + jnp.maximum(under - _WT_BARRIER_ONSET, 0.0) ** 2
    )

    # Offset so a healthy step scores ~1 and a terminated one scores
    # ``done_value`` = 0, making an early trip cost the rest of the horizon.
    return (
        1.0
        - err**2
        - params.pitch_activity_weight * activity
        - _WT_BARRIER_WEIGHT * barrier
    )


def make_wind_turbine_mpc(
    env,
    params,
    horizon: int = 60,
    n_iter: int = 100,
    lr: float = 0.02,
    n_tail: int = 0,
    objective_fn=_wind_turbine_objective,
):
    """Gradient MPC for the NREL 5 MW turbine.

    Gradient-based for the same reason as the aircraft and the column: the
    plant is already differentiable JAX, and the Cp surface is an empirical
    fit that would gain nothing from symbolic re-expression. Optimising pitch
    and torque jointly is the point -- the rate-limited pitch actuator means
    the useful move is often to start pitching *before* the rotor has
    accelerated, which a reactive loop cannot do.

    Two changes make it actually control. The objective is the smooth surrogate
    above rather than the environment's clipped reward, without which the
    controller scored a return of -0.02 against the PID's 393 at every horizon
    tried -- identical to two decimals at 20, 40 and 60, which is the signature
    of an optimiser that is not moving. The horizon is then 60 rather than 20,
    which only matters once the gradient is informative. Measured over 400-step
    episodes the return goes from -0.02 to 387, with no terminations.

    Scored over twelve seeds this reaches 385 against the PID's 392 -- 98%, and
    ahead on 7 of the 12 -- so it is on par with the PID rather than an upper
    bound over it. The remaining gap is one seed that still drops to ~307.

    Two things that look like explanations and are not, both measured: the wind
    forecast (the MPC plans with a fixed key while the episode uses its own, and
    the seed where they disagree scored *higher*), and the inner optimiser
    (Adam, and a decaying step size, are both worse here than the plain one).
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
        n_tail=n_tail,
        objective_fn=objective_fn,
    )


def make_distillation_mpc(
    env, params, horizon: int = 15, n_iter: int = 40, lr: float = 0.08
):
    """Gradient MPC for the distillation column.

    Gradient-based rather than CasADi: the plant is 41 coupled stage balances
    that are already differentiable JAX, and re-expressing them symbolically
    would duplicate the whole model for no gain -- the same rationale as the
    aircraft. Optimises [L_raw, V_raw] jointly, which is the point on an
    ill-conditioned plant: the useful move is a *coordinated* change in reflux
    and boilup, exactly what independent diagonal loops cannot make.
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


def make_ph_mpc(env, params, horizon: int = 20):
    """CasADi/IPOPT MPC for the pH neutralisation CSTR.

    With delta_t = 5 s, horizon = 20 gives 100 s of lookahead -- slightly more
    than one residence time (V/q_total ~ 88 s), so the controller can see a
    change work through the tank.
    """
    return PHCasadiMPC(env, params, horizon=horizon)


def make_hvac_mpc(env, params, horizon: int = 24):
    """CasADi/IPOPT MPC for the single-zone building.

    With delta_t = 900 s, horizon = 24 gives 6 h of lookahead -- enough to see
    the morning setback recovery and the solar peak, which is where the
    anticipation advantage over PID comes from.
    """
    return HVACCasadiMPC(env, params, horizon=horizon)


def make_glass_furnace_mpc(env, params, horizon: int = 60):
    """CasADi/IPOPT MPC for the GlassFurnace (3-zone lumped thermal model).

    With delta_t=30 s, horizon=60 gives 30 min lookahead.  The crown thermal
    time constant is ~15 min, so 2×τ of lookahead is enough to see the next
    scheduled setpoint change and pre-cool / pre-heat accordingly (which PID
    cannot do — that's the whole point of the schedule).
    """
    return GlassFurnaceCasadiMPC(env, params, horizon=horizon)
