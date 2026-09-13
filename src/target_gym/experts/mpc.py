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


def plan_params(spec, params):
    """The params a planner should use for its own internal model.

    Identical to *params* except that anything named in ``spec.noise_fields``
    is zeroed, so the planner predicts the **mean** disturbance rather than one
    invented realisation of it. That is certainty equivalence, and it is the
    standard treatment for additive zero-mean noise.

    Without it the planners rolled the true environment forward under a
    hardcoded ``jax.random.PRNGKey(0)`` while ``rollout`` drives the plant with
    ``PRNGKey(seed)``. On **seed 0 those coincide**, so the planner's simulated
    disturbance was the plant's actual disturbance and the MPC had perfect
    foresight for one seed in ten. On the battery, whose tracked target *was*
    the noise, that was worth 350.4 against 151.8: a median tracking error of
    22 W where the honest figure is 60 630 W. It inflated seed 0 of every
    environment with a gradient or sampling planner, and it inflated the
    published mean of all of them.
    """
    fields = getattr(spec, "noise_fields", ())
    if not fields:
        return params
    return params.replace(**{f: 0.0 for f in fields})


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
        # One jitted entry point, deliberately. A second one for a larger
        # first-solve budget was tried and removed: the batched path never
        # called it, and two jitted functions with the same body compile
        # twice -- about 50 s each for the patrol planner, paid once per
        # process on the per-seed route.
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

    # Iterates are held this far inside the action bounds, as a fraction of the
    # half-range. See ``_optimize`` for why sitting exactly on a bound is fatal.
    _BOUND_MARGIN = 1e-3

    def _descend(self, actions_init: jnp.ndarray, state, n_iter: int) -> jnp.ndarray:
        """Projected gradient descent, kept strictly inside the action bounds.

        The margin is the whole point, and it is not cosmetic. These plants
        saturate: an engine cannot produce less than zero thrust, a valve
        cannot open past fully open. Saturation is written with ``clip`` or
        ``maximum``, and at exactly the kink those hand back a derivative of
        zero, which is a valid subgradient and the wrong one for an optimiser.
        Projecting onto the closed interval parks an action precisely on the
        bound whenever a step overshoots, and the bound is then *absorbing*:
        the derivative there is exactly zero, so gradient descent can never
        move that action again, however it is scaled or preconditioned.

        Measured on ``plane_steps`` seed 2, at the plan where the aircraft
        gives up and glides into the ground with the setpoint 2000 m above it::

            thrust  autodiff d(obj)/d(thrust)   finite difference   objective
            -1.00           0.0000                   +2.998           23.989
            -0.99          +2.9998                   +3.003           24.019
            -0.95          +3.0266                   +3.030           24.140
            -0.50          +3.5376                   +3.545           25.597

        The derivative is right everywhere except on the bound, where the true
        one-sided slope is +3.0 and autodiff returns 0. Thrust sat at exactly
        -1.000 for 800 steps while the elevator went on being optimised
        normally, which is why the controller looked alive the whole time: a
        planner that has stopped searching still emits finite, in-bounds
        actions. Holding the iterates 1e-3 inside the bounds takes the episode
        from terminating at t=1732 to flying all 2400 steps, and the return
        from 1013.6 to 2109.9 against the PID's 1717.6.

        This is the interior-point principle in miniature, and it is the reason
        real NLP solvers keep their iterates off the bounds. Two alternatives
        were measured and are worse. Normalising the gradient per actuator
        rather than over the whole sequence crashes earlier, at t=971, because
        it lets a noisy actuator take a full-size step every iteration. A
        finite-difference line search over a constant offset per actuator does
        work, since it never consults the derivative, but it scores below this
        on both seeds tried and costs ``action_dim * 7`` extra rollouts a step.

        Twelve of the twenty environments with an MPC use this optimiser. The
        four plants among them (battery, boiler drum, distillation, wind
        turbine) were re-recorded after the change and moved by under half a
        point of return, distillation not at all, so the defect was latent for
        them and real only for the aircraft.

        The NaN scrub below is a second instance of the same family, and is
        left as it is: this project has a documented, unlocalised reverse-mode
        NaN in the aircraft dynamics (see ``NAN_TUNERS`` in
        ``tests/experts/test_pid_tuning.py``).
        """
        cost_grad = jax.grad(lambda a: -self._rollout(a, state))
        margin = self._BOUND_MARGIN * 0.5 * (self.action_ub - self.action_lb)
        lb, ub, lr = self.action_lb + margin, self.action_ub - margin, self.lr

        def body(_, actions):
            g = cost_grad(actions)
            # Replace NaN gradients with zero (can arise from numerically
            # unstable rollouts, e.g. near-stall flight dynamics)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            # Clip gradient norm
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-8)
            g = jnp.where(g_norm > 1.0, g / g_norm, g)
            return jnp.clip(actions - lr * g, lb, ub)

        return jax.lax.fori_loop(0, n_iter, body, jnp.clip(actions_init, lb, ub))

    def _optimize(self, actions_init: jnp.ndarray, state) -> jnp.ndarray:
        """Refine a warm-started plan. Two arguments, so it vmaps as it stands."""
        return self._descend(actions_init, state, self.n_iter)

    def solver_report(self) -> dict:
        """No external solver, so no convergence to report.

        Part of the planner interface rather than a special case at the call
        site: the recorder asks every controller for its solver health, and a
        planner that has none should say so rather than raise. Leaving it off
        cost a 44-minute record that crashed on the one sampling planner in the
        suite after every other environment had already finished.
        """
        return {}

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


# IPOPT defaults to 3000 iterations and no time limit. In a receding-horizon
# loop that is not a safety net, it is a hang: one badly conditioned step can
# run for half an hour while its neighbours take a tenth of a second, and the
# episode never finishes. A real MPC has a sample period and returns the best
# iterate it holds when the clock runs out, so ours does the same.
#
# The iteration cap is the one meant to bind. It is deterministic, so a
# baseline recorded on one machine reproduces on another -- which a wall-clock
# cap would not be, since a slower machine would record a different return.
# ``max_cpu_time`` is only a backstop against a solve that is pathological
# rather than merely hard, and sits far above anything a healthy step needs.
IPOPT_MAX_ITER = 150
IPOPT_MAX_CPU_TIME = 60.0

# IPOPT return codes that mean "I stopped because you told me to", as opposed
# to a genuine numerical failure. Both count as non-convergence; separating
# them says whether the cap is doing the work or the problem is broken.
_IPOPT_CAP_STATUSES = frozenset(
    {"Maximum_Iterations_Exceeded", "Maximum_CpuTime_Exceeded"}
)


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
        # Solver health, accumulated over every solve this controller performs.
        # ``reset`` deliberately leaves these alone so one counter covers a
        # whole rollout rather than the last episode of it.
        self.solve_calls = 0
        self.solve_iters = 0
        self.solve_failures = 0
        self.solve_capped = 0
        self.last_return_status = ""
        self._last_u = None
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
        guess = self._save_guess()
        u = np.array(self._mpc.make_step(x0)).flatten()
        if not self._record_solve():
            u = self._fallback(guess, u)
        self._last_u = u
        u_clipped = np.clip(u, -1.0, 1.0)
        return float(u_clipped[0]) if len(u_clipped) == 1 else u_clipped

    def reset(self):
        """Reset so that the next step re-initialises the warm-start."""
        self._initialized = False
        self._last_u = None

    # ------------------------------------------------------------------
    # Shared do_mpc boilerplate
    # ------------------------------------------------------------------

    @staticmethod
    def _quiet_ipopt():
        return {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": IPOPT_MAX_ITER,
            "ipopt.max_cpu_time": IPOPT_MAX_CPU_TIME,
        }

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    #: Typical magnitude of each optimisation variable, by do-mpc kind.
    #: Subclasses override; anything not named is left at 1.0.
    SCALING: dict = {}

    def _apply_scaling(self, mpc) -> None:
        """Tell the solver what a unit is, before ``setup`` freezes the NLP.

        IPOPT auto-scales the objective and the constraints, but not the
        decision variables: step norms, bound handling and the warm start all
        run in whatever units the model happens to use. Left alone, the reactor
        hands it a vector spanning ``rho_ext`` around 0.0016 up to a precursor
        concentration around 377 -- a factor of 605 000, measured over a PID
        episode -- and puts hard bounds on the smallest entry of it. That is a
        badly conditioned KKT system built out of nothing but unit choices, and
        it shows up as iteration counts, which is what makes a seed take twenty
        times its siblings.

        The numbers below are means of ``|x|`` over a PID episode, rounded to
        one figure. They only have to be the right order of magnitude.
        """
        for kind, entries in self.SCALING.items():
            for var, value in entries.items():
                mpc.scaling[kind, var] = float(value)

    # ------------------------------------------------------------------
    # Solver health
    # ------------------------------------------------------------------

    def _record_solve(self) -> bool:
        """Fold the last solve's outcome into the running counters.

        do-mpc neither raises nor warns when IPOPT gives up: it stores the
        failed iterate, hands it back as the action, and warm-starts the next
        step from it. Nothing downstream can tell that apart from a converged
        solve, so an MPC baseline can quietly stop being an upper bound. These
        counters are what ``solver_report`` publishes alongside the return.
        """
        stats = getattr(self._mpc, "solver_stats", None) or {}
        self.solve_calls += 1
        self.solve_iters += int(stats.get("iter_count", 0) or 0)
        status = str(stats.get("return_status", ""))
        capped = status in _IPOPT_CAP_STATUSES
        if capped:
            self.solve_capped += 1
        if stats.get("success", True):
            return True
        self.solve_failures += 1
        self.last_return_status = status
        # A capped solve is still a usable answer: IPOPT was converging and we
        # stopped it, which is the whole point of the cap. A solve that failed
        # for any other reason -- infeasible, restoration failed, invalid
        # number -- returns an iterate that means nothing, and do-mpc will warm
        # start the next step from it and spread the damage.
        return capped

    def _save_guess(self):
        """Snapshot the warm start, so a failed solve cannot poison the next.

        The multipliers only exist once do-mpc has solved at least once, so
        they are read defensively rather than assumed.
        """
        m = self._mpc
        return {
            k: np.array(v)
            for k, v in (
                ("opt_x", m.opt_x_num.master),
                ("lam_g", getattr(m, "lam_g_num", None)),
                ("lam_x", getattr(m, "lam_x_num", None)),
            )
            if v is not None
        }

    def _fallback(self, guess, u):
        """Restore the last good warm start and hold the last good action."""
        m = self._mpc
        m.opt_x_num.master = guess["opt_x"]
        for attr, key in (("lam_g_num", "lam_g"), ("lam_x_num", "lam_x")):
            if key in guess:
                setattr(m, attr, guess[key])
        return u if self._last_u is None else self._last_u

    def solver_report(self) -> dict:
        """Convergence summary for the solves performed so far."""
        calls = max(self.solve_calls, 1)
        return {
            "solver_calls": self.solve_calls,
            "solver_failures": self.solve_failures,
            "solver_capped": self.solve_capped,
            "solver_mean_iters": round(self.solve_iters / calls, 1),
            "solver_last_status": self.last_return_status,
        }


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

    SCALING = {"_x": {"C_a": 1.0, "T": 300.0}}

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
        self._apply_scaling(mpc)
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
        self._apply_scaling(mpc)
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

    SCALING = {"_x": {"h1": 0.25, "h2": 0.25, "h3": 0.25, "h4": 0.25}}

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
        # Both termination bounds, and soft.
        #
        # The plant does not clip these levels, it *ends the episode* when any
        # of them reaches h_min or h_max. Only the lower bound was here, and it
        # was hard, which is backwards on both counts. Hard was wrong because a
        # hard bound the plant can walk the initial state onto makes the NLP
        # infeasible at x0, and IPOPT answers that with a restoration phase and
        # hundreds of iterations rather than an action. Missing h_max was worse:
        # the controller was blind to half of a termination condition it is
        # scored on, so it had no reason not to overflow a tank.
        #
        # Input bounds stay hard, because the optimiser owns those and can
        # always satisfy them. State bounds get slacks, which is the usual
        # division of labour.
        for h in (h1, h2, h3, h4):
            mpc.set_nl_cons(
                f"{h.name()}_min",
                -h,
                ub=-float(p.h_min),
                soft_constraint=True,
                penalty_term_cons=1e3,
            )
            mpc.set_nl_cons(
                f"{h.name()}_max",
                h,
                ub=float(p.h_max),
                soft_constraint=True,
                penalty_term_cons=1e3,
            )

        self._target_h1 = float(p.target_h1_range[0])
        self._target_h2 = float(p.target_h2_range[0])
        p_tpl = mpc.get_p_template(1)

        def p_fun(_t):
            p_tpl["_p", 0, "target_h1"] = self._target_h1
            p_tpl["_p", 0, "target_h2"] = self._target_h2
            return p_tpl

        mpc.set_p_fun(p_fun)
        self._apply_scaling(mpc)
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


# Integral gain and clamp for the furnace's offset-free correction, in kelvin.
# The gain is deliberately slow against a 3960 s open-loop time constant: this
# has to remove a standing offset over hundreds of steps, not chase noise.
_FURNACE_BIAS_GAIN = 0.05
_FURNACE_BIAS_LIMIT = 40.0
_FURNACE_BIAS_RESET = True


#: Checker nodes per chamber in the *controller's* regenerator model. The plant
#: carries four; the controller carries two, and ``_extract_x0`` averages the
#: plant's nodes down in pairs.
#:
#: This is the one knob that decides what the furnace baseline costs to record.
#: The regenerator is two thirds of the controller's state vector, and IPOPT's
#: cost grows superlinearly in problem size: at the plant's four nodes the NLP
#: has 3168 variables and a bad seed ran 5.5 s per step, which is hours per
#: episode against a plant that steps in 92 microseconds.
#:
#: Two nodes rather than one averaged stack. The single stack was what this
#: model carried before, and it left a 2-6 K standing offset because everything
#: downstream of the checker temperatures is nonlinear in them; two nodes keep
#: the hot/cold split that sets air preheat, which is where that error came
#: from, at half the states.
_FURNACE_MPC_REGEN_NODES = 2


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

    SCALING = {
        "_x": {
            "T_crown": 1000.0,
            "T_melt": 1000.0,
            "T_work": 1000.0,
            "m_batch": 10000.0,
            **{f"T_rA{i}": 1000.0 for i in range(_FURNACE_MPC_REGEN_NODES)},
            **{f"T_rB{i}": 1000.0 for i in range(_FURNACE_MPC_REGEN_NODES)},
        },
        "_z": {"T_gas": 2000.0},
    }

    def _build_mpc(self):
        p = self.params
        from target_gym.glass_furnace.env import (
            FUEL_DEAD_TIME_STEPS,
            N_SETPOINTS,
            firing_fraction,
        )

        if not hasattr(self, "_pipeline"):
            nominal = 0.5 * (p.fuel_min + p.fuel_max)
            self._pipeline = np.full(FUEL_DEAD_TIME_STEPS, nominal)

        model = do_mpc.model.Model("continuous")
        SB = 5.670374419e-8
        K = 273.15

        T_crown = model.set_variable("_x", "T_crown")
        T_melt = model.set_variable("_x", "T_melt")
        T_work = model.set_variable("_x", "T_work")
        m_batch = model.set_variable("_x", "m_batch")
        # Both regenerator chambers, at the plant's own node count. The MPC is
        # presented as an upper bound, so it is entitled to the plant's model as
        # well as its state -- it already reads the true state in _extract_x0.
        # The previous version collapsed these two alternating four-node
        # chambers onto one three-node stack "at the cycle average", which is
        # exact only if everything downstream is linear in these temperatures.
        # It is not: measured by check 13 of the model review checklist, the
        # plant ran +0.0275 K per control interval hotter than that model,
        # one-signed on 71% of settled steps, and multiplied by the crown's
        # 132-step time constant that is the 2-6 K standing offset which put
        # this MPC 16% behind its own PID.
        from target_gym.glass_furnace.env import N_REGEN_NODES

        n_regen = _FURNACE_MPC_REGEN_NODES
        T_rA = [model.set_variable("_x", f"T_rA{i}") for i in range(n_regen)]
        T_rB = [model.set_variable("_x", f"T_rB{i}") for i in range(n_regen)]
        T_gas = model.set_variable("_z", "T_gas")  # algebraic: quasi-steady flame
        u_raw = model.set_variable("_u", "u_raw")
        model.set_variable("_tvp", "target_T_crown")
        # Firing actually released, as a fraction of commanded: 1 away from a
        # reversal, near zero while the valves change over. Deterministic in
        # time and therefore known to the controller, which is the point of
        # modelling it -- the dip is a periodic upset a predictive controller
        # can plan through and a PID can only react to.
        firing = model.set_variable("_tvp", "firing")
        # Fuel already in the pipeline. The plant applies what was commanded
        # FUEL_DEAD_TIME_STEPS ago, so the first intervals of any plan are
        # already decided and nothing the optimiser does can change them.
        # ``commit`` is 1 over those intervals and 0 after.
        u_committed = model.set_variable("_tvp", "u_committed")
        commit = model.set_variable("_tvp", "commit")
        # The reversal is a deterministic function of time, so the oracle knows
        # it exactly rather than averaging it away. 0 -> A preheats air.
        a_is_air = model.set_variable("_tvp", "a_is_air")

        m_fuel_free = p.fuel_min + 0.5 * (u_raw + 1.0) * (p.fuel_max - p.fuel_min)
        m_fuel = firing * (commit * u_committed + (1.0 - commit) * m_fuel_free)
        m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
        m_gas = m_fuel + m_air

        eps = p.eps_regen_node

        def _duties(nodes):
            """Exhaust and air duties for one chamber, mirroring the plant.

            Exhaust enters at the hot end and works down; air enters at the cold
            end and works up. Each node exchanges with the stream passing it at
            per-node effectiveness ``eps_regen_node``.
            """
            t_in = T_gas
            q_exh = []
            for node in nodes:  # hot end first
                t_out = t_in - eps * (t_in - node)
                q_exh.append(m_gas * p.c_p_gas * (t_in - t_out))
                t_in = t_out
            t_stack = t_in

            t_in = p.T_ambient
            q_air_rev = []
            for node in reversed(nodes):  # cold end first
                t_out = t_in + eps * (node - t_in)
                q_air_rev.append(-m_air * p.c_p_air * (t_out - t_in))
                t_in = t_out
            return q_exh, list(reversed(q_air_rev)), t_in, t_stack  # noqa: E501

        qA_exh, qA_air, TA_air_out, _ = _duties(T_rA)
        qB_exh, qB_air, TB_air_out, _ = _duties(T_rB)

        # A chamber does one duty at a time, never both at half rate.
        QA = [a_is_air * qa + (1.0 - a_is_air) * qe for qa, qe in zip(qA_air, qA_exh)]
        QB = [(1.0 - a_is_air) * qb + a_is_air * qe for qb, qe in zip(qB_air, qB_exh)]
        T_air = a_is_air * TA_air_out + (1.0 - a_is_air) * TB_air_out
        UA_node = p.U_regen * p.A_regen / n_regen
        # The checker stack's total heat capacity is fixed by the plant; the
        # controller just divides it into fewer, larger nodes.
        C_regen_node = p.C_regen_node * (N_REGEN_NODES / n_regen)

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
        for i in range(n_regen):
            model.set_rhs(
                f"T_rA{i}",
                (QA[i] - UA_node * (T_rA[i] - p.T_ambient)) / C_regen_node,
            )
            model.set_rhs(
                f"T_rB{i}",
                (QB[i] - UA_node * (T_rB[i] - p.T_ambient)) / C_regen_node,
            )
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
        # envelope, six times flatter than the band the reward discriminates
        # over. Against an unchanged fuel penalty the controller duly sold
        # tracking for fuel, and lost to the PID on 7 of 10 seeds.
        #
        # The old form was also non-monotonic: ``((scale - err)/scale)**2`` turns
        # back upward past ``err = scale``, so beyond twice it the objective
        # preferred *more* error. A plain squared normalised error is monotone,
        # smooth, and minimised in the same place as the reward.
        #
        # ``tracking_band``, the same field name the four-tank, the
        # distillation column and the pH loop carry: the error at which
        # tracking is bad for this plant. It read ``params.tracking_scale``,
        # 40 K, left over from when the reward was ``clip(1 - err/40, 0, 1)**2``
        # -- the reward has been log-scaled for a while and that field was dead.
        # 40 K against errors of about 1 K put the tracking term at 6e-4 while
        # the fuel penalty stayed O(1), so the objective was nearly flat in the
        # direction being scored, which is both why this MPC trails its own PID
        # and why IPOPT struggles on it.
        scale = float(p.tracking_band)
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
        self._bias = 0.0
        self._bias_slot = -1
        # Overridden by make_glass_furnace_mpc; defaults here so a directly
        # constructed instance still behaves.
        self._bias_gain = _FURNACE_BIAS_GAIN
        self._bias_reset = _FURNACE_BIAS_RESET
        self._max_steps = int(p.max_steps_in_episode)
        self._n_setpoints = int(N_SETPOINTS)
        p_rev = float(p.reversal_period)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for k in range(self.horizon + 1):
                future = self._current_step + k
                slot = min(
                    (future * self._n_setpoints) // self._max_steps,
                    self._n_setpoints - 1,
                )
                tvp_tpl["_tvp", k, "target_T_crown"] = float(
                    self._target_schedule[slot] + self._bias
                )
                # The reversal is deterministic in time, so the oracle supplies
                # its exact phase across the whole horizon rather than averaging
                # it away: 1 while chamber A preheats the air, 0 while B does.
                cycles = (future * self.mpc_dt) / p_rev
                tvp_tpl["_tvp", k, "a_is_air"] = float(1.0 - (np.floor(cycles) % 2.0))
                tvp_tpl["_tvp", k, "firing"] = float(firing_fraction(future, p, xp=np))
                # The first FUEL_DEAD_TIME_STEPS intervals burn what is already
                # in the pipeline. Beyond that the optimiser decides.
                committed = k < len(self._pipeline)
                tvp_tpl["_tvp", k, "commit"] = 1.0 if committed else 0.0
                tvp_tpl["_tvp", k, "u_committed"] = (
                    float(self._pipeline[k]) if committed else 0.0
                )
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    @staticmethod
    def _coarsen(nodes) -> np.ndarray:
        """The plant's checker nodes averaged onto the controller's, hot end first.

        Both chambers are kept, because they alternate duty and averaging them
        together is what left a standing offset. What is coarsened is the depth
        resolution within a chamber, which is a smooth profile down the stack:
        contiguous groups average to the temperature a node of that size would
        hold, and the group heat capacity is scaled to match in ``_build_mpc``.
        """
        x = np.asarray(nodes, dtype=float)
        n = _FURNACE_MPC_REGEN_NODES
        if x.size == n:
            return x
        return x.reshape(n, x.size // n).mean(axis=1)

    def _extract_x0(self, state):
        """The plant's regenerator state, both chambers, coarsened in depth.

        This once averaged the two chambers together *and* resampled four nodes
        onto three; that cost 2-6 K of standing offset, because everything
        downstream of the checker temperatures is nonlinear in them and the two
        chambers are never at the same temperature. Both chambers are kept now.
        The depth resolution is halved instead, which is what makes the NLP
        affordable: see ``_FURNACE_MPC_REGEN_NODES``.
        """
        return np.concatenate(
            [
                np.array(
                    [
                        float(state.T_crown),
                        float(state.T_melt),
                        float(state.T_work),
                        float(state.m_batch),
                    ]
                ),
                self._coarsen(state.T_rA),
                self._coarsen(state.T_rB),
            ]
        )

    def reset(self):
        """Clear the offset-free bias as well as the warm start.

        Without this the bias earned on one episode is carried into the next,
        where it is a standing setpoint error rather than a correction.
        """
        super().reset()
        self._bias = 0.0
        self._bias_slot = -1

    def _update_setpoint(self, state):
        self._target_schedule = np.asarray(state.target_schedule, dtype=float)
        self._current_step = int(state.time)
        # What the plant will burn over the next intervals regardless of what is
        # decided now. An upper-bound controller reads the true state, and this
        # is part of it.
        self._pipeline = np.asarray(state.fuel_pipeline, dtype=float)

        # Offset-free correction. ``_extract_x0`` collapses the plant's two
        # four-node regenerator chambers onto the model's three nodes by
        # averaging, which is a deliberate model reduction and therefore a
        # structural plant-model mismatch. A finite-horizon MPC with mismatch
        # settles with a steady-state offset; a PID's integrator does not, and
        # over a long episode that is the whole difference between them.
        #
        # Measured on a 1600-step episode before this existed: for the first
        # half of the episode the two are indistinguishable, both still
        # approaching, and from the sixth decile the PID converges to 0.0-0.5 K
        # of error while the MPC plateaus at 2-6 K. Per step that was 0.619
        # against the PID's 0.765 -- a 19% shortfall that the previous 240-step
        # episode was far too short to see, since it ended while both were still
        # on their way.
        #
        # The remedy is the textbook one: integrate the measured tracking error
        # into a bias and shift the setpoint the solver is given, which is the
        # disturbance model of offset-free MPC in its simplest form. The gain is
        # small relative to the plant's 3960 s time constant, and the bias is
        # clamped so a saturated actuator cannot wind it up.
        slot = min(
            (self._current_step * self._n_setpoints) // self._max_steps,
            self._n_setpoints - 1,
        )
        # The bias absorbs model *gain* error as well as a standing disturbance,
        # and gain error is specific to an operating point. Carrying it across a
        # setpoint change applies the previous target's correction to the new
        # one: measured, that put an 11.4 K excursion into the decile after a
        # schedule step, worse there than having no bias at all. So it is
        # dropped when the schedule moves, and re-earned.
        if self._bias_reset and slot != self._bias_slot:
            self._bias_slot = slot
            self._bias = 0.0
        self._bias_slot = slot
        error = float(self._target_schedule[slot]) - float(state.T_crown)
        self._bias = float(
            np.clip(
                self._bias + self._bias_gain * error,
                -_FURNACE_BIAS_LIMIT,
                _FURNACE_BIAS_LIMIT,
            )
        )


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

    SCALING = {
        "_x": {
            "n": 1.0,
            "C0": 100.0,
            "C1": 400.0,
            "C2": 100.0,
            "C3": 70.0,
            "C4": 5.0,
            "C5": 0.8,
            "T_fuel": 1000.0,
            "T_coolant": 600.0,
            "I_hat": 1.0,
            "Xe_hat": 1.0,
            "rho_ext": 0.002,
        }
    }

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
        self._apply_scaling(mpc)
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
        guess = self._save_guess()
        u = np.array(self._mpc.make_step(x0)).flatten()
        if not self._record_solve():
            u = self._fallback(guess, u)
        self._last_u = u
        rho_rate = float(u[0])

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

    SCALING = {"_x": {"T_mass": 20.0, "Q_emitter": 800.0}}

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
        self._apply_scaling(mpc)
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

    SCALING = {"_x": {"Wa": 3e-4, "Wb": 3e-4}, "_z": {"pH": 7.0}}

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
        self._apply_scaling(mpc)
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
        guess = self._save_guess()
        u = np.array(self._mpc.make_step(x0)).flatten()
        if not self._record_solve():
            u = self._fallback(guess, u)
        self._last_u = u
        return float(np.clip(u, -1.0, 1.0)[0])


# ============================================================================
# Factory functions
# ============================================================================


_PLANE_STALL_MARGIN = 1.3  # multiples of stall speed at which the barrier starts
_PLANE_BARRIER_WEIGHT = 10.0


def _plane_objective(state, params):
    """The aircraft's own reward, less a barrier on flying into the stall.

    The altitude reward scores one thing and the aircraft has two actuators, so
    the planner is free to buy altitude with airspeed. Over a 90 s window that
    is the *best* thing it can do when the target is thousands of metres away:
    a zoom climb converts kinetic energy to potential energy far faster than
    the engines can supply it. Measured on ``plane_steps`` seed 0, that is
    exactly what the plan did -- airspeed fell monotonically from 201 to 30 m/s
    over 90 s while the aircraft climbed 3300 m, touched the commanded altitude
    with nothing left, departed at 91 deg angle of attack and hit the ground at
    t=372. It happened on ``plane`` too; there the phugoid that followed
    happened to damp out, which is luck rather than control.

    Angle of attack is the wrong thing to fence, even though it is what
    actually stalls: through that whole manoeuvre it sat at 4-8 deg and only
    crossed 15 deg at t=90, one step from the departure and at the very end of
    the planning window. Airspeed decays through the entire climb, so a floor
    under it is a constraint the planner can see coming and descend.

    The floor is the stall speed at this mass and altitude rather than a fixed
    number, since both move: ``sqrt(2 m g / (rho S CL_max))`` is where the wing
    can no longer carry the weight, and the barrier switches on at 1.3 times
    it, the usual approach margin. Bounded in [0, 1] by construction, which is
    what lets ``done_value`` stay below the worst step the planner can plan, so
    flying into the ground cannot look better than flying slowly.

    Measured on ``plane_steps`` seed 0 over 400 s: return 304 against 83 for
    the unprotected planner, settled error 0.1 m, and airspeed held above
    128 m/s with a worst angle of attack of 7.7 deg. On ``plane``, 226 against
    189. Like the turbine's barrier, the constants barely matter -- weight 30
    scores 299 and a 1.15 margin 299 -- which is the signature of a term
    shaping the approach rather than trading against the objective.

    Two alternatives, both measured and both rejected. Doubling the tail to
    ``n_tail=120`` prices more of the aftermath and does fix seed 0 (274), but
    it costs two thirds again in compute and only moves the horizon at which
    the same trade becomes profitable. Making the planner hold airspeed
    outright, by planning against ``speed_weight=1.0``, flies beautifully for
    400 s (166, angle of attack 2.1 deg) and then crashes anyway at t=1260 on
    the full episode and at t=386 on seed 2: it changes which trim the planner
    settles into without ever putting a floor under the trade.
    """
    from target_gym.plane.env import compute_reward

    reward = compute_reward(state, params)
    speed = jnp.sqrt(state.x_dot**2 + state.z_dot**2)
    v_stall = jnp.sqrt(
        2.0
        * state.m
        * params.gravity
        / (state.rho * params.wings_surface * params.CL_max)
    )
    margin = speed / (_PLANE_STALL_MARGIN * v_stall)
    return reward - _PLANE_BARRIER_WEIGHT * jnp.maximum(1.0 - margin, 0.0) ** 2


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

    Two things make it fly. The objective carries a stall barrier, without
    which the planner trades all its airspeed for altitude on every
    acquisition and departs; see ``_plane_objective``, which is where that is
    measured. And ``n_tail=60``: optimising 30 s of flight and being charged
    for nothing beyond it, the plan climbed hard and left the aircraft outside
    the altitude envelope just past the horizon, settling 654x worse than the
    PID and crashing in one episode of two. Simulating 60 further seconds on
    the held action -- which for this aircraft is close to trim -- prices that
    ending into the objective. Measured over 600-step episodes, settled
    tracking error went from 2949 m to 0.083 m, with no terminations. Sixty is
    the knee: 120 is no better on a fixed setpoint and costs twice.

    The tail alone was not enough, which is why the barrier is here. It fixes
    the ending the plan can *see*; the zoom climb is an ending the plan likes,
    and no affordable horizon changes that.

    ``done_value`` sits below the worst step the objective can score, so a plan
    that reaches the ground is charged for the rest of the horizon rather than
    scoring the 0.0 that a barrier-free positive reward could rely on.
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
        done_value=-(_PLANE_BARRIER_WEIGHT + 1.0),
    )


def _done_value(params) -> float:
    """What a planner on the environment's own reward charges per step once
    the plant has terminated: 0 for the non-negative version-1 reward (the
    forgone reward is the penalty), the failure cost for version 2, whose
    healthy steps are negative."""
    if int(getattr(params, "reward_version", 2)) == 1:
        return 0.0
    return -float(getattr(params, "failure_cost", 0.0))


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

    The objective is the environment's own reward. Under the version-2
    reward, a cost, a healthy step is negative, so a plan that leaves the
    envelope must be charged the failure cost for the rest of the horizon or
    crashing would read as an improvement over flying on; ``done_value`` is
    set to it (version 1 is non-negative and keeps 0).
    """
    return GradientMPC(
        env,
        params,
        action_dim=3,
        action_lb=-1.0,
        action_ub=1.0,
        done_value=_done_value(params),
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
    )


#: Fraction of ``slot_tolerance`` at which the patrol surrogate puts its
#: curvature. See :func:`_patrol_objective`.
_PATROL_ERROR_SCALE = 0.25


def _patrol_objective(state, params):
    """Slot tracking and heading alignment, with a floor under the follower's speed.

    Two departures from the environment's own reward, for the usual two
    reasons.

    The tracking term is ``1 / (1 + (e / scale)**2)`` rather than the shipped
    log-scaled reward. Both are bounded in [0, 1] and both are maximised at
    zero slot error, but a log-scaled reward's gradient decays like ``1/e``.

    ``scale`` is a quarter of ``slot_tolerance``, not the tolerance itself.
    The tolerance is a pass/fail bound; a controller that is actually good
    operates well inside it -- the shipped PID settles at 8-13 m against a 60 m
    tolerance -- so curvature at 60 m leaves the surrogate nearly flat across
    the whole range where the decisions are made. Chosen by measurement rather
    than argument, over two seeds at 300 iterations: a quarter of the tolerance
    scores 175.7, the raw log reward 144.6, and the full tolerance 138.2. The
    precision floor of 3 m is far worse again (19.4 at 50 iterations), so this
    is an interior optimum and not a monotone preference for tighter scaling.

    The barrier is the aircraft objective's, for the same reason it exists
    there. The follower is the same airframe with the same power and stick, and
    the slot can be several hundred metres away at reset, so a planner is free
    to buy position with airspeed and arrive at the slot with nothing left.
    Patrol terminates on the altitude envelope rather than on stall, so a
    departure costs the planner only the steps after it falls out of the sky --
    which a finite horizon may not reach.

    Multiplicative in the alignment factor, as the environment's reward is: a
    wingman flies the slot *parallel* to the lead, not merely at the point.
    """
    from target_gym.patrol.env import heading_alignment, slot_error

    err = slot_error(state) / (_PATROL_ERROR_SCALE * params.slot_tolerance)
    track = 1.0 / (1.0 + err**2)
    align = heading_alignment(state, params)

    f = state.follower
    speed = jnp.sqrt(f.x_dot**2 + f.y_dot**2 + f.z_dot**2)
    v_stall = jnp.sqrt(
        2.0 * f.m * params.gravity / (f.rho * params.wings_surface * params.CL_max)
    )
    margin = speed / (_PLANE_STALL_MARGIN * v_stall)
    penalty = _PLANE_BARRIER_WEIGHT * jnp.maximum(1.0 - margin, 0.0) ** 2
    return track * align - penalty


def make_patrol_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 300,
    lr: float = 0.05,
    n_tail: int = 60,
):
    """Gradient MPC for the formation-keeping follower.

    A ``GradientMPC`` rather than a CasADi one, and that is what makes this
    tractable at all. The obstacle recorded against a patrol MPC was that the
    reference is a *manoeuvring lead*, so a symbolic model would need the
    lead's whole future trajectory wired in as a time-varying parameter. That
    is true of the CasADi route and irrelevant here: the lead is scripted and
    deterministic -- ``step_lead`` advances it with a heading autopilot at a
    fixed ``lead_turn_rate`` -- so a planner that differentiates the true
    ``step_env`` propagates the lead for free, exactly as it propagates the
    follower.

    The horizon is 30 s at the environment's 1 s step, which covers the slot
    capture from a 40 m spawn offset with room for the lead's turn to develop,
    and the settings that go with it are the 2D aircraft's for the reasons that
    file already records. ``n_tail=60`` charges the plan for twice the flight
    it optimises, so it cannot park the follower somewhere that leaves the
    altitude envelope just past the horizon. ``done_value`` sits below the
    worst step this objective can score: with the barrier subtracted the
    objective is no longer non-negative, so a ``done_value`` of 0 would make
    flying out of the envelope score *better* than any penalised step, and the
    planner takes that trade.

    ``n_iter`` is 300 where every other gradient planner here uses 50, and that
    single number is what decides whether this baseline is an upper bound at
    all. Measured over two seeds against a PID scoring ~105: 106.1 at 100
    iterations, 130.3 at 150, 153.8 at 300, 171.5 at 600 with no tail. The
    planner was not stuck, it was stopping early -- 90 decision variables under
    projected gradient descent -- and every objective and horizon variant tried
    before this was being compared at a non-converged optimum, which is why
    none of them looked decisive.

    The other tasks hide this. ``plane3d`` uses the same 50 iterations and wins
    enormously, but its PIDs score 0.16 of ceiling, so an under-converged plan
    clears them anyway. The patrol PID scores 0.50, and a bar that high is what
    made the under-convergence visible.

    The tail earns its keep here rather than costing: at 300 iterations
    ``n_tail=60`` scores 175.7 against 153.8 without it, which is better than
    doubling the iterations to 600 (171.5) and half the cost.
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
        n_tail=n_tail,
        objective_fn=_patrol_objective,
        done_value=-(_PLANE_BARRIER_WEIGHT + 1.0),
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

    def solver_report(self) -> dict:
        """No external solver, so no convergence to report.

        Part of the planner interface rather than a special case at the call
        site: the recorder asks every controller for its solver health, and a
        planner that has none should say so rather than raise. Leaving it off
        cost a 44-minute record that crashed on the one sampling planner in the
        suite after every other environment had already finished.
        """
        return {}

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

    This surrogate was written against a clipped ``clip(1-err/band, 0, 1)**2``
    tracking term that was exactly flat outside the band, leaving the optimiser
    nothing to descend but the degradation and state-of-charge terms, which both
    pull toward doing nothing. That reward is gone -- tracking is log-scaled now,
    and never flat. The surrogate stays anyway, for the reason given in
    :func:`_wind_turbine_objective`: log-scaling fixes the *value*, not the
    gradient, which still decays like ``1/err``. Re-measured over ten seeds
    against the log-scaled reward it is worth a median of +9 (155.8 against
    146.8).

    Read the mean with care in either case. It is carried by one seed where
    lookahead pays enormously (350 against the PID's 164); on the other nine the
    MPC is behind by 4 to 13, for a median of -4 against the PID and 1 win in
    10. So this is a large improvement over descending the reward directly and
    *not* an upper bound -- horizon, iterations and step size were all swept
    without closing the remainder. It is inside the 10% contract tolerance.
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

    The environment used to score power tracking as ``clip(1-err/band, 0, 1)**2``
    minus a pitch-activity penalty: a fine thing to be scored on and a useless
    thing to descend, because one step off the operating point puts the error at
    nearly four times the band, where the term is clipped flat and the only
    surviving gradient belongs to the *penalty*. The optimiser was then correctly
    guided to stop moving the pitch, and returned ~0 for the rest of the episode.

    The reward is log-scaled now and has no flat region, so that premise is gone
    -- but the surrogate is still needed, for a subtler reason. Log-scaling makes
    the reward scale-free in *value* (each halving of the error is worth the same
    increment); it does not make the *gradient* scale-free. Differentiating
    ``1 - log1p(e/f)/log1p(E/f)`` gives ``-1/((f + e) * log1p(E/f))``, which
    decays like ``1/e``: the pull toward the setpoint is weakest exactly where
    the controller is furthest from it. A quadratic in the normalised error has
    the same minimiser and a gradient that instead *grows* with the error.

    Measured over six seeds against the log-scaled reward, that difference is
    still worth almost everything: 341.9 for this surrogate against 172.1 for
    the reward itself (with the barrier below in both). So the surrogate is not a
    workaround for a broken reward any more -- it is a planner-side
    reformulation, which is a normal thing for an MPC to carry.
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
    # Re-checked against the log-scaled reward, where it is still the difference
    # between controlling and tripping: 172.1 with the barrier, 52.4 without,
    # and without it 6 of 6 episodes ended early on the overspeed trip.
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

    The objective is the environment's own reward; see ``make_plane3d_mpc``
    for why ``done_value`` follows the reward version.
    """
    return GradientMPC(
        env,
        params,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        done_value=_done_value(params),
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


def make_glass_furnace_mpc(
    env,
    params,
    horizon: int = 60,
    bias_gain: float = _FURNACE_BIAS_GAIN,
    bias_reset_on_setpoint: bool = _FURNACE_BIAS_RESET,
):
    """CasADi/IPOPT MPC for the GlassFurnace (3-zone lumped thermal model).

    With delta_t=30 s, horizon=60 gives 30 min lookahead.  The crown thermal
    time constant is ~15 min, so 2×τ of lookahead is enough to see the next
    scheduled setpoint change and pre-cool / pre-heat accordingly (which PID
    cannot do — that's the whole point of the schedule).
    """
    mpc = GlassFurnaceCasadiMPC(env, params, horizon=horizon)
    mpc._bias_gain = float(bias_gain)
    mpc._bias_reset = bool(bias_reset_on_setpoint)
    return mpc
