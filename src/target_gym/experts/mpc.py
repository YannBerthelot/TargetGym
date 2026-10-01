"""
Shared machinery for the reference controllers (the MPC slot of each task).

Each task's oracle lives in its own package's ``experts.py``, so changing one
re-records only that package's tasks. This module holds what they share, and
it is in every task's baseline fingerprint, so editing it re-records all of
them.

GradientMPC  (JAX)
    Single-shooting gradient MPC: differentiates through a JAX scan rollout
    and runs gradient descent on the action sequence.  Requires a long horizon
    to be accurate, and suffers from vanishing gradients for stiff systems.

CasadiMPC  (CasADi / IPOPT)
    Proper NLP-based receding-horizon MPC solved by IPOPT via do_mpc.  The
    solver receives analytic Jacobians and finds the exact optimum in each
    window, matching the PC-gym oracle approach.  Each CasADi plant subclasses
    it with its own model.  Requires::

        pip install casadi do-mpc

SamplingMPC  (JAX, gradient-free)
    Cross-entropy-method shooting: samples action sequences, rolls them out,
    refits to the elite fraction.  For plants whose forward rollout is well
    behaved but whose adjoint is not (the cement kiln).

Plus ``plan_params`` (the planner's noise-free copy of the params) and the
helpers that turn a version-2 reward into a planner objective.

Common API::

    mpc = spec.make_mpc(env, params)
    obs, state = env.reset_env(key, params)
    for _ in range(T):
        action = mpc.step(obs, state)   # obs ignored by MPC, kept for symmetry
        obs, state, *_ = env.step_env(key, state, action, params)
    mpc.reset()
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np

try:
    import casadi
    import do_mpc  # noqa: F401  (the CasadiMPC subclasses build on it)

    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False


# ============================================================================
# Gradient MPC (JAX)
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
    lr_end : float or None
        If set, the step decays geometrically from ``lr`` to ``lr_end`` over
        the ``n_iter`` iterations of each solve. The fixed step of a
        normalised descent is also its resolution: it cannot place an action
        closer than about ``lr`` to the optimum, which left steady offsets on
        the 3D aircraft and on distillation (the oracle audit, 2026-10). A
        decaying step keeps the early reach and resolves the end. ``None``
        keeps the fixed step.
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
        lr_end: float | None = None,
        action_dim: int = 1,
        action_lb: float = -1.0,
        action_ub: float = 1.0,
        n_tail: int = 0,
        done_value: float = 0.0,
        objective_fn=None,
        initial_plan_fn=None,
        guide_plan_fn=None,
        move_penalty_fn=None,
    ):
        self.env = env
        self.params = params
        self.horizon = horizon
        self.n_iter = n_iter
        self.lr = lr
        self.lr_end = lr_end
        self.action_dim = action_dim
        self.action_lb = float(action_lb)
        self.action_ub = float(action_ub)
        self.n_tail = int(n_tail)
        self.done_value = float(done_value)
        self.objective_fn = objective_fn
        # ``f(state, params) -> raw action`` (or a whole ``(horizon,
        # action_dim)`` plan) the first plan of an episode is filled with. Without it the plan starts at zero -- mid-range on every
        # actuator -- and the normalised-gradient steps take several env steps
        # to walk it to the operating point: on the wind turbine the first
        # action left a 1.5 MW error where the PID, which starts from the
        # actuator's current position, left 3 kW. A warm start from the
        # actuator state is what a real MPC does at commissioning.
        self.initial_plan_fn = initial_plan_fn
        self._fresh = True
        # ``f(state, params) -> plan`` evaluated at every step: the descent
        # runs from the shifted previous plan as always, and the better of
        # the two plans under the planner's objective is kept. With a
        # stabilising controller's rollout as the guide, the planner's plan is
        # never worse than that controller's under its own model -- which is
        # what makes it an upper bound over it rather than a competitor that
        # can lose to it where the descent stalls (measured on the 2D
        # aircraft: from its own shifted plan it held ten floor-widths worse
        # than the cascaded PID; guided, it cannot).
        self.guide_plan_fn = guide_plan_fn
        # ``f(u_first, u_previous, params) -> cost`` in the objective's units:
        # move suppression on the first action against the one applied last
        # step. An open-loop plan cannot see the activity that re-planning
        # creates -- successive solves disagree, and the applied command
        # jumps where every plan was smooth -- so a reward that prices
        # actuator activity is under-charged in the plan and over-paid in the
        # plant (the turbine's pitch activity ran 3.4x the PID's with the
        # plan predicting less). Priced like that reward term, this closes the
        # gap; it is standard MPC practice (do-mpc's ``rterm``).
        self.move_penalty_fn = move_penalty_fn
        self._u_prev = None
        self._jit_rollout = jax.jit(self._rollout)

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
        u_prev = self._u_prev

        def objective(a):
            total = self._rollout(a, state)
            if self.move_penalty_fn is not None and u_prev is not None:
                total = total - self.move_penalty_fn(a[0], u_prev, self.params)
            return total

        cost_and_grad = jax.value_and_grad(lambda a: -objective(a))
        margin = self._BOUND_MARGIN * 0.5 * (self.action_ub - self.action_lb)

        # The descent is not monotone: a fixed step along a normalised
        # gradient overshoots wherever the objective has an edge -- a
        # tolerance band, a barrier -- and the last iterate can then be far
        # worse than the warm start it began from. Measured on the 2D
        # aircraft (seed 1, step 144 of the test episode): the shifted plan
        # scored -0.003, the plan after 50 iterations -3.31, and three steps
        # later -0.99 against -19.7; taking the descended plan regardless,
        # the planner then reached for the guide, dived 33 m out of the
        # tolerance band and lost to the PID on the hold it had just
        # reached. So the plan returned is the best iterate seen, the warm
        # start included: under its own model the planner never leaves a
        # solve with a worse plan than it entered it with. The steps
        # themselves stay fixed: an accept-only backtracking step was tried
        # and flew the aircraft 55 m/s slow again (running cost 0.15 against
        # the PID's 0.002 on seed 1) -- it makes too little progress per solve.
        lb, ub, lr = self.action_lb + margin, self.action_ub - margin, self.lr
        if self.lr_end is None:

            def step_size(_):
                return lr

        else:
            ratio, denom = float(self.lr_end) / lr, float(max(n_iter - 1, 1))

            def step_size(i):
                return lr * ratio ** (i / denom)

        def body(i, carry):
            actions, best, best_cost = carry
            cost, g = cost_and_grad(actions)
            better = cost < best_cost
            best = jnp.where(better, actions, best)
            best_cost = jnp.where(better, cost, best_cost)
            # Replace NaN gradients with zero (can arise from numerically
            # unstable rollouts, e.g. near-stall flight dynamics)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            # Clip gradient norm
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-8)
            g = jnp.where(g_norm > 1.0, g / g_norm, g)
            return jnp.clip(actions - step_size(i) * g, lb, ub), best, best_cost

        start = jnp.clip(actions_init, lb, ub)
        last, best, best_cost = jax.lax.fori_loop(
            0, n_iter, body, (start, start, jnp.asarray(jnp.inf, start.dtype))
        )
        last_cost, _ = cost_and_grad(last)
        return jnp.where(last_cost < best_cost, last, best)

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
        if self._fresh and self.initial_plan_fn is not None:
            init = jnp.asarray(
                self.initial_plan_fn(state, self.params), dtype=jnp.float32
            )
            if init.ndim == 2:
                self._actions = init  # a whole plan, e.g. a PID's rollout
            else:
                first = jnp.reshape(init, (self.action_dim,))
                self._actions = jnp.broadcast_to(first, (self.horizon, self.action_dim))
        self._fresh = False
        actions_init = jnp.concatenate([self._actions[1:], self._actions[-1:]], axis=0)
        self._actions = self._jit_optimize(actions_init, state)
        if self.guide_plan_fn is not None:
            guide = jnp.asarray(
                self.guide_plan_fn(state, self.params), dtype=jnp.float32
            )
            if self._score(guide, state) > self._score(self._actions, state):
                self._actions = guide
        first = self._actions[0]
        self._u_prev = first
        if self.action_dim == 1:
            return float(first[0])
        return np.array(first)

    def _score(self, plan, state) -> float:
        total = float(self._jit_rollout(plan, state))
        if self.move_penalty_fn is not None and self._u_prev is not None:
            total -= float(self.move_penalty_fn(plan[0], self._u_prev, self.params))
        return total

    def reset(self):
        """Reset the internal action sequence to zeros (or, on the next step,
        to ``initial_plan_fn`` of the state)."""
        self._actions = jnp.zeros((self.horizon, self.action_dim))
        self._fresh = True
        self._u_prev = None
        for fn in (self.initial_plan_fn, self.guide_plan_fn):
            if hasattr(fn, "reset"):
                fn.reset()


# ============================================================================
# CasADi MPC (IPOPT)
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


def _is_v1(params) -> bool:
    return int(getattr(params, "reward_version", 2)) == 1


def _floor_units(params) -> float:
    """The version-2 reward's scale: the tracking cost per step at the floor."""
    return float(getattr(params, "rho_floor_tracking", 1.0)) or 1.0


def _failure_units(params, squared: bool = False) -> float:
    """The failure charge in the planner's units: floor units, and squared
    where the planner squares a linear tracking term (so that it still
    exceeds the worst tracking cost the envelope can produce)."""
    f = float(getattr(params, "failure_cost", 0.0)) / _floor_units(params)
    return f * f if squared else f


def _done_value(params, squared: bool = False) -> float:
    """What a planner on the environment's own reward charges per step once
    the plant has terminated: 0 for the non-negative version-1 reward (the
    forgone reward is the penalty), the failure charge for version 2, whose
    healthy steps are negative -- in the planner's units. Under version 2 no
    plant raises ``terminated`` any more (a trip freezes it at the failure
    cost inside the rollout, ``base.failure_kernel``), so the planner sees the
    trip's cost in the rollout itself and this value is never applied; it is
    kept for the version-1 planners."""
    if _is_v1(params):
        return 0.0
    return -_failure_units(params, squared)


_SMOOTH_K = 0.05  # width of the smoothed max(x, 0) used in CasADi objectives


def _smooth_max(x):
    """``max(x, 0)`` smoothed for IPOPT: ``(x + sqrt(x^2 + k^2)) / 2``."""
    return 0.5 * (x + casadi.sqrt(x * x + _SMOOTH_K * _SMOOTH_K))


class _PIDRolloutPlan:
    """``initial_plan_fn`` / ``guide_plan_fn`` that rolls the shipped PID out
    from the current state over the horizon and hands its action sequence to
    the planner.

    A normalised-gradient planner moves its whole plan by at most ``lr`` per
    iteration, spread over ``horizon * action_dim`` entries, so a solve
    cannot travel far from wherever it starts: on the wind turbine, from a
    constant plan it braked the rotor with the torque (a 700 kW error for the
    whole horizon) because the pitch schedule it needed was out of reach, and
    on the 2D aircraft it parked at the edge of the altitude tolerance 55 m/s
    below cruise. Starting from, and at every step compared against, a
    stabilising controller's plan makes the planner an upper bound over that
    controller under its own model.

    The PID is cold-started for every rollout -- its integrators at zero,
    what a controller switched on in this state would do. A PID stepped
    along the planner's own trajectory instead winds its integrators up on
    a control history that is not its own, and on the aircraft its plan then
    crashed (objective -1e8) while the shipped PID held perfectly.
    """

    def __init__(self, env, make_pid, horizon: int, action_dim: int):
        self.env, self.make_pid = env, make_pid
        self.horizon, self.action_dim = horizon, action_dim
        self.reset()

    def reset(self):
        self._pid = self.make_pid()
        if hasattr(self._pid, "reset"):
            self._pid.reset()
        call = self._pid if callable(self._pid) else self._pid.step
        self._wants_state = len(inspect.signature(call).parameters) >= 2

    def _act(self, pid, obs, state):
        call = pid if callable(pid) else pid.step
        raw = (
            call(np.asarray(obs), state) if self._wants_state else call(np.asarray(obs))
        )
        a = np.reshape(np.asarray(raw, dtype=np.float32), (-1,))
        return (
            a[: self.action_dim]
            if a.shape[0] >= self.action_dim
            else np.resize(a, self.action_dim)
        )

    def __call__(self, state, params):
        env = self.env
        obs = env.get_obs(state, params)
        self.reset()
        pid = self._pid
        plan = []
        s = state
        for _ in range(self.horizon):
            a = self._act(pid, obs, s)
            plan.append(a)
            obs, s, _, term, _ = env.step_env(
                jax.random.PRNGKey(0), s, jnp.asarray(a), params
            )
            if bool(term):
                break
        while len(plan) < self.horizon:
            plan.append(plan[-1])
        return jnp.asarray(np.stack(plan), dtype=jnp.float32)


def _pid_rollout_plan(env, make_pid, horizon: int, action_dim: int):
    return _PIDRolloutPlan(env, make_pid, horizon, action_dim)


def _v2_objective(reward_fn, barrier_fn=None, terms_fn=None, shaping_fn=None):
    """A planner objective on the version-2 reward: the reward in floor units
    (tracking at the floor costs 1 per step) minus a differentiable barrier
    weighted like the failure charge.

    ``terms_fn`` (the environment's ``compute_reward_terms``) is passed for
    the plants whose tracking cost is linear in |error| (p = 1: the wind
    turbine, the battery). Projected descent with a normalised gradient on a
    linear cost is sign descent -- the step never shrinks near the optimum,
    so the plan chatters (measured on the turbine: the rotor speed wandered to
    1.12x rated with the barrier active two thirds of the time, and the hold
    error was 6x the PID's). The tracking term is squared in floor units
    instead, which has the same minimiser and a gradient that vanishes at it;
    the running and failure terms are kept as they are.

    The version-1 surrogates in this module exist because the log-scaled
    reward was flat where a planner needed a gradient. The version-2 reward
    is convex and additive (docs/reward-shaping.md), so the planner can
    descend the plant's own cost -- which is the only way the MPC is an upper
    bound *on that cost*: a surrogate with the version-1 minimiser sells the
    consumption term the new reward charges, and the wind turbine and the
    aircraft measurably lost to their PIDs that way. The barriers stay: a
    terminal state is a boolean and gives the optimiser no derivative pointing
    away from it, so the soft version is weighted at the failure charge
    itself, in the same units.
    """

    def f(state, params):
        floor = _floor_units(params)
        if terms_fn is None:
            obj = reward_fn(state, params) / floor
        else:
            t = terms_fn(state, params)
            track = t["tracking"] / floor
            rest = sum(v for k, v in t.items() if k != "tracking") / floor
            obj = -(track**2 + rest)
        if barrier_fn is not None:
            obj = obj - _failure_units(params, terms_fn is not None) * barrier_fn(
                state, params
            )
        if shaping_fn is not None:
            # Planner-side shaping already expressed in floor units (a
            # regulation preference, not a trip): see the wind turbine.
            obj = obj - shaping_fn(state, params)
        return obj

    return f


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
        return_best: bool = False,
    ):
        self.env = env
        self.params = params
        self.objective = objective
        # Return the best sequence evaluated in the solve (the warm start and
        # the final smoothed mean included) instead of the smoothed mean. The
        # mean is an average of elites, so it can score worse than any of
        # them, and under the model a solve then leaves a worse plan than it
        # entered with. The distribution is refitted exactly as without it.
        self.return_best = return_best
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
        if self.return_best:
            return self._optimize_best(mean, std, state, key)
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

    def _optimize_best(self, mean, std, state, key):
        """``_optimize`` that also tracks the best sequence it evaluates."""
        batch_score = jax.vmap(self._score, in_axes=(0, None))

        def finite(score):
            return jnp.where(jnp.isnan(score), -jnp.inf, score)

        def body(carry, _):
            mean, std, key, best, best_score = carry
            key, sub = jax.random.split(key)
            noise = jax.random.normal(
                sub, (self.n_samples, self.horizon, self.action_dim)
            )
            samples = jnp.clip(
                mean[None] + std[None] * noise, self.action_lb, self.action_ub
            )
            scores = finite(batch_score(samples, state))
            top = jnp.argmax(scores)
            better = scores[top] > best_score
            best = jnp.where(better, samples[top], best)
            best_score = jnp.where(better, scores[top], best_score)
            elite_idx = jnp.argsort(scores)[-self.n_elite :]
            elite = samples[elite_idx]
            new_mean = elite.mean(axis=0)
            new_std = jnp.maximum(elite.std(axis=0), self.min_std)
            mean = self.alpha * mean + (1.0 - self.alpha) * new_mean
            std = self.alpha * std + (1.0 - self.alpha) * new_std
            return (mean, std, key, best, best_score), None

        start = finite(self._score(mean, state))
        (mean, std, _, best, best_score), _ = jax.lax.scan(
            body, (mean, std, key, mean, start), None, length=self.n_iter
        )
        last = finite(self._score(mean, state))
        return jnp.where(last > best_score, mean, best), std

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
