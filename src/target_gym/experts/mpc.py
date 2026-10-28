"""
Shared machinery for the reference controllers (the MPC slot of each task).

Each task's oracle lives in its own package's ``experts.py``, so changing one
re-records only that package's tasks. This module holds what they share, and
it is in every task's baseline fingerprint, so editing it re-records all of
them.

GradientMPC  (JAX)
    Single-shooting gradient MPC: differentiates through a ``lax.scan``
    rollout of ``step_env`` and descends the action sequence.

CasadiMPC  (CasADi / IPOPT)
    Receding-horizon NLP solved by IPOPT through do-mpc, with analytic
    Jacobians. Each CasADi plant subclasses it with its own model.

ShrinkingHorizonNLP  (CasADi / IPOPT)
    One NLP over the rest of the episode on the environment's own discrete
    step, re-solved from the true state every few steps. For deterministic
    plants with episodes short enough to plan whole (the CSTR and the
    four-tank), where it reaches the plant's optimum.

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
    is zeroed, so the planner predicts the **mean** disturbance (certainty
    equivalence). Without it a planner rolling the environment under a fixed
    key would plan on one invented noise path, which is the plant's own path
    on seed 0 and gives the MPC perfect foresight there.
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
        the ``n_iter`` iterations of each solve. A fixed normalised step cannot
        place an action closer than about ``lr`` to the optimum, which leaves a
        steady offset; a decaying one keeps the early reach and resolves the
        end. ``None`` keeps the fixed step.
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

        It assumes holding the last action approximates continuing sensibly,
        so it is enabled per environment (``make_plane_mpc``).
    done_value : float
        Per-step objective charged for every step after the planned rollout
        reaches a terminal state, so a plan that trips the plant early is not
        scored as if it flew on. 0.0 suits an objective that is positive while
        healthy; a negative objective must set this below its worst per-step
        value.
    objective_fn : callable or None
        ``f(state, params) -> scalar`` summed over the rollout in place of the
        environment's reward. It must share the reward's minimiser and be
        smooth where the reward is flat or clipped (``_v2_objective``).
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
        # action_dim)`` plan) for the first plan of an episode. Without it the
        # plan starts mid-range and takes several steps to reach the operating
        # point; a real MPC warm-starts from the actuator state.
        self.initial_plan_fn = initial_plan_fn
        self._fresh = True
        # ``f(state, params) -> plan`` evaluated at every step: the descent
        # runs from the shifted previous plan, and the better of the two under
        # the planner's objective is kept. With a stabilising controller's
        # rollout as the guide, the plan is never worse than that controller's
        # under the planner's model.
        self.guide_plan_fn = guide_plan_fn
        self._jit_rollout = jax.jit(self._rollout)

        self._actions = jnp.zeros((horizon, action_dim))
        # One jitted entry point: two with the same body compile twice, about
        # 50 s each for the patrol planner.
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

        These plants saturate, written with ``clip`` or ``maximum``, whose
        derivative at the kink is zero. An action projected exactly onto a
        bound therefore never moves again: the bound is absorbing. Holding the
        iterates ``_BOUND_MARGIN`` inside the bounds is the interior-point
        principle in miniature, and on the 2D aircraft it was the difference
        between gliding into the ground and flying the episode.

        The NaN scrub below guards against a known, unlocalised reverse-mode
        NaN in the aircraft dynamics (``NAN_TUNERS`` in
        ``tests/experts/test_pid_tuning.py``).
        """
        cost_and_grad = jax.value_and_grad(lambda a: -self._rollout(a, state))
        margin = self._BOUND_MARGIN * 0.5 * (self.action_ub - self.action_lb)

        # The descent is not monotone: a fixed step along a normalised gradient
        # overshoots wherever the objective has an edge (a tolerance band, a
        # barrier). So the plan returned is the best iterate seen, the warm
        # start included, and a solve never leaves a worse plan than it found.
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
        if self.action_dim == 1:
            return float(first[0])
        return np.array(first)

    def _score(self, plan, state) -> float:
        return float(self._jit_rollout(plan, state))

    def reset(self):
        """Reset the internal action sequence to zeros (or, on the next step,
        to ``initial_plan_fn`` of the state)."""
        self._actions = jnp.zeros((self.horizon, self.action_dim))
        self._fresh = True
        for fn in (self.initial_plan_fn, self.guide_plan_fn):
            if hasattr(fn, "reset"):
                fn.reset()


# ============================================================================
# CasADi MPC (IPOPT)
# ============================================================================


# IPOPT defaults to 3000 iterations and no time limit, which in a
# receding-horizon loop can hang one step for half an hour. A real MPC has a
# sample period and applies the best iterate it holds, so ours does too. The
# iteration cap is the one meant to bind, because it is deterministic and a
# record then reproduces on another machine; the CPU-time cap is a backstop.
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
    Receding-horizon MPC solved by IPOPT through do-mpc and CasADi, with
    analytic Jacobians (horizon 5 by default, as in PC-gym's oracle).

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
        decision variables. Without this, units alone make the system badly
        conditioned (the reactor's states span a factor of about 600,000),
        which shows up as iteration counts. ``SCALING`` holds the typical
        ``|x|`` of each variable over a PID episode, to one figure.
        """
        for kind, entries in self.SCALING.items():
            for var, value in entries.items():
                mpc.scaling[kind, var] = float(value)

    # ------------------------------------------------------------------
    # Solver health
    # ------------------------------------------------------------------

    def _record_solve(self) -> bool:
        """Fold the last solve's outcome into the running counters.

        do-mpc neither raises nor warns when IPOPT gives up, so without these
        counters a failed solve looks like a converged one. ``solver_report``
        publishes them alongside the return.
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


# ============================================================================
# Shrinking-horizon NLP (IPOPT)
# ============================================================================


def casadi_step_map(velocity, nx: int, nu: int, delta_t: float, method: str):
    """The first-order step of ``integration.integrate_dynamics`` in CasADi.

    Returns ``F(x, u) -> x_next``, a ``casadi.Function``. ``velocity(x, u)``
    is the plant's right-hand side written in CasADi, and ``method`` is the
    environment's own ``integration_method`` string (``"rk4_1"``,
    ``"euler_2"``, ...), parsed the way ``integrate_dynamics`` parses it, with
    the same stages and the same substep length ``delta_t / n``. So the map
    is the environment's step in float64, wherever the plant is a
    first-order ODE integrated by that function.
    """
    if "euler" in method:
        order, n_sub = "euler", int(method.split("_")[1])
    elif "rk" in method:
        order, n_sub = int(method.split("_")[0][2:]), int(method.split("_")[1])
    else:
        raise ValueError(f"Unknown integration method: {method}")
    if order not in ("euler", 2, 3, 4):
        raise ValueError(f"Unsupported RK order: {order}")
    h = float(delta_t) / n_sub
    x = casadi.MX.sym("x", nx)
    u = casadi.MX.sym("u", nu)

    def f(p):
        return velocity(p, u)

    p = x
    for _ in range(n_sub):
        if order == "euler":
            p = p + h * f(p)
        elif order == 2:
            k1 = f(p)
            p = p + h * f(p + 0.5 * h * k1)
        elif order == 3:
            k1 = f(p)
            k2 = f(p + 0.5 * h * k1)
            k3 = f(p - h * k1 + 2 * h * k2)
            p = p + (h / 6.0) * (k1 + 4 * k2 + k3)
        else:
            k1 = f(p)
            k2 = f(p + 0.5 * h * k1)
            k3 = f(p + 0.5 * h * k2)
            k4 = f(p + h * k3)
            p = p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return casadi.Function("step", [x, u], [p])


def protocol_burn_in(name: str, params) -> int:
    """The first step the protocol scores on task ``name`` at these params.

    ``eval.scored_burn_in``: the hold row's burn-in, capped at half the
    episode, which is what ``eval.evaluate_controller`` scores from. An
    oracle that weights the scored window reads it here so that it
    optimises the window the protocol scores. The burn-in is in every task's
    baseline fingerprint (``provenance.baseline_fingerprint``), so a record
    taken under one burn-in is stale under another.
    """
    from target_gym.eval import scored_burn_in

    return scored_burn_in(name, params)


# IPOPT settings of the shrinking-horizon NLP. The iteration cap is the one
# that binds, and it is deterministic, so a record reproduces on a slower
# machine. Solves average 6 to 12 iterations on the CSTR and the four-tank.
NLP_MAX_ITER = 500
NLP_TOL = 1e-9
NLP_ACCEPTABLE_TOL = 1e-6


class ShrinkingHorizonNLP:
    """
    One NLP over the rest of the episode, re-solved from the true state.

    The oracle for a deterministic plant whose episode is short enough to
    plan whole. Every ``resolve_every`` steps it solves, with IPOPT through
    CasADi's ``Opti``, for the actions from the current step to the end of
    the episode, on ``step_map``, the environment's own discrete step written
    in CasADi: the same integration method and substeps as ``env.py``, in
    float64 (each task's ``experts.py`` builds it and a test holds it equal to
    ``step_env``). In between it applies that plan. There is no model error
    and no horizon to fall short of, so up to float32 rounding in the plant
    the closed loop is the plant's optimum for the objective.

    The objective is the version-2 tracking cost in floor units,
    ``((target - x) / e_floor) ** 2`` summed over the tracked outputs, which is
    the reward's own tracking term where ``e_tol`` is 0 and the exponent 2
    (the constructor refuses anything else). Steps from ``window_start`` on
    weigh 1 and earlier ones ``pre_weight``. ``window_start`` is the step the
    protocol starts scoring from, which the task's factory reads from the
    protocol's own burn-in (``protocol_burn_in``, the hold row in
    ``data/hold_measurements.json``), so the oracle optimises the window the
    protocol scores. A small ``pre_weight`` keeps the approach from being
    free (a window-only objective is degenerate) without letting it trade
    against the window. The cost is divided by the number of steps planned.

    Inputs are bounded to the action box ``[-1, 1]``, hard. ``x_lb`` and
    ``x_ub``, when given, bound every planned state after the current one,
    hard as well: the plant is deterministic and the map exact, so a bound the
    plan keeps is one the plant keeps. Decision variables are scaled by
    ``x_scale``, the typical magnitude of each state.

    The first solve of an episode starts cold, from ``u = 0``; later ones
    warm-start from the previous plan, shifted to the current step. Both roll
    the guess through ``step_map`` for the states' initial values. A solve
    that IPOPT stops at the iteration cap is applied, as ``CasadiMPC`` applies
    one. A solve that fails for any other reason leaves the previous plan in
    place, or, on an episode's first solve, applies the iterate clipped to
    the box. ``solver_report`` counts both, in ``CasadiMPC``'s fields.

    ``state_fields`` names the environment-state fields the map's ``x`` holds,
    in order; ``target_fields`` the fields holding the setpoints, one per
    entry of ``tracked``, the indices into ``x`` of the outputs they set.
    """

    def __init__(
        self,
        env,
        params,
        step_map,
        state_fields,
        target_fields,
        tracked,
        window_start: int,
        x_scale,
        x_lb=None,
        x_ub=None,
        pre_weight: float = 1e-3,
        resolve_every: int = 1,
        tol: float = NLP_TOL,
        acceptable_tol: float = NLP_ACCEPTABLE_TOL,
        max_iter: int = NLP_MAX_ITER,
    ):
        if not _CASADI_AVAILABLE:
            raise ImportError(
                "casadi is required for ShrinkingHorizonNLP: pip install casadi"
            )
        if float(getattr(params, "e_tol", 0.0)) != 0.0 or (
            float(getattr(params, "tracking_exponent", 2.0)) != 2.0
        ):
            raise ValueError(
                "ShrinkingHorizonNLP's objective is the reward's tracking term "
                "only for a quadratic cost with no tolerance band"
            )
        self.env, self.params = env, params
        self.step_map = step_map
        self.state_fields = tuple(state_fields)
        self.target_fields = tuple(target_fields)
        self.tracked = tuple(int(i) for i in tracked)
        self.nx = int(step_map.size1_in(0))
        self.action_dim = int(step_map.size1_in(1))
        if len(self.state_fields) != self.nx:
            raise ValueError("state_fields must name every entry of the map's x")
        if len(self.target_fields) != len(self.tracked):
            raise ValueError("one target field per tracked output")
        self.steps = int(params.max_steps_in_episode)
        # The first solve of an episode plans all of it.
        self.horizon = self.steps
        self.window_start = int(window_start)
        self.x_scale = np.asarray(x_scale, dtype=float).reshape(self.nx)
        self.x_lb = None if x_lb is None else np.asarray(x_lb, float).reshape(self.nx)
        self.x_ub = None if x_ub is None else np.asarray(x_ub, float).reshape(self.nx)
        self.e_floor = np.broadcast_to(
            np.asarray(params.e_floor, dtype=float), (len(self.tracked),)
        )
        self.pre_weight = float(pre_weight)
        self.resolve_every = max(int(resolve_every), 1)
        self._ipopt = {
            "print_level": 0,
            "sb": "yes",
            "tol": float(tol),
            "acceptable_tol": float(acceptable_tol),
            "max_iter": int(max_iter),
            "mu_strategy": "adaptive",
        }
        # Solver health over every solve, as ``CasadiMPC`` keeps it: ``reset``
        # leaves these alone so one count covers a whole rollout.
        self.solve_calls = 0
        self.solve_iters = 0
        self.solve_failures = 0
        self.solve_capped = 0
        self.last_return_status = ""
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, _obs, state):
        """The action for the current state. ``_obs`` is ignored."""
        t = int(state.time)
        if self._plan is not None and t < self._t0:
            self._plan = None  # a new episode without a reset
        if t >= self.steps:  # past the planned episode: hold the last action
            u = np.zeros(self.action_dim) if self._plan is None else self._plan[:, -1]
        else:
            stale = self._plan is None or t - self._t0 >= self._plan.shape[1]
            if stale or (t - self._t0) % self.resolve_every == 0:
                self._solve(state, t)
            u = self._plan[:, t - self._t0]
        return float(u[0]) if self.action_dim == 1 else u

    def reset(self):
        """Forget the plan, so the next step solves from a cold start."""
        self._plan = None
        self._t0 = 0

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

    # ------------------------------------------------------------------
    # The solve
    # ------------------------------------------------------------------

    def _guess(self, t: int, n: int) -> np.ndarray:
        """The previous plan from step ``t`` on, or zeros on a cold start."""
        if self._plan is None:
            return np.zeros((self.action_dim, n))
        guess = self._plan[:, t - self._t0 :]
        if guess.shape[1] < n:
            pad = np.repeat(guess[:, -1:], n - guess.shape[1], axis=1)
            guess = np.concatenate([guess, pad], axis=1)
        return guess[:, :n]

    def _solve(self, state, t: int) -> None:
        F, nx, nu = self.step_map, self.nx, self.action_dim
        x0 = np.array([float(getattr(state, f)) for f in self.state_fields])
        targets = [float(getattr(state, f)) for f in self.target_fields]
        n = self.steps - t
        scale = self.x_scale
        S = casadi.DM(scale)

        opti = casadi.Opti()
        X = opti.variable(nx, n + 1)  # states over scale
        U = opti.variable(nu, n)
        opti.subject_to(X[:, 0] == x0 / scale)
        for k in range(n):
            opti.subject_to(X[:, k + 1] == F(X[:, k] * S, U[:, k]) / S)
        opti.subject_to(opti.bounded(-1.0, U, 1.0))
        if self.x_lb is not None or self.x_ub is not None:
            lb = np.full(nx, -np.inf) if self.x_lb is None else self.x_lb
            ub = np.full(nx, np.inf) if self.x_ub is None else self.x_ub
            for k in range(1, n + 1):
                opti.subject_to(opti.bounded(lb / scale, X[:, k], ub / scale))
        J = 0
        for k in range(n):
            # Action k sets the state the step's cost is charged on, k + 1.
            w = 1.0 if t + k >= self.window_start else self.pre_weight
            for j, i in enumerate(self.tracked):
                J += w * ((targets[j] - X[i, k + 1] * scale[i]) / self.e_floor[j]) ** 2
        opti.minimize(J / n)

        guess = self._guess(t, n)
        xg = np.zeros((nx, n + 1))
        xg[:, 0] = x0
        for k in range(n):
            xg[:, k + 1] = np.array(F(xg[:, k], guess[:, k])).ravel()
        opti.set_initial(X, xg / scale[:, None])
        opti.set_initial(U, guess)
        opti.solver("ipopt", {"print_time": 0}, dict(self._ipopt))
        try:
            u = np.array(opti.solve().value(U))
            ok = True
        except RuntimeError:
            u = np.array(opti.debug.value(U))
            ok = False
        stats = opti.stats()
        status = str(stats.get("return_status", ""))
        self.solve_calls += 1
        self.solve_iters += int(stats.get("iter_count", 0) or 0)
        capped = status in _IPOPT_CAP_STATUSES
        if capped:
            self.solve_capped += 1
        if not ok:
            self.solve_failures += 1
            self.last_return_status = status
            if not capped and self._plan is not None:
                return  # keep the previous plan rather than a failed iterate
        self._plan = np.clip(u.reshape(nu, n), -1.0, 1.0)
        self._t0 = t


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
    plant raises ``terminated`` any more (a trip is charged the trip cost and
    the plant restarts, ``base.failure_kernel``), so the planner sees the
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

    A normalised-gradient planner cannot travel far from where a solve
    starts, so from a constant plan it can miss a manoeuvre the plant needs.
    Starting from, and at every step compared against, a stabilising
    controller's plan makes the planner an upper bound over that controller
    under its own model.

    The PID is cold-started for every rollout, as a controller switched on in
    this state would be: one stepped along the planner's trajectory winds its
    integrators up on a control history that is not its own.
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
    the plants whose tracking cost is linear in |error| (p = 1). Normalised
    descent on a linear cost never shrinks its step near the optimum, so the
    plan chatters; the tracking term is squared in floor units instead (same
    minimiser, a gradient that vanishes at it), and the other terms are kept.

    The version-2 reward is convex and additive, so the planner descends the
    plant's own cost, which is what makes the MPC an upper bound on that
    cost. A trip is a boolean with no derivative pointing away from it, so
    the barrier stays, weighted at the failure charge in the same units.
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

    For plants whose forward rollout is well behaved but whose adjoint is not:
    on the cement kiln, reverse-mode gradients through the Arrhenius term and
    the kiln's transport delay overflow to NaN after about eight steps. So
    this samples instead: draw action sequences, roll them out, keep the elite
    fraction, refit, repeat. Everything is vmapped and jitted.
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
