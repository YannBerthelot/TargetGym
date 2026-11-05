"""
Shared machinery for the reference controllers (the MPC slot of each task).

Each task's oracle lives in its own package's ``oracle.py``, so changing one
re-records only that package's oracle rows. This module holds what they
share, and it is in every task's oracle fingerprint, so editing it
re-records all of them.

GradientMPC  (JAX)
    Single-shooting gradient MPC: differentiates through a ``lax.scan``
    rollout of ``step_env`` and descends the action sequence.

NewtonMPC  (JAX, second order)
    Box-constrained iterative LQR on the environment's own step: a Newton
    method on the action sequence, with feedback in its line search. Receding
    horizon, or the rest of the episode when the step weights end there.
    The oracle of the process and industrial plants.

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

import dataclasses
import inspect
import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

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

    Sensor and actuator noise go too: the oracle stays a full-state bound, and
    what it reads through ``get_obs`` is the exact state.
    """
    fields = (*getattr(spec, "noise_fields", ()), "sensor_noise_scale", "action_noise")
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
# Newton MPC (JAX): box-constrained iterative LQR
# ============================================================================

_PLAN_KEY = jax.random.PRNGKey(0)
#: Step lengths the line search tries along the Newton step, all at once.
#: Zero keeps the current plan, so a solve never leaves a worse one.
_LINE_SEARCH = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 1e-3, 0.0)


def _is_float(leaf) -> bool:
    return jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)


def _to_x(state):
    """Every floating-point leaf of a state, flattened into one vector."""
    leaves = jax.tree_util.tree_leaves(state)
    return jnp.concatenate([jnp.ravel(v) for v in leaves if _is_float(v)])


def _from_x(x, state):
    """``state`` with its floating-point leaves read back from ``x``; the
    others (the clock, keys) are kept."""
    leaves, treedef = jax.tree_util.tree_flatten(state)
    out, i = [], 0
    for v in leaves:
        v = jnp.asarray(v)
        if _is_float(v):
            out.append(jnp.reshape(x[i : i + v.size], v.shape).astype(v.dtype))
            i += v.size
        else:
            out.append(v)
    return jax.tree_util.tree_unflatten(treedef, out)


def _box_qp(H, g, lo, hi):
    """``argmin 0.5 d'Hd + g'd`` over ``lo <= d <= hi``, H positive definite.

    Exact for the one or two actions these plants have: every pattern of free
    and bound coordinates is solved and the best feasible one kept. Returns
    the step and the mask of free coordinates."""
    m = g.shape[0]
    patterns = jnp.array(list(itertools.product((0, 1, 2), repeat=m)))

    def candidate(p):
        free = (p == 0).astype(H.dtype)
        fixed = jnp.where(p == 1, lo, jnp.where(p == 2, hi, 0.0))
        Hf = free[:, None] * H * free[None, :] + jnp.diag(1.0 - free)
        d = jnp.linalg.solve(Hf, -free * (g + H @ fixed)) * free + fixed
        feasible = jnp.all((d >= lo - 1e-9) & (d <= hi + 1e-9))
        return d, jnp.where(feasible, 0.5 * d @ H @ d + g @ d, jnp.inf), free

    d, value, free = jax.vmap(candidate)(patterns)
    best = jnp.argmin(value)
    return d[best], free[best]


@dataclasses.dataclass(frozen=True)
class _NewtonConfig:
    """What a solve is compiled for. Hashed by ``jax.jit``, so a second
    planner with the same configuration reuses the compilation: the
    environment is the registry's shared instance, and the functions are
    module-level."""

    env: object
    horizon: int
    action_dim: int
    stage_fn: object
    action_map: object
    action_unmap: object
    guide_step: object


def _stage(cfg, params, state, v):
    """``(next state, cost)`` of applying decision ``v`` in ``state``."""
    u = v if cfg.action_map is None else cfg.action_map(state, v, params)
    u = jnp.reshape(u, (cfg.action_dim,))
    a = u[0] if cfg.action_dim == 1 else u
    if cfg.stage_fn is not None:
        return cfg.stage_fn(cfg.env, state, a, params)
    _, new, reward, _, _ = cfg.env.step_env(_PLAN_KEY, state, a, params)
    floor = getattr(params, "rho_floor_tracking", 1.0)
    return new, -reward / jnp.where(floor > 0, floor, 1.0)


@partial(jax.jit, static_argnames="cfg")
def _newton_solve(cfg, params, guide, s0, U, X, K, w, n_iter):
    """``n_iter`` iterations of box-constrained iLQR from ``s0``.

    The start is the previous plan ``U`` replayed with its feedback ``K``
    around its states ``X`` (an unstable plant drifts off an open-loop
    replay), or the guide's rollout when that costs less. Returns the plan,
    its states, its feedback gains and its cost."""
    m = cfg.action_dim

    def rollout(V):
        def f(s, vw):
            s2, c = _stage(cfg, params, s, vw[0])
            return s2, (s, vw[1] * c)

        _, (S, c) = jax.lax.scan(f, s0, (V, w))
        return S, jnp.sum(c)

    def closed_loop(Ub, Xb, Kb, alpha, kff):
        def f(s, inp):
            ub, kk, KK, xb = inp
            v = jnp.clip(ub + alpha * kk + KK @ (_to_x(s) - xb), -1.0, 1.0)
            return _stage(cfg, params, s, v)[0], v

        return jax.lax.scan(f, s0, (Ub, kff, Kb, Xb))[1]

    U = closed_loop(U, X, K, 0.0, jnp.zeros_like(U))
    S, J = rollout(U)
    if cfg.guide_step is not None:
        gparams, gstate = guide

        def g(carry, _):
            s, gs = carry
            u, gs = cfg.guide_step(gparams, gs, cfg.env.get_obs(s, params))
            v = jnp.clip(jnp.reshape(u, (m,)), -1.0, 1.0)
            if cfg.action_unmap is not None:
                v = jnp.clip(jnp.reshape(cfg.action_unmap(s, v, params), (m,)), -1, 1)
            return (_stage(cfg, params, s, v)[0], gs), v

        Ug = jax.lax.scan(g, (s0, gstate), None, length=cfg.horizon)[1]
        Sg, Jg = rollout(Ug)
        take = Jg < J
        U, J = jnp.where(take, Ug, U), jnp.where(take, Jg, J)
        S = jax.tree_util.tree_map(lambda a, b: jnp.where(take, a, b), Sg, S)

    def linearise(s, v, wk):
        x = _to_x(s)
        nx = x.shape[0]

        def split(z):
            return _from_x(z[:nx], s), z[nx:]

        z = jnp.concatenate([x, v])
        F = jax.jacfwd(lambda z: _to_x(_stage(cfg, params, *split(z))[0]))(z)
        cost = lambda z: wk * _stage(cfg, params, *split(z))[1]  # noqa: E731
        g, Hc = jax.grad(cost)(z), jax.hessian(cost)(z)
        return (
            F[:, :nx],
            F[:, nx:],
            g[:nx],
            g[nx:],
            Hc[:nx, :nx],
            Hc[nx:, nx:],
            Hc[nx:, :nx],
        )

    def iteration(_, carry):
        U, S, J, mu, K = carry
        A, B, lx, lu, lxx, luu, lux = jax.vmap(linearise)(S, U, w)

        def backward(carry, k):
            Vx, Vxx = carry
            Qx, Qu = lx[k] + A[k].T @ Vx, lu[k] + B[k].T @ Vx
            Qxx = lxx[k] + A[k].T @ Vxx @ A[k]
            Quu = luu[k] + B[k].T @ Vxx @ B[k]
            Quu = 0.5 * (Quu + Quu.T)
            Qux = lux[k] + B[k].T @ Vxx @ A[k]
            # Levenberg-Marquardt, relative to the curvature at this step.
            scale = jnp.mean(jnp.abs(jnp.diag(Quu)))
            reg = Quu + (mu * scale + 1e-12 * (1.0 + scale)) * jnp.eye(m)
            kff, free = _box_qp(reg, Qu, -1.0 - U[k], 1.0 - U[k])
            Hf = free[:, None] * reg * free[None, :] + jnp.diag(1.0 - free)
            Kk = -free[:, None] * jnp.linalg.solve(Hf, free[:, None] * Qux)
            Vx = Qx + Kk.T @ Quu @ kff + Kk.T @ Qu + Qux.T @ kff
            Vxx = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
            return (Vx, 0.5 * (Vxx + Vxx.T)), (kff, Kk)

        nx = lx.shape[1]
        steps = jnp.arange(cfg.horizon)[::-1]
        zeros = (jnp.zeros(nx), jnp.zeros((nx, nx)))
        kff, Knew = jax.tree_util.tree_map(
            lambda a: a[::-1], jax.lax.scan(backward, zeros, steps)[1]
        )
        Xb = jax.vmap(_to_x)(S)
        alphas = jnp.asarray(_LINE_SEARCH)
        Us = jax.vmap(lambda a: closed_loop(U, Xb, Knew, a, kff))(alphas)
        Ss, Js = jax.vmap(rollout)(Us)
        Js = jnp.where(jnp.isfinite(Js), Js, jnp.inf)
        best = jnp.argmin(Js)
        better = Js[best] < J
        pick = lambda new, old: jnp.where(better, new, old)  # noqa: E731
        S = jax.tree_util.tree_map(lambda a, b: pick(a[best], b), Ss, S)
        # A full step taken relaxes the damping; a failed search raises it.
        mu = jnp.where(better, jnp.where(best == 0, mu * 0.3, mu), mu * 10.0)
        mu = jnp.clip(mu, 1e-9, 1e6)
        return pick(Us[best], U), S, pick(Js[best], J), mu, pick(Knew, K)

    U, S, J, _, K = jax.lax.fori_loop(
        0, n_iter, iteration, (U, S, J, jnp.float32(1e-6), K)
    )
    return U, jax.vmap(_to_x)(S), K, J


class NewtonMPC:
    """Second-order MPC: box-constrained iterative LQR on ``step_env``.

    Each solve is Newton's method on the action sequence: the step is
    linearised along the plan, the stage cost expanded to second order, and
    a Riccati sweep backwards gives a step and a feedback gain at every
    stage; the line search rolls the plant forward under that feedback, so
    an unstable plant is not driven off by an open-loop replay. The step
    solves a small box-constrained QP, exactly (Tassa, Mansard and Todorov,
    ICRA 2014), so the actions stay in ``[-1, 1]`` and a bound is not
    absorbing as it is for a projected gradient. Near the optimum it
    converges quadratically, which is what lets it hold a plant to float32
    rounding, as the IPOPT oracles it replaced did, where ``GradientMPC``
    leaves a step-size offset.

    The model is the environment's own ``step_env`` on the full state, with
    ``plan_params``: no restated physics and no model error beyond the
    noise. The stage cost is the step's reward in floor units, or
    ``stage_fn(env, state, action, params) -> (next_state, cost)`` where the
    reward needs a planner's version (a smooth stand-in for a kink, a
    barrier). ``action_map(state, v, params)`` turns a decision ``v`` in
    ``[-1, 1]`` into the environment's action, for actuators planned by rate
    (``action_unmap`` inverts it for the guide).

    ``weights``, if given, weighs each step of the episode, read from the
    step the plan starts at: zeros past the episode's end make a plan whose
    ``horizon`` is the episode shrink with it. ``guide`` is a pure
    controller (``experts.pid.Controller``): each solve starts from the
    better of the previous plan and the guide's rollout, so the plan is never
    worse than the guide under the planner's model. Every solve runs a fixed
    number of iterations, ``n_iter_cold`` for an episode's first and
    ``n_iter`` after, so a record reproduces on another machine; both are
    arguments, not compile-time constants, and every planner with the same
    configuration shares one compilation.
    """

    def __init__(
        self,
        env,
        params,
        horizon: int,
        n_iter: int = 10,
        n_iter_cold: int | None = None,
        action_dim: int = 1,
        weights=None,
        resolve_every: int = 1,
        guide=None,
        stage_fn=None,
        action_map=None,
        action_unmap=None,
    ):
        self.env, self.params = env, params
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.n_iter = int(n_iter)
        self.n_iter_cold = self.n_iter if n_iter_cold is None else int(n_iter_cold)
        self.weights = None if weights is None else np.asarray(weights, np.float32)
        self.resolve_every = max(int(resolve_every), 1)
        self.guide = guide
        self._cfg = _NewtonConfig(
            env=env,
            horizon=self.horizon,
            action_dim=self.action_dim,
            stage_fn=stage_fn,
            action_map=action_map,
            action_unmap=action_unmap,
            guide_step=None if guide is None else guide.step,
        )
        self.solve_calls = 0
        self.solve_iters = 0
        self.reset()

    def reset(self):
        """Forget the plan, so the next step solves from a cold start."""
        self._U = None
        self._t0 = 0

    def solver_report(self) -> dict:
        """Solves performed so far, and the fixed iterations each ran."""
        calls = max(self.solve_calls, 1)
        return {
            "solver_calls": self.solve_calls,
            "solver_mean_iters": round(self.solve_iters / calls, 1),
        }

    def _weights(self, t: int) -> jnp.ndarray:
        if self.weights is None:
            return jnp.ones(self.horizon, jnp.float32)
        w = self.weights[t : t + self.horizon]
        return jnp.asarray(np.pad(w, (0, self.horizon - w.shape[0])))

    def _solve(self, state, t: int, cold: bool) -> None:
        H, m = self.horizon, self.action_dim
        if cold:
            U = jnp.zeros((H, m))
            X = jnp.broadcast_to(_to_x(state), (H,) + _to_x(state).shape)
            K = jnp.zeros((H, m, X.shape[1]))
        else:
            shift = t - self._t0

            def shifted(a):
                return jnp.concatenate([a[shift:], jnp.repeat(a[-1:], shift, 0)])

            U, X, K = shifted(self._U), shifted(self._X), shifted(self._K)
        n = self.n_iter_cold if cold else self.n_iter
        guide = None
        if self.guide is not None:
            guide = (self.guide.params, self.guide.init_state)
        self._U, self._X, self._K, _ = _newton_solve(
            self._cfg, self.params, guide, state, U, X, K, self._weights(t), n
        )
        self._t0 = t
        self.solve_calls += 1
        self.solve_iters += n

    def step(self, _obs, state):
        """The action for the true ``state``; ``_obs`` is ignored."""
        t = int(state.time)
        cold = self._U is None or t < self._t0 or t - self._t0 >= self.horizon
        if cold or (t - self._t0) % self.resolve_every == 0:
            self._solve(state, t, cold)
        v = self._U[t - self._t0]
        cfg = self._cfg
        u = v if cfg.action_map is None else cfg.action_map(state, v, self.params)
        u = np.clip(np.asarray(u, dtype=float).reshape(self.action_dim), -1.0, 1.0)
        return float(u[0]) if self.action_dim == 1 else u


def protocol_burn_in(name: str, params) -> int:
    """The first step the protocol scores on task ``name`` at these params.

    ``eval.scored_burn_in``: the hold row's burn-in, capped at half the
    episode, which is what ``eval.evaluate_controller`` scores from. An
    oracle that weights the scored window reads it here so that it
    optimises the window the protocol scores. The burn-in is in both of
    every task's record fingerprints (``provenance``), so a record taken
    under one burn-in is stale under another.
    """
    from target_gym.eval import scored_burn_in

    return scored_burn_in(name, params)


def window_weights(steps: int, window_start: int, pre_weight: float, horizon: int):
    """``NewtonMPC`` weights for a plan over the rest of a ``steps``-step
    episode that the protocol scores from ``window_start``.

    Steps in the window weigh 1 and earlier ones ``pre_weight``: a small
    weight keeps the approach from being free (a window-only objective is
    degenerate) without letting it trade against the window. Steps past the
    episode's end weigh 0, so a plan ``horizon`` long shrinks with the
    episode. Indexed by the step of the action, which sets the state the
    step's cost is charged on.
    """
    w = np.zeros(int(steps) + int(horizon), dtype=np.float32)
    w[: int(steps)] = 1.0
    w[: int(window_start)] = pre_weight
    return w


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
