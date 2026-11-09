"""The receding-horizon oracle of every task on the shared solver.

A task states its planner problem as a ``TaskModel`` in its own ``oracle.py``;
``Oracle`` runs it as the accepted specs execute it:

- horizon: the task's summary in ``data/oracle_horizons/`` (Cooper's T_c
  and P), shrinking as H_k = min(P, N_ep - k) at sample k
  (oracle-horizon-protocol.md, H12);
- execution: one solve every T_c, its first move held over the T_c, on the
  noise-free params the caller passes (``plan_params``); a cold start puts
  every planned state at the measured one and holds the reset actuators
  (oracle-solver.md, G8), a warm start shifts the last plan, a warm solve
  that fails or hits the cap is solved again cold (G11), and a solve that
  still fails applies what it started from, so the previous plan's next move
  (owner decision, oracle-audit.md: never the expert);
- smoothing: the K5 widths in ``data/oracle_widths/``, with K7's
  continuation on cold solves (oracle-smoothing-protocol.md);
- certificate: per solve, the solver's (KKT error, iterations, capped,
  failed, violation) and K8's replay of the planned moves on the exact
  ``step_env`` from the same start.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import records
from target_gym.oracles.horizon import episode_samples, horizon_at
from target_gym.oracles.shooting import Shooting, StageModel

try:
    from jax import enable_x64
except ImportError:  # JAX 0.6; 0.10 removed the experimental name
    from jax.experimental import enable_x64


#: G10: the tie-break's worst case over an episode, every move of every solve
#: at full size, in reward units: what an error of half the resolution costs
#: in one step, (q / 2)^2 / e_floor^2 with e_floor clamped at the resolution q.
TIE_BUDGET = 0.25


class TaskModel(NamedTuple):
    """A task's planner problem. ``step(d, x, u, t)`` is one task step of the
    planner's model from ``x`` under the raw move ``u``, ``t`` steps into the
    plan, and returns ``(x_next, cost, y, tracked, guards)``: the reward's
    smooth terms on the state reached, the hinge arguments then the limited
    quantities (``StageModel``), the tracked outputs, and the arguments of the
    dropped guards (T4), each at its edge at 0 and inside above it. ``d``
    holds ``task`` (``data`` of the solve's start), ``widths`` (the T5
    sites') and ``end`` (the steps the plan covers). ``move_rows(x, u,
    steps)``: rows >= 0 bounding each move from its stage's state over a
    hold of ``steps`` task steps (T1r, ``StageModel``), and
    ``move_scale(steps)`` each move's reach over that hold, its range if not
    given. A reward not in floors gives ``tie_budget``, ``TIE_BUDGET`` in its
    units. A plant with a spare move (G12) gives ``rest(x)``, the state's
    distance from its resting target, 1 at the edge of its limit: its square
    costs the plan the budget over an episode spent there, and K8's replay
    never sees it. ``action(state, u)`` is what the env is sent, each task step,
    for a move that is not a raw command, such as a rate (``oracles.rates``):
    the raw move itself if not given, and K8's replay sends the same."""

    nx: int
    nu: int
    state_x: Callable  # env state -> x
    data: Callable  # env state -> the solve's exogenous data
    step: Callable
    hold: Callable  # env state -> the raw moves that hold its actuators
    tracked: Callable  # env state -> its tracked outputs
    resolution: tuple  # q_y of each tracked output
    hinges: tuple = ()
    limits: tuple = ()
    move_rows: Callable | None = None
    move_scale: Callable | None = None
    tie_budget: float = TIE_BUDGET
    rest: Callable | None = None
    action: Callable | None = None


def float64(tree):
    """Floats to float64 and int32 to int64, for a solve under x64."""

    def cast(a):
        a = jnp.asarray(a)
        if jnp.issubdtype(a.dtype, jnp.floating):
            return a.astype(jnp.float64)
        return a.astype(jnp.int64) if a.dtype == jnp.int32 else a

    return jax.tree_util.tree_map(cast, tree)


def scored_run(fn, *args):
    """``fn(*args)`` as the scored run computes it: in float32 with x64 off,
    whatever the caller's mode. ``jax.random.normal`` draws other values under
    x64, so a disturbance path read off ``step_env`` under x64 is not the
    episode's (a ceiling's weather or turbulence). Called eagerly, not traced."""

    def cast(a):
        a = np.asarray(a)
        if np.issubdtype(a.dtype, np.floating):
            return a.astype(np.float32)
        return a.astype(np.int32) if a.dtype == np.int64 else a

    with enable_x64(False):
        return jax.tree_util.tree_map(
            np.asarray, fn(*jax.tree_util.tree_map(cast, args))
        )


def raw_of(value, lo, hi):
    """The raw move in [-1, 1] that ``convert_raw_action_to_range`` maps to
    ``value``, clipped to the box."""
    return jnp.clip(2.0 * (value - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def softplus_clip(z, lo, hi, w):
    """clip(z, lo, hi) smoothed by the softplus of width ``w`` (K2):
    z - p_w(z - hi) + p_w(lo - z); the exact clip at ``w`` = 0, with the
    width guarded so that branch's derivative stays finite (JAX FAQ)."""
    v = jnp.where(w > 0.0, w, 1.0)
    smooth = z - v * jax.nn.softplus((z - hi) / v) + v * jax.nn.softplus((lo - z) / v)
    return jnp.where(w > 0.0, smooth, jnp.clip(z, lo, hi))


def hinge_costs(hinges, y):
    """The reward's value of the epigraph terms at hinge arguments ``y``."""
    out = jnp.zeros((), y.dtype)
    for i, h in enumerate(hinges):
        r = jnp.abs(y[i]) if h.absolute else y[i]
        out = out + h.weight * jnp.maximum(r - h.tol, 0.0) ** h.power
    return out


class Oracle:
    """The oracle of task ``name``: ``step(obs, state)`` reads the true state.

    ``params`` are the planner's (noise zeroed by ``plan_params``, as the
    runners pass them), cast to float64; ``build(env, params)`` states the
    task's ``TaskModel`` on them, and K8's replay runs ``env.step_env`` on
    them too.
    ``certificates`` keeps every solve's certificate, ``solver_report`` sums
    them for the record.
    """

    def __init__(self, env, params, name: str, build: Callable):
        self.env = env
        horizon = records.load_row("oracle_horizons", name)
        self.tc = int(horizon["T_c_steps"])
        self.n_ep = episode_samples(int(params.max_steps_in_episode), self.tc)
        self.P = min(int(horizon["P"]), self.n_ep)  # H12: never past the episode
        smoothing = records.load_row("oracle_widths", name)
        sites = smoothing["sites"].values()
        self.widths = np.array([s["width"] for s in sites], float)
        self.scales = np.array([s["scale"] for s in sites], float)
        with enable_x64():
            self.params = float64(params)
            self.model = model = build(env, self.params)
            # K11: each limit backed off by the planner's error on it (none
            # recorded where nothing is smoothed: the model is exact there).
            back = smoothing.get("limit_backoffs") or [0.0] * len(model.limits)
            rows, rest, scale = model.move_rows, model.rest, model.move_scale
            resting = model.tie_budget / float(params.max_steps_in_episode)

            def planned(d, x, u, t):
                x1, cost, y = model.step(d, x, u, t)[:3]
                return x1, cost if rest is None else cost + resting * rest(x1) ** 2, y

            stage = StageModel(
                step=planned, block=self.tc,
                lo=(-1.0,) * model.nu, hi=(1.0,) * model.nu, hinges=model.hinges,
                limits=tuple(
                    q._replace(backoff=float(b)) for q, b in zip(model.limits, back)
                ),
                move_rows=None if rows is None else lambda x, u: rows(x, u, self.tc),
                move_scale=None if scale is None else scale(self.tc),
                tie=model.tie_budget / (model.nu * self.n_ep),
            )  # fmt: skip
            self.shooting = Shooting(stage, self.P, model.nx)
        self._certify = jax.jit(self._replay)
        self.certificates: list[dict] = []
        self.reset()

    def reset(self):
        """Forget the plan: the next step solves cold."""
        self._plan, self._move, self._last = None, None, None

    def _rungs(self) -> list:
        """K7: the widths of a cold solve's continuation, a tenth of each
        site's scale down to its width by tenths; one rung with no site."""
        if not len(self.widths):
            return [self.widths]
        n = int(np.max(np.round(np.log10(self.scales / self.widths))))
        return [
            np.maximum(self.scales * 10.0**-r, self.widths) for r in range(1, n + 1)
        ]

    def step(self, _obs, state):
        """The raw move for the true ``state``; ``_obs`` is ignored."""
        t = int(state.time)
        clock = int(getattr(state, "block_clock", t))
        # A restart (a trip's fresh draw, or a new episode) moves a clock back.
        last = self._last or (t, clock)
        cold = self._plan is None or t < last[0] or clock < last[1]
        if cold or t % self.tc == 0:
            self._solve(state, t, cold)
        self._last = (t, clock)
        u = np.clip(np.asarray(self._move, float), -1.0, 1.0)
        if self.model.action is not None:
            with enable_x64():
                u = self.model.action(float64(state), jnp.asarray(u))
            u = np.clip(np.asarray(u, float), -1.0, 1.0)
        return float(u[0]) if u.size == 1 else u

    def _solve(self, state, t: int, cold: bool):
        k = min(t // self.tc, self.n_ep - 1)
        active = horizon_at(self.P, self.n_ep, k)
        with enable_x64():
            st = float64(state)
            x0, task = self.model.state_x(st), self.model.data(st)
            d = {"task": task, "end": active * self.tc}
            hold = self.model.hold(st)
            plan = self._attempt(d, st, x0, hold, active, cold)
            c = plan.certificate
            if not cold and (bool(c.failed) or bool(c.capped)):
                # G11: a warm start jams when the active set must change, its
                # slacks a hair off their bounds, where a cold solve succeeds.
                self.certificates[-1]["retried"] = True
                again = self._attempt(d, st, x0, hold, active, True)
                if not bool(again.certificate.failed):
                    plan = again
        self._plan = plan
        self._move = np.asarray(self.shooting.moves(self._plan)[0])

    def _attempt(self, d, st, x0, hold, active: int, cold: bool):
        """One solve, through K7's rungs when cold; its last plan."""
        plans, rungs = [], self._rungs() if cold else [self.widths]
        for i, widths in enumerate(rungs):
            d["widths"] = jnp.asarray(widths)
            t0 = time.perf_counter()
            if i == 0:
                args = () if cold else (self._plan,)
                plan = self.shooting.solve(d, x0, hold, *args, active=active)
            else:
                plan = self.shooting.resolve(d, x0, hold, plans[-1], active=active)
            jax.block_until_ready(plan.z)
            plans.append(plan)
            seconds = time.perf_counter() - t0
            self._record(plan, seconds, d, st, x0, active, i == len(rungs) - 1)
        return plans[-1]

    def _record(self, plan, seconds, d, st, x0, active, applies):
        c = plan.certificate
        moves = self.shooting.moves(plan)
        replay = {
            k: v.item() for k, v in self._certify(d, st, x0, moves, active).items()
        }
        self.certificates.append({
            "seconds": seconds, "iterations": int(c.iterations),
            "error": float(c.error), "capped": bool(c.capped),
            "failed": bool(c.failed), "violation": float(c.violation),
            "shift": float(c.shift), "retried": False, "applies": applies, **replay,
        })  # fmt: skip

    def _replay(self, d, st, x0, moves, active):
        """K8 for the planned ``moves``, each held one T_c: the planner's model
        and ``step_env`` run from the same start, over the planned steps."""
        m, b, n = self.model, self.tc, self.P * self.tc
        ts = jnp.arange(n)
        live, us = ts < active * b, moves[ts // b]

        def planned(x, t):
            x1, cost, y, tracked, guards = m.step(d, x, us[t], t)
            low = jnp.min(jnp.asarray(guards, x.dtype), initial=jnp.inf)
            return x1, (cost + hinge_costs(m.hinges, y), jnp.asarray(tracked), low)

        key = jax.random.PRNGKey(0)  # the noise is zeroed; only a trip draws

        def replayed(s, t):
            sent = us[t] if m.action is None else m.action(s, us[t])
            _, s1, r, _, info = self.env.step_env(key, s, sent, self.params)
            s1 = jax.tree_util.tree_map(lambda a, b: a.astype(b.dtype), s1, s)
            return s1, (r, jnp.asarray(m.tracked(s1)), info["tripped"])

        cost, y_plan, low = jax.lax.scan(planned, x0, ts)[1]
        r, y_env, trip = jax.lax.scan(replayed, st, ts)[1]
        R_plan = -jnp.sum(jnp.where(live, cost, 0.0))
        R_env = jnp.sum(jnp.where(live, r, 0.0))
        q = jnp.asarray(m.resolution, x0.dtype)
        gap = jnp.where(live[:, None], jnp.abs(y_plan - y_env) / q, 0.0)
        return {
            "reward_plan": R_plan,
            "reward_replay": R_env,
            # Relative to the larger reward, or to 1, a step at the floor.
            "replay_gap": jnp.abs(R_plan - R_env)
            / jnp.maximum(1.0, jnp.maximum(jnp.abs(R_plan), jnp.abs(R_env))),
            "output_gap": jnp.max(gap),
            "replay_trip": jnp.any(live & trip),
            "guard_crossed": jnp.any(live & (low <= 0.0)),
        }

    def solver_report(self) -> dict:
        """The certificates summed for the record: the solves, those a cold
        retry superseded (G11), and over the rest the failed and capped ones,
        the worst KKT error and row violation over the finite values (a failed
        solve's start may hold none: -1 then). Over the plans that applied,
        K7's wider rungs left out: the worst replay gaps, and the solves whose
        replay tripped or whose plan crossed a dropped guard (K9's flag). Mean
        iterations are over every solve."""
        every = self.certificates
        c = [x for x in every if not x["retried"]]
        applied = [x for x in c if x["applies"]]  # K7's wider rungs only start it
        calls = max(len(every), 1)

        def worst(key, solves=c):
            return max((v for x in solves if np.isfinite(v := x[key])), default=-1.0)

        return {
            "solver_calls": len(every),
            "solver_retries": len(every) - len(c),
            "solver_failures": sum(x["failed"] for x in c),
            "solver_capped": sum(x["capped"] for x in c),
            "solver_mean_iters": round(sum(x["iterations"] for x in every) / calls, 1),
            "solver_max_error": worst("error", [x for x in c if not x["failed"]]),
            "solver_max_violation": worst("violation"),
            "solver_max_replay_gap": worst("replay_gap", applied),
            "solver_max_output_gap": worst("output_gap", applied),
            "solver_replay_trips": sum(x["replay_trip"] for x in applied),
            "solver_guard_flags": sum(x["guard_crossed"] for x in applied),
        }
