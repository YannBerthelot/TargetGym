"""Run any controller on the benchmark: one policy interface, every task.

A controller is a :class:`Policy` - batched over episodes, its memory an
explicit carry::

    carry = policy.reset(key, obs, task)            # obs (n, obs_dim)
    action, carry = policy.act(carry, obs, t)       # action (n, n_act)

``task`` is a :class:`TaskInfo`: what the controller is told about the task
(the action bounds, which observation channels are the tracked outputs and
their targets, the sample time, the tracking floor, the episode length, and
any further declared information a benchmark family adds in ``extras``).
The true state is withheld; a policy that sets ``oracle = True`` - a
full-state ceiling, as the shipped MPC is - receives it as ``act(...,
state=state)``, so a benchmark row that sees more than a plant instruments
says so in its own code.

:func:`run_policy` runs a policy on one task, :func:`run_policy_on_benchmark`
on every registered task (or a chosen few), and both score the episodes with
this library's protocol (:mod:`target_gym.eval`). Episodes are PAIRED: the
reset states and every step's key depend only on ``(key, n, T)``, so two
policies run on the same key meet the same plants, the same set-point
schedules and the same noise, and their per-episode costs can be compared
episode by episode. The convention: ``k_reset, k_steps = split(key)``, the
episodes reset on ``split(k_reset, n)`` and step ``t`` of episode ``i`` on
``split(k_steps, T n).reshape(T, n, 2)[t, i]`` (the convention of
TargetFoundation's paired runs, which a policy run here reproduces action for
action, the rewards to float rounding); the policy's own key is
``fold_in(key, 1)``. Each step gets its own key (gymnax's convention; the
older single-episode helpers here reuse one key for a whole episode, so their
numbers on a stochastic plant are not these).

A policy may be written in JAX (jit its own ``act``; the loop runs on the
host so a policy that cannot be traced still fits) or in plain numpy.
:class:`CallablePolicy` adapts the shipped ``(obs, state) -> action``
controllers (``runners.baseline_policy``), so the PID reference runs through
the same function as everything else::

    from target_gym.benchmark import run_policy_on_benchmark, shipped_policy
    pid = run_policy_on_benchmark(shipped_policy("pid"), n_episodes=8)
    mine = run_policy_on_benchmark(MyPolicy(), n_episodes=8)
    mine["cstr"]["metrics"]["gain"], pid["cstr"]["metrics"]["gain"]

A policy that cannot run a task (a single-input controller on a
two-actuator plant) raises :class:`Unsupported` from ``reset``; the task's
row then carries the reason instead of numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

from target_gym.eval import Episode, evaluate, hold_settings


class Unsupported(Exception):
    """Raised by ``Policy.reset`` for a task the policy cannot run."""


@dataclass(frozen=True)
class TaskInfo:
    """What a controller is told about a task, per episode where it varies.

    ``e_floor`` is ``(n, n_out)`` and ``dt`` / ``max_steps`` ``(n,)``, so a
    family whose episodes are different plants (per-episode parameters) is
    described by the same object as a single plant run ``n`` times.
    ``extras`` carries declared information a benchmark family adds beyond
    this library's (a trip envelope on the error, the loop's direction); a
    policy that needs one of them names it."""

    name: str
    n_episodes: int
    n_act: int
    value_index: tuple[int, ...]
    target_index: tuple[int, ...]
    action_low: np.ndarray
    action_high: np.ndarray
    dt: np.ndarray
    e_floor: np.ndarray
    max_steps: np.ndarray
    extras: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    """A controller as the benchmark runs it (module docstring)."""

    name: str

    def reset(self, key: Any, obs: Any, task: TaskInfo) -> Any: ...

    def act(self, carry: Any, obs: Any, t: int) -> tuple[Any, Any]: ...


def _as_tuple(index) -> tuple[int, ...]:
    return (
        tuple(int(i) for i in index)
        if isinstance(index, (tuple, list))
        else (int(index),)
    )


def task_info(
    name: str,
    env,
    params,
    n: int,
    per_episode_params: bool = False,
    extras: Mapping[str, Any] | None = None,
) -> TaskInfo:
    """The :class:`TaskInfo` of ``n`` episodes of ``env`` under ``params``
    (shared, or batched ``(n, ...)`` with ``per_episode_params``)."""
    import jax

    from target_gym.registry import control_step_seconds

    first = jax.tree.map(lambda x: x[0], params) if per_episode_params else params
    space = env.action_space(first)
    shape = space.shape if space.shape else (1,)
    vi, ti = _as_tuple(env.obs_value_index), _as_tuple(env.obs_target_index)

    def rows(value, width: int | None = None) -> np.ndarray:
        """``(n, width)``: a shared value repeated, a per-episode one as is."""
        v = np.asarray(value, float).reshape((n if per_episode_params else 1), -1)
        return np.broadcast_to(v, (n, v.shape[1] if width is None else width)).copy()

    if per_episode_params:
        dt = np.asarray(
            [
                control_step_seconds(env, jax.tree.map(lambda x, i=i: x[i], params))
                for i in range(n)
            ]
        )
    else:
        dt = np.full(n, control_step_seconds(env, params))
    return TaskInfo(
        name=name,
        n_episodes=n,
        n_act=int(np.prod(shape)),
        value_index=vi,
        target_index=ti,
        action_low=np.broadcast_to(np.asarray(space.low, float), shape).copy(),
        action_high=np.broadcast_to(np.asarray(space.high, float), shape).copy(),
        dt=dt,
        e_floor=rows(getattr(params, "e_floor", np.inf), len(vi)),
        max_steps=rows(params.max_steps_in_episode)[:, 0].astype(int),
        extras=dict(extras or {}),
    )


def target_changes(targets: np.ndarray, first: np.ndarray) -> np.ndarray:
    """``(T,)`` bool: a new target became active at the step, by
    :func:`target_gym.eval.run_episode`'s rule - some target moved by more than
    5 % of its first value's magnitude since the previous step. ``targets``
    ``(T, n_out)``, ``first`` the targets at reset."""
    targets = np.asarray(targets, float)
    prev = np.vstack([np.asarray(first, float)[None], targets[:-1]])
    span = np.maximum(np.abs(targets[0]), 1e-9)
    return (np.abs(targets - prev) > 0.05 * span).any(axis=1)


def run_policy(
    policy: Policy,
    env,
    params,
    task: TaskInfo,
    key,
    n_steps: int | None = None,
    per_episode_params: bool = False,
    record: Mapping[str, Callable[[Any, Any, Any], Any]] | None = None,
) -> dict[str, Any]:
    """``task.n_episodes`` closed-loop episodes of ``policy`` on ``env``.

    Returns per-episode, per-step arrays ``(n, T)``: ``reward``, ``tripped``
    (``info["tripped"]``, False where the plant reports none), ``in_band``
    (every tracked error within ``e_floor``), ``target_change``, ``action``
    ``(n, T, n_act)``, ``value`` / ``target`` ``(n, T, n_out)`` (the measured
    outputs and their targets after the step), ``terminated`` (latched from
    the step on which the episode terminated), and ``info_error`` where the
    plant reports its true tracking error; per episode ``length`` (its own
    ``max_steps``, cut at a termination); ``terms`` (name -> ``(n, T)``) where
    the plant reports its reward terms; the policy's final ``carry``;
    ``wall_s``. A terminated episode is
    frozen: its plant stops, its reward is 0 afterwards and its record ends.

    ``record``: named per-step quantities a study wants beside the reward,
    each ``fn(state, params, tripped) -> scalar`` for one episode (vmapped
    here) on the state after the step, its parameters, and whether the step
    tripped (False where the plant reports no trips; after a trip the state is
    the restarted plant, so ``tripped`` is how a function scores that step).
    Returned as ``records`` (name -> ``(n, T)``)."""
    import jax
    import jax.numpy as jnp

    n = task.n_episodes
    T = int(task.max_steps.max()) if n_steps is None else int(n_steps)
    pa = 0 if per_episode_params else None
    k_reset, k_steps = jax.random.split(key)
    step_keys = jax.random.split(k_steps, T * n).reshape(T, n, 2)
    reset = jax.jit(jax.vmap(env.reset_env, in_axes=(0, pa)))
    step = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, pa)))
    obs, state = reset(jax.random.split(k_reset, n), params)
    first_target = np.asarray(obs)[:, list(task.target_index)]
    carry = policy.reset(jax.random.fold_in(key, 1), obs, task)
    oracle = bool(getattr(policy, "oracle", False))
    terms_fn = getattr(env, "reward_terms", None)
    terms_b = None if terms_fn is None else jax.jit(jax.vmap(terms_fn, in_axes=(0, pa)))
    record_b = {
        k: jax.jit(jax.vmap(fn, in_axes=(0, pa, 0))) for k, fn in (record or {}).items()
    }
    records_rec: dict[str, list] = {k: [] for k in record_b}
    vi, ti = list(task.value_index), list(task.target_index)
    done = np.zeros(n, bool)
    length = np.minimum(task.max_steps, T).astype(int)
    rec: dict[str, list] = {
        k: [] for k in ("reward", "tripped", "terminated", "action", "value", "target")
    }
    err_rec, terms_rec = [], {}
    t0 = time.perf_counter()
    for t in range(T):
        if oracle:
            action, carry = policy.act(carry, obs, t, state=state)
        else:
            action, carry = policy.act(carry, obs, t)
        action = jnp.reshape(jnp.asarray(action, jnp.float32), (n, task.n_act))
        keys = step_keys[t]
        obs_n, state_n, r, term, info = step(keys, state, action, params)
        live = jnp.asarray(~done)

        def freeze(new, old, live=live):
            return jnp.where(
                jnp.reshape(live, live.shape + (1,) * (new.ndim - 1)), new, old
            )

        obs, state = freeze(obs_n, obs), jax.tree.map(freeze, state_n, state)
        o = np.asarray(obs)
        rec["reward"].append(np.where(done, 0.0, np.asarray(r, float)))
        tripped = info.get("tripped") if isinstance(info, dict) else None
        tripped = np.zeros(n, bool) if tripped is None else np.asarray(tripped, bool)
        rec["tripped"].append(tripped & ~done)
        for k, fn in record_b.items():
            records_rec[k].append(
                np.asarray(fn(state, params, jnp.asarray(tripped)), float)
            )
        rec["action"].append(np.asarray(action))
        rec["value"].append(o[:, vi])
        rec["target"].append(o[:, ti])
        if isinstance(info, dict) and "error" in info:
            err_rec.append(np.asarray(info["error"], float))
        step_terms = info.get("reward_terms") if isinstance(info, dict) else None
        if step_terms is None and terms_b is not None:
            step_terms = terms_b(state, params)
        for k, v in (step_terms or {}).items():
            terms_rec.setdefault(k, []).append(np.asarray(v, float))
        newly = np.asarray(term, bool) & ~done
        length = np.where(newly, np.minimum(length, t + 1), length)
        done |= np.asarray(term, bool)
        rec["terminated"].append(done.copy())
    out: dict[str, Any] = {k: np.stack(v, axis=1) for k, v in rec.items()}
    err = np.abs(out["value"] - out["target"])
    band = task.e_floor[:, None, :]
    out["in_band"] = (err <= band).all(axis=-1)
    out["target_change"] = np.stack(
        [target_changes(out["target"][i], first_target[i]) for i in range(n)]
    )
    if err_rec:
        out["info_error"] = np.stack(err_rec, axis=1)
    out["terms"] = {k: np.stack(v, axis=1) for k, v in terms_rec.items()}
    out["records"] = {k: np.stack(v, axis=1) for k, v in records_rec.items()}
    out["length"] = length
    out["carry"] = carry  # the policy's final carry, for its own diagnostics
    out["wall_s"] = time.perf_counter() - t0
    return out


def episodes_of(rollout: Mapping[str, Any]) -> list[Episode]:
    """The protocol's :class:`~target_gym.eval.Episode` records of a
    :func:`run_policy` rollout, each cut at its own length."""
    out = []
    for i, L in enumerate(np.asarray(rollout["length"], int)):
        out.append(
            Episode(
                cost=-np.asarray(rollout["reward"][i, :L], float),
                in_band=np.asarray(rollout["in_band"][i, :L], bool),
                target_change=np.asarray(rollout["target_change"][i, :L], bool),
                failed=np.asarray(rollout["tripped"][i, :L], bool),
                terms={
                    k: np.asarray(v[i, :L], float) for k, v in rollout["terms"].items()
                },
            )
        )
    return out


def score(
    rollout: Mapping[str, Any],
    burn_in,
    rho_floor: float | None = None,
) -> dict[str, Any]:
    """The protocol's numbers for a rollout: ``metrics`` (``eval.evaluate``
    over the episodes pooled, the library's convention for one plant run
    several times; with a ``burn_in`` per episode, the mean over episodes of
    each one's own metrics) and the per-episode ``gain`` / ``failure_rate``
    arrays, for paired comparisons between policies run on the same seed;
    ``records``: each recorded quantity's mean over each episode's own steps
    (the burn-in included), per episode."""
    eps = episodes_of(rollout)
    b = np.asarray(burn_in, int)
    per = [
        evaluate(
            [e], burn_in=int(np.broadcast_to(b, (len(eps),))[i]), rho_floor=rho_floor
        )
        for i, e in enumerate(eps)
    ]
    if b.ndim == 0:
        metrics = evaluate(eps, burn_in=int(b), rho_floor=rho_floor)
    else:
        keys = [k for k in per[0] if np.isscalar(per[0][k])]
        metrics = {k: float(np.nanmean([m[k] for m in per])) for k in keys}
    lengths = np.asarray(rollout["length"], int)
    records = {
        k: np.asarray([v[i, :L].mean() for i, L in enumerate(lengths)], float)
        for k, v in (rollout.get("records") or {}).items()
    }
    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "gain": np.asarray([m["gain"] for m in per], float),
        "failure_rate": np.asarray([m["failure_rate"] for m in per], float),
        "records": records,
    }


def run_policy_on_benchmark(
    policy: Policy,
    tasks: Iterable[str] | None = None,
    n_episodes: int = 8,
    seed: int = 0,
    n_steps: int | None = None,
    test_params: bool = True,
    declare: Callable[[str, Any, Any], Mapping[str, Any]] | None = None,
    record: Callable[[str, Any, Any], Mapping[str, Callable] | None] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """``policy`` on every registered task (or ``tasks``), ``n_episodes`` each,
    scored by the protocol at the task's burn-in (``eval.hold_settings``,
    capped at half the episode as ``eval.evaluate_controller`` does).

    Returns per task ``{"metrics", "gain", "failure_rate", "n_episodes",
    "n_steps", "wall_s"}`` (``metrics["failure_rate"]`` pooled over the
    episodes' cycles, ``failure_rate`` per episode) and ``"diagnostics"``
    where the policy has a ``diagnostics(carry) -> dict`` method (called on
    the final carry), or ``{"unsupported": reason}`` when the policy raised
    :class:`Unsupported`. The task's key is ``fold_in(PRNGKey(seed), task
    index in the registry)``, so a task's episodes do not depend on which
    other tasks were run.

    ``test_params`` (default): the task as the registry defines it,
    ``spec.make_test_params()`` - the parameters this library records its
    baselines and measures its burn-ins on. They differ from the
    environments' defaults on every aircraft task (episodes of 200-1200 steps
    against 10 000) and on the reactor (864 against 8640); on five aircraft
    tasks they are the task itself (plane, plane_energy and plane_sine share
    one environment and differ only there). False runs ``env.default_params``,
    the environment's own defaults. ``declare(name, env, params)``: the information a
    study declares to the controller beyond this library's
    (:attr:`TaskInfo.extras`), per task. ``record(name, env, params)``: the
    per-step quantities to record on that task (:func:`run_policy`'s
    ``record``), returned per episode in the row's ``records``."""
    import jax

    from target_gym.registry import REGISTRY

    names = list(REGISTRY) if tasks is None else list(tasks)
    order = {name: i for i, name in enumerate(REGISTRY)}
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = REGISTRY[name]
        env = spec.make_env()
        params = spec.make_test_params() if test_params else env.default_params
        extras = None if declare is None else declare(name, env, params)
        info = task_info(name, env, params, n_episodes, extras=extras)
        key = jax.random.fold_in(jax.random.PRNGKey(seed), order[name])
        try:
            rollout = run_policy(
                policy, env, params, info, key, n_steps=n_steps,
                record=None if record is None else record(name, env, params),
            )  # fmt: skip
        except Unsupported as e:
            results[name] = {"unsupported": str(e)}
            if verbose:
                print(f"  {name}: unsupported ({e})", flush=True)
            continue
        T = int(rollout["reward"].shape[1])
        burn_in = min(hold_settings(name)["burn_in"], T // 2)
        row = score(rollout, burn_in, rho_floor=getattr(params, "rho_floor", None))
        row.update(n_episodes=n_episodes, n_steps=T, wall_s=float(rollout["wall_s"]))
        if hasattr(policy, "diagnostics"):
            row["diagnostics"] = policy.diagnostics(rollout["carry"])
        results[name] = row
        if verbose:
            m = row["metrics"]
            print(
                f"  {name}: gain {m['gain']:.4g}, failure rate {m['failure_rate']:.3f} "
                f"({row['wall_s']:.0f} s)",
                flush=True,
            )
    return results


class CallablePolicy:
    """A :class:`Policy` from per-episode ``(obs, state) -> action`` callables
    (``make()`` builds a fresh one for each episode at reset). They run one
    episode at a time on the host; ``oracle`` passes each its episode's true
    state (the shipped MPC reads it)."""

    def __init__(
        self,
        make: Callable[[TaskInfo], Callable | None],
        name: str,
        oracle: bool = False,
    ):
        self.make, self.name, self.oracle = make, name, oracle

    def reset(self, key, obs, task: TaskInfo):
        del key, obs
        calls = [self.make(task) for _ in range(task.n_episodes)]
        if any(c is None for c in calls):
            raise Unsupported(f"{self.name} is not shipped for {task.name}")
        return calls

    def act(self, carry, obs, t, state=None):
        import jax

        o = np.asarray(obs)
        out = []
        for i, call in enumerate(carry):
            s = None if state is None else jax.tree.map(lambda x, i=i: x[i], state)
            out.append(np.atleast_1d(np.asarray(call(o[i], s), float)))
        return np.stack(out), carry


def shipped_policy(kind: str = "pid", test_params: bool = True) -> CallablePolicy:
    """The controller this library ships for each task (``"pid"`` or
    ``"mpc"``, ``runners.baseline_policy``) as a :class:`Policy`; a task
    without one, or not registered here, is :class:`Unsupported`. The MPC
    plans on the task's parameters, so ``test_params`` must match the run's
    (:func:`run_policy_on_benchmark`: the registered task by default)."""
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy

    def make(task: TaskInfo):
        spec = REGISTRY.get(task.name)
        if spec is None:  # not one of this library's tasks: nothing is shipped for it
            return None
        params = (
            spec.make_test_params() if test_params else spec.make_env().default_params
        )
        return baseline_policy(spec, kind, params)

    return CallablePolicy(make, name=kind, oracle=(kind == "mpc"))


__all__ = [
    "CallablePolicy",
    "Policy",
    "TaskInfo",
    "Unsupported",
    "episodes_of",
    "run_policy",
    "run_policy_on_benchmark",
    "score",
    "shipped_policy",
    "target_changes",
    "task_info",
]
