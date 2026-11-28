"""Scoring runs: a task scored on its stream pinned to each scenario.

docs/proposals/long-strata.md, R2, as the owner cut it on 2026-11-24 ("a scenario,
a seed or a day of simulation stays only if removing it would hide a named
behaviour or change which controller looks better"). A task with a
:data:`PROTOCOLS` entry is scored on the scenarios it keeps, a dropped scenario's
weight going to the twin named for it; each seed is one run of the stream pinned
to the scenario, from the reset as it is, scored over ``days`` from its first
step; a seed's cost is its mean cost per step over the run (J4), and with an
``edge_charge`` the plant's stock lost from the reset to the run's last step, spread
over its steps, less what it gained with ``edge_credit`` (the reactor: the boron its
core owes, owner, 2026-11-28). Ten seeds, and ten more for a scenario whose room
interval contains 1, once (:func:`needs_more`), set before any result.

No burn-in: on the reactor the slowest thing the reference leaves is the phase of
its calm-day boron cycle, about 8 days, which never decays, so a burn-in would
remove nothing; 7 days score one cycle. Measured in docs/proposals/reactor-axial.md
(the evaluation cut).

    out = run("reactor", "baseload", "pid", seeds=seeds("reactor"))

This module is in no fingerprint: it decides how a task is scored, not what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PROTOCOLS",
    "Protocol",
    "edge",
    "needs_more",
    "policy",
    "run",
    "run_functional",
    "run_params",
    "scored_steps",
    "seeds",
    "stream",
    "weights",
]


@dataclass(frozen=True)
class Protocol:
    """``days`` scored a run, ``seeds`` a scenario and ``more`` once where the
    stop rule asks; ``folds`` maps a dropped scenario to the kept twin that takes
    its weight; ``first`` is the first seed. ``edge_charge`` names the plant's
    method that values its stock (``stored_value(state, params)``): a run is
    charged the value lost from its reset to its last step, spread over its steps,
    and with ``edge_credit`` credited the value gained (``evaluation.Evaluation``'s
    edge charge, read from the reset since a run has no unscored step)."""

    days: float
    seeds: int = 10
    more: int = 10
    folds: dict = field(default_factory=dict)
    first: int = 100
    source: str = ""
    edge_charge: str | None = None
    edge_credit: bool = False


PROTOCOLS = {
    "reactor": Protocol(
        days=7.0,
        folds={"edf_short_hold": "edf_daily", "krsko_shallow": "krsko_slow"},
        source="owner, 2026-11-24: 7 days; the source's standard day of each twin "
        "pair (EDF's 3 to 8 h hold, Krsko's day 1); owner, 2026-11-28: the boron "
        "the core owes at the end against the reset's, charged and credited",
        edge_charge="stored_value",
        edge_credit=True,
    ),
}


def _spec(name):
    from target_gym.registry import REGISTRY

    if name not in PROTOCOLS:
        raise KeyError(f"{name} has no scoring-run protocol")
    return REGISTRY[name]


def weights(name: str) -> dict[str, float]:
    """The kept scenarios' shares of operating time, each dropped twin's folded
    into the scenario named for it; they sum to 1."""
    from target_gym import scenarios

    folds = PROTOCOLS[name].folds
    out: dict[str, float] = {}
    for s, w in scenarios.weights(_spec(name)).items():
        kept = folds.get(s, s)
        out[kept] = out.get(kept, 0.0) + w
    return out


def scored_steps(name: str) -> int:
    """The control steps a run scores."""
    from target_gym.registry import control_step_seconds

    spec = _spec(name)
    step = control_step_seconds(spec.make_env(), spec.make_test_params())
    return int(round(PROTOCOLS[name].days * 86400.0 / step))


def seeds(name: str, extended: bool = False) -> list[int]:
    """A scenario's seeds: the first ``seeds``, then ``more`` new ones."""
    p = PROTOCOLS[name]
    return list(range(p.first, p.first + p.seeds + (p.more if extended else 0)))


def needs_more(reference, other) -> bool:
    """The stop rule: a scenario takes ``more`` seeds when the 95 % interval of
    its room, ``reference`` over ``other`` paired seed for seed (``scoring``'s
    bootstrap), contains 1."""
    from target_gym.scoring import headline

    lo, hi = headline({"s": {"policy": other, "reference": reference}}, {"s": 1.0})[
        "saving_ci"
    ]
    return bool(lo <= 0.0 <= hi)


def stream(name: str, scenario: str, steps: int | None = None, params=None):
    """The stream pinned to ``scenario`` with no burn-in, ending after ``steps``
    (the protocol's by default), and its params: ``params``, or the scored ones."""
    from target_gym import scenarios
    from target_gym.stream import StreamEnv, _event_rate, _per_step

    spec = _spec(name)
    env, suite = spec.make_env(), scenarios.suite(spec)
    params = spec.make_test_params() if params is None else params
    pin = [s.name for s in suite].index(scenario)
    scored = scored_steps(name) if steps is None else int(steps)
    # The plant knows the run's length: a core drawn at reset stays inside its
    # load-following range for all of it (owner, 2026-11-25).
    params = params.replace(max_steps_in_episode=scored)
    from target_gym.registry import control_step_seconds

    hours = control_step_seconds(env, params) / 3600.0
    expected = _per_step(_event_rate(suite[pin], True), hours) * scored
    out = StreamEnv(
        env, params, suite, pin=pin, b_min=0, scored=scored,
        max_events=int(np.ceil(expected)),
    )  # fmt: skip
    return out, params


def run_params(name: str, scenario: str, steps: int | None = None, params=None):
    """The task's params as the run's scenario has them, with the run's length:
    what a planner is built on, so it plans to the run's end."""
    s, params = stream(name, scenario, steps, params)
    pin = [x.name for x in s.scenarios].index(scenario)
    scored = scored_steps(name) if steps is None else int(steps)
    return s.scenario_params(params, pin).replace(max_steps_in_episode=scored)


def policy(name: str, scenario: str, kind: str, steps=None, params=None):
    """A shipped policy for a run: ``pid`` (the reference), ``mpc`` (the planner on
    the mean path), ``ceiling`` (the planner with the outside world foreseen) or
    ``null`` (the reset's own move held). Planners read the task's state."""
    from target_gym.benchmark import CallablePolicy
    from target_gym.runners.runners import baseline_policy

    spec = _spec(name)
    p = run_params(name, scenario, steps, params)
    if kind == "pid":
        return CallablePolicy(lambda _t: baseline_policy(spec, "pid", p), name=kind)
    env = spec.make_env()
    if kind == "null":

        def make(_task):
            held = {}

            def act(obs, state):
                if "u" not in held:
                    held["u"] = np.asarray(env.initial_action(state.env, p))
                return held["u"]

            return act

    elif kind in ("mpc", "ceiling"):
        from target_gym.experts.mpc import plan_params

        def make(_task):
            if kind == "mpc":
                planner = spec.make_mpc(env, plan_params(spec, p))
            else:
                planner = spec.make_mpc(env, p, seed=0)
            planner.reset()
            return lambda obs, state: planner.step(obs, state.env)

    else:
        raise ValueError(f"unknown policy kind {kind!r}")
    return CallablePolicy(make, name=kind, oracle=True)


def edge(name: str, before, after, steps: int):
    """Each run's end charge per step: the plant's stock valued ``before`` (at the
    reset) less ``after`` (at the last step), over ``steps``; a gain counts only
    where the protocol credits it, and nothing where it names no stock."""
    p = PROTOCOLS[name]
    if p.edge_charge is None:
        return np.zeros(np.shape(before))
    lost = np.asarray(before, float) - np.asarray(after, float)
    return (lost if p.edge_credit else np.maximum(lost, 0.0)) / steps


def _value(name: str, env):
    """The plant's stock function on a run's state, or None (``edge_charge``)."""
    method = PROTOCOLS[name].edge_charge
    if method is None:
        return None
    of = getattr(env.env, method)
    return lambda state, params: of(state.env, params)


def run(
    name, scenario, kind_or_policy, seeds, steps=None, terms=False, params=None
) -> dict:
    """``{"seeds", "costs", "trips"}``: each seed's mean cost per step over the run
    and whether it tripped; with ``terms``, each reward term's mean per step too.
    ``kind_or_policy`` is a :func:`policy` kind or any ``benchmark`` policy."""
    from target_gym.benchmark import run_policy, task_info

    s, params = stream(name, scenario, steps, params)
    pol = (
        policy(name, scenario, kind_or_policy, steps, params)
        if isinstance(kind_or_policy, str)
        else kind_or_policy
    )
    n = len(seeds)
    T = scored_steps(name) if steps is None else int(steps)
    task = task_info(name, s.env, params, n)
    value = _value(name, s)
    record = None if value is None else {"stock": lambda st, p, _: value(st, p)}
    roll = run_policy(pol, s, params, task, seeds=list(seeds), n_steps=T, record=record)
    cost = -roll["reward"].mean(axis=1)
    if value is not None:
        cost = cost + edge(name, _reset_values(s, params, seeds, value),
                           roll["records"]["stock"][:, -1], T)  # fmt: skip
    out = {
        "seeds": [int(x) for x in seeds],
        "costs": [float(c) for c in cost],
        "trips": [int(np.any(t)) for t in roll["tripped"]],
    }
    if terms:
        out.update(
            {k: [float(x) for x in v.mean(axis=1)] for k, v in roll["terms"].items()}
        )
    return out


def _reset_values(stream, params, seeds, value) -> np.ndarray:
    """The stock at each seed's reset, as ``run_policy`` resets it."""
    import jax

    from target_gym.benchmark import episode_keys

    keys = jax.numpy.stack(
        [episode_keys(jax.random.PRNGKey(int(x)), 1, 1)[0][0] for x in seeds]
    )
    # Reset as run_policy resets, under jit and vmap, so float32 agrees with it.
    at = jax.jit(jax.vmap(lambda k: value(stream.reset_env(k, params)[1], params)))
    return np.asarray(at(keys), float)


def run_functional(name, scenario, act, memory, seeds, steps=None, params=None) -> dict:
    """:func:`run` for a policy written as ``act(memory, obs) -> (action, memory)``
    in JAX (a ``Controller``'s step), compiled once and run on every seed at once:
    the same seeds, worlds and costs as :func:`run`, much faster."""
    import jax
    import jax.numpy as jnp

    from target_gym.benchmark import episode_keys

    s, params = stream(name, scenario, steps, params)
    T = scored_steps(name) if steps is None else int(steps)
    keys = jnp.stack(
        [episode_keys(jax.random.PRNGKey(int(x)), 1, 1)[0][0] for x in seeds]
    )

    value = _value(name, s)

    def one(key):
        obs, state = s.reset_env(key, params)
        before = jnp.float32(0.0) if value is None else value(state, params)

        def step(carry, _):
            obs, state, m = carry
            u, m = act(m, obs)
            obs, state, r, _, info = s.step_env(None, state, u, params)
            return (obs, state, m), (-r, info["tripped"])

        (_, last, _), (cost, tripped) = jax.lax.scan(
            step, (obs, state, memory), None, length=T
        )
        after = jnp.float32(0.0) if value is None else value(last, params)
        return cost.mean(), tripped.any(), before, after

    cost, tripped, before, after = jax.jit(jax.vmap(one))(keys)
    cost = np.asarray(cost, float) + edge(name, before, after, T)
    return {
        "seeds": [int(x) for x in seeds],
        "costs": [float(c) for c in cost],
        "trips": [int(t) for t in np.asarray(tripped)],
    }
