"""Scoring runs: a task scored on its stream pinned to each scenario.

docs/proposals/long-strata.md, R2, as the owner cut it on 2026-11-24 ("a scenario,
a seed or a day of simulation stays only if removing it would hide a named
behaviour or change which controller looks better"). A task with a
:data:`PROTOCOLS` entry is scored on the scenarios it keeps, a dropped scenario's
weight going to the twin named for it; each seed is one run of the stream pinned
to the scenario, from the reset as it is, scored over ``days`` from its first
step; a seed's cost is its mean cost per step over the run (J4). Ten seeds, and
ten more for a scenario whose room interval contains 1, once (:func:`needs_more`),
set before any result.

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
    "needs_more",
    "policy",
    "run",
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
    its weight; ``first`` is the first seed."""

    days: float
    seeds: int = 10
    more: int = 10
    folds: dict = field(default_factory=dict)
    first: int = 100
    source: str = ""


PROTOCOLS = {
    "reactor": Protocol(
        days=7.0,
        folds={"edf_short_hold": "edf_daily", "krsko_shallow": "krsko_slow"},
        source="owner, 2026-11-24: 7 days; the source's standard day of each twin "
        "pair (EDF's 3 to 8 h hold, Krsko's day 1); no priced end",
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


def stream(name: str, scenario: str, steps: int | None = None):
    """The stream pinned to ``scenario`` with no burn-in, ending after ``steps``
    (the protocol's by default), and its params."""
    from target_gym import scenarios
    from target_gym.stream import StreamEnv, _event_rate, _per_step

    spec = _spec(name)
    env, params, suite = spec.make_env(), spec.make_test_params(), scenarios.suite(spec)
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


def run_params(name: str, scenario: str, steps: int | None = None):
    """The task's params as the run's scenario has them, with the run's length:
    what a planner is built on, so it plans to the run's end."""
    s, params = stream(name, scenario, steps)
    pin = [x.name for x in s.scenarios].index(scenario)
    scored = scored_steps(name) if steps is None else int(steps)
    return s.scenario_params(params, pin).replace(max_steps_in_episode=scored)


def policy(name: str, scenario: str, kind: str, steps: int | None = None):
    """A shipped policy for a run: ``pid`` (the reference), ``mpc`` (the planner on
    the mean path), ``ceiling`` (the planner with the outside world foreseen) or
    ``null`` (the reset's own move held). Planners read the task's state."""
    from target_gym.benchmark import CallablePolicy
    from target_gym.runners.runners import baseline_policy

    spec = _spec(name)
    p = run_params(name, scenario, steps)
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


def run(name, scenario, kind_or_policy, seeds, steps=None, terms=False) -> dict:
    """``{"seeds", "costs", "trips"}``: each seed's mean cost per step over the run
    and whether it tripped; with ``terms``, each reward term's mean per step too.
    ``kind_or_policy`` is a :func:`policy` kind or any ``benchmark`` policy."""
    from target_gym.benchmark import run_policy, task_info

    s, params = stream(name, scenario, steps)
    pol = (
        policy(name, scenario, kind_or_policy, steps)
        if isinstance(kind_or_policy, str)
        else kind_or_policy
    )
    n = len(seeds)
    T = scored_steps(name) if steps is None else int(steps)
    task = task_info(name, s.env, params, n)
    roll = run_policy(pol, s, params, task, seeds=list(seeds), n_steps=T)
    out = {
        "seeds": [int(x) for x in seeds],
        "costs": [float(c) for c in -roll["reward"].mean(axis=1)],
        "trips": [int(np.any(t)) for t in roll["tripped"]],
    }
    if terms:
        out.update(
            {k: [float(x) for x in v.mean(axis=1)] for k, v in roll["terms"].items()}
        )
    return out
