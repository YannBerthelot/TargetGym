"""Scenario suites: the operating windows a task is evaluated on.

docs/proposals/task-charter.md, Amendment 3, rule 20. The plants run continuously, so
an episode is a window of operation. Each task has a suite of scenarios drawn from its
family's sources: a normal day, each documented upset, load or grade changes, known
schedules. A scenario is a set of parameter overrides on the scored configuration;
inside it, timing and sizes are still drawn from the family's ranges by the seed.

A task declares its suite on its registry entry (``EnvSpec.scenarios``), as a
callable returning the scenarios, kept in its own package so its fingerprint covers
it. A task that declares none has one scenario, :data:`NORMAL`.

Run a policy on every scenario of a task::

    from target_gym import scenarios
    runs = scenarios.run(spec, lambda params: policy, seeds=10)
    per_seed = scenarios.costs(runs)   # {scenario: cost per step, per seed}

``target_gym.scoring.headline`` turns those costs, with the reference's and the
ceiling's, into the task's headline.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np

__all__ = ["NORMAL", "Scenario", "costs", "params_for", "run", "suite", "weights"]


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One operating window: ``overrides`` on the scored configuration, the
    ``source`` (a ``REALITY.md`` reference) that shows this event happening, and its
    ``weight``, its share of real operating time (any positive scale; ``weights``
    turns them into shares). The headline weighs scenarios by it (charter J4).

    In a stream (``target_gym.stream``) a scenario lasts a memoryless time of mean
    ``dwell_h`` hours, and its events start at ``events_per_h`` per hour, both from
    the plant record (long-strata draft 4, S1). ``stream_overrides`` are parameters
    only a continuing plant reads, applied after ``overrides``: how long an event
    holds before it returns, or turning off the window's own event.

    ``pinned_events_per_h``, where set, is the rate a scoring run pinned to the
    scenario (R2) draws at instead. A pinned run is one endless visit, and where a
    draw is absorbed while an event of its kind is under way, a visit's start
    delivers more than its long run: the pinned rate delivers per scored hour what
    the stream delivers per hour in the scenario (owner, 2026-11-12)."""

    name: str
    overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    source: str = ""
    description: str = ""
    weight: float = 1.0
    dwell_h: float | None = None
    events_per_h: float = 0.0
    pinned_events_per_h: float | None = None
    stream_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)


NORMAL = Scenario("normal", {}, description="the scored configuration as it is")


def suite(spec) -> tuple[Scenario, ...]:
    """``spec``'s scenarios, checked: not empty, unique names, and every override a
    parameter of the task."""
    declared = tuple(spec.scenarios()) if spec.scenarios is not None else (NORMAL,)
    if not declared:
        raise ValueError(f"{spec.name}: the scenario suite is empty")
    names = [s.name for s in declared]
    if len(set(names)) != len(names):
        raise ValueError(f"{spec.name}: scenario names repeat: {names}")
    fields = set(vars(spec.make_test_params()))
    for s in declared:
        unknown = sorted(set(s.overrides) - fields)
        if unknown:
            raise ValueError(f"{spec.name}: scenario {s.name!r} sets no such {unknown}")
        if not (np.isfinite(s.weight) and s.weight > 0):
            raise ValueError(f"{spec.name}: scenario {s.name!r} weight {s.weight}")
    return declared


def weights(spec) -> dict[str, float]:
    """Each scenario's share of operating time, summing to 1."""
    declared = suite(spec)
    total = sum(s.weight for s in declared)
    return {s.name: s.weight / total for s in declared}


def params_for(spec, scenario: Scenario, **extra):
    """The scored configuration with ``scenario``'s overrides, then ``extra``."""
    return spec.make_test_params(**{**scenario.overrides, **extra})


def run(spec, make_policy, seeds: int, **extra) -> dict[str, list]:
    """``{scenario: episodes}`` on seeds ``0 .. seeds - 1`` of each scenario.

    ``make_policy(params)`` returns a fresh policy for one episode, so a stateful
    controller starts clean and a reference can read its scenario's params.
    ``extra`` overrides apply to every scenario, after its own.
    """
    from target_gym import eval as E

    out = {}
    for s in suite(spec):
        p = params_for(spec, s, **extra)
        out[s.name] = [
            E.run_episode(spec, p, make_policy(p), seed=k) for k in range(seeds)
        ]
    return out


def costs(runs: Mapping[str, list]) -> dict[str, np.ndarray]:
    """Each scenario's cost per step, one value per seed, over the whole episode."""
    from target_gym import eval as E

    return {
        name: np.array([E.gain([ep], 0)[0] for ep in episodes])
        for name, episodes in runs.items()
    }
