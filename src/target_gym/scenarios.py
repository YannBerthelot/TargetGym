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

__all__ = ["NORMAL", "Scenario", "costs", "params_for", "run", "suite"]


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One operating window: ``overrides`` on the scored configuration, and the
    ``source`` (a ``REALITY.md`` reference) that shows this event happening."""

    name: str
    overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    source: str = ""
    description: str = ""


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
    return declared


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
