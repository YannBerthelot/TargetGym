"""What an evaluation reads of a task's episodes: its scored window and its seeds.

Owner, 2026-11-24, the evaluation cut (long-strata note R2 and R6). A task may
declare an ``Evaluation`` on its registry entry; one that declares none scores
every step on the ladder's seeds, as before.

This lives apart from ``target_gym.scenarios`` on purpose: the aircraft tasks
hash that module into their version stamps, and nothing here changes what an
environment is (a task's own declaration sits in its package, where its
fingerprint covers it).
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np

__all__ = ["LADDER_SEEDS", "Evaluation", "costs", "line_seeds", "of"]
__all__ += ["saving_interval", "scored_cost", "scored_window"]

#: The seeds every line ran on before a task declared its own (the ladder's, J3).
LADDER_SEEDS = tuple(range(100, 110))


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """What an evaluation reads of a task's episodes.

    The first ``unscored_steps`` burn the plant in from its reset, and the steps
    after the scored ones are the ceiling's tail, charged in its plan so it does not
    spend at the end a stock the next days pay for: an evaluation reads the mean
    cost per step over the ``scored_steps`` between (all the rest when None). Lines
    run on ``seeds``, and on ``extra_seeds`` too in a scenario where the reference's
    saving over the ceiling is unclear (``line_seeds``)."""

    unscored_steps: int = 0
    scored_steps: int | None = None
    seeds: tuple[int, ...] = LADDER_SEEDS
    extra_seeds: tuple[int, ...] = ()


def of(spec) -> Evaluation:
    """``spec``'s declared evaluation, or the whole episode on the ladder's seeds."""
    declared = getattr(spec, "evaluation", None)
    return declared() if declared is not None else Evaluation()


def scored_window(spec, n_steps: int) -> slice:
    """The steps an evaluation reads of an ``n_steps`` episode of ``spec``."""
    ev = of(spec)
    end = n_steps if ev.scored_steps is None else ev.unscored_steps + ev.scored_steps
    if not 0 <= ev.unscored_steps < end <= n_steps:
        raise ValueError(
            f"{spec.name}: scored steps {ev.unscored_steps} to {end} do not fit an "
            f"episode of {n_steps}"
        )
    return slice(ev.unscored_steps, end)


def scored_cost(spec, cost) -> float:
    """One episode's mean cost per step over ``spec``'s scored window."""
    cost = np.asarray(cost, float)
    return float(cost[scored_window(spec, len(cost))].mean())


def costs(runs: Mapping[str, list], spec) -> dict[str, np.ndarray]:
    """Each scenario's cost per step over ``spec``'s scored window, one value per
    seed (``target_gym.scenarios.costs`` reads the whole episode)."""
    return {
        name: np.array([scored_cost(spec, ep.cost) for ep in episodes])
        for name, episodes in runs.items()
    }


def saving_interval(savings) -> tuple[float, float]:
    """The 95 % Student t interval of the mean of paired per-seed ``savings``."""
    from scipy import stats

    x = np.asarray(savings, float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - half), float(x.mean() + half)


def line_seeds(spec, savings=None) -> tuple[int, ...]:
    """The seeds a scenario's lines run on (the seed rule, owner, 2026-11-24):
    ``spec``'s seeds, and its extra seeds too when ``savings``, the reference's cost
    less the ceiling's on each of those seeds, has an interval that touches 0."""
    ev = of(spec)
    if savings is None or not ev.extra_seeds:
        return ev.seeds
    lo, hi = saving_interval(savings)
    return ev.seeds + ev.extra_seeds if lo <= 0.0 <= hi else ev.seeds
