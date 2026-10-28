"""Every scored episode captures the expert reaching its hold.

The expert is the reference a policy is scored against (``scoring.compare``),
so the window ``eval.hold_mask`` scores has to hold the expert's steady state,
not the tail of its first move. The rule, agreed with the user on 2026-10-27:

1. Every target cycle with scored hold steps lasts at least twice the task's
   fixed settle ``S`` (``eval.fixed_settle``: the slower of the expert's and
   the oracle's measured settles). The scorer then drops the whole settle in
   every cycle instead of cutting it to half the cycle, and at least half of
   each cycle is hold.
2. The expert's scored hold, on the tracking cost and floored at the ceiling
   as ``scoring.score`` floors it, is at most twice its hold on the same task
   run twice as long. Two is the settle rule's own factor: a level more than
   twice the later one is still the approach. It catches a measured settle
   that reads short because its level came from a window still converging.

A cycle runs from one target change to the next, the first from the reset
(``target_gym.eval``'s module docstring). The burn-in keeps its cap at half
the episode; both checks count the hold after it. Seeds 100-102, never the
scored 0-9, so no episode length is fitted to the seeds policies are scored on.
"""

from __future__ import annotations

import numpy as np
import pytest

from target_gym import eval as E
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy
from target_gym.scoring import ceiling

pytestmark = pytest.mark.large

SEEDS = (100, 101, 102)

# Known gaps, each owned by the session rebuilding that task's expert, whose
# PR lengthens the episode to its new expert and removes the mark. Strict, so a
# task that starts passing says so. None is open.
CYCLE_GAPS: dict[str, str] = {}
DOUBLING_GAPS = dict(CYCLE_GAPS)
# compressor_surge's schedule ends at step 1200 and then holds its last levels,
# so a run twice as long adds 1200 quiet steps and its hold falls 2.2x without
# any settling (3.6x with its blocks stretched instead). Its expert is at 0.1x
# its late level 80 steps into a block, inside the 100-step fixed settle.
DOUBLING_EXEMPT = {"compressor_surge": "its schedule ends at step 1200"}


def _names(gaps, exempt=None):
    exempt = exempt or {}
    marks = {n: pytest.mark.xfail(strict=True, reason=r) for n, r in gaps.items()}
    marks |= {n: pytest.mark.skip(reason=r) for n, r in exempt.items()}
    return [pytest.param(n, marks=marks.get(n, ())) for n in sorted(REGISTRY)]


def _expert_episodes(name, params):
    spec = REGISTRY[name]
    return [
        E.run_episode(spec, params, baseline_policy(spec, "pid", params), seed=s)
        for s in SEEDS
    ]


def _cycles(ep):
    changes = np.flatnonzero(ep.target_change)
    bounds = np.concatenate([[0], changes, [len(ep.cost)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _hold(name, params, episodes):
    burn_in = E.scored_burn_in(name, params)
    level = E.hold(episodes, burn_in, "tracking", E.fixed_settle(name))[0]
    return max(level, ceiling(name, params))


@pytest.fixture(scope="module")
def scored():
    """The expert's episodes on each task's scored configuration, run once."""
    cache = {}

    def get(name):
        if name not in cache:
            params = REGISTRY[name].make_test_params()
            cache[name] = params, _expert_episodes(name, params)
        return cache[name]

    return get


@pytest.mark.parametrize("name", _names(CYCLE_GAPS))
def test_every_cycle_scored_as_hold_lasts_twice_the_settle(name, scored):
    params, episodes = scored(name)
    settle = E.fixed_settle(name)
    burn_in = E.scored_burn_in(name, params)
    short = []
    for seed, ep in zip(SEEDS, episodes):
        held = E.hold_mask(ep, burn_in, settle)
        assert held.any(), f"{name} seed {seed}: the scored window holds no step"
        short += [
            (seed, a, b - a)
            for a, b in _cycles(ep)
            if held[a:b].any() and b - a < 2 * settle
        ]
    assert not short, (
        f"{name}: (seed, start, length) of cycles shorter than twice the settle "
        f"{settle}: {short}. Lengthen the episode or the target period."
    )


@pytest.mark.parametrize("name", _names(DOUBLING_GAPS, DOUBLING_EXEMPT))
def test_the_expert_holds_its_level_when_the_episode_doubles(name, scored):
    params, episodes = scored(name)
    longer = params.replace(max_steps_in_episode=2 * int(params.max_steps_in_episode))
    scored_hold = _hold(name, params, episodes)
    later_hold = _hold(name, longer, _expert_episodes(name, longer))
    assert scored_hold <= 2.0 * later_hold, (
        f"{name}: the expert's scored hold is {scored_hold / later_hold:.2f}x its "
        "hold on an episode twice as long, so the scored window still holds its "
        "approach. Lengthen the episode."
    )
