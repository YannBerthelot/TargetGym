"""The process-plant experts against their spec, on seeds they were not tuned on.

Contract, set before measuring (docs/proposals/scoring-and-experts.md, "The
experts, plant by plant"): an expert is the loop practice runs, tuned by SIMC
from step tests on training seeds 100-109. Checked here on held-out seeds
200-209 of the scored configuration:

- it never trips the plant;
- once settled (after the plant's scored burn-in and fixed settle), it holds
  every tracked output within its instrument's resolution, which is the
  ceiling's hold cost (``target_gym.scoring.ceiling``).

The second is what a loop that cannot be told apart from perfect on the
plant's own instruments achieves. Practice does not always reach it, and
where it does not the miss is a finding, recorded as a strict xfail with the
measured value.
"""

import numpy as np
import pytest

from target_gym import eval as E
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy
from target_gym.scoring import ceiling

HELD_OUT = range(200, 210)


def _finding(reason):
    return pytest.mark.xfail(strict=True, reason=f"finding: {reason}")


def _episodes(name):
    spec = REGISTRY[name]
    p = spec.make_test_params()
    eps = [
        E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=s)
        for s in HELD_OUT
    ]
    return eps, p


#: Tuned at the steepest curve, the loop no longer cycles, and it is slow.
PROBE = "it holds 230 against a ceiling of 1, tuned at q2 = 0 as the plan says"
#: 8 trips on 4 of 10 seeds.
THERMOWELL = "the datasheet's inner PD trips 8 times on 4 of the 10 seeds"

PLANTS = [
    "first_order",
    "cstr",
    "ph_neutralization",
    pytest.param("unstable_cstr", marks=_finding(THERMOWELL)),
    "four_tank",
    "distillation",
]


@pytest.mark.large
@pytest.mark.parametrize("name", PLANTS)
def test_the_expert_never_trips_the_plant(name):
    eps, _ = _episodes(name)
    failed = [s for s, ep in zip(HELD_OUT, eps) if np.any(ep.failed)]
    assert not failed, f"{name}: the expert tripped the plant on seeds {failed}"


@pytest.mark.large
@pytest.mark.parametrize(
    "name",
    [
        "first_order",
        pytest.param(
            "cstr",
            marks=_finding(
                "the recipe holds 4,509 against a ceiling of 1; the 1 min analyser "
                "keeps the outer loop slow, and PR 8 moves this floor"
            ),
        ),
        pytest.param("ph_neutralization", marks=_finding(PROBE)),
        pytest.param(
            "unstable_cstr",
            marks=_finding(
                "the recipe holds 1.3e4 on average (worst seed 4.7e4) against a "
                "ceiling of 1, mostly the cost of its trips"
            ),
        ),
        pytest.param(
            "four_tank",
            marks=_finding(
                "the recipe holds 35.8 against a ceiling of 2 under the load "
                "steps, and PR 8 looks at this ceiling"
            ),
        ),
        pytest.param(
            "distillation",
            marks=_finding(
                "the recipe holds 3,942 against a ceiling of 2, on chromatographs "
                "5 min late, and PR 8 moves this floor"
            ),
        ),
    ],
)
def test_the_expert_holds_within_the_instruments_resolution(name):
    eps, p = _episodes(name)
    burn_in, settle = E.scored_burn_in(name, p), E.fixed_settle(name)
    holds = [E.hold([ep], burn_in, "tracking", settle)[0] for ep in eps]
    limit = ceiling(name, p)
    assert np.mean(holds) <= limit, (
        f"{name}: holds {np.mean(holds):.3g} on average (worst seed "
        f"{max(holds):.3g}) against a ceiling of {limit:.3g}"
    )
