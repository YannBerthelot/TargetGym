"""A task's headline: the frequency-weighted saving over the reference.

Contract, from the owner's scoring decision of 2026-11-08 (charter J4):

- The scenarios are strata. Each has a weight, its real share of operating time. Weights
  are shares whatever their scale, and every one is positive.
- A task's expected cost per step is ``J = sum_s w_s * mean(cost | s)``, the mean over
  each scenario's seeds, since a bad seed costs real money and is not trimmed.
- The headline is ``saving = 1 - J_agent / J_reference``: 0 at the reference, above 0
  cheaper. Its interval resamples seeds within each scenario, keeping agent and
  reference paired, so an agent exactly 10 % cheaper on every seed has an interval of
  exactly 10 %.
- Beside it: each scenario's own saving, the worst scenario, and the probability of
  improvement, the weighted share of seeds where the agent costs less.
- ``beats_reference``, when trips are given: the interval lies above 0, and the agent
  trips no more than the reference in any scenario.
- ``gap_closed``, when ceilings are given: ``(b - x) / (b - c)`` on each scenario's
  means, reported, never the headline.
- A reference that costs nothing anywhere leaves no headline (a solved task): NaN.
- Weights must cover exactly the scenarios costed. Costs are non-negative.

Expected values are worked by hand in each test.
"""

import math

import numpy as np
import pytest

from target_gym.scoring import gap_closed, headline


def side(policy, reference, **extra):
    return {"policy": policy, "reference": reference, **extra}


def test_the_reference_closes_nothing_and_the_ceiling_closes_all():
    # b 10, c 0: x 10 gives 0, x 0 gives 1, x 5 gives (10 - 5) / 10 = 0.5.
    np.testing.assert_allclose(gap_closed([10.0, 0.0, 5.0], 10.0, 0.0), [0, 1, 0.5])


def test_no_gap_when_the_reference_is_at_the_ceiling():
    assert math.isnan(gap_closed([3.0], 4.0, 4.0)[0])


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_a_negative_or_nan_cost_is_an_error(bad):
    with pytest.raises(ValueError):
        gap_closed([bad], 10.0, 0.0)


def test_saving_weighs_scenarios_by_their_share_of_operating_time():
    # J_agent = 0.99 * 9 + 0.01 * 30 = 9.21, J_ref = 0.99 * 10 + 0.01 * 20 = 10.1,
    # so the saving is 1 - 9.21 / 10.1.
    result = headline(
        {
            "normal": side([9.0, 9.0], [10.0, 10.0]),
            "upset": side([30.0] * 2, [20.0] * 2),
        },
        {"normal": 0.99, "upset": 0.01},
    )
    assert result["saving"] == pytest.approx(1 - 9.21 / 10.1)


def test_weights_are_shares_whatever_their_scale():
    # Weights 3 and 1 are shares 0.75 and 0.25: J_agent = 6.75 + 7.5 = 14.25 and
    # J_ref = 7.5 + 5 = 12.5, so the saving is 1 - 14.25 / 12.5 = -0.14.
    result = headline(
        {
            "normal": side([9.0, 9.0], [10.0, 10.0]),
            "upset": side([30.0] * 2, [20.0] * 2),
        },
        {"normal": 3.0, "upset": 1.0},
    )
    assert result["saving"] == pytest.approx(-0.14)
    # Each scenario's own saving: 1 - 9 / 10 = 0.1 and 1 - 30 / 20 = -0.5.
    assert result["per_scenario"] == pytest.approx({"normal": 0.1, "upset": -0.5})
    assert result["worst"] == "upset"


def test_a_bad_seed_counts_in_full():
    # Seeds 5, 5, 5, 110 have a mean of 31.25: the saving is 1 - 31.25 / 10 = -2.125.
    result = headline(
        {"normal": side([5.0, 5.0, 5.0, 110.0], [10.0] * 4)}, {"normal": 1}
    )
    assert result["saving"] == pytest.approx(-2.125)


def test_probability_of_improvement_is_the_weighted_share_of_seeds_won():
    # normal (0.75): 3 of 4 seeds won; upset (0.25): none. 0.75 * 0.75 = 0.5625.
    result = headline(
        {
            "normal": side([9.0, 9.0, 9.0, 11.0], [10.0] * 4),
            "upset": side([21.0, 21.0], [20.0, 20.0]),
        },
        {"normal": 0.75, "upset": 0.25},
    )
    assert result["probability_of_improvement"] == pytest.approx(0.5625)


def test_an_agent_cheaper_by_the_same_share_on_every_seed_has_a_tight_interval():
    # 0.9 times the reference on every seed: any resample of paired seeds gives
    # exactly 0.1, so the interval is 0.1 to 0.1.
    reference = np.array([8.0, 10.0, 12.0, 15.0])
    result = headline({"normal": side(0.9 * reference, reference)}, {"normal": 1.0})
    assert result["saving_ci"] == pytest.approx((0.1, 0.1))


def trips(policy, reference):
    return {"policy_trips": policy, "reference_trips": reference}


def test_beating_the_reference_needs_a_positive_interval_and_no_more_trips():
    reference = np.array([8.0, 10.0, 12.0, 15.0])
    cheaper = side(0.9 * reference, reference, **trips([0, 0, 0, 0], [0, 0, 0, 0]))
    tripping = side(0.9 * reference, reference, **trips([0, 1, 0, 0], [0, 0, 0, 0]))
    dearer = side(1.1 * reference, reference, **trips([0, 0, 0, 0], [0, 0, 0, 0]))
    assert headline({"normal": cheaper}, {"normal": 1})["beats_reference"] is True
    assert headline({"normal": tripping}, {"normal": 1})["beats_reference"] is False
    assert headline({"normal": dearer}, {"normal": 1})["beats_reference"] is False


def test_gap_closed_is_reported_on_each_scenarios_means():
    # b 10, x 9, c 6: (10 - 9) / (10 - 6) = 0.25.
    result = headline(
        {"normal": side([9.0, 9.0], [10.0, 10.0], ceiling=[6.0, 6.0])}, {"normal": 1}
    )
    assert result["gap_closed"] == pytest.approx({"normal": 0.25})


def test_a_reference_that_costs_nothing_leaves_no_headline():
    result = headline({"calm": side([1.0], [0.0])}, {"calm": 1.0})
    assert math.isnan(result["saving"])


@pytest.mark.parametrize(
    "weights",
    [{"normal": 1.0}, {"normal": 1.0, "upset": 0.0}, {"normal": 1.0, "upset": -1.0}],
)
def test_weights_must_cover_the_scenarios_and_be_positive(weights):
    costs = {"normal": side([9.0], [10.0]), "upset": side([9.0], [10.0])}
    with pytest.raises(ValueError):
        headline(costs, weights)
