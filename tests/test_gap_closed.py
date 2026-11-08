"""The headline: the share of the gap an agent closes, per scenario.

Contract, from docs/proposals/task-charter.md, Amendment 3, rule 21:

- ``gap_closed(x, b, c) = (b - x) / (b - c)`` per seed, with ``x`` the agent's cost,
  ``b`` the reference's and ``c`` the ceiling's: 0 at the reference, 1 at the ceiling,
  above 1 past the ceiling (a ceiling is a local optimum), negative above the
  reference. NaN where ``b <= c``: there is no gap to close.
- ``headline(costs)``: each scenario scores the IQM of its seeds' gaps, leaving out
  seeds with no gap. The headline is the plain mean over scenarios, whatever their seed
  counts, so a rare upset counts as much as a normal day. A scenario with no gap on any
  seed is solved: it is listed, and left out of the mean. With every scenario solved
  the headline is NaN.
- Costs are non-negative: a negative or NaN cost is an error.

Expected values are worked by hand in each test.
"""

import math

import numpy as np
import pytest

from target_gym.scoring import gap_closed, headline


def test_the_reference_closes_nothing_and_the_ceiling_closes_all():
    # b 10, c 0: x 10 gives 0, x 0 gives 1, x 5 gives (10 - 5) / 10 = 0.5.
    np.testing.assert_allclose(gap_closed([10.0, 0.0, 5.0], 10.0, 0.0), [0, 1, 0.5])


def test_worse_than_the_reference_is_negative_and_past_the_ceiling_above_one():
    # b 10, c 2: x 18 gives (10 - 18) / 8 = -1, x 0 gives 10 / 8 = 1.25.
    np.testing.assert_allclose(gap_closed([18.0, 0.0], 10.0, 2.0), [-1.0, 1.25])


def test_no_gap_when_the_reference_is_at_the_ceiling():
    assert math.isnan(gap_closed([3.0], 4.0, 4.0)[0])


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_a_negative_or_nan_cost_is_an_error(bad):
    with pytest.raises(ValueError):
        gap_closed([bad], 10.0, 0.0)


def costs(policy, reference, ceiling):
    return {"policy": policy, "reference": reference, "ceiling": ceiling}


def test_scenarios_count_equally_whatever_their_seed_counts():
    # "normal" has 8 seeds at gap 0.5, "upset" 4 seeds at gap 0: the mean of the
    # two scenarios is 0.25, where pooling seeds would give 12 seeds, 4 / 12 = 0.33.
    result = headline(
        {
            "normal": costs([5.0] * 8, [10.0] * 8, [0.0] * 8),
            "upset": costs([10.0] * 4, [10.0] * 4, [0.0] * 4),
        }
    )
    assert result["per_scenario"] == pytest.approx({"normal": 0.5, "upset": 0.0})
    assert result["headline"] == pytest.approx(0.25)


def test_a_scenario_scores_the_interquartile_mean_of_its_seeds():
    # Gaps 0.5, 0.5, 0.5 and -10 (one bad seed): sorted, the middle two are 0.5
    # and 0.5, so the IQM is 0.5.
    result = headline({"normal": costs([5.0, 5.0, 5.0, 110.0], 10.0, 0.0)})
    assert result["per_scenario"]["normal"] == pytest.approx(0.5)


def test_a_scenario_with_no_gap_is_solved_and_left_out_of_the_mean():
    result = headline(
        {
            "normal": costs([5.0, 5.0], [10.0, 10.0], [0.0, 0.0]),
            "calm": costs([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
        }
    )
    assert result["solved"] == ["calm"]
    assert result["headline"] == pytest.approx(0.5)


def test_every_scenario_solved_gives_no_headline():
    result = headline({"calm": costs([1.0], [0.0], [0.0])})
    assert math.isnan(result["headline"])
