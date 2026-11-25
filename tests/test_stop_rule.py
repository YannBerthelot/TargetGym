"""The scoring seeds' stop rule (owner, 2026-11-24, written before any result).

Contract: the scoring seeds come in blocks of 10 from 200 (``charter.ANCHOR_SEEDS``
is the first). After each block, a paired bootstrap that resamples seeds within
each scenario (J4) gives a 95 % interval on a ratio of two weighted costs, J_a over
J_b, with J = sum over scenarios of w_s times the scenario's mean. The run stops at
the first block whose interval excludes the threshold (1 for a room or a saving),
and at 50 seeds otherwise, which it reports as undecided. A block's seeds are always
the earlier blocks' plus 10 more, so a stopped run is a prefix of a longer one.
"""

import numpy as np
import pytest

from target_gym import charter

W = {"normal": 0.8, "upset": 0.2}


class Measure:
    """A paired measure from per-seed values: ``a[seed]`` and ``b[seed]`` in every
    scenario. Records the seed tuples it is asked for."""

    def __init__(self, a, b):
        self.a, self.b, self.calls = a, b, []

    def __call__(self, seeds):
        self.calls.append(tuple(seeds))
        pick = lambda f: np.array([f(s) for s in seeds], float)  # noqa: E731
        return {s: pick(self.a) for s in W}, {s: pick(self.b) for s in W}


def test_a_clear_ratio_stops_after_the_first_ten_seeds():
    # Every seed costs 1.2 against 1.0: every resample gives 1.2.
    m = Measure(lambda s: 1.2, lambda s: 1.0)
    out = charter.stop_seeds(m, W)
    assert m.calls == [tuple(range(200, 210))]
    assert out["seeds"] == tuple(range(200, 210)) == charter.ANCHOR_SEEDS
    assert out["decided"]
    assert out["ratio"] == pytest.approx(1.2)
    assert out["interval"] == pytest.approx([1.2, 1.2])


def test_a_ratio_of_one_runs_every_block_and_is_reported_undecided():
    m = Measure(lambda s: 1.0, lambda s: 1.0)
    out = charter.stop_seeds(m, W)
    assert m.calls == [tuple(range(200, 200 + n)) for n in (10, 20, 30, 40, 50)]
    assert out["seeds"] == tuple(range(200, 250))
    assert not out["decided"]


def test_a_spread_first_block_takes_a_second_block_to_decide():
    # Seeds 200 to 209 alternate 2.0 and 0.5 against 1.0: a resample of ten with
    # three or fewer 2.0s has a mean under 1 (about 17 % of resamples), so the
    # interval spans 1. Seeds 210 to 219 all cost 1.5: a resample of twenty falls
    # under 1 only with nearly all its picks on the 0.5s, far rarer than 2.5 %.
    def a(s):
        return 1.5 if s >= 210 else (2.0 if s % 2 == 0 else 0.5)

    m = Measure(a, lambda s: 1.0)
    out = charter.stop_seeds(m, W)
    assert m.calls == [tuple(range(200, 210)), tuple(range(200, 220))]
    assert out["decided"]
    assert out["interval"][0] > 1.0
    assert out["ratio"] == pytest.approx((5 * 2.0 + 5 * 0.5 + 10 * 1.5) / 20)


def test_the_ratio_weighs_each_scenario_by_its_share():
    # normal costs 1.0 against 1.0, upset 3.0 against 1.0: J_a = 0.8 + 0.6 = 1.4,
    # J_b = 1.0.
    def measure(seeds):
        n = len(seeds)
        a = {"normal": np.full(n, 1.0), "upset": np.full(n, 3.0)}
        return a, {s: np.full(n, 1.0) for s in W}

    out = charter.stop_seeds(measure, W)
    assert out["ratio"] == pytest.approx(1.4)
    assert out["seeds"] == charter.ANCHOR_SEEDS


def test_a_ratio_clearly_under_the_threshold_also_stops():
    out = charter.stop_seeds(Measure(lambda s: 0.9, lambda s: 1.0), W)
    assert out["decided"] and out["interval"][1] < 1.0
    assert out["seeds"] == charter.ANCHOR_SEEDS
