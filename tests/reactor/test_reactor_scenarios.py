"""The reactor's scenario suite (charter rule 20).

Contract: 4 to 10 scenarios, each a day the family's sources show (REALITY.md,
section 6) and citing it; inside each, the day has the documented shape: its depth
and its ramp rate, and a hold only where the source has one.

Weights (REALITY.md section 7): EDF's units spend 250 h a year load following
(IEA 2019 workshop, EDF, slide 7). An EDF day is off full power for its hold, 5.5 h
on average, plus two 25 min ramps: 6.33 h, so 39.5 days a year, a share of
39.5 / 365 = 0.108 of days. Baseload days take the rest, 0.892. The five
load-follow days share 0.108 equally (ours: no source splits it), 0.0216 each.
"""

import jax
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.reactor.env import scheduled_load
from target_gym.registry import REGISTRY

SPEC = REGISTRY["reactor"]

# name: (lowest load, ramp in fraction per hour, hold in h or None for none)
SHAPES = {
    "edf_daily": (0.5, 1.2, (3.0, 8.0)),
    "edf_short_hold": (0.5, 1.2, (2.0, 3.0)),
    "krsko_slow": (0.5, 0.08, (0.0, 0.0)),
    "krsko_shallow": (0.75, 0.04, (0.0, 0.0)),
    "fast_return": (0.5, 3.0, (3.0, 8.0)),
    "baseload": (1.0, None, None),
}


def test_the_suite_is_the_familys_days():
    suite = scenarios.suite(SPEC)
    assert 4 <= len(suite) <= 10
    assert {s.name for s in suite} == set(SHAPES)
    for s in suite:
        assert "REALITY.md" in s.source


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_each_day_has_its_documented_shape(name):
    low, rate, hold = SHAPES[name]
    (s,) = [x for x in scenarios.suite(SPEC) if x.name == name]
    p = scenarios.params_for(SPEC, s)
    env = SPEC.env_factory()
    for seed in range(5):
        _, st = env.reset_env(jax.random.PRNGKey(seed), p)
        t = np.arange(0, 24 * 3600, 60.0)
        load = np.asarray(scheduled_load(t, st.ramp_down_at, st.ramp_up_at, p))
        assert load.min() >= low - 1e-6
        if rate is None:
            assert load.min() == pytest.approx(1.0)
            continue
        bottom = float(st.ramp_down_at) + (1.0 - low) / float(p.load_ramp_rate)
        at_bottom = scheduled_load(bottom, st.ramp_down_at, st.ramp_up_at, p)
        assert float(at_bottom) == pytest.approx(low, abs=1e-5)
        assert float(p.load_ramp_rate) * 3600.0 == pytest.approx(rate)
        held = float(st.ramp_up_at - st.ramp_down_at) - (1 - low) / p.load_ramp_rate
        assert hold[0] * 3600.0 - 1.0 <= held <= hold[1] * 3600.0 + 1.0


def test_each_day_weighs_its_share_of_operating_time():
    w = scenarios.weights(SPEC)
    assert w["baseload"] == pytest.approx(0.892, abs=0.002)
    for name in SHAPES:
        if name != "baseload":
            assert w[name] == pytest.approx(0.0216, abs=0.0005)
