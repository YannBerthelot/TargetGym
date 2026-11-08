"""The 2D aircraft tasks' floors and failure cost, which their oracle reads.

The oracle itself is on the shared solver: ``tests/oracles``.
"""

import pytest

from target_gym import records, registry


@pytest.mark.parametrize("name", ["plane_sine", "plane_energy"])
def test_the_failure_cost_is_twice_the_altitude_envelopes_cost(name):
    """In floor units, across the 12 192 m (40 000 ft) altitude envelope."""
    p = registry.get(name).make_test_params()
    assert p.failure_cost == pytest.approx(2.0 * (12192.0 / p.e_floor) ** 2)


def test_a_crash_costs_an_hour_forty_thousand_feet_off_the_level():
    """``plane`` (Amendment 2): the ground is the one trip, and no source prices
    it; ours, an hour at the band cost of 12 192 m off the level, past +-65 ft."""
    p = registry.get("plane").make_test_params()
    band = 65.0 * 0.3048
    assert p.failure_cost == pytest.approx(((12192.0 - band) / band) ** 2)
    assert p.restart_steps == 3600


def test_a_hold_inside_the_bands_is_the_best_hold():
    """``plane`` (Amendment 2): precision inside +-65 ft and +-10 kt earns
    nothing, so the best hold costs 0, whatever an oracle once held."""
    p = registry.get("plane").make_test_params()
    assert p.rho_floor == p.rho_floor_tracking == 0.0


@pytest.mark.parametrize("name", ["plane_sine", "plane_energy"])
def test_the_nea_reference_is_the_recorded_hold(name):
    """rho_floor is the oracle's lowest per-seed altitude hold in floor units
    (``scripts/measure_hold.py``)."""
    p = registry.get(name).make_test_params()
    (hold,) = records.load_row("hold_measurements", name)["mpc"]["e_hold_min"]
    assert p.rho_floor == p.rho_floor_tracking
    assert p.rho_floor == pytest.approx((hold / p.e_floor) ** 2, rel=0.02)
