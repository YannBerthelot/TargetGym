"""The 2D aircraft tasks' floors and failure cost, which their oracle reads.

The oracle itself is on the shared solver: ``tests/oracles``.
"""

import pytest

from target_gym import records, registry

TASKS = ["plane", "plane_sine", "plane_energy"]


@pytest.mark.parametrize("name", TASKS)
def test_the_failure_cost_is_twice_the_altitude_envelopes_cost(name):
    """In floor units, across the 12 192 m (40 000 ft) altitude envelope."""
    p = registry.get(name).make_test_params()
    assert p.failure_cost == pytest.approx(2.0 * (12192.0 / p.e_floor) ** 2)


@pytest.mark.parametrize("name", TASKS)
def test_the_nea_reference_is_the_recorded_hold(name):
    """rho_floor is the oracle's lowest per-seed altitude hold in floor units
    (``scripts/measure_hold.py``)."""
    p = registry.get(name).make_test_params()
    (hold,) = records.load_row("hold_measurements", name)["mpc"]["e_hold_min"]
    assert p.rho_floor == p.rho_floor_tracking
    assert p.rho_floor == pytest.approx((hold / p.e_floor) ** 2, rel=0.02)
