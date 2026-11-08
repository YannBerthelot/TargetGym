"""The hvac expert: a heating curve built as Rehau builds one [12, p. 7].

Contract (hvac/experts.py, from REALITY.md):
- the supply is a straight line in the outdoor temperature's 24 h mean
  [9, 13], through the heating limit (supply equals the room setpoint when
  the mean reaches it) and the design point (the supply that holds the
  setpoint at the design outdoor temperature, from the building's steady
  state);
- it reads the 24 h mean, not the instantaneous outdoor reading;
- the supply stays inside the valve's range.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.hvac.env import HVACParams, steady_state
from target_gym.hvac.experts import make_hvac_curve, make_hvac_pid_controller

P = HVACParams()


def _supply(obs):
    ctrl = make_hvac_pid_controller()
    raw, _ = ctrl.act(ctrl.init_state, jnp.asarray(obs, jnp.float32))
    span = P.T_supply_max - P.T_supply_min
    return float(P.T_supply_min + 0.5 * (raw[0] + 1.0) * span)


def _obs(T_out_24h, T_out=0.0):
    return [21.0, T_out, T_out_24h, 22.0, 21.5, 22.0, 0.0, 1.0, 23.5]


def test_at_the_heating_limit_the_supply_is_the_room_setpoint():
    assert _supply(_obs(21.0)) == pytest.approx(21.0, abs=1e-3)


def test_at_the_design_point_the_supply_holds_the_setpoint():
    *_, design = steady_state(P, -10.0, 21.0)
    assert _supply(_obs(-10.0)) == pytest.approx(float(design), abs=1e-3)


def test_the_curve_reads_the_days_mean_not_the_instant():
    assert _supply(_obs(0.0, T_out=-15.0)) == pytest.approx(
        _supply(_obs(0.0, T_out=10.0)), abs=1e-6
    )


def test_the_curve_is_straight_between_its_points():
    s = [_supply(_obs(t)) for t in (-10.0, 5.5, 21.0)]
    np.testing.assert_allclose(s[1], 0.5 * (s[0] + s[2]), atol=1e-3)


def test_the_supply_stays_inside_the_valves_range():
    assert _supply(_obs(-200.0)) == pytest.approx(P.T_supply_max, abs=1e-3)
    assert _supply(_obs(60.0)) == pytest.approx(P.T_supply_min, abs=1e-3)


def test_an_offset_shifts_the_curve_in_parallel():
    from target_gym.hvac.experts import curve_supply

    base, shifted = make_hvac_curve(), make_hvac_curve(offset=-2.0)
    for t in (-10.0, 0.0, 10.0):
        assert float(curve_supply(shifted, t)) == pytest.approx(
            float(curve_supply(base, t)) - 2.0
        )
