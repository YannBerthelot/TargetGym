"""The battery's expert: a price-aware baseline planner on the BMS's report.

Contract (owner, 2026-11-15; energy/battery/experts.py):

- It declares the open period's baseline so that the reported stored energy
  reaches a target over that period, after the locked baselines have run
  (NESO-GUIDE p. 37, item 4). Positive exports.
- The target is the window's low edge (the export requirement over the
  capacity) plus the 2 sigma back-off, unless the open period's day-ahead price
  makes storing profitable, below the base price times the round-trip
  efficiency. Then it is the high edge (one less the import requirement over the
  capacity) less the back-off. Energy sold is valued at the base price, so holding
  more than the requirement only loses the round trip (Sioshansi et al. 2009,
  sec. 2: a device charges 1/eta per unit it discharges).

Numbers below: capacity 3.854 MWh, baseline limit 0.4 * 1.927 / 1.4 = 0.5506 MW,
period 0.5 h, base GBP 74/MWh, efficiency 0.92 (buy below GBP 68.08), back-off
2 x 0.0501 = 0.1002.
"""

import jax.numpy as jnp
import pytest

from target_gym.energy.battery.experts import (
    BatteryPIDParams,
    BatteryPIDState,
    battery_pid_step,
)

P = BatteryPIDParams()
CAP = 3.854
#: Each requirement at 0.35 of the capacity: the window is 0.35 to 0.65.
REQ = 0.35 * CAP
#: soc 0.5 against a target 0.4502 (0.35 + 0.1002): 0.0498 x 3.854 MWh over 0.5 h
#: is 0.38386 MW, raw 0.38386 / 0.5506.
RAW = 0.0498 * CAP / 0.5 / 0.5506


def _obs(soc, price=74.0, locked=(0.0, 0.0, 0.0), to_next=1800.0, req=(REQ, REQ)):
    obs = jnp.zeros(17)
    obs = obs.at[0].set(soc).at[6:9].set(jnp.asarray(locked)).at[11].set(to_next)
    obs = obs.at[9].set(req[0]).at[10].set(req[1])
    return obs.at[13:17].set(74.0).at[16].set(price)


def _act(obs):
    action, _ = battery_pid_step(P, BatteryPIDState(integral=0.0), obs)
    return float(action[0])


def test_at_a_flat_price_it_holds_the_back_off_above_the_export_requirement():
    assert _act(_obs(0.4502)) == pytest.approx(0.0, abs=1e-4)


def test_at_a_flat_price_it_returns_the_gap_over_one_period():
    assert _act(_obs(0.5)) == pytest.approx(RAW, rel=1e-3)


def test_a_deficit_raises_the_low_edge():
    """An export requirement of 0.40 x capacity: target 0.5002, so soc 0.5002 holds."""
    assert _act(_obs(0.5002, req=(0.40 * CAP, REQ))) == pytest.approx(0.0, abs=1e-4)


def test_a_price_below_the_round_trip_breakeven_fills_to_the_import_side():
    """Target 0.65 - 0.1002 = 0.5498: the same gap the other way."""
    assert _act(_obs(0.5, price=-10.0)) == pytest.approx(-RAW, rel=1e-3)


@pytest.mark.parametrize(
    "price, sign",
    [(68.0, -1.0), (68.2, 1.0), (300.0, 1.0)],
    ids=["just-below-breakeven", "just-above-breakeven", "spike"],
)
def test_the_breakeven_is_the_base_price_times_the_efficiency(price, sign):
    """74 x 0.92 = 68.08. A high price keeps the low target: the energy above the
    requirement is already sold down to it."""
    assert _act(_obs(0.5, price=price)) == pytest.approx(sign * RAW, rel=1e-3)


def test_locked_exports_are_taken_off_first():
    """0.38386 MW locked for the next period already moves the 0.19193 MWh."""
    assert _act(_obs(0.5, locked=(0.0, 0.38386, 0.0))) == pytest.approx(0.0, abs=1e-3)


def test_the_action_is_clipped_to_the_baseline_limit():
    assert _act(_obs(0.95)) == pytest.approx(1.0)


def test_the_action_has_a_batch_and_an_action_axis():
    action, _ = battery_pid_step(
        P, BatteryPIDState(integral=0.0), jnp.stack([_obs(0.5), _obs(0.6)])
    )
    assert action.shape == (2, 1)
