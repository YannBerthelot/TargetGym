"""The battery's expert: a baseline planner on the BMS's report.

Contract (NESO-GUIDE p. 37, item 4): it declares the open period's baseline so
that the reported stored energy returns to the middle of the window over that
period, after the locked baselines have run. Positive exports.
"""

import jax.numpy as jnp
import pytest

from target_gym.energy.battery.experts import (
    BatteryPIDParams,
    BatteryPIDState,
    battery_pid_step,
)

P = BatteryPIDParams()


def _obs(soc, locked=(0.0, 0.0, 0.0), to_next=1800.0):
    obs = jnp.zeros(13)
    obs = obs.at[0].set(soc).at[6:9].set(jnp.asarray(locked)).at[11].set(to_next)
    return obs


def _act(obs):
    action, _ = battery_pid_step(P, BatteryPIDState(integral=0.0), obs)
    return float(action[0])


def test_at_the_middle_with_nothing_locked_it_declares_zero():
    assert _act(_obs(0.5)) == pytest.approx(0.0, abs=1e-6)


def test_above_the_middle_it_exports_and_below_it_imports():
    assert _act(_obs(0.55)) > 0.0 > _act(_obs(0.45))


def test_it_returns_the_gap_over_one_period():
    """0.05 of 3.854 MWh is 0.1927 MWh, over 0.5 h is 0.3854 MW; the limit is
    0.4 * 1.927 / 1.4 = 0.5506 MW, so raw 0.7."""
    assert _act(_obs(0.55)) == pytest.approx(0.3854 / 0.5506, rel=1e-3)


def test_locked_exports_are_taken_off_first():
    """0.3854 MW locked for the next period already moves the 0.1927 MWh."""
    assert _act(_obs(0.55, locked=(0.0, 0.3854, 0.0))) == pytest.approx(0.0, abs=1e-3)


def test_the_action_has_a_batch_and_an_action_axis():
    action, _ = battery_pid_step(
        P, BatteryPIDState(integral=0.0), jnp.stack([_obs(0.5), _obs(0.6)])
    )
    assert action.shape == (2, 1)
