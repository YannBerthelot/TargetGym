"""Fuel burnup and the dilution carried over from earlier days (owner, 2026-11-10).

Contracts, from the source before the code:

- The fuel loses reactivity as it burns, met by diluting boron: BEAVRS (MIT, a
  measured Westinghouse 4-loop PWR, 3411 MWth), Table 24, letdown of 623 to 0 ppm
  from 52 to 326 EFPD in cycle 1 (2.27 ppm a day) and 918 to 104 ppm from 13 to 257
  EFPD in cycle 2 (3.34 ppm a day). Each day draws a rate in 2.3 to 3.3 ppm a day,
  lost at that rate through the whole day as if at full power (PHYSICS.md, D9).
- The day starts with up to one day of burnup not yet diluted (ours: the operator
  dilutes at least once a day): the plant is lined out at a Tavg below its program
  by that debt, ``debt * boron_worth / (alpha_coolant + alpha_fuel)``: at a fixed
  power the fuel sits a fixed step above the coolant and cools with it, 8/33 K per
  ppm here.

Hand values: 3.3 ppm a day of debt is 0.8 K; 3.3 ppm a day for 6 h is 0.825 ppm,
0.2 K.
"""

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams

K_PER_PPM = 8.0 / 33.0


def reset(seed=0, **changes):
    changes.setdefault("demand_sigma", 0.0)
    changes.setdefault("initial_xenon_power_range", (1.0, 1.0))
    changes.setdefault("ramp_down_range", (1e6, 1e6))
    env, p = Reactor(), ReactorParams(**changes)
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    return env, p, obs, s


def hold(env, p, s, steps):
    """Rods where they are, no boron: the Tavg error, program minus Tavg."""
    rods = np.array([1.0 - 2.0 * float(s.rod_in), 0.0], np.float32)
    for k in range(steps):
        obs, s, _, _, _ = env.step_env(jax.random.PRNGKey(k), s, rods, p)
    return float(s.T_ref - 0.5 * (s.T_coolant + s.T_cold))


def test_six_hours_of_burnup_cool_tavg_by_its_lost_reactivity():
    """Against the same day with no burnup, so xenon and the rest cancel."""
    steps = 6 * 120  # 30 s steps
    env, p, _, s = reset(burnup_range=(3.3, 3.3), carried_days_range=(0.0, 0.0))
    burnt = hold(env, p, s, steps)
    env, p, _, s = reset(burnup_range=(0.0, 0.0), carried_days_range=(0.0, 0.0))
    clean = hold(env, p, s, steps)
    assert burnt - clean == pytest.approx(0.825 * K_PER_PPM, abs=0.02)


def test_the_day_starts_cold_by_the_dilution_not_yet_done():
    env, p, obs, s = reset(burnup_range=(3.3, 3.3), carried_days_range=(1.0, 1.0))
    cold = float(s.T_ref - 0.5 * (s.T_coolant + s.T_cold))  # the plant, not a reading
    assert cold == pytest.approx(3.3 * K_PER_PPM, abs=0.01)


def test_the_carried_debt_starts_lined_out():
    """With no burnup during the day, Tavg sits where it started."""
    env, p, obs, s = reset(burnup_range=(0.0, 0.0), carried_days_range=(1.0, 1.0))
    start = float(s.T_ref - 0.5 * (s.T_coolant + s.T_cold))
    assert hold(env, p, s, 20) == pytest.approx(start, abs=0.02)


@pytest.mark.parametrize("seed", range(5))
def test_the_draws_stay_inside_the_sourced_ranges(seed):
    env, p, obs, s = reset(seed)
    rate = float(s.burn_rate) * 86400.0  # ppm a day
    assert 2.3 <= rate <= 3.3
    cold = float(s.T_ref - 0.5 * (s.T_coolant + s.T_cold))
    assert 0.0 <= cold <= rate * K_PER_PPM + 1e-3
