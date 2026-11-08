"""The building needs its heater, on a day's energy balance.

Contract, from the review's finding and the heating-season irradiance PHYSICS.md
cites: ``solar_peak`` is a seasonal mean, not a clear-sky peak.

- Free gains stay well below the losses. A day of half-sine solar over 12 h at
  175 W/m2 is ``175 * 12 * 2 / pi`` = 1337 Wh/m2, or 18.4 kWh through 25 m2 at g
  0.55. An office's occupancy, 08:00 to 18:00 [6], adds ``(8 * 10 + 2 * 14) *
  150`` = 16.2 kWh: 34.6 kWh. Holding 20 C over a 5 C day loses ``159.4 * 15 *
  24`` = 57.4 kWh, so 22.8 kWh is the heater's, two fifths of the demand.
- Left off, the zone settles where gains balance losses, ``5 + 1442 / 159.4`` =
  14.0 C on a day's mean, 7 K under the band.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.hvac.env import (
    HVACParams,
    internal_gain,
    solar_gain,
    total_heat_loss_coefficient,
)
from target_gym.hvac.env_jax import BuildingHVAC

STEPS_PER_DAY = 96  # 15 min


def _kwh_per_day(gain, params):
    times = jnp.arange(STEPS_PER_DAY)
    return float(jax.vmap(lambda t: gain(t, params))(times).sum()) * 0.25 / 1000.0


def test_a_days_free_gains_leave_two_fifths_of_the_loss_to_the_heater():
    params = HVACParams()
    free = _kwh_per_day(solar_gain, params) + _kwh_per_day(internal_gain, params)
    loss = float(total_heat_loss_coefficient(params)) * (20.0 - 5.0) * 24.0 / 1000.0
    assert free == pytest.approx(34.6, abs=0.2)
    assert loss == pytest.approx(57.4, abs=0.1)


STEPS = 34 * STEPS_PER_DAY


def test_a_building_left_unheated_settles_at_its_free_running_mean():
    """The last of 34 days. The slab adds 9e7 J/K to the 2.5e7 J/K of the
    5R1C mass, so the building's time constant is 1.15e8 / 159.4 = 200 h, and
    34 days is four of them: 2 % of the 5 K start is left, and the 0.5 % of
    the gain the standard sends back through the glazing takes 0.05 K. The
    weather is off."""
    params, env = (
        HVACParams(T_out_noise_std=0.0, sensor_noise_scale=0.0),
        BuildingHVAC(),
    )
    _, state = env.reset_env(jax.random.PRNGKey(0), params)

    def body(s, _):
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(1), s, -jnp.ones(1), params)
        return s, s.T_air

    _, air = jax.lax.scan(body, state, None, length=STEPS)
    assert float(np.mean(air[-STEPS_PER_DAY:])) == pytest.approx(14.0, abs=0.3)
