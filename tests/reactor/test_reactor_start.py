"""How a day starts: yesterday's axial xenon tilt (the physics review's finding 3,
owner 2026-11-10).

Contracts, before the code:

- A unit that load-follows starts its day from where yesterday left its axial
  xenon: on a share of days equal to load following's, reset draws the iodine and
  xenon tilts as one pair from a table of load-follow end-of-day states, every pair
  equally likely; otherwise yesterday was baseload and the tilts start at 0
  (PHYSICS.md section 4).
- The plant starts consistent with that tilt: xenon heavier in the top half pushes
  power down the core, so the starting delta I is negative, and the reading shows it.
"""

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams

TABLE = ((0.001, 0.02), (-0.003, -0.01), (0.0, 0.0))


@pytest.mark.parametrize("seed", range(6))
def test_after_a_load_follow_day_the_tilts_are_one_row_of_the_table(seed):
    env, p = Reactor(), ReactorParams(initial_tilts=TABLE, initial_tilt_share=1.0)
    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    pair = (float(s.I_tilt), float(s.Xe_tilt))
    assert any(np.allclose(pair, row, atol=1e-7) for row in TABLE)


@pytest.mark.parametrize("seed", range(3))
def test_after_a_baseload_day_there_is_no_tilt(seed):
    env, p = Reactor(), ReactorParams(initial_tilts=TABLE, initial_tilt_share=0.0)
    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    assert (float(s.I_tilt), float(s.Xe_tilt)) == (0.0, 0.0)


def test_xenon_heavier_at_the_top_starts_the_power_low_in_the_core():
    env = Reactor()
    p = ReactorParams(
        initial_tilts=((0.0, 0.02),), initial_tilt_share=1.0, sensor_noise_scale=0.0
    )
    obs, s = env.reset_env(jax.random.PRNGKey(0), p)
    assert float(s.Xe_tilt) == pytest.approx(0.02)
    assert float(s.delta_I) < -0.005
    assert float(obs[6]) == pytest.approx(float(s.delta_I), abs=1e-6)
