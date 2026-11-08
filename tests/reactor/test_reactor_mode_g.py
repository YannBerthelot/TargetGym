"""The reactor's rods as EDF's Mode G (Muniglia et al., ICAPP 2016, section II.C;
arXiv 2107.11201, section 2).

Contract:

- three inputs: the power-shimming group's demand, the temperature group R's demand,
  the boron flow; +1 is all the way out for both groups;
- R stays in its manoeuvring band, 27 steps of a 260-step core at the top, whatever
  it is asked;
- the shimming group moves at most 60 steps/min, R at most 72 steps/min;
- the shimming group's worth is the power defect from full power to none, so its
  calibration puts it fully out at full power (derived: the plant's own balance);
- R's band is worth 2 x 2.8 K of the moderator coefficient, 30 pcm/K: 168 pcm (ours:
  the band spans the speed program's full scale both ways);
- the observation adds R's position, on the same scale as the shimming group's;
- reset is critical with the shimming group on its calibration and R mid-band.
"""

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    compute_next_state,
    r_worth,
    rod_balance,
    shim_position,
    steady_state_temperatures,
)

BAND = 27.0 / 260.0


def reset(seed=0, **changes):
    env, p = Reactor(), ReactorParams(**changes)
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    return env, p, obs, s


def sub_step(env, p, s, action):
    return compute_next_state(np.asarray(action, np.float32), s, p,
                              integration_method=env.integration_method)[0]  # fmt: skip


def test_the_action_has_three_inputs_and_the_observation_r():
    env, p, obs, s = reset()
    assert env.action_space(p).shape == (3,)
    assert obs.shape == (11,)
    assert float(obs[10]) == pytest.approx(1.0 - 2.0 * float(s.r_in) / BAND, abs=1e-6)


@pytest.mark.parametrize("demand", [-1.0, 1.0])
def test_r_stays_in_its_band(demand):
    env, p, obs, s = reset()
    for _ in range(300):
        s = sub_step(env, p, s, [1.0, demand, 0.0])
    assert 0.0 <= float(s.r_in) <= BAND + 1e-7
    assert float(s.r_in) == pytest.approx(BAND if demand < 0 else 0.0, abs=1e-6)


def test_the_groups_move_at_their_speeds():
    env, p, obs, s = reset()
    s1 = sub_step(env, p, s, [-1.0, -1.0, 0.0])
    # 5 s: 60 steps/min is 5 steps, 72 steps/min is 6 steps, of 260.
    assert float(s1.rod_in - s.rod_in) == pytest.approx(5.0 / 260.0, rel=1e-4)
    assert float(s1.r_in - s.r_in) == pytest.approx(6.0 / 260.0, rel=1e-4)


def test_the_shimming_group_is_worth_the_power_defect():
    p = ReactorParams()

    def needed(load):
        T_fuel, T_hot, T_cold, _ = steady_state_temperatures(load, p)
        return float(rod_balance(T_fuel, 0.5 * (T_hot + T_cold), 1.0, p))

    assert -p.rho_ext_min == pytest.approx(needed(1.0) - needed(0.0), rel=1e-3)
    assert float(shim_position(1.0, p)) == pytest.approx(0.0, abs=1e-4)
    # Its S-curve is half its worth at half the stroke, and half the defect is at 50 %.
    assert float(shim_position(0.5, p)) == pytest.approx(0.5, abs=0.02)


def test_r_band_is_worth_168_pcm():
    p = ReactorParams()
    assert float(r_worth(BAND, p) - r_worth(0.0, p)) == pytest.approx(-168e-5, rel=1e-3)


def test_reset_is_critical_with_r_mid_band():
    env, p, obs, s = reset(demand_sigma=0.0)  # no wobble on the turbine
    assert float(s.r_in) == pytest.approx(BAND / 2.0, rel=1e-5)
    hold = [1.0 - 2.0 * float(s.rod_in), 0.0, 0.0]
    for _ in range(120):
        s = sub_step(env, p, s, hold)
    assert float(s.n) == pytest.approx(1.0, abs=2e-3)
