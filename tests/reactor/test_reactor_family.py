"""The reactor against its family's documented behaviour (charter rule 19).

Contracts, from sources, written before the runs:

- Rod gain (Krsko, NENE 2022 paper 911, figs 3 and 6, a Westinghouse 2-loop PWR):
  through a 100-50-100 day at about 8 %/h, bank D travels from 225 to about 190
  steps (35 steps) while boron covers the rest, and delta I stays inside +-5 %
  around the target all day. So in a member of this family, the bank's whole
  travel moves delta I by no more than the band's width, 10 %, and the same day,
  rods for delta I inside that travel and boron for criticality, stays inside
  +-5 % (read off a figure: 0.01 slack).
- Xenon's swing on a load cut (Krsko fig. 5, the same day): at the 50 % point the
  mean xenon reads 110 against 96 at full power, 1.146 times. Tolerance 0.05: two
  readings off a figure, about 2 in 100 each, and the 50 % point's time to half an
  hour, while xenon climbs about 2 % an hour.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import flux_tilt, moderator_tilt, rod_tilt

STROKE = 228.0
PARK = 3.0  # 225 steps withdrawn, Krsko fig. 3
KRSKO_TRAVEL = 35.0  # 225 to 190 steps, fig. 3's axis and trace
KRSKO_BAND = 0.05  # +-5 % delta I, section 2
FIGURE_SLACK = 0.01  # reading a delta I trace off fig. 6


@pytest.mark.parametrize("n", [1.0, 0.5])
def test_the_banks_whole_travel_moves_delta_i_by_less_than_the_band(n):
    p = ReactorParams()
    rise = p.P_thermal_ref * n / p.m_dot_cp

    def delta_I(steps_in):
        eps = rod_tilt(steps_in / STROKE, p) + moderator_tilt(rise, 0.0, p)
        return n * float(flux_tilt(0.0, 1.0, n, eps, p))

    swing = delta_I(PARK) - delta_I(PARK + KRSKO_TRAVEL)
    assert 0.0 < swing <= 2.0 * KRSKO_BAND


@pytest.fixture(scope="module")
def krsko_day():
    """Krsko's day 1: 100 to 50 % at 8 %/h and straight back. As in the paper, bank
    D is positioned for delta I but only inside its 35-step travel, and boron
    trims Tavg for criticality. The two loops' gains are ours: a procedure, not
    the plant; a slow trim, since boron arrives 5 to 15 min late."""
    env = Reactor()
    p = ReactorParams(
        demand_sigma=0.0,
        initial_xenon_power_range=(1.0, 1.0),
        load_ramp_rate=0.08 / 3600.0,
    )
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    down_at = 3600.0
    s = s.replace(
        ramp_down_at=jnp.asarray(down_at, s.n.dtype),
        ramp_up_at=jnp.asarray(down_at + 0.5 / p.load_ramp_rate, s.n.dtype),
    )

    def step(carry, _):
        s, steps_in = carry
        # 0.1 step per 30 s step per 1 % of delta I, inside the bank's travel.
        steps_in = jnp.clip(steps_in + 10.0 * s.delta_I, PARK, PARK + KRSKO_TRAVEL)
        rods = 1.0 - 2.0 * steps_in / STROKE
        error = s.T_ref - 0.5 * (s.T_coolant + s.T_cold)  # K, positive when cold
        boron = jnp.clip(-0.1 * error, -1.0, 1.0)  # dilute when cold
        a = jnp.stack([rods, boron]).astype(jnp.float32)
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
        return (s, steps_in), (s.delta_I, s.time, s.Xe_hat, s.target_n)

    start = (s, jnp.asarray(PARK, s.n.dtype))
    _, trace = jax.lax.scan(step, start, None, length=p.max_steps_in_episode)
    return [np.asarray(x) for x in trace]


def test_a_krsko_day_with_the_banks_travel_stays_inside_the_band(krsko_day):
    dI, t, _, _ = krsko_day
    assert int(np.sum(t == 0)) == 0  # no trip
    assert float(np.max(np.abs(dI))) <= KRSKO_BAND + FIGURE_SLACK


def test_xenon_swings_on_the_load_cut_as_krskos(krsko_day):
    _, _, xenon, load = krsko_day
    bottom = int(np.argmin(load))
    assert xenon[bottom] == pytest.approx(110.0 / 96.0, abs=0.05)
