"""The turbine's limit on the steam it gets (owner, 2026-11-15).

Contract, from the plant record (REALITY.md section 3), written before the code. A
planner free to run Tavg under its program loaded the turbine on steam a real one
could not: nothing in the plant tied the load to the steam's pressure.

- The steam is saturated at the steam generator's temperature. The steam table
  gives 5.503, 6.417, 7.442 and 8.588 MPa at 270, 280, 290 and 300 C; the plant's
  curve, ``ln p = a - b / T`` through the first and the last (b = 4618.5 K), must
  give the two between to 0.1 %.
- A condensing turbine passes steam in proportion to its inlet pressure over the
  square root of its inlet temperature (Stodola's cone law with a negligible outlet
  pressure), and with its valves wide open it passes 105 % of its rated flow at its
  rated steam (Wolf Creek's valves-wide-open heat balance). Rated steam is the
  full-load program's, 580.2 - 22 = 558.2 K. So the most load the turbine takes,
  by hand: 1.05 at 558.2 K; at 555.2 K, 1.05 x exp(4618.5 x (1 / 558.2 - 1 /
  555.2)) x sqrt(558.2 / 555.2) = 1.05 x 0.95628 x 1.00270 = 1.00680; at 552.2 K,
  1.05 x 0.91402 x 1.00542 = 0.96492.
- On its program the turbine is never held back: the steam is warmer at part load
  (564.8 - 6.6 x load), so the valves pass 1.134 at 15 % of load, 1.099 at half
  load, 1.05 at full load.
- A turbine on its plan whose steam turns 6 K colder than rated takes 0.96492 of
  rated load, not its plan's 1.0, and the 0.03508 it does not take is lost output:
  $22.222 a second per unit of load, $3.90 over a 5 s sub-step.
- The limit clears itself: the steam generator closes on the primary in C_sg /
  UA_sg = 7.3 s, and the turbine reloads 2.5 % a step, so three control steps later
  the turbine is back on its plan.

Every plant here is quiet: no load wobble, no sensor noise, no burnup, no debt carried
in, at the start of its cycle, xenon balanced with full load, on full load all day.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    NOT_RELOADING,
    compute_next_state,
    compute_reward_terms,
    steady_state_temperatures,
    steam_pressure,
    turbine_capacity,
)

P = ReactorParams(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
    initial_xenon_power_range=(1.0, 1.0),
    ramp_down_range=(1e6, 1e6),
)
RATED = 558.2  # K, the steam at full load on the program


@pytest.mark.parametrize(
    "celsius, mpa", [(270.0, 5.503), (280.0, 6.417), (290.0, 7.442), (300.0, 8.588)]
)
def test_the_steams_pressure_follows_the_steam_table(celsius, mpa):
    at = lambda c: float(steam_pressure(c + 273.15, P, np))  # noqa: E731
    assert at(celsius) / at(270.0) == pytest.approx(mpa / 5.503, rel=1e-3)


@pytest.mark.parametrize(
    "colder, capacity", [(0.0, 1.05), (3.0, 1.00680), (6.0, 0.96492)]
)
def test_the_valves_wide_open_pass_less_on_colder_steam(colder, capacity):
    out = float(turbine_capacity(RATED - colder, P, np))
    assert out == pytest.approx(capacity, rel=1e-4)


@pytest.mark.parametrize("load, capacity", [(0.15, 1.134), (0.5, 1.099), (1.0, 1.05)])
def test_on_its_program_the_turbine_is_never_held_back(load, capacity):
    steam = float(steady_state_temperatures(load, P)[3])
    assert steam == pytest.approx(564.8 - 6.6 * load, abs=1e-3)
    assert float(turbine_capacity(steam, P, np)) == pytest.approx(capacity, rel=1e-3)


def _full_load():
    """A plant lined out at full load on its plan, and the action that holds it."""
    _, s = Reactor().reset_env(jax.random.PRNGKey(0), P)
    assert float(s.T_sg) == pytest.approx(RATED, abs=0.01)
    return s, jnp.array([1.0 - 2.0 * float(s.rod_in), 0.0, 1.0])


def test_a_turbine_on_cold_steam_takes_what_its_valves_pass():
    s, hold = _full_load()
    cold = s.replace(T_sg=s.T_sg - 6.0)
    after, _ = compute_next_state(hold, cold, P)
    assert float(after.pickup) == pytest.approx(0.96492, rel=1e-3)
    assert float(after.target_n) == pytest.approx(0.96492, rel=1e-3)
    assert float(compute_reward_terms(after, P)["tracking"]) == pytest.approx(
        3.90, abs=0.03
    )
    # On rated steam the same step leaves the turbine on its plan, at no cost.
    on, _ = compute_next_state(hold, s, P)
    assert float(on.pickup) > 0.5 * NOT_RELOADING
    assert float(compute_reward_terms(on, P)["tracking"]) == pytest.approx(
        0.0, abs=1e-3
    )


def test_the_turbine_is_back_on_its_plan_once_the_steam_is():
    env = Reactor()
    s, hold = _full_load()
    s = s.replace(T_sg=s.T_sg - 6.0)
    key, under = jax.random.PRNGKey(0), []
    for _ in range(3):
        s = env.step_env(key, s, hold, P)[1]
        under.append(float(s.pickup) < 0.5 * NOT_RELOADING)
    assert under[0]  # the first step ends with the turbine under its plan
    assert not under[-1]
