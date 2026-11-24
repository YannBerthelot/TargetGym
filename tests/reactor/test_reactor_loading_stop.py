"""The turbine's loading stop on a low Tavg (interlock C-16; owner, 2026-11-16).

Contract, from the plant record (REALITY.md section 4), written before the code.
What the interlock does is Vogtle's safety report's (UFSAR 7.7.1.4.3, 7.7.2.4 and
Table 7.7.1-1, read in full); its setpoint, 10 F, is a training card's (the owner's
pick, flagged in the record).

- The interlock compares the coolant's average temperature with the program's
  reference at the turbine's load, and stops the turbine's loading once Tavg has
  fallen its setpoint under the reference: 10 F, 5.556 K. So a turbine at 60 % of
  load under its plan, asked to load at its fastest (5 %/min, 0.41667 % of rated
  load over a 5 s sub-step), is at 60.41667 % a sub-step later with Tavg 5.5 K
  under its program, and still at 60 % with Tavg 5.6 K under it. A Tavg over its
  program does not stop it: with Tavg 6 K over, it loads.
- It stops the loading and nothing else (the report's table: automatic loading
  stops until the condition clears): the same turbine asked to unload goes to
  59.58333 %, and a turbine on a flat plan stays on it.
- It does not latch: loading can begin again once a rise in the coolant's
  temperature has cleared the interlock. A stopped turbine loads again on the next
  sub-step once Tavg is back within its setpoint.
- What it is for: it keeps the coolant's temperature from dropping too far in a
  rapid return on the moderator alone. By hand at the start of the cycle, with the rods
  and the boron still: a unit of load takes Tavg 94.2 K down (3 pcm/K of Doppler on
  the fuel's 471.2 K over the 15 pcm/K of both feedbacks) while its program rises
  15.4 K, so Tavg falls 109.6 K under its program per unit of load, and a turbine
  loading from a line-out at 50 % reaches the stop 5.556 / 109.6 = 5.07 % of rated
  load up, at 55.07 %, no earlier: the plant's heat only lags. The lag: the heat to
  shed for a unit of load is 2.36e11 J (the primary's 1.4e9 J/K over 94.2 K, the
  steam generator's 1e9 J/K over 94.2 + 22 K, less the fuel's 3.3e7 J/K over the
  377 K it warms) against 3e9 W per unit of load not yet followed, 79 s, in which
  the turbine loads 6.6 %; a control step's sampling adds 2.5 %. So, asked to load
  for ten minutes, the turbine stops between 55 % and 55.07 + 6.6 + 2.5 = 64.2 %,
  where the plan is at full load, and the xenon it then burns off clears the stop
  by well under 1 % of load in 2.5 min.

Every plant here is quiet: no load wobble, no sensor noise, no burnup, no debt carried
in, at the start of its cycle, xenon balanced with the load it starts at.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import NOT_RELOADING, compute_next_state, tavg_program

QUIET = dict(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
)
#: Full load all day, xenon balanced with it.
BASELOAD = ReactorParams(
    ramp_down_range=(1e6, 1e6), initial_xenon_power_range=(1.0, 1.0), **QUIET
)
#: Lined out at 50 %, xenon balanced with it; the plan's return starts at 600 s at
#: 2 %/min and is at full load from 2100 s (as in ``test_reactor_turbine.py``).
AT_THE_HOLD = ReactorParams(
    ramp_down_range=(-3000.0, -3000.0),
    hold_range=(2100.0, 2100.0),
    initial_xenon_power_range=(0.5, 0.5),
    **QUIET,
)
LOAD, UNLOAD = 1.0, -1.0
SUB_STEP = 0.05 / 60.0 * 5.0  # of rated load: 5 %/min over 5 s


def _under_its_plan(under, load=0.6):
    """A turbine at ``load`` under a full-load plan, Tavg ``under`` K under the
    program at that load, and the action that holds the rods with the turbine
    asked for ``turbine``."""
    _, s = Reactor().reset_env(jax.random.PRNGKey(0), BASELOAD)
    T_ref = jnp.float32(tavg_program(load, BASELOAD))
    shift = T_ref - under - 0.5 * (s.T_coolant + s.T_cold)
    s = s.replace(
        pickup=s.pickup * 0 + load,
        T_ref=T_ref,
        T_coolant=s.T_coolant + shift,
        T_cold=s.T_cold + shift,
    )
    return s, lambda turbine: jnp.array([1.0 - 2.0 * float(s.rod_in), 0.0, turbine])


@pytest.mark.parametrize(
    "under, after",
    [(-6.0, 0.6 + SUB_STEP), (0.0, 0.6 + SUB_STEP), (5.5, 0.6 + SUB_STEP), (5.6, 0.6)],
)
def test_the_turbine_stops_loading_with_tavg_ten_fahrenheit_under_its_program(
    under, after
):
    s, act = _under_its_plan(under)
    out, _ = compute_next_state(act(LOAD), s, BASELOAD)
    assert float(out.pickup) == pytest.approx(after, abs=1e-6)


def test_the_stop_leaves_the_turbine_free_to_unload():
    s, act = _under_its_plan(6.0)
    out, _ = compute_next_state(act(UNLOAD), s, BASELOAD)
    assert float(out.pickup) == pytest.approx(0.6 - SUB_STEP, abs=1e-6)


def test_the_stop_leaves_a_turbine_on_a_flat_plan_on_it():
    s, act = _under_its_plan(6.0, load=1.0)
    s = s.replace(pickup=s.pickup * 0 + NOT_RELOADING)
    out, _ = compute_next_state(act(LOAD), s, BASELOAD)
    assert float(out.pickup) > 0.5 * NOT_RELOADING
    assert float(out.target_n) == pytest.approx(1.0, abs=1e-6)


def test_a_stopped_turbine_loads_again_once_tavg_is_back():
    s, act = _under_its_plan(5.6)
    stopped, _ = compute_next_state(act(LOAD), s, BASELOAD)
    assert float(stopped.pickup) == pytest.approx(0.6, abs=1e-6)
    back = stopped.T_ref - 5.5 - 0.5 * (stopped.T_coolant + stopped.T_cold)
    warmer = stopped.replace(
        T_coolant=stopped.T_coolant + back, T_cold=stopped.T_cold + back
    )
    out, _ = compute_next_state(act(LOAD), warmer, BASELOAD)
    assert float(out.pickup) == pytest.approx(0.6 + SUB_STEP, abs=1e-6)


def test_a_return_on_the_moderator_alone_stops_a_few_percent_up():
    env, p = Reactor(), AT_THE_HOLD
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    hold = 1.0 - 2.0 * float(s.rod_in)  # the rods where they are, no boron
    loads, tripped = [], False
    for k in range(80):  # 30 min held at 50 %, then 10 min asked to load
        action = np.array([hold, 0.0, 0.0 if k < 60 else LOAD], np.float32)
        obs, s, _, _, info = step(jax.random.PRNGKey(k), s, action, p)
        loads.append(float(obs[3]))
        tripped = tripped or bool(info["tripped"])
    assert not tripped
    assert loads[59] == pytest.approx(0.5, abs=1e-6)
    assert 0.55 <= loads[-1] <= 0.642
    assert loads[-1] - loads[-6] < 0.005
