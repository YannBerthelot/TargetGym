"""The turbine's loading rate, the reactor's third action (owner, 2026-11-15).

Contract, from the plant record (REALITY.md sections 3 and 6), written before the code:

- The secondary operator "has to set the power target and the power variation rate of
  the turbine according to the grid dispatcher instructions", and the reactor's power
  follows the turbine's (OAPS, sections 1.2 and 2.1). So the third action is the
  turbine's loading rate: +1 loads at the turbine's fastest rate, -1 unloads at it and
  0 holds the load.
- The fastest rate is 5 % of rated load a minute, the EUR's fastest return (REALITY.md
  section 6): 2.5 % of rated load over a 30 s step.
- The dispatch's plan is the most the turbine takes, so at +1 it follows the plan, and
  it follows the plan down whatever the action.
- The plan the turbine does not take is lost output at the spot price, $80/MWh on
  1 GWe, as the trip's and the derate's are: $22.222 a second per unit of load, so
  $111.11 per 5 s sub-step.
- The turbine does not unload below the load a restart comes back at, 15 % (Doel-3).

Every plant here is quiet: no load wobble, no sensor noise, no burnup, no debt carried
in, at the start of its cycle, xenon balanced with the load it starts at.
"""

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import compute_reward_terms

QUIET = dict(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
)
XE_FULL = dict(initial_xenon_power_range=(1.0, 1.0))  # xenon balanced with full power
BASELOAD = dict(ramp_down_range=(1e6, 1e6), **XE_FULL)  # full load all day
#: The plan starts down at 600 s, at the default 2 %/min.
DOWN_AT_600 = dict(ramp_down_range=(600.0, 600.0), **XE_FULL)
#: The plan's ramp down ran before the day began (it started 3000 s ago and took
#: 1500 s at 2 %/min), so the day starts lined out at 50 %, xenon balanced with it,
#: and the return starts at 600 s: -3000 + 1500 + 2100.
AT_THE_HOLD = dict(
    ramp_down_range=(-3000.0, -3000.0),
    hold_range=(2100.0, 2100.0),
    initial_xenon_power_range=(0.5, 0.5),
)
RETURN_AT = 600.0  # s
PLAN_RATE = 0.02 / 60.0  # of rated load a second


def reset(seed=0, **changes):
    env, p = Reactor(), ReactorParams(**{**QUIET, **changes})
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    return env, p, obs, s


def run(env, p, s, turbine, steps):
    """``steps`` control steps with the rods held where they are, no boron and the
    turbine's action ``turbine``: the load read each step, the power, the tracking
    cost and whether any step tripped."""
    hold = 1.0 - 2.0 * float(s.rod_in)
    loads, powers, tracking, tripped = [], [], [], False
    for k in range(steps):
        action = np.array([hold, 0.0, turbine], np.float32)
        obs, s, _, _, info = env.step_env(jax.random.PRNGKey(k), s, action, p)
        loads.append(float(obs[3]))
        powers.append(float(obs[0]))
        tracking.append(float(info["reward_terms"]["tracking"]))
        tripped = tripped or bool(info["tripped"])
    return s, np.array(loads), np.array(powers), np.array(tracking), tripped


def test_the_action_is_rods_boron_and_the_turbines_loading_rate():
    env, p = Reactor(), ReactorParams()
    space = env.action_space(p)
    assert space.shape == (3,)
    np.testing.assert_array_equal(np.asarray(space.low), -np.ones(3))
    np.testing.assert_array_equal(np.asarray(space.high), np.ones(3))


def test_at_full_rate_the_turbine_follows_the_plan():
    """The plan starts down at 600 s at 2 %/min; step k ends at 30 (k + 1) s."""
    env, p, obs, s = reset(**DOWN_AT_600)
    _, loads, _, tracking, tripped = run(env, p, s, turbine=1.0, steps=30)
    t = 30.0 * (np.arange(30) + 1)
    plan = 1.0 - PLAN_RATE * np.maximum(t - 600.0, 0.0)
    assert not tripped
    np.testing.assert_allclose(loads, plan, atol=1e-5)
    assert plan[-1] == pytest.approx(0.9)  # 300 s of ramp by the last step


def test_the_turbine_follows_the_plan_down_whatever_the_action():
    env, p, obs, s = reset(**DOWN_AT_600)
    _, loads, _, _, _ = run(env, p, s, turbine=0.0, steps=30)
    t = 30.0 * (np.arange(30) + 1)
    np.testing.assert_allclose(
        loads, 1.0 - PLAN_RATE * np.maximum(t - 600.0, 0.0), atol=1e-5
    )


def test_at_zero_the_turbine_holds_its_load_and_the_reactor_stays_with_it():
    """Held at 50 % while the plan returns: the load and the power stay at 50 %."""
    env, p, obs, s = reset(**AT_THE_HOLD)
    assert float(obs[3]) == pytest.approx(0.5, abs=1e-6)
    _, loads, powers, _, tripped = run(env, p, s, turbine=0.0, steps=40)
    assert not tripped
    np.testing.assert_allclose(loads, 0.5, atol=1e-6)
    np.testing.assert_allclose(powers[-10:], 0.5, atol=0.01)


def test_the_plan_the_turbine_does_not_take_is_lost_output():
    """Step 30 runs from 900 s to 930 s. Its sub-steps end 305, 310, ... 330 s into
    the plan's return, where the plan is 2 %/min above the 50 % the turbine holds:
    the shortfalls sum to (6 x 300 + 5 x 21) / 3000 = 0.635 of rated load, at
    $111.11 per sub-step, $70.56. Nothing else is charged: Tavg stays on its program
    and the offset inside its band."""
    env, p, obs, s = reset(**AT_THE_HOLD)
    _, _, _, tracking, _ = run(env, p, s, turbine=0.0, steps=31)
    np.testing.assert_allclose(tracking[:19], 0.0, atol=1e-6)  # before the return
    assert tracking[30] == pytest.approx(70.556, rel=1e-3)


def test_a_turbine_under_its_plan_is_charged_per_unit_of_load():
    """Lined out at full load with the turbine capped at 90 %: 0.1 of rated load at
    $111.11 per sub-step."""
    env, p, obs, s = reset(**BASELOAD)
    terms = compute_reward_terms(s.replace(pickup=s.pickup * 0 + 0.9), p)
    assert float(terms["tracking"]) == pytest.approx(11.111, rel=1e-4)


def test_the_fastest_loading_is_five_percent_a_minute():
    """Held for 30 min, the plan is at 90 %, 40 % above the turbine; four steps at
    full rate then load 4 x 2.5 %."""
    env, p, obs, s = reset(**AT_THE_HOLD)
    s, loads, _, _, _ = run(env, p, s, turbine=0.0, steps=60)
    assert loads[-1] == pytest.approx(0.5, abs=1e-6)
    _, loads, _, _, _ = run(env, p, s, turbine=1.0, steps=4)
    np.testing.assert_allclose(loads, [0.525, 0.55, 0.575, 0.6], atol=1e-5)


def test_the_fastest_unloading_is_five_percent_a_minute():
    env, p, obs, s = reset(**BASELOAD)
    _, loads, _, _, _ = run(env, p, s, turbine=-1.0, steps=4)
    np.testing.assert_allclose(loads, [0.975, 0.95, 0.925, 0.9], atol=1e-5)


def test_half_the_action_is_half_the_rate():
    env, p, obs, s = reset(**BASELOAD)
    _, loads, _, _, _ = run(env, p, s, turbine=-0.5, steps=4)
    np.testing.assert_allclose(loads, [0.9875, 0.975, 0.9625, 0.95], atol=1e-5)


def test_the_turbine_does_not_unload_below_the_restart_load():
    """A day planned at 20 % all through, lined out there. Full unloading takes
    2.5 % a step and stops at the 15 % a restart comes back at."""
    env, p, obs, s = reset(
        load_min=0.2,
        ramp_down_range=(-1e4, -1e4),
        hold_range=(1e6, 1e6),
        initial_xenon_power_range=(0.2, 0.2),
    )
    assert p.restart_load == pytest.approx(0.15)
    assert float(obs[3]) == pytest.approx(0.2, abs=1e-6)
    _, loads, _, _, tripped = run(env, p, s, turbine=-1.0, steps=4)
    assert not tripped
    np.testing.assert_allclose(loads, [0.175, 0.15, 0.15, 0.15], atol=1e-5)
