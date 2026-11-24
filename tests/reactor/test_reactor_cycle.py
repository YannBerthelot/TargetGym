"""The core's point in its fuel cycle (owner, 2026-11-15).

Contract, from the plant record (REALITY.md sections 3 and 8; PHYSICS.md section 3),
written before the code:

- A core burns through its cycle at a steady rate, and EDF's units keep their full
  load-following range up to 90 % of the cycle (IEA 2019, EDF, slides 6 and 12). So a
  day's core sits at a point drawn evenly over the first 90 % of its cycle, and the
  operator knows it: observation 13, the fraction of the cycle burnt.
- Boron falls in a straight line through the cycle (BEAVRS, Table 24: 623 to 0 ppm and
  918 to 104 ppm, both straight in burnup), from Krsko's 1690 ppm at the start (fig. 3)
  to 0 at the end.
- "The lower the boron concentration, the less boron is removed for a given volume of
  injected fresh water" (OAPS, section 2.2). Feed and bleed of well-mixed coolant
  removes ``1 - exp(-dilution_flow / primary_mass x 30 s)`` = 0.11993 % of the boron
  in a 30 s step at full flow, whatever the boron.
- The moderator coefficient grows more negative as boron falls, and each ppm is worth
  more. One coordinate moves both in straight lines: -12 pcm/K and -8 pcm/ppm at the
  start, -60 pcm/K and -10.9 pcm/ppm at the end (ours, inside the family's 0 to
  -80 pcm/K and 6.7 to 10.9 pcm/ppm).
- At a fixed load the fuel moves with the coolant, so a ppm of boron moves Tavg by
  ``-boron_worth / (alpha_coolant + alpha_fuel)``: -8 / 15 K at the start, and at 80 %
  of the cycle -10.32 / 53.4 K (-12 - 0.8 x 48 = -50.4 pcm/K with Doppler's -3;
  -8 - 0.8 x 2.9 = -10.32 pcm/ppm).
- Mode A's fastest ramp is "1-2 %NP/min at the beginning of the fuel cycle and only
  0.1-0.3 %NP/min at 80 % of the fuel cycle" (OAPS, section 2.2): 3.3 to 20 times
  slower.
- Burnup moves the core along its cycle: a ppm of boron burnt is 1 / 1690 of it.
- A stream's core refuels when it reaches the end of the load-following range, and
  starts again at the range's start, uncharged; a window never refuels (owner,
  2026-11-11 and 2026-11-15).
- A trip's restart keeps the core: its cycle point is not drawn again.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    core_at,
    cycle_point,
    rod_balance,
    steady_state_temperatures,
)

QUIET = dict(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    initial_tilt_share=0.0,
    initial_xenon_power_range=(1.0, 1.0),
    ramp_down_range=(1e6, 1e6),  # full load all day
)
CYCLE, PLAN = 13, 14  # observation channels


def reset(seed=0, **changes):
    env, p = Reactor(), ReactorParams(**{**QUIET, **changes})
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    return env, p, obs, s


def at_cycle(c, **changes):
    return reset(cycle_range=(c, c), **changes)


def rollout(env, p, s, actions):
    """The plant under ``actions``, one row a step: the observations, the true
    boron and Tavg after each step, and whether it tripped."""
    keys = jax.random.split(jax.random.PRNGKey(7), actions.shape[0])

    def step(s, x):
        a, k = x
        obs, s, _, _, info = env.step_env(k, s, a, p)
        tavg = 0.5 * (s.T_coolant + s.T_cold)
        return s, (obs, s.boron, tavg, info["tripped"])

    s, out = jax.jit(lambda s: jax.lax.scan(step, s, (actions, keys)))(s)
    return s, tuple(np.asarray(x) for x in out)


def hold(s, steps, boron=0.0):
    """Rods where they are, a boron flow, the turbine on its plan."""
    row = [1.0 - 2.0 * float(s.rod_in), boron, 1.0]
    return jnp.tile(jnp.asarray(row, jnp.float32), (steps, 1))


# --- where a day's core sits ---


def test_a_days_core_sits_in_the_first_ninety_percent_of_its_cycle():
    env, p = Reactor(), ReactorParams()
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    obs, states = jax.vmap(lambda k: env.reset_env(k, p))(keys)
    points = np.asarray(obs[:, CYCLE])
    assert points.min() >= 0.0 and points.max() <= 0.9
    # Drawn evenly: 32 draws leave the bottom or the top fifth empty once in
    # 3000 times, (0.7 / 0.9) ** 32.
    assert points.min() < 0.2 and points.max() > 0.7
    np.testing.assert_allclose(
        points, np.asarray(jax.vmap(lambda s: cycle_point(s, p))(states)), atol=1e-6
    )


@pytest.mark.parametrize("c, ppm", [(0.0, 1690.0), (0.5, 845.0), (0.9, 169.0)])
def test_boron_falls_in_a_straight_line_through_the_cycle(c, ppm):
    """Lined out at full load with xenon in balance and the rods parked, the core is
    critical on its cycle point's boron."""
    env, p, obs, s = at_cycle(c)
    assert float(s.boron) == pytest.approx(ppm, abs=0.1)
    assert float(obs[CYCLE]) == pytest.approx(c, abs=1e-6)


# --- what boron can do ---


@pytest.mark.parametrize("c, drop", [(0.0, 2.0268), (0.8, 0.40536)])
def test_full_dilution_removes_less_boron_late_in_the_cycle(c, drop):
    """0.11993 % of the boron a step: 2.0268 ppm of 1690, 0.40536 ppm of 338. The
    first step the water reaches the core is a whole step of full flow."""
    env, p, obs, s = at_cycle(c)
    actions = jnp.concatenate([hold(s, 3, boron=-1.0), hold(s, 40)])
    _, (_, boron, _, tripped) = rollout(env, p, s, actions)
    assert not tripped.any()
    steps = np.diff(np.concatenate([[float(s.boron)], boron]))
    first = int(np.argmax(steps < 0.0))
    assert steps[first] == pytest.approx(-drop, rel=5e-3)


@pytest.mark.parametrize(
    "c, dilute, k_per_ppm", [(0.0, 2, -8.0 / 15.0), (0.8, 10, -10.32 / 53.4)]
)
def test_a_ppm_of_boron_moves_tavg_less_late_in_the_cycle(c, dilute, k_per_ppm):
    """About 4 ppm of dilution, then 35 min for the water to arrive (15 min at most)
    and Tavg to settle: 2.2 K at the start of the cycle, 0.78 K at 80 %."""
    env, p, obs, s = at_cycle(c)
    actions = jnp.concatenate([hold(s, dilute, boron=-1.0), hold(s, 70)])
    end, (_, boron, tavg, tripped) = rollout(env, p, s, actions)
    assert not tripped.any()
    moved = float(boron[-1]) - float(s.boron)
    assert moved == pytest.approx(-4.04, abs=0.05)
    warmed = float(tavg[-1]) - float(0.5 * (s.T_coolant + s.T_cold))
    assert warmed / moved == pytest.approx(k_per_ppm, rel=0.05)


@pytest.mark.parametrize(
    "c, mtc, worth, boron",
    [
        (0.0, -12.0e-5, -8.0e-5, 1690.0),
        (0.8, -50.4e-5, -10.32e-5, 338.0),
        (1.0, -60.0e-5, -10.9e-5, 0.0),
    ],
)
def test_the_coefficients_move_together_in_straight_lines(c, mtc, worth, boron):
    core = core_at(c, ReactorParams())
    assert float(core.alpha_coolant) == pytest.approx(mtc, rel=1e-5)
    assert float(core.boron_worth) == pytest.approx(worth, rel=1e-5)
    assert float(core.boron_ref) == pytest.approx(boron, abs=1e-3)


def carried_rate(c):
    """% of rated load a minute that full dilution carries on the program: its
    reactivity a second over the power defect per unit of load, at the cycle point's
    full-power boron."""
    core = core_at(c, ReactorParams())

    def critical(load):
        T_fuel, T_hot, T_cold, _ = steady_state_temperatures(load, core)
        return float(rod_balance(T_fuel, 0.5 * (T_hot + T_cold), 1.0, core))

    defect = (critical(1.0) - critical(0.5)) / 0.5
    dilution = core.dilution_flow / core.primary_mass * core.boron_ref
    return float(dilution * abs(core.boron_worth) / abs(defect) * 60.0 * 100.0)


def test_boron_carries_far_slower_ramps_at_eighty_percent_of_the_cycle():
    """OAPS: 1 to 2 %/min at the start, 0.1 to 0.3 %/min at 80 %, so 3.3 to 20 times
    slower. By hand the model gives 32.4 pcm/min over 16.4 pcm/% = 1.97 %/min at the
    start and 8.37 over 22.4 = 0.37 %/min at 80 %, 5.3 times slower: the ratio is
    the source's; the late rate sits a quarter above the source's 0.3 (PHYSICS.md
    section 2)."""
    start, late = carried_rate(0.0), carried_rate(0.8)
    assert 1.0 <= start <= 2.0
    assert 3.3 <= start / late <= 20.0


# --- the core moves along its cycle ---


def test_burnup_moves_the_core_along_its_cycle():
    """3.38 ppm a day is 0.002 of 1690 ppm."""
    env, p, obs, s = at_cycle(0.5, burnup_range=(3.38, 3.38))
    _, (obs, _, _, tripped) = rollout(env, p, s, hold(s, 2880))
    assert not tripped.any()
    assert float(obs[-1, CYCLE]) == pytest.approx(0.502, abs=2e-5)


REFUEL = dict(cycle_range=(0.5, 0.5002), burnup_range=(3.38, 3.38))
PER_STEP = 0.002 / 2880.0  # of the cycle, at 3.38 ppm a day


def test_a_streams_core_refuels_at_the_end_of_its_range():
    """The range is 0.0002 of the cycle wide, 288 steps at 3.38 ppm a day. The core
    is renewed on the step it reaches the end, back at the range's start and
    critical on that point's boron."""
    env, p, obs, s = reset(refuels=True, **REFUEL)
    due = int(np.ceil((0.5002 - float(obs[CYCLE])) / PER_STEP))
    assert due > 10  # the draw leaves the core some way to go
    _, (obs, boron, _, _) = rollout(env, p, s, hold(s, due + 5))
    points = obs[:, CYCLE]
    assert np.all(np.diff(points[: due - 2]) > 0.0)
    back = int(np.argmax(np.diff(points) < 0.0)) + 1
    assert abs(back - (due - 1)) <= 1
    assert float(points[back]) == pytest.approx(0.5, abs=1e-5)
    assert float(boron[back]) == pytest.approx(845.0, abs=0.5)


def test_a_window_never_refuels():
    env, p, obs, s = reset(**REFUEL)
    _, (obs, _, _, _) = rollout(env, p, s, hold(s, 300))
    assert np.all(np.diff(obs[:, CYCLE]) > 0.0)
    assert float(obs[-1, CYCLE]) > 0.5002


def test_a_restart_keeps_the_cores_cycle_point():
    env, p = Reactor(), ReactorParams(sensor_noise_scale=0.0)
    obs, s = env.reset_env(jax.random.PRNGKey(3), p)
    before = float(obs[CYCLE])
    # The whole primary and the steam side at 700 K: the hot leg stays past its
    # 610 K limit through the step's first sub-step, a scram.
    hot = s.T_coolant * 0 + 700.0
    scram = s.replace(T_coolant=hot, T_cold=hot, T_sg=hot)
    action = jnp.asarray([1.0 - 2.0 * float(s.rod_in), 0.0, 1.0], jnp.float32)
    obs, s, _, _, info = env.step_env(jax.random.PRNGKey(4), scram, action, p)
    assert bool(info["tripped"])
    assert float(obs[CYCLE]) == pytest.approx(before, abs=1e-5)
