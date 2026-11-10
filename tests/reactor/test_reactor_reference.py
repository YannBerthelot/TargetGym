"""The reference's xenon tool and its override's handback.

Contracts (Amendment 3's reactor row; OAPS section 1 for the operator's xenon
tool; Shinskey's bumpless transfer for the override):

- The xenon tool runs the textbook iodine and xenon equations on the observed
  power. Started in balance, it follows the plant's mean xenon through a load
  cut (the plant's means add only the tilt's small burn term), and while xenon
  builds it asks for dilution, since xenon's poison must be met by less boron.
- The reference, bank D on the offset, never sits deeper than the rod insertion
  limit (the owner's condition for adopting it, 2026-11-08). The saturation
  override's handback test went with the Mode A reference it tested.
- The xenon tool has play (owner, 2026-11-10; PHYSICS.md section 6): the boron it
  did not move is a debt, read as the Tavg shift it leaves at steady power,
  ``debt * boron_worth / (alpha_coolant + alpha_fuel)``: at a fixed power the fuel
  is a fixed step above the coolant, so it cools with it (here 8/15, 0.5333 K per
  ppm). Inside Tavg's free zone, the 1.6 K dead band on either side of the program
  (the physics review's finding 1, 2026-11-10), each edge pulled in by the shift
  over the 600 s lead, it moves nothing; past it, only the excess.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import rod_insertion_limit
from target_gym.reactor.experts import (
    boron_can_follow,
    make_reactor_pid_controller,
    xenon_feedforward,
    xenon_play,
)


def test_xenon_tool_follows_the_plants_xenon_through_a_load_cut():
    env = Reactor()
    # Rods held with no boron through the cut run Tavg hot for hours; the DNB
    # limit's clock and the hot-leg envelope are lifted so no restart resets the
    # xenon being followed (at the start-of-cycle MTC the hold runs the hot leg
    # past 610 K).
    p = ReactorParams(
        demand_sigma=0.0,
        initial_xenon_power_range=(1.0, 1.0),
        dnb_allowed=1e9,
        T_coolant_max=1e9,
    )
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(ramp_down_at=jnp.float32(0.0), ramp_up_at=jnp.float32(1e8))
    hold = jnp.array([1.0 - 2.0 * float(s.rod_in), 0.0], jnp.float32)

    def step(c, _):
        s, I, Xe, first = c
        flow, I, Xe = xenon_feedforward(I, Xe, s.n, s.boron, first, p, 30.0)
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, hold, p)
        return (s, I, Xe, False), (s.Xe_hat, Xe, flow)

    start = (s, jnp.float32(0.0), jnp.float32(0.0), True)
    _, (plant, tool, flow) = jax.lax.scan(step, start, None, length=6 * 120)
    plant, tool, flow = map(np.asarray, (plant, tool, flow))
    assert plant[-1] > 1.05  # the cut did build xenon
    # Explicit 30 s steps on hour-long constants, and no tilt term: within 2 %.
    np.testing.assert_allclose(tool[::60], plant[::60], rtol=0.02)
    assert np.all(flow[120:300] < 0.0)  # building, 1 to 2.5 h in: dilute


@pytest.mark.parametrize("seed", [100, 104, 108])
def test_the_reference_keeps_the_rods_inside_their_insertion_limit(seed):
    """The owner's condition for the procedure (2026-11-08): bank D holds the offset
    only from above its insertion limit, on the EDF day."""
    env, p = Reactor(), ReactorParams()
    expert = make_reactor_pid_controller()
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)

    def step(c, k):
        s, obs, m = c
        a, m = expert.step(expert.params, m, obs)
        obs, s, _, _, _ = env.step_env(k, s, a, p)
        return (s, obs, m), (s.rod_in - rod_insertion_limit(s.n, p), s.time)

    _, (past, t) = jax.lax.scan(step, (s, obs, expert.init_state), keys)
    assert float(np.max(np.asarray(past))) <= 1e-6
    assert int(np.sum(np.asarray(t) == 0)) == 0  # no trip


# The rods' action band (owner, 2026-11-08; PHYSICS.md section 6): a plant's rods sit
# still on a baseload day while the offset is well inside its band, and the reference
# still keeps the band on every load-follow day, never spending the penalty clock's
# allowed hour.


def _day(scenario, seed, **changes):
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    spec = REGISTRY["reactor"]
    (sc,) = [s for s in scenarios.suite(spec) if s.name == scenario]
    p = scenarios.params_for(spec, sc, **changes)  # scored: sensor noise on
    env, expert = Reactor(), make_reactor_pid_controller()
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)

    def step(c, k):
        s, obs, m = c
        a, m = expert.step(expert.params, m, obs)
        obs, s1, _, _, _ = env.step_env(k, s, a, p)
        moved = jnp.abs(s1.rod_in - s.rod_in)
        out = (moved, s1.delta_I / s1.n, s1.ao_outside, s1.time, a[1])
        return (s1, obs, m), out

    _, out = jax.jit(lambda c: jax.lax.scan(step, c, keys))((s, obs, expert.init_state))
    return p, [np.asarray(x) for x in out]


@pytest.mark.parametrize("seed", [100, 104, 108])
def test_the_rods_sit_still_on_a_baseload_day(seed):
    p, (moved, ao, _, _, _) = _day("baseload", seed)
    assert float(np.max(np.abs(ao - p.ao_ref))) <= 0.025  # inside the action band
    # Rods move in whole steps; the float32 action round trip leaves 2e-6 of one.
    assert float(np.sum(moved)) * 228.0 < 0.01


@pytest.mark.parametrize(
    "scenario",
    ["edf_daily", "edf_short_hold", "krsko_slow", "krsko_shallow", "fast_return"],
)
def test_the_reference_keeps_the_band_on_every_load_follow_day(scenario):
    for seed in (100, 104, 108):
        p, (_, _, outside, t, _) = _day(scenario, seed)
        assert int(np.sum(t == 0)) == 0  # no trip
        assert float(outside[-1]) < p.ao_allowed


# --- the xenon tool's play ---
# Hand values: boron_worth -8e-5 per ppm, alpha_coolant -12e-5 and alpha_fuel -3e-5
# per K, so a ppm of debt is 8/15 = 0.5333 K. Each edge, 1.6 K, is a debt of 3 ppm,
# on the cold side and the hot side alike, at every power.

K_PER_PPM = 8.0 / 15.0
EDGE = 1.6 / K_PER_PPM  # 3 ppm


def _play(debt, rate):
    p = ReactorParams()
    return xenon_play(debt, rate, p, dt=30.0, lead=600.0, dead_band=1.6)


@pytest.mark.parametrize("debt", [-2.0, 0.0, 2.0])
def test_inside_the_free_zone_the_xenon_tool_moves_no_boron(debt):
    moved, left = _play(debt, 0.0)
    assert float(moved) == 0.0
    assert float(left) == pytest.approx(debt)


@pytest.mark.parametrize(
    "debt, moved_ppm",
    [
        (-4.0, -1.0),  # 2.13 K cold, past 3 ppm by 1: dilute 1 ppm
        (4.0, 1.0),  # 2.13 K hot: the same band, borate 1 ppm
    ],
)
def test_past_the_free_zone_the_tool_moves_only_the_excess(debt, moved_ppm):
    moved, left = _play(debt, 0.0)
    assert float(moved) == pytest.approx(moved_ppm, abs=1e-4)
    assert float(left) == pytest.approx(debt - moved_ppm, abs=1e-4)


def test_the_edges_are_pulled_in_by_the_lead():
    """Xenon building at 0.001 ppm/s: 0.03 ppm more debt in the step, and the edge
    600 s ahead is 0.6 ppm closer, at -2.4. From -1.0 nothing moves; from -2.5 the
    debt reaches -2.53 and the 0.13 past -2.4 is diluted."""
    moved, left = _play(-1.0, -0.001)
    assert float(moved) == 0.0
    assert float(left) == pytest.approx(-1.03, abs=1e-5)
    moved, left = _play(-2.5, -0.001)
    assert float(moved) == pytest.approx(-0.13, abs=1e-4)
    assert float(left) == pytest.approx(-2.4, abs=1e-4)


def test_the_hot_edge_is_the_dead_band_not_the_dnb_limit():
    """At full power the DNB limit is 2.83 K over the program, past the 1.6 K band:
    the tool acts at the band."""
    moved, _ = _play(EDGE - 0.1, 0.0)
    assert float(moved) == 0.0
    moved, _ = _play(EDGE + 0.1, 0.0)
    assert float(moved) == pytest.approx(0.1, abs=1e-3)


@pytest.mark.parametrize("seed", [100, 104])
def test_with_xenon_in_balance_a_baseload_day_moves_no_boron(seed):
    """Nothing is charged on a balanced baseload day, so nothing is done. No burnup:
    its daily dilution is a chore of its own."""
    _, (*_, flow) = _day(
        "baseload",
        seed,
        initial_xenon_power_range=(1.0, 1.0),
        burnup_range=(0.0, 0.0),
        carried_days_range=(0.0, 0.0),
    )
    assert float(np.max(np.abs(flow))) == 0.0


# --- rods on Tavg through a fast ramp (owner, 2026-11-10; PHYSICS.md section 6) ---
# Hand values: on the program the fuel sits 243.6 K hotter at full load than at half
# (the physics review, finding 1) and Tavg 7.7 K hotter, so the power defect is
# 3e-5 x 243.6 + 12e-5 x 7.7 = 823.2 pcm per 50 % of load, 205.8 ppm per unit of load
# at 8 pcm/ppm. At 1690 ppm full dilution moves 10 / 2.5e5 x 1690 = 0.0676 ppm/s and
# full boration 3 / 2.5e5 x (7800 - 1690) = 0.0733 ppm/s.


@pytest.mark.parametrize(
    "rate, follows",
    [
        (0.0, True),
        (-0.08 / 3600.0, True),  # Krsko's 8 %/h down: 0.0046 ppm/s of boration
        (0.08 / 3600.0, True),  # and back up: 0.0046 ppm/s of dilution
        (-0.05 / 60.0, False),  # 5 %/min down: 0.1715 ppm/s, past 0.0733
        (0.05 / 60.0, False),  # 5 %/min up: past 0.0676
    ],
)
def test_boron_follows_a_ramp_only_within_the_lines_rate(rate, follows):
    assert bool(boron_can_follow(rate, 1690.0, ReactorParams())) is follows
