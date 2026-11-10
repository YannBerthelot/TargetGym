"""The compressor as a continuing plant (long-strata draft 4, S1, S2).

Contract:
- the consumers' demand keeps moving however long the plant runs: past the
  schedule's six blocks, a fresh level each block, within the scenario's
  ``demand_range``, ramping from the level before at the scenario's ramp, so the
  opening never jumps;
- the plant does not read how long it has run: from one state, the same step a
  hundred million steps later moves the plant and the valve, and reads the surge
  counter, the same way;
- a demand move started now (``start_event``) ramps from the opening now to a
  fresh level within the range, at its scenario's ramp through a switch;
- two policies on one seed meet one outside world (S2): every ``OUTSIDE_WORLD``
  field but the surge record, the scenarios and the events, through trips and
  switches. The record (``surge_times``, ``surges``) is the plant's own: a trip's
  restart carries it with the world (``base.failure_kernel``), and a policy moves
  it.

Oracles: the scenarios' declared ranges and ramps, the block's length, the
valve's dead time (0.15 s, longer than a step).

The task is not opted in (``STREAM_READY``) while the shared conformance suite's
rates, 2 events an hour over 2 N steps, give a 0.1 s plant 0.13 events, and the
suite reads the surge record as outside world: the S2 check here runs at rates
scaled to the 2-minute window, the record aside.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import scenarios
from target_gym.compressor_surge.env import schedule_ramp, scheduled_demand
from target_gym.registry import REGISTRY
from target_gym.stream import StreamEnv

SPEC = REGISTRY["compressor_surge"]
ENV = SPEC.make_env()
STREAM = StreamEnv(ENV, SPEC.make_test_params(), scenarios.suite(SPEC))
NAMES = [s.name for s in STREAM.scenarios]
STEP = jax.jit(ENV.step_env)


def _params(name):
    return STREAM.scenario_params(STREAM.default_params, NAMES.index(name))


def _opening(state, params):
    """The consumers' scheduled opening the state is at, at its schedule's ramp."""
    return scheduled_demand(
        state.demand_levels,
        state.block_clock * params.delta_t,
        params,
        ramp=schedule_ramp(state, params),
    )


def _hold(state, params):
    """The raw action that holds the drive and the recycle where they are."""
    return jnp.array(
        [
            2.0 * (state.N_ramp - params.N_min) / (params.N_max - params.N_min) - 1.0,
            2.0 * state.x - 1.0,
        ]
    )


#: Steps each run below takes: one compile serves every test.
RUN = 410


@jax.jit
def _openings(state, action, params):
    """The scheduled opening after each of ``RUN`` steps at ``action``."""

    def body(s, k):
        _, s, *_ = ENV.step_env(k, s, action, params)
        return s, _opening(s, params)

    keys = jax.random.split(jax.random.PRNGKey(9), RUN)
    return jax.lax.scan(body, state, keys)


def test_the_demand_keeps_moving_past_the_schedule_s_six_blocks():
    p = _params("load_swings")
    _, s = ENV.reset_env(jax.random.PRNGKey(0), p)
    block = int(p.demand_block_steps)  # 200 steps, 20 s
    end = 6 * block  # the schedule's six blocks
    s = s.replace(time=jnp.int32(end - 10), block_clock=jnp.int32(end - 10))
    seen = np.asarray(_openings(s, _hold(s, p), p)[1])
    lo, hi = p.demand_range
    assert lo - 1e-6 <= seen.min() and seen.max() <= hi + 1e-6
    # A ramp moves at most the range per ramp: no jump at a block or a renewal.
    assert np.abs(np.diff(seen)).max() <= (hi - lo) * p.delta_t / p.demand_ramp_s + 1e-5
    # Each of the two blocks after the six settles on a level of its own.
    plateaus = seen[10 + block - 1 :: block][:2]
    assert len(set(np.round(plateaus, 6).tolist())) == 2


def test_the_plant_does_not_read_how_long_it_has_run():
    # Low demand with the recycle shut surges at once [C1; C4], and the valve
    # history holds an opening command issued one step ago.
    p = _params("normal").replace(demand_range=(0.2, 0.2), demand_sigma=0.0)
    _, s = ENV.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(x=jnp.asarray(0.0), valve_cmds=jnp.zeros_like(s.valve_cmds))
    shut = _hold(s, p).at[1].set(-1.0)
    for k in range(200):
        obs, s, *_ = STEP(jax.random.PRNGKey(k), s, shut, p)
        if float(obs[7]) >= 1.0:  # the counter reads a surge
            break
    assert float(obs[7]) >= 1.0
    open_ = shut.at[1].set(1.0)
    _, s, *_ = STEP(jax.random.PRNGKey(300), s, open_, p)
    key = jax.random.PRNGKey(301)
    now = STEP(key, s, open_, p)
    later = STEP(key, s.replace(time=s.time + 100_000_000), open_, p)
    # The plant, the valve and the counter; the deviation's next draw is keyed
    # on the step, so it is not compared.
    for field in ("m_c", "dp", "N", "N_ramp", "x"):
        assert float(getattr(now[1], field)) == float(getattr(later[1], field)), field
    assert float(now[0][7]) == float(later[0][7])


def test_a_demand_move_started_now_ramps_to_a_fresh_level_in_range():
    p = _params("fast_approach")
    _, s = ENV.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(time=jnp.int32(500), block_clock=jnp.int32(500))
    before = float(_opening(s, p))
    s = ENV.start_event(s, jax.random.PRNGKey(1), p)
    assert float(_opening(s, p)) == before
    ramp = int(np.ceil(p.demand_ramp_s / p.delta_t))
    seen = np.asarray(_openings(s, _hold(s, p), p)[1])[: ramp + 5]
    lo, hi = p.demand_range
    assert seen[-1] != before and lo <= seen[-1] <= hi
    np.testing.assert_allclose(seen[ramp:], seen[-1], rtol=1e-6)


def test_a_demand_ramp_keeps_its_scenario_s_rate_through_a_switch():
    slow, normal = _params("slow_approach"), _params("normal")
    assert slow.demand_ramp_s > normal.demand_ramp_s  # 15 s against 5 s
    _, s = ENV.reset_env(jax.random.PRNGKey(0), normal)
    s = ENV.start_event(s, jax.random.PRNGKey(1), slow)
    seen = np.asarray(_openings(s, _hold(s, normal), normal)[1])
    block = int(slow.demand_block_steps)  # the move's block: 200 steps
    level, start = seen[block - 2], float(_opening(s, normal))
    steps = int(round(slow.demand_ramp_s / slow.delta_t))  # 150 steps
    # Linear from the onset over the slow ramp, though the switch says 5 s.
    np.testing.assert_allclose(
        seen[steps // 2 - 1], start + 0.5 * (level - start), rtol=1e-5
    )
    np.testing.assert_allclose(seen[steps - 1 : block - 1], level, rtol=1e-6)


#: The outside world a policy cannot move: ``OUTSIDE_WORLD`` but the surge record.
WORLD = tuple(f for f in ENV.OUTSIDE_WORLD if f not in ("surge_times", "surges"))


def test_two_policies_meet_one_world_through_trips_and_switches():
    fast = [
        dataclasses.replace(c, dwell_h=0.002, events_per_h=360.0)
        for c in STREAM.scenarios
    ]
    stream = StreamEnv(ENV, STREAM.default_params, fast)  # ~7 s dwells, an event a 10 s
    params = stream.default_params
    n = int(params.max_steps_in_episode) // 2  # 60 s

    @jax.jit
    def roll(side):
        action = jnp.full((2,), side)
        _, state = stream.reset(jax.random.PRNGKey(3), params)

        def body(state, k):
            _, state, _, term, trunc, info = stream.step(k, state, action, params)
            world = {f: getattr(state.env, f) for f in WORLD}
            return state, (
                term | trunc,
                info["scenario"],
                info["event"],
                info["tripped"],
                world,
            )

        return jax.lax.scan(
            body, state, jax.random.split(jax.random.PRNGKey(side > 0), n)
        )[1]

    low, high = roll(-1.0), roll(1.0)
    assert not bool(low[0].any()) and not bool(high[0].any())
    np.testing.assert_array_equal(low[1], high[1])
    np.testing.assert_array_equal(low[2], high[2])
    assert int(low[2].sum()) > 0 and len(set(np.asarray(low[1]).tolist())) > 1
    assert bool(low[3].any()) != bool(high[3].any())  # one of them trips
    for field in WORLD:
        np.testing.assert_array_equal(
            np.asarray(low[4][field]), np.asarray(high[4][field]), err_msg=field
        )
