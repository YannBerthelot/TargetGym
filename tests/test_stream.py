"""The stream: a task run as a continuing plant (docs/proposals/long-strata.md, Part A).

Contract (draft 4, S1, S2, R2):
- a stream never signals ``terminated``; ``truncated`` fires only at a configured end
  (a stream length, or a scoring run's end), never at the task's window length N;
- scenarios switch as a memoryless chain whose long-run time shares are the weights,
  re-entering the current scenario allowed; each step runs on its scenario's params;
- events start at the scenario's rate, through the task's ``start_event`` hook;
- the outside world comes from the stream's seed and the step alone: the policy and the
  learner's key never move it (common random numbers);
- the observation is the task's own: no clock, no scenario label;
- a scoring run pins one scenario, burns in for B in [b_min, 2 b_min), places exactly
  ``count`` events uniformly over S, and ends with ``truncated`` after B + S steps.

Oracles: the weights and rates the test declares, a stationary chain's time shares,
and Poisson counts with their standard deviation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from target_gym.base import EnvParams
from target_gym.scenarios import Scenario
from target_gym.stream import StreamEnv, event_counts

HOUR = 3600.0


@struct.dataclass
class ToyParams(EnvParams):
    delta_t: float = HOUR  # one step is one hour, so rates per hour are per step
    max_steps_in_episode: int = 10  # the task's window, which a stream must ignore
    continuing: bool = True
    gain: float = 0.5
    limit: float = 5.0


@struct.dataclass
class ToyState:
    time: int
    x: jnp.ndarray
    level: jnp.ndarray
    outside: jnp.ndarray  # driven by the step key only: the outside world
    gain_seen: jnp.ndarray  # the gain the last step ran on


class Toy:
    """A tank relaxing to a level an event shifts; it trips past ``limit``."""

    def reset_env(self, key, params):
        z = jnp.zeros(())
        s = ToyState(0, z, z, z, jnp.asarray(params.gain, float))
        return jnp.stack([s.x]), s

    def step_env(self, key, s, action, params):
        x = s.x + params.gain * (s.level - s.x) + action[0]
        tripped = jnp.abs(x) > params.limit
        x = jnp.where(tripped, 0.0, x)  # a trip restarts the plant, nothing ends
        outside = s.outside + jax.random.normal(key)
        new = ToyState(s.time + 1, x, s.level, outside, jnp.asarray(params.gain))
        return jnp.stack([x]), new, -(x**2), jnp.array(False), {"tripped": tripped}

    def start_event(self, s, key, params):
        return s.replace(level=jax.random.uniform(key, minval=0.5, maxval=1.0))

    def action_space(self, params=None):
        return None


def hold(obs):
    return jnp.zeros(1)


def push(obs):
    return 6.0 * jnp.ones(1)  # trips the toy on every step


@partial(jax.jit, static_argnums=(0, 1))
def rollout(stream, policy, obs, state, keys):
    """``len(keys)`` steps of ``stream.step``: the stacked infos, terminated,
    truncated, the task's states and the observations. Jitted once per stream."""
    params = stream.default_params

    def body(carry, k):
        obs, state = carry
        obs, state, _, term, trunc, info = stream.step(k, state, policy(obs), params)
        keep = {key: info[key] for key in ("scenario", "event", "scored", "tripped")}
        return (obs, state), (keep, term, trunc, state.env, obs)

    return jax.lax.scan(body, (obs, state), keys)[1]


def run(stream, steps, policy=hold, seed=0, learner_seed=0):
    """``steps`` steps of a stream started from seed ``seed``."""
    obs, state = stream.reset(jax.random.PRNGKey(seed), stream.default_params)
    keys = jax.random.split(jax.random.PRNGKey(1000 + learner_seed), steps)
    return rollout(stream, policy, obs, state, keys)


def suite(weights, dwells=None, rates=None, gains=None):
    n = len(weights)
    return [
        Scenario(
            f"s{i}",
            {"gain": gains[i]} if gains else {},
            weight=weights[i],
            dwell_h=None if dwells is None else dwells[i],
            events_per_h=0.0 if rates is None else rates[i],
        )
        for i in range(n)
    ]


def test_a_stream_never_signals_done_at_the_window_or_at_a_trip():
    stream = StreamEnv(Toy(), ToyParams(), suite([1.0]))
    # A push of 6 a step trips the plant on every step.
    info, term, trunc, _, _ = run(stream, 1000, policy=push)
    assert bool(info["tripped"].all())
    assert not bool(term.any()) and not bool(trunc.any())


def test_a_configured_end_truncates_there_and_only_there():
    stream = StreamEnv(Toy(), ToyParams(), suite([1.0]), length=37)
    _, _, trunc, _, _ = run(stream, 100)
    assert np.flatnonzero(np.asarray(trunc)).tolist() == [36, 73]


@pytest.mark.parametrize(
    "weights, dwells",
    [
        # Equal dwells: a draw that skipped the current scenario would give the
        # first one 49 % of the time instead of 90 % (review, E2).
        ((0.9, 0.05, 0.05), (10.0, 10.0, 10.0)),
        ((0.5, 0.3, 0.2), (5.0, 15.0, 30.0)),
    ],
)
def test_scenario_time_shares_are_their_weights(weights, dwells):
    stream = StreamEnv(Toy(), ToyParams(), suite(weights, dwells))
    info = run(stream, 100_000)[0]
    shares = np.bincount(np.asarray(info["scenario"]), minlength=3) / 100_000
    # Standard deviation of a share is about sqrt(2 w (1 - w) D / T) < 0.01 here.
    np.testing.assert_allclose(shares, weights, atol=0.03)


def test_each_step_runs_on_its_scenario_s_params():
    gains = (0.1, 0.5, 0.9)
    stream = StreamEnv(
        Toy(), ToyParams(), suite((0.4, 0.3, 0.3), (5.0,) * 3, None, gains)
    )
    info, _, _, env, _ = run(stream, 2000)
    used = np.asarray(gains)[np.asarray(info["scenario"])]
    np.testing.assert_allclose(np.asarray(env.gain_seen), used, rtol=1e-6)
    assert len(set(np.asarray(info["scenario"]).tolist())) == 3


def test_events_arrive_at_the_scenario_s_rate():
    rate = 0.01  # per hour, so per step
    stream = StreamEnv(Toy(), ToyParams(), suite([1.0], None, [rate]))
    info = run(stream, 100_000)[0]
    expected = rate * 100_000  # Poisson mean 1000, standard deviation about 32
    assert abs(int(info["event"].sum()) - expected) < 4 * np.sqrt(expected)


def test_two_policies_and_two_learner_keys_meet_the_same_world():
    stream = StreamEnv(
        Toy(), ToyParams(), suite((0.6, 0.4), (15.0, 15.0), (0.05, 0.05))
    )
    calm = run(stream, 3000, seed=4, learner_seed=0)
    wild = run(stream, 3000, push, seed=4, learner_seed=1)
    for key in ("scenario", "event"):
        np.testing.assert_array_equal(calm[0][key], wild[0][key])
    np.testing.assert_array_equal(calm[3].outside, wild[3].outside)
    assert bool(wild[0]["tripped"].any()) and not bool(calm[0]["tripped"].any())


def test_another_seed_is_another_world():
    stream = StreamEnv(Toy(), ToyParams(), suite((0.6, 0.4), (15.0, 15.0)))
    a, b = run(stream, 3000, seed=4), run(stream, 3000, seed=5)
    assert not np.array_equal(a[3].outside, b[3].outside)


def test_the_observation_is_the_task_s_own():
    stream = StreamEnv(Toy(), ToyParams(), suite((0.6, 0.4), (15.0, 15.0)))
    _, _, _, env, obs = run(stream, 50)
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(env.x)[:, None])


def test_a_scoring_run_pins_its_scenario_burns_in_and_ends():
    b_min, scored, count = 50, 200, 3
    stream = StreamEnv(
        Toy(),
        ToyParams(),
        suite((0.6, 0.4), (15.0, 15.0), (0.02, 0.02)),
        pin=1,
        b_min=b_min,
        scored=scored,
        max_events=4,
    )
    start = jax.jit(stream.start)
    burn_events, burn_steps = 0, 0
    for seed in range(20):
        _, state = start(jax.random.PRNGKey(seed), count)
        b = int(state.burn_in)
        assert b_min <= b < 2 * b_min
        # One scan length for every seed, so it compiles once; read the first B + S.
        keys = jax.random.split(jax.random.PRNGKey(7), 2 * b_min + scored)
        out = rollout(stream, hold, jnp.zeros(1), state, keys)
        info, term, trunc = jax.tree_util.tree_map(
            lambda a: a[: b + scored], (out[0], out[1], out[2])
        )
        assert set(np.asarray(info["scenario"]).tolist()) == {1}
        scoring = np.asarray(info["scored"])
        assert scoring.tolist() == [False] * b + [True] * scored
        assert int(info["event"][b:].sum()) == count
        assert np.flatnonzero(np.asarray(trunc)).tolist() == [b + scored - 1]
        assert not bool(term.any())
        burn_events += int(info["event"][:b].sum())
        burn_steps += b
    # Events run through the burn-in at the stream's rate (review, S1).
    expected = 0.02 * burn_steps
    assert abs(burn_events - expected) < 4 * np.sqrt(expected)


@pytest.mark.parametrize(
    "expected, n, counts",
    [(2.3, 10, [2] * 7 + [3] * 3), (0.25, 4, [0, 0, 0, 1]), (4.0, 3, [4, 4, 4])],
)
def test_event_counts_take_the_floor_and_ceiling_with_the_right_mean(
    expected, n, counts
):
    assert sorted(event_counts(expected, n).tolist()) == counts


def test_a_stream_refuses_what_it_cannot_run():
    with pytest.raises(ValueError, match="continuing"):
        StreamEnv(Toy(), ToyParams(continuing=False), suite([1.0]))
    with pytest.raises(ValueError, match="dwell"):
        StreamEnv(Toy(), ToyParams(), suite((0.5, 0.5)))

    class NoHook(Toy):
        start_event = None

    with pytest.raises(ValueError, match="start_event"):
        StreamEnv(NoHook(), ToyParams(), suite([1.0], None, [0.1]))
