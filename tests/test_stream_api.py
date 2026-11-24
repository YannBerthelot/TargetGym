"""The stream behind the front door: ``tg.make_stream`` and
``tg.make_gymnasium(name, stream=True)``.

Contract (``target_gym.stream``'s module docstring; docs/proposals/long-strata.md,
Part A; ``make_gymnasium``'s docstring):

- ``tg.make_stream(name)`` is the task as a continuing plant: it steps as an
  environment does (six values) and never signals ``terminated`` or
  ``truncated``, however long it runs, unless it is given a ``length``;
- the observation is the task's own, with no clock and no scenario label;
- ``tg.make_gymnasium(name, stream=True)`` is that plant behind the stateful
  Gymnasium API, for libraries that expect it;
- the stream runs the task's scored parameters over its scenario suite, so it
  takes no parameter overrides;
- a task not wired for the stream is refused, by name.
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.registry import REGISTRY

NAME = "first_order"  # the fixture task: 150 steps a window, a 4 s compile


def _window(name):
    return int(REGISTRY[name].make_test_params().max_steps_in_episode)


def test_make_stream_runs_past_the_window_without_an_end():
    stream = tg.make_stream(NAME)
    steps = _window(NAME) + 20
    action = np.zeros(stream.action_space().shape, np.float32)
    obs, state = stream.reset(jax.random.PRNGKey(0))
    ends = []
    for t in range(steps):
        out = stream.step(jax.random.PRNGKey(t), state, action)
        assert len(out) == 6
        obs, state, reward, terminated, truncated, info = out
        ends.append(bool(terminated) or bool(truncated))
    assert not any(ends)
    assert int(state.time) == steps


def test_a_stream_given_a_length_ends_there():
    stream = tg.make_stream(NAME, length=7)
    action = np.zeros(stream.action_space().shape, np.float32)
    obs, state = stream.reset(jax.random.PRNGKey(0))
    truncated_at = []
    for t in range(7):
        obs, state, _, terminated, truncated, _ = stream.step(
            jax.random.PRNGKey(t), state, action
        )
        assert not bool(terminated)
        truncated_at.append(bool(truncated))
    assert truncated_at == [False] * 6 + [True]


def test_the_stream_s_observation_is_the_task_s_own():
    env, params = tg.make(NAME)
    stream = tg.make_stream(NAME)
    obs, _ = stream.reset(jax.random.PRNGKey(0))
    assert obs.shape == env.observation_space(params).shape


def test_the_gymnasium_stream_runs_past_the_window_without_an_end():
    env = tg.make_gymnasium(NAME, stream=True)
    plain = tg.make_gymnasium(NAME)
    assert env.observation_space.shape == plain.observation_space.shape
    assert env.action_space.shape == plain.action_space.shape
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    action = np.zeros(env.action_space.shape, np.float32)
    for _ in range(_window(NAME) + 20):
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert isinstance(reward, float) and np.isfinite(reward)
    assert obs.shape == env.observation_space.shape


def test_the_gymnasium_stream_passes_gymnasium_s_api_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(tg.make_gymnasium(NAME, stream=True), skip_render_check=True)


def test_two_gymnasium_streams_on_one_seed_meet_one_world():
    """Common random numbers (S2): the seed fixes the world, whatever is done."""
    quiet, busy = (tg.make_gymnasium(NAME, stream=True) for _ in range(2))
    quiet.reset(seed=3)
    busy.reset(seed=3)
    low = np.full(quiet.action_space.shape, -1.0, np.float32)
    high = np.full(quiet.action_space.shape, 1.0, np.float32)
    for _ in range(40):
        *_, a = quiet.step(low)
        *_, b = busy.step(high)
        assert int(a["scenario"]) == int(b["scenario"])
        assert bool(a["event"]) == bool(b["event"])


def test_a_gymnasium_stream_takes_no_parameter_overrides():
    with pytest.raises(ValueError, match="scored"):
        tg.make_gymnasium(NAME, stream=True, max_steps_in_episode=10)


def test_a_gymnasium_stream_refuses_a_render_mode_it_cannot_serve():
    with pytest.raises(ValueError, match="render"):
        tg.make_gymnasium(NAME, stream=True, render_mode="rgb_array")


def test_a_task_not_wired_for_the_stream_is_refused_by_name():
    unwired = [
        n
        for n, s in REGISTRY.items()
        if not getattr(s.make_env(), "STREAM_READY", False)
    ]
    if not unwired:
        pytest.skip("every registered task is wired for the stream")
    with pytest.raises(ValueError, match=unwired[0]):
        tg.make_stream(unwired[0])
