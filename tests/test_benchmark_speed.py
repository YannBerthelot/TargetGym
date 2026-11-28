"""The throughput benchmark backs a published claim, so it is tested.

The README quotes steps-per-second figures produced by
``python -m target_gym.benchmark_speed``. A benchmark that has silently stopped
working, or that reports a number computed from a rollout shorter than it
claims, would keep those figures on the page while meaning nothing.

These run tiny batches: what is checked is that the benchmark measures what it
says, not how fast this particular machine is.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import pytest

from target_gym.benchmark_speed import benchmark_env, main
from target_gym.registry import REGISTRY

SMALL = dict(steps=8, batch_size=4)


@pytest.mark.parametrize("name", ["cstr", "hvac", "plane"])
def test_benchmark_reports_a_positive_rate(name):
    spec = REGISTRY[name]
    params = spec.make_test_params(max_steps_in_episode=50)
    rate = benchmark_env(spec.make_env(), params, **SMALL)

    assert jnp.isfinite(rate), f"{name}: benchmark returned {rate}"
    assert rate > 0.0, f"{name}: non-positive throughput"


def test_rate_counts_every_environment_in_the_batch(monkeypatch):
    """The reported figure must count the whole batch, not one environment.

    A benchmark that timed the batch but divided by ``steps`` alone would
    under-report by the batch size and still look entirely plausible.

    Checked against a controlled clock rather than by comparing two real runs.
    The wall-clock version of this was flaky: throughput on a machine running
    the rest of the suite in parallel is not a stable quantity, which is the
    same reason tests/test_performance_contracts.py refuses to time anything.
    """
    import target_gym.benchmark_speed as bs

    # the benchmark takes the best of several trials, so the clock is cycled
    # rather than being a fixed-length list: every trial takes exactly 2 s
    ticks = itertools.cycle([0.0, 2.0])
    monkeypatch.setattr(bs.time, "perf_counter", lambda: next(ticks))

    spec = REGISTRY["cstr"]
    params = spec.make_test_params(max_steps_in_episode=50)
    env = spec.make_env()

    small = bs.benchmark_env(env, params, steps=8, batch_size=4)
    large = bs.benchmark_env(env, params, steps=8, batch_size=32)

    # identical elapsed time, 8x the environments -> 8x the steps per second
    assert large == pytest.approx(8 * small, rel=1e-6), (
        f"{small:.0f} -> {large:.0f} steps/s for an 8x larger batch; the batch "
        "size is not being counted"
    )


def test_cli_runs(monkeypatch, capsys):
    """``python -m target_gym.benchmark_speed --envs ...`` still works."""
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_speed", "--envs", "cstr", "--batch", "4", "--steps", "8"],
    )
    main()
    out = capsys.readouterr().out
    assert "cstr" in out, f"benchmark printed no row for cstr:\n{out}"


# What the benchmark times (outside review M4, 2026-11-27): gymnax's ``step``
# with its auto-reset, as an RL loop calls it, a key of its own for every
# environment and step, random actions inside the action space, and everything
# ``step`` returns kept, so nothing it times can be compiled away. The rate
# published from a state-only run with one key and zero actions was 6 to 45
# times too high.


def _short_cstr(steps=8, batch=3):
    import jax

    import target_gym.benchmark_speed as bs

    spec = REGISTRY["cstr"]
    env = spec.make_env()
    # Episodes of 5 steps, so the auto-reset runs inside an 8-step rollout.
    params = spec.make_test_params(max_steps_in_episode=5)
    key = jax.random.PRNGKey(0)
    keys, actions = bs.draws(env, params, key, steps, batch)
    return bs, env, params, keys, actions


def test_every_environment_and_step_has_its_own_key_and_a_random_action():
    import numpy as np

    _, env, params, keys, actions = _short_cstr()
    flat = np.asarray(keys).reshape(8 * 3, -1)
    assert len({tuple(k) for k in flat.tolist()}) == 8 * 3
    space = env.action_space(params)
    a = np.asarray(actions)
    assert a.shape == (8, 3) + tuple(space.shape)
    assert np.all(a >= np.asarray(space.low)) and np.all(a <= np.asarray(space.high))
    assert np.unique(a).size > 1


def test_the_rollout_keeps_everything_step_returns():
    import jax
    import numpy as np

    bs, env, params, keys, actions = _short_cstr()
    reset_keys = jax.random.split(jax.random.PRNGKey(1), 3)
    _, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)
    _, totals = bs.rollout(env, params, state, keys, actions)
    # The same steps, one Python iteration each: what they returned, summed.
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    want = {"obs": 0.0, "reward": 0.0, "terminated": 0.0, "truncated": 0.0}
    s = state
    for t in range(8):
        obs, s, reward, terminated, truncated, _ = step(keys[t], s, actions[t], params)
        want["obs"] += float(np.sum(obs))
        want["reward"] += float(np.sum(reward))
        want["terminated"] += float(np.sum(terminated))
        want["truncated"] += float(np.sum(truncated))
    assert want["truncated"] > 0  # the episodes ended and reset inside the run
    for k, v in want.items():
        assert float(totals[k]) == pytest.approx(v, rel=1e-5), k
