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
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 50})
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
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 50})
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
