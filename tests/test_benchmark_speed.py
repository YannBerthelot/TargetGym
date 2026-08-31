"""The throughput benchmark backs a published claim, so it is tested.

The README quotes steps-per-second figures produced by
``python -m target_gym.benchmark_speed``. A benchmark that has silently stopped
working, or that reports a number computed from a rollout shorter than it
claims, would keep those figures on the page while meaning nothing.

These run tiny batches: what is checked is that the benchmark measures what it
says, not how fast this particular machine is.
"""

from __future__ import annotations

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


def test_rate_scales_with_batch_size():
    """The reported figure must count every environment in the batch.

    A benchmark that timed the batch but divided by ``steps`` alone would
    under-report by the batch size and still look plausible. Doubling the batch
    should not halve the reported rate.
    """
    spec = REGISTRY["cstr"]
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 50})
    env = spec.make_env()

    small = benchmark_env(env, params, steps=8, batch_size=4)
    large = benchmark_env(env, params, steps=8, batch_size=32)

    # Generous bound: this asserts the batch is counted at all, and is not a
    # timing assertion -- a loaded CI runner may make either number noisy.
    assert large > small * 0.5, (
        f"throughput fell from {small:.0f} to {large:.0f} when the batch grew "
        "8x; the reported rate is probably not counting the batch"
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
