"""``GradientMPC``: the solve never returns a plan worse than its warm start."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.experts.mpc import GradientMPC
from target_gym.registry import REGISTRY


@pytest.mark.parametrize("lr", [0.05, 2.0])
def test_descent_returns_no_worse_a_plan_than_it_started_from(lr):
    """With a step large enough to overshoot on every iteration, the last
    iterate is worse than the warm start; the plan returned must not be."""
    spec = REGISTRY["first_order"]
    env, p = spec.make_env(), spec.make_test_params()
    lo = float(np.min(env.action_space(p).low))
    hi = float(np.max(env.action_space(p).high))
    mpc = GradientMPC(
        env, p, action_dim=1, action_lb=lo, action_ub=hi, horizon=10, n_iter=20, lr=lr
    )
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    for seed in range(3):
        start = jax.random.uniform(
            jax.random.PRNGKey(seed), (10, 1), minval=lo, maxval=hi
        )
        plan = mpc._jit_optimize(start, state)
        assert mpc._score(plan, state) >= mpc._score(start, state) - 1e-6
        assert bool(jnp.all(plan >= lo)) and bool(jnp.all(plan <= hi))
