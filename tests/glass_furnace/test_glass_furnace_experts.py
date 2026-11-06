"""The glass furnace's oracle on protocol seed 2, where the oracle audit
(2026-10) measured the protocol cost falling from 0.326 to 0.171 per step and
the mean window error from 0.38 K to 0.17 K.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts.mpc import plan_params

SPEC = registry.get("glass_furnace")


@pytest.mark.large
@pytest.mark.xfail(
    strict=True,
    reason="Finding (2026-11-06): under the fuel slew the oracle holds 8.84 K, no "
    "trips, against 0.25 K. A planner gets no gradient from ``actuate`` at "
    "command == held or past the rate limit, so its plan stalls. With that "
    "patched and the plan made by rate it holds 0.29 K. PR 9 rebuilds it.",
)
def test_the_oracle_holds_the_crown_on_protocol_seed_2():
    """The protocol's construction (``runners.baseline_policy``: the factory
    on ``plan_params``, then ``reset``) and its rollout (``PRNGKey(seed)`` on
    every step) on seed 2. The audit measured a mean |crown error| of 0.17 K
    over the scored window (the 800 steps after the protocol's burn-in),
    against 0.38 K for the oracle before it, with no trip."""
    env = SPEC.make_env()
    params = SPEC.make_test_params()
    oracle = SPEC.make_mpc(env, plan_params(SPEC, params))
    oracle.reset()
    key = jax.random.PRNGKey(2)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    errors = []
    for t in range(int(params.max_steps_in_episode)):
        u = oracle.step(obs, state)
        obs, state, _, _, info = step(key, state, jnp.atleast_1d(u), params)
        assert not bool(info.get("tripped", False))
        if t >= 800:
            errors.append(abs(float(state.target_T_crown) - float(state.T_crown)))
    assert np.mean(errors) < 0.25
