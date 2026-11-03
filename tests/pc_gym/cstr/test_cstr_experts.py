"""The CSTR's oracle: ``NewtonMPC`` over the rest of the episode, on the
environment's own step.

The fast tests hold the factory to the protocol: the plan covers the episode
and weighs the window the protocol scores. The slow test is the closed loop on
the protocol (``eval.evaluate_controller``, seeds 0-2), against the oracle
audit's measurement (2026-10): 9.4e-9 per step, against 0.317 for the oracle
it replaced.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym.experts.mpc import plan_params, window_weights
from target_gym.pc_gym.cstr.oracle import make_cstr_mpc
from target_gym.registry import REGISTRY

SPEC = REGISTRY["cstr"]


def test_the_oracle_plans_the_episode_on_the_protocols_window():
    """The factory reads the window from the protocol's own burn-in: the
    first 12 steps of the 150 weigh 1e-3, and the plan is the episode."""
    p = plan_params(SPEC, SPEC.make_test_params())
    mpc = make_cstr_mpc(SPEC.make_env(), p)
    assert E.scored_burn_in("cstr", p) == 12
    assert (mpc.resolve_every, mpc.horizon) == (1, 150)
    np.testing.assert_allclose(mpc.weights, window_weights(150, 12, 1e-3, 150))


def test_a_short_closed_loop_solves_every_step_and_resets_cold():
    """Eight steps, eight solves, every action in the box; after ``reset``
    the next action is a fresh oracle's."""
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=8)
    mpc = make_cstr_mpc(env, p)
    _, start = env.reset_env(jax.random.PRNGKey(0), p)
    state = start
    step = jax.jit(env.step_env)
    for _ in range(8):
        u = mpc.step(None, state)
        assert -1.0 <= u <= 1.0
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.asarray([u]), p)
    assert mpc.solver_report()["solver_calls"] == 8
    mpc.reset()
    fresh = make_cstr_mpc(env, p).step(None, start)
    assert mpc.step(None, start) == pytest.approx(fresh, abs=1e-6)


@pytest.mark.large
def test_the_protocol_gain_is_at_the_plants_optimum():
    """Measured 9.42e-9 per step with zero trips, 15 s for three seeds; the
    oracle it replaced gave 0.317."""
    m = E.evaluate_controller("cstr", "mpc", seeds=3)
    assert m["failure_rate"] == 0.0
    assert m["gain"] < 1e-6
