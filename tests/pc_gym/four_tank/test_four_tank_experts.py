"""The four-tank's oracle: ``NewtonMPC`` over the rest of the episode, on
the environment's own step.

The fast tests hold the factory to the protocol: the plan covers the episode,
weighs the window the protocol scores and re-solves every 25 steps. The slow
test is the closed loop on the protocol (``eval.evaluate_controller``, seeds
0-2), against the oracle audit's measurement (2026-10): 3.1e-8 per step,
against 0.0074 for the oracle it replaced.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym.experts.mpc import plan_params, window_weights
from target_gym.pc_gym.four_tank.oracle import make_four_tank_mpc
from target_gym.registry import REGISTRY

SPEC = REGISTRY["four_tank"]


def test_the_oracle_plans_the_episode_on_the_protocols_window():
    """The factory reads the window from the protocol's own burn-in: the
    burn-in row's 270 steps, under half the 1200-step episode, weigh 1e-3,
    and the plan is the episode."""
    p = plan_params(SPEC, SPEC.make_test_params())
    mpc = make_four_tank_mpc(SPEC.make_env(), p)
    assert E.scored_burn_in("four_tank", p) == 270
    assert (mpc.resolve_every, mpc.horizon) == (25, 1200)
    np.testing.assert_allclose(mpc.weights, window_weights(1200, 270, 1e-3, 1200))


def test_a_short_closed_loop_solves_once_per_period():
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)
    mpc = make_four_tank_mpc(env, p, resolve_every=25)
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    for _ in range(30):
        u = np.asarray(mpc.step(None, state))
        assert u.shape == (2,) and np.all(np.abs(u) <= 1.0)
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.asarray(u), p)
    assert mpc.solver_report()["solver_calls"] == 2


@pytest.mark.large
def test_the_protocol_gain_is_at_the_plants_optimum():
    """Measured 3.08e-8 per step with zero trips, 28 s for three seeds; the
    oracle it replaced gave 0.0074."""
    m = E.evaluate_controller("four_tank", "mpc", seeds=3)
    assert m["failure_rate"] == 0.0
    assert m["gain"] < 1e-6
