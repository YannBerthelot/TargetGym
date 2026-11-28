"""The reactor's MPC on the observation (v1 item 5, as the owner read it on 2026-11-28).

Contract, written before the code:

- the record command builds it as ``observation_only_mpc(env, params, plan)``:
  ``params`` the run's own configuration, noise included, ``plan`` the planner's,
  its process noise zeroed; it returns a policy with ``reset()`` and
  ``__call__(obs)`` that is handed the observation and nothing else;
- it is the day planner fed the shared estimator's belief in place of the true
  state (``estimation.ObservationOnly``);
- so on a baseload hour it runs the plant: no trip, every action in the box, and the
  belief it plans from tracks what the observation hides or reads late. Xenon is
  within a quarter of the reset's spread of it (8.3e-3 of full power's balance), and
  boron within half a ppm, where the reset spreads it over hundreds of ppm and the
  boronmeter reads it 20 min late.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scoring_runs as S
from target_gym.experts.mpc import plan_params
from target_gym.reactor import Reactor
from target_gym.registry import REGISTRY
from target_gym.runners.runners import reads_state


def _built(steps):
    from target_gym.reactor.oracle import observation_only_mpc

    spec, env = REGISTRY["reactor"], Reactor()
    params = S.run_params("reactor", "baseload", steps=steps)
    return env, params, observation_only_mpc(env, params, plan_params(spec, params))


@pytest.mark.medium
def test_a_baseload_hour_on_the_observation_tracks_the_hidden_core():
    env, params, policy = _built(120)
    assert not reads_state(policy)
    policy.reset()
    key = jax.random.PRNGKey(5)
    obs, state = env.reset_env(key, params)
    step, actions, trips = jax.jit(env.step_env), [], 0
    for t in range(120):
        u = np.asarray(policy(obs), float)
        actions.append(u)
        obs, state, _, _, info = step(
            jax.random.fold_in(key, t), state, jnp.asarray(u, jnp.float32), params
        )
        trips += int(info["tripped"])
    belief = policy.belief.state
    assert trips == 0
    assert np.all(np.abs(np.asarray(actions)) <= 1.0)
    assert abs(float(belief.Xe_hat - state.Xe_hat)) < 2e-3
    assert abs(float(belief.boron - state.boron)) < 0.5
