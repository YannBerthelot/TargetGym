"""The pH oracle reads the buffer flow and forecasts it.

Its model takes q2 as a time-varying parameter: the env's OU mean decay from
the current value. Planning on the nominal flow instead made almost all of its
hold error (0.0146 pH against 0.0022 on the protocol seeds), so the closed-loop
check below (in the slow job) would catch a return to it.
"""

import jax
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts.mpc import plan_params
from target_gym.pc_gym.ph_neutralization.experts import q2_forecast

pytest.importorskip("casadi")


def test_the_forecast_is_the_noise_free_q2_path():
    """Control interval k is integrated with q2 after k + 1 env updates, so
    with the noise off the forecast is exactly the path the plant takes."""
    spec = registry.get("ph_neutralization")
    env = spec.make_env()
    params = spec.make_test_params().replace(q2_noise_std=0.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    state = state.replace(q2=1.6)  # well off nominal
    forecast = q2_forecast(float(state.q2), params, horizon=20)
    step = jax.jit(env.step_env)
    path = []
    for _ in range(21):
        _, state, *_ = step(key, state, np.zeros(1), params)
        path.append(float(state.q2))
    np.testing.assert_allclose(forecast, path, rtol=1e-5)


@pytest.mark.slow
def test_the_oracle_holds_well_below_the_nominal_model():
    """Seed 1 is the protocol seed where the buffer drifts furthest: planning on
    the nominal flow held 0.022 pH there, reading it holds 0.0031."""
    spec = registry.get("ph_neutralization")
    env = spec.make_env()
    params = spec.make_test_params()
    oracle = spec.make_mpc(env, plan_params(spec, params))
    oracle.reset()
    key = jax.random.PRNGKey(1)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    errors = []
    for t in range(int(params.max_steps_in_episode)):
        action = np.atleast_1d(oracle.step(obs, state))
        obs, state, *_ = step(key, state, action, params)
        if t >= 108:  # the protocol's burn-in on this plant
            errors.append(abs(float(state.target_pH) - float(state.pH)))
    assert np.mean(errors) < 0.006
    assert oracle.solver_report()["solver_failures"] == 0
