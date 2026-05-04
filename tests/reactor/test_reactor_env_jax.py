import jax
import jax.numpy as jnp
import pytest

from target_gym.reactor.env import (
    N_GROUPS,
    N_SETPOINTS,
    ReactorParams,
    ReactorState,
    steady_state_precursors,
    steady_state_xenon,
)
from target_gym.reactor.env_jax import CONTROL_PERIOD, Reactor


def _make_state(params=None, **overrides) -> ReactorState:
    if params is None:
        params = ReactorParams()
    I_hat_eq, Xe_hat_eq = steady_state_xenon(1.0, params)
    defaults = dict(
        time=0,
        n=1.0,
        C=steady_state_precursors(1.0, params),
        T_fuel=params.initial_T_fuel,
        T_coolant=params.initial_T_coolant,
        I_hat=I_hat_eq,
        Xe_hat=Xe_hat_eq,
        target_n=0.8,
        target_schedule=jnp.full((N_SETPOINTS,), 0.8),
        demand_key=jax.random.PRNGKey(42),
        rho_ext=jnp.zeros(()),
    )
    defaults.update(overrides)
    return ReactorState(**defaults)


def test_reset_env_returns_obs_and_state():
    env = Reactor()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    assert obs.shape == env.obs_shape
    assert isinstance(state, ReactorState)
    assert state.time == 0

    p = env.default_params
    assert p.initial_n_range[0] <= float(state.n) <= p.initial_n_range[1]
    assert p.target_n_range[0] <= float(state.target_n) <= p.target_n_range[1]
    assert state.C.shape == (N_GROUPS,)
    # Precursors are initialised at the steady-state value for n.
    expected_C = steady_state_precursors(state.n, p)
    assert jnp.allclose(state.C, expected_C)


def test_step_env_advances_state():
    env = Reactor()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    obs2, state2, reward, done, info = env.step_env(key, state, jnp.array([0.0]))
    assert obs2.shape == env.obs_shape
    assert isinstance(state2, ReactorState)
    assert state2.time == state.time + CONTROL_PERIOD
    assert jnp.isfinite(reward)
    assert done.dtype == jnp.bool_
    assert "last_state" in info
    assert isinstance(info["last_state"], ReactorState)


@pytest.mark.parametrize("method", ["euler_100", "rk2_50", "rk4_50"])
def test_integration_methods_run(method):
    env = Reactor(integration_method=method)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key)
    _, new_state, _, _, _ = env.step_env(key, state, jnp.array([0.0]))
    # No NaN/Inf under any integration method
    assert jnp.isfinite(new_state.n)
    assert jnp.all(jnp.isfinite(new_state.C))
    assert jnp.isfinite(new_state.T_fuel)
    assert jnp.isfinite(new_state.T_coolant)


def test_action_and_observation_space():
    env = Reactor()

    a = env.action_space()
    assert a.shape == (1,)
    assert jnp.all(a.low == -1.0)
    assert jnp.all(a.high == 1.0)

    o = env.observation_space(env.default_params)
    assert o.shape == env.obs_shape

    s = env.state_space(env.default_params)
    assert s.shape == len(ReactorState.__dataclass_fields__)


def test_get_obs_hides_fuel_and_precursors():
    env = Reactor()
    params = env.default_params
    # Unusual values so we can verify they don't leak through.
    state = _make_state(
        params=params,
        T_fuel=jnp.asarray(1234.5),
        C=jnp.array([9999.0, 8888.0, 7777.0, 6666.0, 5555.0, 4444.0]),
    )
    obs = env.get_obs(state)
    assert obs.shape == (4,)
    # None of the hidden values should appear in the obs.
    for hidden in (1234.5, 9999.0, 8888.0, 7777.0, 6666.0, 5555.0, 4444.0):
        assert not bool(jnp.any(jnp.isclose(obs, hidden)))


def test_obs_matches_manual_call():
    env = Reactor()
    state = _make_state()
    obs1 = env.get_obs(state)
    obs2 = env.get_obs(state, env.default_params)
    assert jnp.allclose(obs1, obs2)


def test_full_short_episode_is_finite():
    """A short episode with zero rod motion should remain finite throughout."""
    env = Reactor()
    params = ReactorParams(max_steps_in_episode=40)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    for _ in range(40):
        _, state, reward, done, _ = env.step_env(key, state, jnp.array([0.0]), params)
        assert jnp.isfinite(state.n)
        assert jnp.all(jnp.isfinite(state.C))
        assert jnp.isfinite(state.T_fuel)
        assert jnp.isfinite(state.T_coolant)
        assert jnp.isfinite(reward)
        if bool(done):
            break


def test_ou_demand_evolves_target():
    """OU demand process produces varying targets within the configured range."""
    env = Reactor()
    params = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    # Initial target is within range
    assert params.target_n_range[0] - 1e-6 <= float(state.target_n)
    assert float(state.target_n) <= params.target_n_range[1] + 1e-6

    # Step for a while and collect targets — they should vary
    targets = [float(state.target_n)]
    for _ in range(200):
        _, state, _, done, _ = env.step_env(key, state, jnp.array([0.0]), params)
        targets.append(float(state.target_n))
        assert params.target_n_range[0] - 1e-6 <= targets[-1]
        assert targets[-1] <= params.target_n_range[1] + 1e-6

    # Target should have changed (OU has nonzero sigma)
    assert max(targets) - min(targets) > 0.001

    # Different seeds produce different demand trajectories
    unique_targets = set()
    for seed in range(10):
        _, st = env.reset_env(jax.random.PRNGKey(seed), params)
        unique_targets.add(float(st.target_n))
    assert len(unique_targets) > 1


def test_observation_scales_rod_to_unit_range():
    """rho_ext_norm in the obs is in [-1, 1] after rescaling."""
    env = Reactor()
    params = env.default_params
    state_mid = _make_state(params=params, rho_ext=jnp.zeros(()))
    # At rho_ext = 0, the normalised value is not necessarily 0 because
    # the rod range is asymmetric ([-0.010, +0.003]); it should still be
    # within [-1, 1].
    obs_mid = env.get_obs(state_mid)
    assert -1.0 <= float(obs_mid[2]) <= 1.0

    state_max = _make_state(params=params, rho_ext=jnp.asarray(params.rho_ext_max))
    obs_max = env.get_obs(state_max)
    assert float(obs_max[2]) == pytest.approx(1.0, abs=1e-4)

    state_min = _make_state(params=params, rho_ext=jnp.asarray(params.rho_ext_min))
    obs_min = env.get_obs(state_min)
    assert float(obs_min[2]) == pytest.approx(-1.0, abs=1e-4)
