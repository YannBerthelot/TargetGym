import jax
import jax.numpy as jnp
import pytest

from target_gym.reactor.env import (
    ReactorParams,
    ReactorState,
    steady_state_precursors,
    steady_state_temperatures,
    steady_state_xenon,
)
from target_gym.reactor.env_jax import CONTROL_PERIOD, Reactor


def _make_state(params=None, **overrides) -> ReactorState:
    if params is None:
        params = ReactorParams()
    I_hat_eq, Xe_hat_eq = steady_state_xenon(1.0, params)
    T_fuel, T_hot, T_cold, T_sg = steady_state_temperatures(1.0, params)
    defaults = dict(
        time=0,
        physics_time=0,
        n=1.0,
        C=steady_state_precursors(1.0, params),
        T_fuel=T_fuel,
        T_coolant=T_hot,
        T_cold=T_cold,
        T_sg=T_sg,
        I_hat=I_hat_eq,
        Xe_hat=Xe_hat_eq,
        target_n=0.8,
        T_ref=570.0,
        load_wobble=0.0,
        ramp_down_at=3600.0,
        ramp_up_at=18000.0,
        demand_key=jax.random.PRNGKey(42),
        rho_ext=jnp.zeros(()),
        rho_ext_cmd=0.0,
    )
    defaults.update(overrides)
    return ReactorState(**defaults)


def test_reset_draws_inside_its_ranges_at_steady_precursors():
    env = Reactor()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key)

    p = env.default_params
    assert float(state.n) == pytest.approx(1.0)
    assert float(state.target_n) == pytest.approx(1.0)
    assert p.ramp_down_range[0] <= float(state.ramp_down_at) <= p.ramp_down_range[1]
    hold = float(state.ramp_up_at - state.ramp_down_at) - 600.0
    assert p.hold_range[0] <= hold + 1e-2
    assert hold <= p.hold_range[1] + 1e-2
    # Precursors are initialised at the steady-state value for n.
    expected_C = steady_state_precursors(state.n, p)
    assert jnp.allclose(state.C, expected_C)


def test_the_physics_clock_runs_a_control_period_per_env_step():
    env = Reactor()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key)

    _, state2, _, _, _ = env.step_env(key, state, jnp.array([0.0]))
    assert state2.physics_time == state.physics_time + CONTROL_PERIOD


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
    # None of the hidden values should appear in the obs.
    for hidden in (1234.5, 9999.0, 8888.0, 7777.0, 6666.0, 5555.0, 4444.0):
        assert not bool(jnp.any(jnp.isclose(obs, hidden)))


def test_the_load_wobbles_inside_its_day():
    """The OU part moves the load while the schedule holds it at full."""
    env = Reactor()
    params = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    # Well before any ramp: the day says 1, and the wobble can only go below it.
    targets = [float(state.target_n)]
    _jstep = jax.jit(env.step_env)
    for _ in range(30):
        _, state, _, _, _ = _jstep(key, state, jnp.array([0.0]), params)
        targets.append(float(state.target_n))
    assert max(targets) <= 1.0
    assert min(targets) > 0.95
    assert max(targets) - min(targets) > 1e-4


def test_seeds_draw_different_days():
    env = Reactor()
    days = {
        float(env.reset_env(jax.random.PRNGKey(seed), env.default_params)[1].ramp_up_at)
        for seed in range(10)
    }
    assert len(days) == 10


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


def test_time_limit_counts_env_steps_and_rollout_stops_there():
    """Regression for the truncation bug that understated the baselines 5x.

    ``max_steps_in_episode`` used to be in physics steps while the shipped
    rollout counted env steps, so it ran past the limit and scored a frozen
    plant at a fraction of the reward. Now the limit is env steps: the episode
    ends exactly there, and the reward in the last step is the same order as in
    the first.
    """
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import rollout

    spec = REGISTRY["reactor"]
    # Version-1 reward: non-negative, so "a tenth of the reward" reads as it did.
    params = spec.make_test_params(max_steps_in_episode=30, reward_version=1)
    env = spec.make_env()
    obs, state = env.reset_env(jax.random.PRNGKey(0), params)
    step = jax.jit(env.step_env)
    for _ in range(30):
        _, state, _, _, _ = step(jax.random.PRNGKey(0), state, jnp.zeros((1,)), params)
    assert int(state.time) == 30
    assert int(state.physics_time) == 30 * CONTROL_PERIOD
    assert bool(env.is_truncated(state, params))

    _, _, rewards = rollout(spec, params, lambda obs: jnp.zeros((1,)), seed=0)
    assert len(rewards) == 30
    # Past the old (physics-step) limit the plant is still live and scored in
    # full: no order-of-magnitude drop between the first and the last step.
    assert rewards[-1] > 0.1 * rewards[0]


def test_demand_noise_is_fresh_every_sub_step():
    """The OU demand keys its noise on the physics clock, so two consecutive
    env steps must not replay the same draws (which is what folding on the
    env-step clock would do)."""
    env = Reactor()
    params = env.default_params
    obs, state = env.reset_env(jax.random.PRNGKey(3), params)
    step = jax.jit(env.step_env)
    _, s1, _, _, _ = step(jax.random.PRNGKey(0), state, jnp.zeros((1,)), params)
    _, s2, _, _, _ = step(jax.random.PRNGKey(0), s1, jnp.zeros((1,)), params)
    d1 = float(s1.target_n) - float(state.target_n)
    d2 = float(s2.target_n) - float(s1.target_n)
    assert d1 != d2


def test_the_plant_runs_in_float64_under_x64():
    """Under x64 the reset draws a float64 state and the step keeps it, so a
    planner can roll the plant in double precision; the float32 plant is
    unchanged (its recorded returns hold)."""
    try:
        from jax import enable_x64
    except ImportError:  # JAX 0.6; 0.10 removed the experimental name
        from jax.experimental import enable_x64

    with enable_x64():
        env = Reactor()
        _, state = env.reset_env(jax.random.PRNGKey(0), env.default_params)
        _, state, reward, _, _ = env.step_env(
            jax.random.PRNGKey(1), state, jnp.array([0.1]), env.default_params
        )
        floats = {
            leaf.dtype
            for leaf in jax.tree.leaves(state)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
        }
    assert floats == {jnp.dtype("float64")}
    assert reward.dtype == jnp.float64
