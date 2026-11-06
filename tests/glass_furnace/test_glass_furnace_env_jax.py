import jax
import jax.numpy as jnp

from target_gym.glass_furnace.env import (
    FUEL_DEAD_TIME_STEPS,
    N_REGEN_NODES,
    N_SETPOINTS,
    GlassFurnaceState,
)
from target_gym.glass_furnace.env_jax import GlassFurnace


def _make_state(**overrides) -> GlassFurnaceState:
    """Build a GlassFurnaceState with sensible defaults for the new schema."""
    defaults = dict(
        time=0,
        T_gas=1640.0,
        T_crown=1577.0,
        T_melt=1497.0,
        T_work=1227.0,
        T_rA=jnp.linspace(1300.0, 400.0, N_REGEN_NODES),
        T_rB=jnp.linspace(1300.0, 400.0, N_REGEN_NODES),
        m_batch=12000.0,
        target_T_crown=1587.0,
        target_schedule=jnp.full((N_SETPOINTS,), 1587.0),
        m_pull_disturbance=jnp.zeros(()),
        lhv_deviation=jnp.zeros(()),
        T_crown_meas=1587.0,
        fuel_pipeline=jnp.full((FUEL_DEAD_TIME_STEPS,), 0.6),
        fuel_flow=0.6,
        T_air_preheat=1250.0,
        T_stack=500.0,
    )
    defaults.update(overrides)
    return GlassFurnaceState(**defaults)


def test_reset_places_the_furnace_on_a_consistent_operating_point():
    env = GlassFurnace()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key)

    # Initial crown temperature within the configured initial range
    assert (
        env.default_params.initial_T_crown_range[0]
        <= float(state.T_crown)
        <= env.default_params.initial_T_crown_range[1]
    )
    # Target within target range
    assert (
        env.default_params.target_T_crown_range[0]
        <= float(state.target_T_crown)
        <= env.default_params.target_T_crown_range[1]
    )
    # Reset places the furnace on its settled orbit
    # (test_glass_furnace_operating_point.py), and that ordering must be
    # physical: flame > crown > melt > work.
    assert state.T_gas > state.T_crown > state.T_melt > state.T_work
    # Checker stack starts with its hot-to-cold gradient.
    assert float(state.T_rA[0]) > float(state.T_rA[-1])


def test_observation_hides_unmeasurable_states():
    """Only real plant instrumentation is observable.

    obs = [T_crown_meas, T_air_preheat, fuel_pct, reversal_phase, target, pull].
    Glass-zone temperatures, the checker profile, the batch blanket mass and
    the gas's heating value are all hidden. The crown reading is the
    *instrument*, which lags the true crown temperature.
    """
    env = GlassFurnace()
    sentinels = dict(
        T_melt=1234.5,
        T_work=5678.9,
        m_batch=4321.0,
        lhv_deviation=8765.0,
        T_rA=jnp.full((N_REGEN_NODES,), 2468.0),
    )
    state = _make_state(**sentinels)
    obs = env.get_obs(state)
    for hidden in (1234.5, 5678.9, 4321.0, 8765.0, 2468.0):
        assert not bool(
            jnp.any(jnp.isclose(obs, hidden))
        ), f"hidden state {hidden} leaked into the observation"
