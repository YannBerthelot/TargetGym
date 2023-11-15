"""Make sure that functions are jax jitable/vmapable"""
import jax
import jax.numpy as jnp

from plane.env import (
    compute_next_state,
)
from plane.env import EnvParams, EnvState
from plane.utils import list_to_array


def test_compute_next_state():
    """This tests all functions in dynamics"""
    params = EnvParams()
    state = EnvState(
        x=0,
        x_dot=0,
        z=0,
        z_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=params.initial_mass + params.initial_fuel_quantity,
        power=0,
        fuel=params.initial_fuel_quantity,
        rho=params.air_density_at_sea_level,
        t=0,
    )
    powers = jnp.ones(10)
    params = list_to_array([params for _ in range(10)])
    state = list_to_array([state for _ in range(10)])

    jax.vmap(jax.jit(compute_next_state), in_axes=0)(powers, state, params)
