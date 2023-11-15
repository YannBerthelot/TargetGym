"""Make sure that every function is jax jitable/vmapable"""
import jax
import jax.numpy as jnp

from plane.dynamics import (
    compute_drag,
    compute_next_state,
    compute_weight,
    compute_z_drag_coefficient,
)
from plane.env import EnvParams, EnvState
from plane.utils import list_to_array


def test_compute_drag():
    S = jnp.array([2, 3])
    C = jnp.array([10, 11])
    V = jnp.array([5, 4])
    rho = jnp.array([0.01, 0.02])
    expected_drag = 0.5 * rho[0] * S[0] * C[0] * V[0] ** 2
    out = jax.vmap(compute_drag, in_axes=0)(S, C, V, rho)


def test_compute_z_drag_coefficient():
    C_z_max = 0.9
    threshold_alpha = 15
    stall_alpha = 20
    min_alpha = 5
    alpha = jnp.array(
        [stall_alpha + 0.1, threshold_alpha + 0.1, min_alpha + 0.1, min_alpha - 0.01]
    )
    C_z_max = jnp.array([C_z_max for _ in range(len(alpha))])
    threshold_alpha = jnp.array([threshold_alpha for _ in range(len(alpha))])
    stall_alpha = jnp.array([stall_alpha for _ in range(len(alpha))])
    min_alpha = jnp.array([min_alpha for _ in range(len(alpha))])
    out = jax.vmap(compute_z_drag_coefficient, in_axes=0)(
        alpha, C_z_max, threshold_alpha, stall_alpha, min_alpha
    )


def test_compute_next_state():
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

    jax.vmap(compute_next_state, in_axes=0)(powers, state, params)
