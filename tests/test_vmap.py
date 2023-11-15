"""Make sure that every function is jax jitable/vmapable"""
import jax
import jax.numpy as jnp

from plane.dynamics import (compute_drag, compute_weight,
                            compute_z_drag_coefficient)


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
