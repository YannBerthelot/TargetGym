# test_integrate_dynamics.py
import math

import jax.numpy as jnp
import pytest

from target_gym.integration import (
    integrate_dynamics,  # <-- adjust import to where your function lives
)


def test_first_order_constant_velocity():
    """dx/dt = 1 → x(t) = x0 + t"""
    compute_velocity = lambda p: (jnp.array([1.0]), None)
    positions = jnp.array([0.0])
    delta_t = 1.0

    for method in ["euler_1", "rk2_1", "rk3_1", "rk4_1", "rk4_10"]:
        new_p, metrics = integrate_dynamics(
            positions, delta_t, method, compute_velocity=compute_velocity
        )
        assert jnp.allclose(new_p, jnp.array([1.0]), atol=1e-6)


def test_second_order_constant_acceleration():
    """v' = g, p' = v with g=-9.8"""
    g = -9.8
    compute_acceleration = lambda v, p: (jnp.array([g]), None)

    positions = jnp.array([0.0])
    velocities = jnp.array([10.0])
    delta_t = 1.0

    v_exact = velocities + g * delta_t
    p_exact = positions + velocities * delta_t + 0.5 * g * delta_t**2

    for method in ["euler_1", "rk2_1", "rk3_1", "rk4_1", "rk4_5"]:
        v, p, metrics = integrate_dynamics(
            positions,
            delta_t,
            method,
            velocities=velocities,
            compute_acceleration=compute_acceleration,
        )
        if method.startswith("euler"):
            assert jnp.allclose(v, v_exact, atol=1e-6)
            # don't check p vs exact, just sanity check it's finite
            assert jnp.isfinite(p).all()
        else:
            assert jnp.allclose(v, v_exact, atol=1e-6)
            assert jnp.allclose(p, p_exact, atol=1e-6)


def test_second_order_substep_convergence():
    """Harmonic oscillator: x'' = -x, exact solution cos(t)."""

    def compute_acceleration(v, p):
        return -p, None

    positions = jnp.array([1.0])  # x(0) = 1
    velocities = jnp.array([0.0])  # v(0) = 0
    delta_t = 0.1
    exact = jnp.array([math.cos(delta_t)])  # true solution

    # Single RK4 step
    _, p1, _ = integrate_dynamics(
        positions,
        delta_t,
        "rk4_1",
        velocities=velocities,
        compute_acceleration=compute_acceleration,
    )

    # More substeps should reduce error
    _, p10, _ = integrate_dynamics(
        positions,
        delta_t,
        "rk4_10",
        velocities=velocities,
        compute_acceleration=compute_acceleration,
    )

    err1 = abs(p1 - exact)
    err10 = abs(p10 - exact)
    assert err10 < err1 + 1e-7


def test_invalid_method():
    """Unknown method should raise ValueError."""
    with pytest.raises(ValueError):
        integrate_dynamics(
            jnp.array([0.0]), 1.0, "foobar", compute_velocity=lambda p: p
        )


def test_missing_arguments():
    """Missing velocity/acceleration definitions should raise ValueError."""
    with pytest.raises(ValueError):
        integrate_dynamics(jnp.array([0.0]), 1.0, "euler_1")
