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


# ── The exponential step, "exp_N", and steppers registered by name ──────────
#
# Contract (owner, 2026-11-24, "Exact step for linear parts"):
#
# - ``exp_N`` takes N steps of ``y + h phi1(h J) f(y)``, J the field's Jacobian
#   at ``y`` (the exponential Rosenbrock-Euler step). Where the field is affine,
#   ``f(y) = A y + b``, that is the exact solution whatever the step's length.
#   On a smooth field it is of order 2.
# - A caller that knows its Jacobian's structure passes ``drift(rate, jacobian,
#   h)``, the change of the state over ``h`` under the linearised field, in
#   place of the default matrix exponential.
# - It works in the second-order form, and under ``jit``, ``vmap`` and ``grad``.
# - ``STEPPERS`` maps a name to a step ``(field, y, h, drift) -> (y, aux)``;
#   ``"<name>_<N>"`` then takes N of them.
#
# Expected values are closed forms, worked in the comments.

import jax  # noqa: E402
import numpy as np  # noqa: E402

from target_gym import integration  # noqa: E402

#: float32, one step: a few ulps of values of order 1.
F32 = 2e-6


def _spiral(decay, turn, forcing):
    """``y' = A y + b`` with ``A = [[-decay, -turn], [turn, -decay]]``: its
    solution turns about ``y_eq = -A^-1 b`` by ``turn * t`` and shrinks by
    ``exp(-decay * t)``. Complex modes, so no real eigenbasis helps the step."""
    A = np.array([[-decay, -turn], [turn, -decay]])
    b = np.asarray(forcing, float)
    y_eq = -np.linalg.solve(A, b)

    def exact(y0, t):
        c, s = np.cos(turn * t), np.sin(turn * t)
        return y_eq + np.exp(-decay * t) * np.array([[c, -s], [s, c]]) @ (y0 - y_eq)

    field = lambda y: (jnp.asarray(A, y.dtype) @ y + jnp.asarray(b, y.dtype), None)
    return field, exact


@pytest.mark.parametrize("length", [0.1, 3.0, 40.0])
def test_exponential_step_is_exact_on_an_affine_field_whatever_its_length(length):
    field, exact = _spiral(decay=0.7, turn=2.0, forcing=[1.0, -0.5])
    y0 = np.array([2.0, -1.0])
    new, _ = integrate_dynamics(
        jnp.asarray(y0, jnp.float32), length, "exp_1", compute_velocity=field
    )
    np.testing.assert_allclose(new, exact(y0, length), atol=F32)


def test_rk4_is_not_exact_on_that_field_at_a_long_step():
    """The same field at ``h |lambda|`` of about 6: RK4 leaves it by far more
    than rounding, which is what the exponential step is for."""
    field, exact = _spiral(decay=0.7, turn=2.0, forcing=[1.0, -0.5])
    y0 = np.array([2.0, -1.0])
    new, _ = integrate_dynamics(
        jnp.asarray(y0, jnp.float32), 3.0, "rk4_1", compute_velocity=field
    )
    assert np.max(np.abs(np.asarray(new) - exact(y0, 3.0))) > 1e-2


def test_exponential_step_is_of_order_two_on_a_curved_field():
    """``y' = -y^2`` from 1 has ``y(t) = 1 / (1 + t)``. Halving the step divides
    an order-2 error by 4; 3 to 5 leaves room for the next term."""
    field = lambda y: (-y * y, None)
    errors = []
    for count in (8, 16, 32):
        new, _ = integrate_dynamics(
            jnp.array([1.0], jnp.float32), 1.0, f"exp_{count}", compute_velocity=field
        )
        errors.append(abs(float(new[0]) - 0.5))
    assert 3.0 < errors[0] / errors[1] < 5.0
    assert errors[0] > 20 * F32  # the errors compared are above rounding


def test_exponential_step_takes_the_callers_drift():
    """On ``y' = -k y`` the drift over ``h`` is ``(exp(-k h) - 1) y``. A caller
    passing that closed form gets the default's answer; one passing a drift
    of zero stays put, so the callable is what moves the state."""
    k, h, y0 = 0.8, 2.5, jnp.array([3.0], jnp.float32)
    field = lambda y: (-k * y, None)

    def closed_form(rate, jacobian, h):
        return jnp.expm1(jacobian[0, 0] * h) / jacobian[0, 0] * rate

    default, _ = integrate_dynamics(y0, h, "exp_1", compute_velocity=field)
    own, _ = integrate_dynamics(
        y0, h, "exp_1", compute_velocity=field, drift=closed_form
    )
    still, _ = integrate_dynamics(
        y0, h, "exp_1", compute_velocity=field, drift=lambda r, j, h: 0.0 * r
    )
    np.testing.assert_allclose(own, 3.0 * np.exp(-k * h), atol=F32)
    np.testing.assert_allclose(default, own, atol=F32)
    np.testing.assert_allclose(still, y0, atol=0.0)


def test_exponential_step_in_the_second_order_form():
    """``v' = g``, ``p' = v`` is affine: ``v = v0 + g t`` and ``p = p0 + v0 t +
    g t^2 / 2`` exactly. With 10 m/s up, g = -9.8 and t = 2: v = -9.6, p = 0.4."""
    v, p, _ = integrate_dynamics(
        jnp.array([0.0], jnp.float32),
        2.0,
        "exp_1",
        velocities=jnp.array([10.0], jnp.float32),
        compute_acceleration=lambda v, p: (jnp.full_like(v, -9.8), None),
    )
    np.testing.assert_allclose(v, [-9.6], atol=1e-5)
    np.testing.assert_allclose(p, [0.4], atol=1e-5)


def test_exponential_step_under_jit_vmap_and_grad():
    """``y' = -k y`` over ``h``: ``y0 exp(-k h)``, and ``-h y0 exp(-k h)`` for
    its derivative in ``k``. With y0 = 2, h = 1.5: at k = 0.4, 1.0976 and
    -1.6464; at k = 1.0, 0.4463 and -0.6694."""

    def end(k, y0):
        field = lambda y: (-k * y, None)
        return integrate_dynamics(y0, 1.5, "exp_1", compute_velocity=field)[0][0]

    ks, y0 = jnp.array([0.4, 1.0], jnp.float32), jnp.array([2.0], jnp.float32)
    values = jax.jit(jax.vmap(end, in_axes=(0, None)))(ks, y0)
    slopes = jax.jit(jax.vmap(jax.grad(end), in_axes=(0, None)))(ks, y0)
    np.testing.assert_allclose(values, 2.0 * np.exp(-1.5 * np.asarray(ks)), rtol=1e-5)
    np.testing.assert_allclose(slopes, -3.0 * np.exp(-1.5 * np.asarray(ks)), rtol=1e-5)


def test_a_registered_stepper_is_taken_by_its_name():
    """Three steps of a stepper that adds ``h`` move the state by the whole
    interval, and its ``aux`` comes back, one per step as the other families
    return it; the name is free again afterwards."""
    integration.STEPPERS["plus"] = lambda field, y, h, drift: (y + h, 5.0 * h)
    try:
        new, aux = integrate_dynamics(
            jnp.array([1.0]), 6.0, "plus_3", compute_velocity=lambda y: (y, None)
        )
    finally:
        del integration.STEPPERS["plus"]
    np.testing.assert_allclose(new, [7.0])
    np.testing.assert_allclose(aux, [10.0, 10.0, 10.0])
    with pytest.raises(ValueError):
        integrate_dynamics(
            jnp.array([1.0]), 6.0, "plus_3", compute_velocity=lambda y: (y, None)
        )
