from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


def integrate_dynamics(
    positions: jnp.ndarray,
    delta_t: float,
    method: str = "rk4_1",
    velocities: Optional[jnp.ndarray] = None,
    compute_acceleration: Optional[
        Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, Any]]
    ] = None,
    compute_velocity: Optional[Callable[[jnp.ndarray], Tuple[jnp.ndarray, Any]]] = None,
):
    """
    General integrator for ODEs.

    Supports:
    - Second-order dynamics (velocities + accelerations) if `velocities` and `compute_acceleration` are given.
    - First-order dynamics (positions + velocities) if `compute_velocity` is given.

    Args:
        positions: jnp.ndarray, current position/state
        delta_t: float, integration step size
        method: str, either "euler_N" or "rkK_N" (K = 2–4, N = substeps)
        velocities: jnp.ndarray, current velocities (required for 2nd-order)
        compute_acceleration: function (v, p) -> (a, metrics), required for 2nd-order
        compute_velocity: function (p) -> velocity, required for 1st-order

    Returns:
        new_velocities (or None if 1st-order),
        new_positions,
        metrics (from the last step)
    """

    # Parse method string
    if "euler" in method:
        order, n_substeps = "euler", int(method.split("_")[1])
    elif "rk" in method:
        order, n_substeps = int(method.split("_")[0][2:]), int(method.split("_")[1])
    else:
        raise ValueError(f"Unknown integration method: {method}")

    h = delta_t / n_substeps

    # ----------------------------
    # Second-order system
    # ----------------------------
    def second_order_step(v, p, h):
        def euler(v, p, h):
            a, metrics = compute_acceleration(v, p)
            v_new = v + h * a
            p_new = p + h * v_new
            return v_new, p_new, metrics

        def rk2(v, p, h):
            a1, _ = compute_acceleration(v, p)
            v1 = v
            a2, metrics = compute_acceleration(v + 0.5 * h * a1, p + 0.5 * h * v1)
            v2 = v + 0.5 * h * a1
            v_new = v + h * a2
            p_new = p + h * v2
            return v_new, p_new, metrics

        def rk3(v, p, h):
            a1, _ = compute_acceleration(v, p)
            v1 = v
            a2, _ = compute_acceleration(v + 0.5 * h * a1, p + 0.5 * h * v1)
            v2 = v + 0.5 * h * a1
            a3, metrics = compute_acceleration(
                v - h * a1 + 2 * h * a2, p - h * v1 + 2 * h * v2
            )
            v3 = v - h * a1 + 2 * h * a2
            v_new = v + (h / 6.0) * (a1 + 4 * a2 + a3)
            p_new = p + (h / 6.0) * (v1 + 4 * v2 + v3)
            return v_new, p_new, metrics

        def rk4(v, p, h):
            a1, _ = compute_acceleration(v, p)
            v1 = v
            a2, _ = compute_acceleration(v + 0.5 * h * a1, p + 0.5 * h * v1)
            v2 = v + 0.5 * h * a1
            a3, _ = compute_acceleration(v + 0.5 * h * a2, p + 0.5 * h * v2)
            v3 = v + 0.5 * h * a2
            a4, metrics = compute_acceleration(v + h * a3, p + h * v3)
            v4 = v + h * a3
            v_new = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
            p_new = p + (h / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
            return v_new, p_new, metrics

        if order == "euler":
            return euler(v, p, h)
        elif order == 2:
            return rk2(v, p, h)
        elif order == 3:
            return rk3(v, p, h)
        elif order == 4:
            return rk4(v, p, h)
        else:
            raise ValueError(f"Unsupported RK order: {order}")

    # ----------------------------
    # First-order system
    # ----------------------------
    def first_order_step(p, h):
        def euler(p, h):
            v, metrics = compute_velocity(p)
            return p + h * v, metrics

        def rk2(p, h):
            k1, _ = compute_velocity(p)
            k2, metrics = compute_velocity(p + 0.5 * h * k1)
            return p + h * k2, metrics

        def rk3(p, h):
            k1, _ = compute_velocity(p)
            k2, _ = compute_velocity(p + 0.5 * h * k1)
            k3, metrics = compute_velocity(p - h * k1 + 2 * h * k2)
            return p + (h / 6.0) * (k1 + 4 * k2 + k3), metrics

        def rk4(p, h):
            k1, _ = compute_velocity(p)
            k2, _ = compute_velocity(p + 0.5 * h * k1)
            k3, _ = compute_velocity(p + 0.5 * h * k2)
            k4, metrics = compute_velocity(p + h * k3)
            return p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), metrics

        if order == "euler":
            return euler(p, h)
        elif order == 2:
            return rk2(p, h)
        elif order == 3:
            return rk3(p, h)
        elif order == 4:
            return rk4(p, h)
        else:
            raise ValueError(f"Unsupported RK order: {order}")

    # ----------------------------
    # Run integration with substeps
    # ----------------------------
    if compute_velocity is not None:  # first-order mode

        def step_fn(p, _):
            p_new, metrics = first_order_step(p, h)
            return p_new, metrics

        new_positions, metrics = jax.lax.scan(
            step_fn, positions, xs=None, length=n_substeps
        )
        return new_positions, metrics

    elif (
        compute_acceleration is not None and velocities is not None
    ):  # second-order mode

        def step_fn(carry, _):
            v, p = carry
            v_new, p_new, metrics = second_order_step(v, p, h)
            v_new = v_new.reshape(v.shape)
            p_new = p_new.reshape(p.shape)
            return (v_new, p_new), metrics

        # if jnp.ndim(velocities) == 0:
        #     velocities = velocities.reshape((1,))
        # if jnp.ndim(positions) == 0:
        #     positions = positions.reshape((1,))
        (new_velocities, new_positions), metrics = jax.lax.scan(
            step_fn, (velocities, positions), xs=None, length=n_substeps
        )
        return new_velocities, new_positions, metrics

    else:
        raise ValueError(
            "Must provide either (compute_velocity) or (velocities + compute_acceleration)."
        )
