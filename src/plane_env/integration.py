from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def compute_velocity_and_pos_from_acceleration_integration(
    velocities: jnp.ndarray,
    positions: jnp.ndarray,
    delta_t: float,
    compute_acceleration: Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, Any]],
    method: str = "rk4_1",
):
    """
    Integrates velocities and positions given an acceleration function.

    Supports:
    - Semi-implicit Euler integration ("euler_N", with N substeps)
    - Runge-Kutta integration ("rkK_N", with order K and N substeps)

    Args:
        velocities: jnp.ndarray, current velocity state
        positions: jnp.ndarray, current position state
        delta_t: float, integration step size
        compute_acceleration: function (velocities, positions) -> (accelerations, metrics)
        method: str, either "euler_N" or "rkK_N" where K is order (2–4) and N is substeps

    Returns:
        new_velocities: updated velocities
        new_positions: updated positions
        metrics: auxiliary metrics (from the last call to compute_acceleration)
    """

    def euler_step(v, p, h):
        a, metrics = compute_acceleration(v, p)
        v_new = v + a * h
        p_new = p + v_new * h
        return v_new, p_new, metrics

    def rk2_step(v, p, h):
        a1, _ = compute_acceleration(v, p)
        v1 = v

        a2, metrics = compute_acceleration(v + 0.5 * a1 * h, p + 0.5 * v1 * h)
        v2 = v + 0.5 * a1 * h

        v_new = v + h * a2
        p_new = p + h * v2
        return v_new, p_new, metrics

    def rk3_step(v, p, h):
        a1, _ = compute_acceleration(v, p)
        v1 = v

        a2, _ = compute_acceleration(v + 0.5 * a1 * h, p + 0.5 * v1 * h)
        v2 = v + 0.5 * a1 * h

        a3, metrics = compute_acceleration(
            v - h * a1 + 2 * h * a2, p - h * v1 + 2 * h * v2
        )
        v3 = v - h * a1 + 2 * h * a2

        v_new = v + (h / 6.0) * (a1 + 4 * a2 + a3)
        p_new = p + (h / 6.0) * (v1 + 4 * v2 + v3)
        return v_new, p_new, metrics

    def rk4_step(v, p, h):
        a1, _ = compute_acceleration(v, p)
        v1 = v

        a2, _ = compute_acceleration(v + 0.5 * a1 * h, p + 0.5 * v1 * h)
        v2 = v + 0.5 * a1 * h

        a3, _ = compute_acceleration(v + 0.5 * a2 * h, p + 0.5 * v2 * h)
        v3 = v + 0.5 * a2 * h

        a4, metrics = compute_acceleration(v + a3 * h, p + v3 * h)
        v4 = v + a3 * h

        v_new = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        p_new = p + (h / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        return v_new, p_new, metrics

    # Parse method string
    if "euler" in method:
        order = "euler"
        n_substeps = int(method.split("_")[1])
    elif "rk" in method:
        parts = method.split("_")
        order = int(parts[0][2:])  # e.g. "rk3" -> 3
        n_substeps = int(parts[1])
    else:
        raise ValueError(f"Unknown integration method: {method}")

    h = delta_t / n_substeps

    def step_fn(carry, _):
        v, p = carry
        if order == "euler":
            v_new, p_new, metrics = euler_step(v, p, h)
        elif order == 2:
            v_new, p_new, metrics = rk2_step(v, p, h)
        elif order == 3:
            v_new, p_new, metrics = rk3_step(v, p, h)
        elif order == 4:
            v_new, p_new, metrics = rk4_step(v, p, h)
        else:
            raise ValueError(f"Unsupported RK order: {order}")
        return (v_new, p_new), metrics

    (new_velocities, new_positions), metrics = jax.lax.scan(
        f=step_fn, init=(velocities, positions), xs=None, length=n_substeps
    )

    return new_velocities, new_positions, metrics
