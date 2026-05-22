"""
3D airplane dynamics extending the 2D longitudinal model with roll (bank) physics.

Reuses aerodynamic functions from the 2D plane module. Adds:
- 3D force decomposition (lift tilted by bank angle)
- Roll moment from ailerons + roll damping
- Heading derived from 3D velocity vector
"""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from gymnax.environments import EnvParams

from target_gym.plane.dynamics import (
    EPS,
    aero_coefficients,
    aero_coefficients_with_mach,
    aero_mach_factors,
    clamp_altitude,
    compute_air_density_from_altitude,
    compute_drag,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_next_power,
    compute_next_stick,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    compute_weight,
)
from target_gym.utils import compute_norm_from_coordinates, norm2, norm3


def compute_next_aileron(requested_aileron, current_aileron, delta_t):
    """Smooth aileron response (same dynamics as stick)."""
    aileron_diff = requested_aileron - current_aileron
    current_aileron += 0.9 * delta_t * aileron_diff
    return current_aileron


def compute_velocity_3d(x_dot, y_dot, z_dot):
    """Total airspeed from 3D velocity components."""
    return norm3(x_dot, y_dot, z_dot)


def compute_horizontal_speed(x_dot, y_dot):
    """Horizontal speed magnitude."""
    return norm2(x_dot, y_dot)


def compute_gamma_3d(x_dot, y_dot, z_dot):
    """Flight path angle from 3D velocity vector."""
    V_horiz = compute_horizontal_speed(x_dot, y_dot)
    return jnp.arctan2(z_dot, V_horiz)


def compute_psi(x_dot, y_dot):
    """Heading (ground track) from horizontal velocity components."""
    return jnp.arctan2(y_dot, x_dot)


def compute_alpha_3d(theta, x_dot, y_dot, z_dot):
    """
    Angle of attack in the longitudinal plane.
    AoA = pitch - flight path angle.
    """
    gamma = compute_gamma_3d(x_dot, y_dot, z_dot)
    alpha = theta - gamma
    return jnp.arctan2(jnp.sin(alpha), jnp.cos(alpha)), gamma


def newton_second_law_3d(
    thrust: float,
    lift: float,
    drag: float,
    P: float,
    gamma: float,
    theta: float,
    phi: float,
    psi: float,
) -> tuple[float, float, float]:
    """
    3D Newton's second law. Computes net forces in world frame (x, y, z).

    The lift vector is perpendicular to the velocity and rotated by the bank
    angle phi around the velocity axis:
      - cos(phi) component in the vertical plane ("up")
      - sin(phi) component in the horizontal plane ("right")
    """
    # Velocity direction
    v_hat = jnp.array(
        [
            jnp.cos(gamma) * jnp.cos(psi),
            jnp.cos(gamma) * jnp.sin(psi),
            jnp.sin(gamma),
        ]
    )

    # Drag: opposite velocity
    F_drag = -drag * v_hat

    # Lift decomposition with bank angle
    # "Up" perpendicular to velocity in the vertical plane
    lift_up = jnp.array(
        [
            -jnp.sin(gamma) * jnp.cos(psi),
            -jnp.sin(gamma) * jnp.sin(psi),
            jnp.cos(gamma),
        ]
    )
    # "Right" perpendicular (horizontal, perpendicular to heading)
    lift_right = jnp.array(
        [
            -jnp.sin(psi),
            jnp.cos(psi),
            0.0,
        ]
    )

    lift_dir = jnp.cos(phi) * lift_up + jnp.sin(phi) * lift_right
    F_lift = lift * lift_dir

    # Thrust along body axis (pitch + heading)
    t_hat = jnp.array(
        [
            jnp.cos(theta) * jnp.cos(psi),
            jnp.cos(theta) * jnp.sin(psi),
            jnp.sin(theta),
        ]
    )
    F_thrust = thrust * t_hat

    # Weight: downward
    F_weight = jnp.array([0.0, 0.0, -P])

    F_total = F_drag + F_lift + F_thrust + F_weight
    return F_total[0], F_total[1], F_total[2]


def compute_acceleration_3d(
    velocities: jnp.ndarray,
    positions: jnp.ndarray,
    action: tuple,
    params: EnvParams,
    clip: bool = False,
    min_clip_boundaries: Optional[tuple] = None,
    max_clip_boundaries: Optional[tuple] = None,
) -> tuple[float]:
    """
    Compute linear and angular accelerations for the 3D aircraft.

    velocities: [x_dot, y_dot, z_dot, theta_dot, phi_dot]
    positions:  [x, y, z, theta, phi]
    action:     (thrust, stick, aileron)

    Returns: (accelerations [a_x, a_y, a_z, alpha_pitch, alpha_roll], metrics)
    """
    xp = jnp
    thrust, stick, aileron = action
    x_dot, y_dot, z_dot, _, phi_dot = velocities
    _, _, z, theta, phi = positions

    # Derived angles
    alpha, gamma = compute_alpha_3d(theta, x_dot, y_dot, z_dot)
    psi = compute_psi(x_dot, y_dot)

    m = params.initial_mass
    rho = compute_air_density_from_altitude(z)
    V = compute_velocity_3d(x_dot, y_dot, z_dot)
    M = compute_Mach_from_velocity_and_speed_of_sound(
        velocity=V,
        speed_of_sound=compute_speed_of_sound_from_altitude(z),
    )

    P = compute_weight(m, params.gravity)

    # Pre-compute Mach factors once; reused across wings/stabilizer/elevator.
    beta, drag_rise = aero_mach_factors(M, params)
    alpha_deg = xp.rad2deg(alpha)
    stick_deg = xp.rad2deg(stick)

    # ====================================================
    # WINGS
    # ====================================================
    C_z_w, C_x_w = aero_coefficients_with_mach(alpha_deg, beta, drag_rise, params)
    lift_wings = compute_drag(S=params.wings_surface, C=C_z_w, V=V, rho=rho)
    drag_wings = compute_drag(S=params.wings_surface, C=C_x_w, V=V, rho=rho)
    M_wings = lift_wings * params.moment_arm_wings

    # ====================================================
    # STABILIZER
    # ====================================================
    C_z_s, C_x_s = aero_coefficients_with_mach(alpha_deg - 3.0, beta, drag_rise, params)
    lift_stab = compute_drag(S=params.stabilizer_surface, C=C_z_s, V=V, rho=rho)
    drag_stab = compute_drag(S=params.stabilizer_surface, C=C_x_s, V=V, rho=rho)
    F_stab = lift_stab - drag_stab
    M_stabilizer = -F_stab * params.moment_arm_stabilizer

    # ====================================================
    # ELEVATOR
    # ====================================================
    C_z_e, C_x_e = aero_coefficients_with_mach(
        alpha_deg - stick_deg - 3.0, beta, drag_rise, params
    )
    lift_elev = compute_drag(S=params.elevator_surface, C=C_z_e, V=V, rho=rho)
    drag_elev = compute_drag(S=params.elevator_surface, C=C_x_e, V=V, rho=rho)
    F_elev = lift_elev * xp.cos(stick) - drag_elev * xp.sin(stick)
    M_elevator = -F_elev * params.moment_arm_stabilizer

    # ====================================================
    # AILERON (roll moment from differential lift)
    # ====================================================
    # Ailerons create roll by differential lift: one wing goes up, the other
    # down.  The net roll moment is proportional to dynamic pressure, aileron
    # area, deflection angle, and moment arm.  At zero deflection the moment
    # is zero (no differential lift).
    q = 0.5 * rho * V**2
    M_aileron = (
        q
        * params.aileron_surface
        * params.cl_alpha
        * xp.rad2deg(aileron)
        * params.moment_arm_aileron
    )

    # Roll damping: opposes roll rate
    # M_damp = C_lp * (p * b / (2V)) * q * S * b
    V_safe = jnp.maximum(V, 1.0)  # avoid division by zero
    M_roll_damping = (
        params.C_lp
        * (phi_dot * params.wingspan / (2.0 * V_safe))
        * q
        * params.wings_surface
        * params.wingspan
    )

    # ====================================================
    # TOTAL MOMENTS & FORCES
    # ====================================================
    M_pitch = M_wings + M_stabilizer + M_elevator
    M_roll = M_aileron + M_roll_damping

    drag_total = drag_wings + drag_stab + drag_elev
    lift_total = lift_wings + lift_stab + lift_elev

    F_x, F_y, F_z = newton_second_law_3d(
        thrust=thrust,
        lift=lift_total,
        drag=drag_total,
        P=P,
        gamma=gamma,
        theta=theta,
        phi=phi,
        psi=psi,
    )

    metrics = (drag_total, lift_total, C_x_e, C_z_e, F_x, F_y, F_z)
    accelerations = xp.array(
        [
            F_x / m,
            F_y / m,
            F_z / m,
            M_pitch / params.I,
            M_roll / params.I_x,
        ]
    )

    if clip:
        assert min_clip_boundaries is not None
        assert max_clip_boundaries is not None
        accelerations = jnp.clip(
            accelerations,
            jnp.array(min_clip_boundaries),
            jnp.array(max_clip_boundaries),
        )
    return accelerations, metrics
