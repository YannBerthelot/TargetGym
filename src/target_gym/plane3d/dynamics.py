"""
3D airplane dynamics extending the 2D longitudinal model with roll (bank) physics.

Reuses aerodynamic functions from the 2D plane module. Adds:
- 3D force decomposition (lift tilted by bank angle)
- Roll moment from ailerons + roll damping, which a rolling gust drives
- Heading derived from 3D velocity vector
"""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from gymnax.environments import EnvParams

from target_gym.plane.dynamics import (
    compute_air_density_from_altitude,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_speed_of_sound_from_altitude,
    compute_weight,
    longitudinal_airframe,
    pitch_inertia,
    read_angles,
)
from target_gym.utils import compute_norm_from_coordinates


def rolling_gust(key, speed, params):
    """The roll rate turbulence puts on the wing this step, in rad/s (PHYSICS.md).

    MIL-F-8785C's Dryden spectrum of ``p_g`` starts at
    ``Phi_0 = sigma_w^2 / (V L) 0.8 (pi L / 4b)^(1/3)`` and rolls off at
    ``pi V / 4b``, about 20 rad/s, far above the step. So a step holds a white
    draw, and its size keeps ``Phi_0``: a held draw of std s has
    ``Phi_0 = s^2 dt / pi``. ``sigma_w`` is the stationary std of the gust the
    step advances, so a task's turbulence scales both axes together.
    """
    dt, theta = params.delta_t, params.turbulence_theta
    retained = (1.0 - theta * dt) ** 2
    # ``turbulence_sigma`` is the OU innovation, or on the amended tasks the
    # Dryden gust's stationary std itself (plane3d/PHYSICS.md).
    sigma_w2 = jnp.where(
        getattr(params, "amended", False),
        params.turbulence_sigma**2,
        params.turbulence_sigma**2 * dt / (1.0 - retained),
    )
    span = params.wingspan
    length = params.turbulence_scale_length
    level = sigma_w2 / (speed * length)
    level = level * 0.8 * (jnp.pi * length / (4.0 * span)) ** (1.0 / 3.0)
    return jnp.sqrt(jnp.pi * level / dt) * jax.random.normal(key)


def compute_velocity_3d(x_dot, y_dot, z_dot):
    """Total airspeed from 3D velocity components."""
    return compute_norm_from_coordinates(jnp.array((x_dot, y_dot, z_dot)))


def compute_horizontal_speed(x_dot, y_dot):
    """Horizontal speed magnitude."""
    return compute_norm_from_coordinates(jnp.array((x_dot, y_dot)))


def compute_gamma_3d(x_dot, y_dot, z_dot):
    """Flight path angle from 3D velocity vector."""
    V_horiz = compute_horizontal_speed(x_dot, y_dot)
    return jnp.arctan2(z_dot, V_horiz)


def compute_psi(x_dot, y_dot):
    """Heading (ground track) from horizontal velocity components."""
    return jnp.arctan2(y_dot, x_dot)


def read_angles_3d(state, params):
    """``state`` with the flight-path angle and track an IRS works out from its
    measured velocities (``plane.dynamics.read_angles``)."""
    return read_angles(
        state,
        params,
        gamma=compute_gamma_3d(state.x_dot, state.y_dot, state.z_dot),
        psi=compute_psi(state.x_dot, state.y_dot),
    )


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


@partial(
    jax.jit, static_argnames=["clip", "min_clip_boundaries", "max_clip_boundaries"]
)
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

    Aerodynamics (and the thrust direction) use the **air-relative** velocity
    ``V_ground - wind`` (wind from ``params.wind_x/y/z`` — a physics-engine
    property inherited by every plane-based env): angle of attack, flight-path
    angle, the heading used to orient the forces, airspeed, Mach and dynamic
    pressure all derive from it.  In coordinated flight the nose points into the
    air-relative wind, so under a crosswind the ground track differs from the
    heading (crab).  The accelerations still act on the ground velocity and
    position integrates it, so ``wind = 0`` reproduces the original behaviour.
    """
    xp = jnp
    thrust, stick, aileron = action
    x_dot, y_dot, z_dot, theta_dot, phi_dot = velocities
    _, _, z, theta, phi = positions

    # Air-relative velocity drives all aerodynamics and the body/thrust axis.
    air_x = x_dot - params.wind_x
    air_y = y_dot - params.wind_y
    air_z = z_dot - params.wind_z

    # Derived angles (air-relative)
    alpha, gamma = compute_alpha_3d(theta, air_x, air_y, air_z)
    psi = compute_psi(air_x, air_y)

    m = params.initial_mass
    rho = compute_air_density_from_altitude(z)
    V = compute_velocity_3d(air_x, air_y, air_z)
    M = compute_Mach_from_velocity_and_speed_of_sound(
        velocity=V,
        speed_of_sound=compute_speed_of_sound_from_altitude(z),
    )

    P = compute_weight(m, params.gravity)

    lift_total, drag_total, M_pitch, (C_x_e, C_z_e) = longitudinal_airframe(
        alpha, V, M, rho, theta_dot, stick, params
    )

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
        * params.aileron_effectiveness
        * xp.rad2deg(aileron)
        * params.moment_arm_aileron
    )

    # Roll damping: opposes the roll rate relative to the air, which a rolling
    # gust moves, so it is the term through which turbulence reaches the bank.
    # M_damp = C_lp * ((p - p_g) * b / (2V)) * q * S * b
    V_safe = jnp.maximum(V, 1.0)  # avoid division by zero
    M_roll_damping = (
        params.C_lp
        * ((phi_dot - params.gust_roll_rate) * params.wingspan / (2.0 * V_safe))
        * q
        * params.wings_surface
        * params.wingspan
    )

    # ====================================================
    # TOTAL MOMENTS & FORCES
    # ====================================================
    M_roll = M_aileron + M_roll_damping

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
            M_pitch / pitch_inertia(m, params),
            M_roll / (m * params.roll_gyration_radius**2),
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
