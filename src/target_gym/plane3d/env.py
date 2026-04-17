"""
3D airplane environment state, parameters, and transition logic.

The physics (state transition) are shared across all 3D tasks.
Task-specific reward, observation, and reset logic live in the
individual environment classes (env_jax.py).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.plane.dynamics import (
    compute_air_density_from_altitude,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_next_power,
    compute_next_stick,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
)
from target_gym.plane3d.dynamics import (
    compute_acceleration_3d,
    compute_alpha_3d,
    compute_next_aileron,
    compute_psi,
    compute_velocity_3d,
)

SPEED_OF_SOUND = 343.0


@struct.dataclass
class PlaneState3D(EnvState):
    x: float
    x_dot: float
    y: float
    y_dot: float
    z: float
    z_dot: float
    theta: float
    theta_dot: float
    phi: float
    phi_dot: float
    psi: float
    alpha: float
    gamma: float
    m: float
    power: float
    stick: float
    aileron: float
    fuel: float
    # Task targets — semantics depend on the task variant
    target_altitude: float
    target_heading: (
        float  # heading: desired heading; fig8: orientation angle; circle: unused
    )
    target_x: float  # circle/fig8: center x; heading: unused (0)
    target_y: float  # circle/fig8: center y; heading: unused (0)
    target_radius: float  # circle: radius; fig8: lobe radius; heading: unused (0)

    @property
    def rho(self):
        return compute_air_density_from_altitude(self.z)

    @property
    def speed_of_sound(self):
        return compute_speed_of_sound_from_altitude(self.z)

    @property
    def M(self):
        return compute_Mach_from_velocity_and_speed_of_sound(
            compute_velocity_3d(self.x_dot, self.y_dot, self.z_dot),
            self.speed_of_sound,
        )


@struct.dataclass
class PlaneParams3D(EnvParams):
    gravity: float = 9.81
    initial_mass: float = 73_500.0
    thrust_output_at_sea_level: float = 240_000.0
    air_density_at_sea_level: float = 1.225
    frontal_surface: float = 12.6
    wings_surface: float = 122.6
    C_x0: float = 0.095
    C_z0: float = 0.9
    initial_fuel_quantity: float = 23860 / 1.25
    specific_fuel_consumption: float = 17.5 / 1000

    # Aero coefficients (shared with 2D)
    cl_alpha: float = 0.04
    cl0: float = 0.2
    cd0: float = 0.02
    k: float = 0.045
    aoa_stall: float = 15.0
    CL_max: float = 1.5
    M_crit: float = 0.80
    k_drag: float = 5.0

    speed_of_sound: float = SPEED_OF_SOUND
    I: float = 9_000_000  # Iyy (pitch)
    I_x: float = 2_500_000  # Ixx (roll)
    moment_arm_stabilizer: float = 15.0
    moment_arm_wings: float = 1.5
    stabilizer_surface: float = 27
    elevator_surface: float = 10

    # Roll-specific A320 parameters
    wingspan: float = 35.8
    aileron_surface: float = 6.0
    moment_arm_aileron: float = 14.0
    C_lp: float = -0.4  # roll damping derivative

    max_steps_in_episode: int = 10_000
    min_alt: float = 0.0
    max_alt: float = 40_000.0 / 3.281
    target_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    target_heading_range: Tuple[float, float] = (-3.14159, 3.14159)
    target_radius_range: Tuple[float, float] = (8_000.0, 12_000.0)  # m
    # Figure-8: half-amplitude of the altitude twist (meters).  The curve
    # altitude is z_mean ± this value, so the two crossover passes differ
    # by 2× this.  200 m ≈ 660 ft — gentle enough for an A320 but enough
    # to require coordinated altitude+heading control.
    figure8_altitude_amplitude: float = 200.0
    # Random orientation of the lemniscate (radians).  ±15° by default.
    figure8_angle_range: Tuple[float, float] = (-0.26, 0.26)
    initial_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    initial_z_dot: float = 0.0
    initial_x_dot: float = 200.0
    initial_y_dot: float = 0.0
    initial_theta_dot: float = 0.0
    initial_theta: float = 0.0
    initial_phi: float = 0.0
    initial_phi_dot: float = 0.0
    initial_heading: float = 0.0
    initial_power: float = 1.0
    initial_stick: float = 0.0
    initial_aileron: float = 0.0

    delta_t: float = 1.0


# ─── Shared helpers ──────────────────────────────────────


def check_is_terminal_3d(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Return (terminated, truncated) flags."""
    terminated = xp.logical_or(state.z <= params.min_alt, state.z >= params.max_alt)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def altitude_reward(state, params, xp=jnp):
    """Altitude tracking component, shared by all tasks."""
    max_alt_diff = params.max_alt - params.min_alt
    return xp.float_power(
        (max_alt_diff - xp.abs(state.target_altitude - state.z)) / max_alt_diff,
        10.0,
    )


def terminal_penalty(state, params, xp=jnp):
    """Check terminal and return penalty flag."""
    return xp.logical_or(state.z <= params.min_alt, state.z >= params.max_alt)


# ─── Heading task reward ────────────────────────────────


def compute_reward_heading(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: altitude tracking * heading tracking."""
    done_alt = terminal_penalty(state, params, xp)
    alt_r = altitude_reward(state, params, xp)
    heading_diff = xp.abs(wrap_angle(state.psi - state.target_heading))
    heading_r = (1.0 - heading_diff / jnp.pi) ** 2
    return xp.where(done_alt, -1.0 * params.max_steps_in_episode, alt_r * heading_r)


# ─── Circle task reward ─────────────────────────────────


def distance_to_circle(state: PlaneState3D):
    """Signed distance from aircraft to the target circle (positive = outside)."""
    dx = state.x - state.target_x
    dy = state.y - state.target_y
    dist_to_center = jnp.sqrt(dx**2 + dy**2)
    return dist_to_center - state.target_radius


def compute_reward_circle(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: altitude tracking * proximity to the circle path."""
    done_alt = terminal_penalty(state, params, xp)
    alt_r = altitude_reward(state, params, xp)
    d = xp.abs(distance_to_circle(state))
    # Normalize by radius so reward doesn't depend on circle size
    circle_r = xp.exp(-0.5 * (d / (state.target_radius * 0.1)) ** 2)
    return xp.where(done_alt, -1.0 * params.max_steps_in_episode, alt_r * circle_r)


# ─── Figure-8 task: twisted 3D lemniscate ───────────────
#
# The lemniscate of Bernoulli is parametrised as:
#   x(τ) = a·cos(τ) / (1 + sin²τ)
#   y(τ) = a·sin(τ)·cos(τ) / (1 + sin²τ)
# for τ ∈ [0, 2π].  The 3D twist adds a sinusoidal altitude:
#   z(τ) = z_mean + Δz·sin(τ)
# so the two crossover passes (τ=π/2 at z_mean+Δz and τ=3π/2 at z_mean-Δz)
# are at different altitudes.  Viewed from above it is still a figure-8,
# but in 3D the path is unambiguous — no two branches share the same (x,y,z).
#
# The whole curve is rotated in the horizontal plane by target_heading
# (the orientation angle, randomised at reset).

_N_CURVE_SAMPLES = 400


def _sample_twisted_lemniscate(state: PlaneState3D, params: PlaneParams3D):
    """Return (curve_x, curve_y, curve_z) arrays for the twisted lemniscate."""
    a = state.target_radius
    cx, cy = state.target_x, state.target_y
    z_mean = state.target_altitude
    dz = params.figure8_altitude_amplitude
    orientation = state.target_heading  # repurposed for figure-8

    tau = jnp.linspace(0, 2.0 * jnp.pi, _N_CURVE_SAMPLES, endpoint=False)
    denom = 1.0 + jnp.sin(tau) ** 2
    base_x = a * jnp.cos(tau) / denom
    base_y = a * jnp.sin(tau) * jnp.cos(tau) / denom

    # Rotate by orientation angle
    cos_o = jnp.cos(orientation)
    sin_o = jnp.sin(orientation)
    curve_x = cx + base_x * cos_o - base_y * sin_o
    curve_y = cy + base_x * sin_o + base_y * cos_o
    curve_z = z_mean + dz * jnp.sin(tau)
    return curve_x, curve_y, curve_z


def nearest_point_on_twisted_lemniscate(state: PlaneState3D, params: PlaneParams3D):
    """Find nearest point on the 3D twisted lemniscate.

    Returns (nearest_dx, nearest_dy, nearest_dz, dist, tangent_heading)
    where (dx, dy, dz) is the vector from aircraft to nearest curve point
    and tangent_heading is the heading of the curve tangent at that point
    (flipped if anti-aligned with aircraft velocity for consistent direction).
    """
    curve_x, curve_y, curve_z = _sample_twisted_lemniscate(state, params)

    dx = curve_x - state.x
    dy = curve_y - state.y
    dz = curve_z - state.z
    dists_sq = dx**2 + dy**2 + dz**2
    idx = jnp.argmin(dists_sq)

    nearest_dx = dx[idx]
    nearest_dy = dy[idx]
    nearest_dz = dz[idx]
    dist = jnp.sqrt(dists_sq[idx] + 1e-8)

    # Tangent via central finite differences (wrapping around)
    idx_next = (idx + 1) % _N_CURVE_SAMPLES
    idx_prev = (idx - 1) % _N_CURVE_SAMPLES
    tx = curve_x[idx_next] - curve_x[idx_prev]
    ty = curve_y[idx_next] - curve_y[idx_prev]

    # Flip tangent if anti-aligned with aircraft velocity
    dot = tx * state.x_dot + ty * state.y_dot
    sign = jnp.where(dot >= 0, 1.0, -1.0)
    tangent_heading = jnp.arctan2(sign * ty, sign * tx)

    return nearest_dx, nearest_dy, nearest_dz, dist, tangent_heading


def distance_to_lemniscate(state: PlaneState3D):
    """2D distance from aircraft to the flat lemniscate (used in tests)."""
    a = state.target_radius
    cx, cy = state.target_x, state.target_y
    t = jnp.linspace(-0.99 * jnp.pi / 4, 0.99 * jnp.pi / 4, 200)
    r = a * jnp.sqrt(jnp.maximum(jnp.cos(2 * t), 0.0))
    lx_r = cx + r * jnp.cos(t)
    ly_r = cy + r * jnp.sin(t)
    lx_l = cx - r * jnp.cos(t)
    ly_l = cy - r * jnp.sin(t)
    all_x = jnp.concatenate([lx_r, lx_l])
    all_y = jnp.concatenate([ly_r, ly_l])
    dists = jnp.sqrt((state.x - all_x) ** 2 + (state.y - all_y) ** 2 + 1e-8)
    return jnp.min(dists)


def compute_reward_figure8(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: Gaussian on 3D distance to the twisted lemniscate.

    Pure shape tracking — no moving reference, no shape backstop.  The 3D
    twist makes crossovers unambiguous (different altitudes), so the reward
    has a single global optimum: fly along the curve.
    """
    done_alt = terminal_penalty(state, params, xp)
    _, _, _, dist, _ = nearest_point_on_twisted_lemniscate(state, params)
    sigma = state.target_radius * 0.1
    track_r = xp.exp(-0.5 * (dist / sigma) ** 2)
    return xp.where(done_alt, -1.0 * params.max_steps_in_episode, track_r)


# ─── Observation helpers ────────────────────────────────


def get_obs_heading(state: PlaneState3D, xp=jnp):
    """Observation for heading task (15 values)."""
    return xp.stack(
        [
            state.x_dot,
            state.y_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.phi,
            state.phi_dot,
            state.gamma,
            state.psi,
            state.target_altitude,
            state.target_heading,
            state.power,
            state.stick,
            state.aileron,
        ]
    )


def get_obs_circle(state: PlaneState3D, xp=jnp):
    """
    Observation for circle task (17 values).
    Includes relative position to circle center and target radius.
    """
    return xp.stack(
        [
            state.x_dot,
            state.y_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.phi,
            state.phi_dot,
            state.gamma,
            state.psi,
            state.target_altitude,
            state.x - state.target_x,  # relative x to center
            state.y - state.target_y,  # relative y to center
            state.target_radius,
            state.power,
            state.stick,
            state.aileron,
        ]
    )


def get_obs_figure8(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """
    Observation for figure-8 task (19 values).

    Provides the vector from aircraft to the nearest point on the 3D
    twisted lemniscate (nearest_dx, nearest_dy, nearest_dz) plus the
    tangent heading at that point.

    Layout:
      [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
       gamma, psi, target_altitude, target_radius,
       nearest_dx, nearest_dy, nearest_dz, tangent_heading,
       power, stick, aileron]
    """
    ndx, ndy, ndz, _, tang_hdg = nearest_point_on_twisted_lemniscate(state, params)
    return xp.stack(
        [
            state.x_dot,
            state.y_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.phi,
            state.phi_dot,
            state.gamma,
            state.psi,
            state.target_altitude,
            state.target_radius,
            ndx,
            ndy,
            ndz,
            tang_hdg,
            state.power,
            state.stick,
            state.aileron,
        ]
    )


# ─── Shared state transition ────────────────────────────


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state_3d(
    power_requested: float,
    stick_requested: float,
    aileron_requested: float,
    state: PlaneState3D,
    params: PlaneParams3D,
    integration_method: str = "rk4_1",
):
    """Compute next state using the 3D dynamics model."""
    dt = params.delta_t
    power = compute_next_power(power_requested, state.power, dt)
    stick = compute_next_stick(stick_requested, state.stick, dt)
    aileron = compute_next_aileron(aileron_requested, state.aileron, dt)

    thrust = compute_thrust_output(
        power=power,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
        rho=state.rho,
        M=state.M,
    )

    positions = jnp.array([state.x, state.y, state.z, state.theta, state.phi])
    velocities = jnp.array(
        [
            state.x_dot,
            state.y_dot,
            state.z_dot,
            state.theta_dot,
            state.phi_dot,
        ]
    )

    _compute_acceleration = partial(
        compute_acceleration_3d,
        action=(thrust, stick, aileron),
        params=params,
        clip=True,
        min_clip_boundaries=(-100, -100, -100, -1.5, -1.5),
        max_clip_boundaries=(100, 100, 100, 1.5, 1.5),
    )

    (x_dot, y_dot, z_dot, theta_dot, phi_dot), (x, y, z, theta, phi), metrics = (
        integrate_dynamics(
            velocities=velocities,
            positions=positions,
            delta_t=dt,
            compute_acceleration=_compute_acceleration,
            method=integration_method,
        )
    )

    alpha, gamma = compute_alpha_3d(theta, x_dot, y_dot, z_dot)
    psi = compute_psi(x_dot, y_dot)
    m = params.initial_mass + state.fuel

    new_state = PlaneState3D(
        x=x,
        x_dot=x_dot,
        y=y,
        y_dot=y_dot,
        z=z,
        z_dot=z_dot,
        theta=theta,
        theta_dot=theta_dot,
        phi=phi,
        phi_dot=phi_dot,
        psi=psi,
        alpha=alpha,
        gamma=gamma,
        m=m,
        power=power,
        stick=stick,
        aileron=aileron,
        fuel=state.fuel,
        time=state.time + 1,
        target_altitude=state.target_altitude,
        target_heading=state.target_heading,
        target_x=state.target_x,
        target_y=state.target_y,
        target_radius=state.target_radius,
    )
    return new_state, metrics
