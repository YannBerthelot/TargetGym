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
    advance_gust,
    compute_air_density_from_altitude,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_next_power,
    compute_next_stick,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    step_key,
    total_wind_3d,
)
from target_gym.plane3d.dynamics import (
    compute_acceleration_3d,
    compute_alpha_3d,
    compute_next_aileron,
    compute_psi,
    compute_velocity_3d,
)


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
    # Ornstein-Uhlenbeck turbulence gust (m/s); total wind = params.wind_* + gust.
    gust_x: float = 0.0
    gust_y: float = 0.0
    gust_z: float = 0.0

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

    # Actuator lags, as first-order rates per second. TUNED -- not sourced.
    # PlaneParams3D does not inherit from PlaneParams, so these are declared
    # here as well; the values match the 2D aircraft's.
    power_response_rate: float = 0.05
    stick_response_rate: float = 0.9
    aileron_response_rate: float = 0.9

    # Aero coefficients (shared with 2D)
    # Finite-wing lift-curve slope, derived (PHYSICS.md 4):
    #   a = a0 / (1 + a0/(pi*AR*e))  with a0 = 2*pi/rad, AR = 34.1^2/122.6 = 9.48,
    #   e = 0.85  ->  5.034 /rad = 0.0879 /deg.
    # Was 0.04 /deg (-54%), which both inflated cruise trim AoA to 4.06 deg
    # (A320 flies ~2 deg) and disagreed with `k`, which correctly encodes the
    # same AR. With the derived slope, cl0 + cl_alpha*aoa_stall = 1.52 ~ CL_max,
    # so cl0/aoa_stall/CL_max/cl_alpha become mutually consistent.
    cl_alpha: float = 0.08786  # per deg
    cl0: float = 0.2
    cd0: float = 0.02
    k: float = 0.045
    aoa_stall: float = 15.0
    CL_max: float = 1.5
    # Negative-stall limit. A cambered wing stalls asymmetrically: the negative
    # branch bottoms out near -10 deg AoA. Without this the corrected (steeper)
    # lift slope lets CL run away negative -- it was previously masked by the
    # too-shallow slope, not actually bounded.
    CL_min: float = -1.0
    # Degrees beyond `aoa_stall` at which the separation sigmoid is centred, so
    # that lift *peaks at* aoa_stall rather than being halved there.
    aoa_stall_width: float = 3.0
    M_crit: float = 0.80
    # Shock stall above M_crit (PHYSICS.md D3): the attainable lift falls
    # once shocks form, rather than rising with the Prandtl-Glauert factor.
    # k = 4.0 puts CL_max back to its low-speed value by M 0.9 and at 40 %
    # of it by M 0.95, which is the right order for lift divergence.
    k_shock_stall: float = 4.0
    shock_stall_floor: float = 0.25
    k_drag: float = 5.0

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

    # Steady mean wind (world frame, m/s).  Aerodynamics use the air-relative
    # velocity V_ground - (wind + gust); wind is not observed (crab in crosswind).
    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_z: float = 0.0
    # Ornstein-Uhlenbeck turbulence: sigma = gust std (m/s), theta = mean-
    # reversion rate (1/s).  sigma = 0 (default) => steady wind, no turbulence.
    turbulence_sigma: float = 0.0
    turbulence_theta: float = 0.2
    # Linear wind shear: horizontal wind gains ``wind_shear_x``/``wind_shear_y``
    # m/s per metre of altitude above ``shear_ref_alt`` (0 => no shear).
    wind_shear_x: float = 0.0
    wind_shear_y: float = 0.0
    shear_ref_alt: float = 0.0

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
    """Reward: multiplicative altitude * heading, both with ^10 sharp shaping.

    Mirrors Plane (2D) altitude reward and extends it with a heading factor,
    so the task is "Plane + heading" rather than an additive simplification.
    Multiplicative composition requires both objectives to be met; ^10 makes
    the reward numerically discriminative. Crash penalty matches Plane 2D.
    """
    done_alt = terminal_penalty(state, params, xp)
    max_alt_diff = params.max_alt - params.min_alt
    alt_base = xp.clip(
        (max_alt_diff - xp.abs(state.target_altitude - state.z)) / max_alt_diff,
        0.0,
        1.0,
    )
    alt_r = alt_base**10
    heading_diff = xp.abs(wrap_angle(state.psi - state.target_heading))
    heading_base = xp.clip(1.0 - heading_diff / jnp.pi, 0.0, 1.0)
    heading_r = heading_base**10
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
    key=None,
):
    """Compute next state using the 3D dynamics model.

    Wind is a physics-engine property: total wind = steady ``params.wind_x/y/z``
    plus an Ornstein-Uhlenbeck turbulence gust (advanced with ``key`` when
    ``params.turbulence_sigma > 0``).  Aerodynamics/engine-Mach use the
    air-relative velocity while position/observations stay in the ground frame.
    ``wind = 0`` with ``turbulence_sigma = 0`` reproduces the original behaviour;
    ``key=None`` freezes the gust (constant wind).
    """
    dt = params.delta_t
    power = compute_next_power(
        power_requested, state.power, dt, params.power_response_rate
    )
    stick = compute_next_stick(
        stick_requested, state.stick, dt, params.stick_response_rate
    )
    aileron = compute_next_aileron(aileron_requested, state.aileron, dt)

    # Total wind = steady mean + altitude shear + OU turbulence gust.
    gust = advance_gust(
        jnp.array([state.gust_x, state.gust_y, state.gust_z]),
        params.turbulence_theta,
        params.turbulence_sigma,
        dt,
        step_key(key, state.time),
    )
    total_wind_x, total_wind_y, total_wind_z = total_wind_3d(
        state.z, gust[0], gust[1], gust[2], params
    )
    eff_params = params.replace(
        wind_x=total_wind_x, wind_y=total_wind_y, wind_z=total_wind_z
    )

    # Engine ram/Mach effects depend on airspeed, not ground speed.
    air_speed = compute_velocity_3d(
        state.x_dot - total_wind_x,
        state.y_dot - total_wind_y,
        state.z_dot - total_wind_z,
    )
    M_air = compute_Mach_from_velocity_and_speed_of_sound(
        air_speed, state.speed_of_sound
    )
    thrust = compute_thrust_output(
        power=power,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
        rho=state.rho,
        M=M_air,
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
        params=eff_params,
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
    # Total all-up mass. ``initial_fuel_quantity`` is a *component* of
    # ``initial_mass``, not an addition to it, and matches the mass the
    # dynamics actually integrate (see compute_acceleration). Once fuel
    # burn is implemented this becomes initial_mass - burned.
    m = params.initial_mass

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
        gust_x=gust[0],
        gust_y=gust[1],
        gust_z=gust[2],
    )
    return new_state, metrics
