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

from target_gym import reward as R
from target_gym.base import NO_RESTART, EnvParams, EnvState
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
from target_gym.utils import log_scaled_reward


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

    #: Steps of downtime left after a trip (``base.failure_kernel``); 0 when healthy.
    downtime: int = 0


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

    # span^2 / S for the A320. PlaneParams3D does not inherit from PlaneParams,
    # so the shared aerodynamics needs its own copy; the value matches.
    aspect_ratio: float = 9.48

    # Control-surface effectiveness (the flap parameter tau). A surface
    # deflected by delta does not change the section's incidence by delta: only
    # the hinged part of the chord turns, so the effective change is tau*delta.
    # Thin-aerofoil theory gives tau ~ 0.4 for a control surface occupying a
    # quarter of the chord, which is the usual aileron proportion.
    #
    # Without it the model applied the *wing's* lift-curve slope directly to the
    # deflection, implying a section lift change of 2.20 at full aileron --
    # larger than the whole wing's CL_max of 1.5, and giving a roll rate of
    # 84 deg/s against an A320's 25-30.
    aileron_effectiveness: float = 0.4

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

    # Reward precision floors -- the finest error each sensor can resolve.
    # Below these the reward saturates, because "improving" further is noise.
    precision_floor: float = 1.0  # m, barometric altimeter resolution
    heading_precision_floor: float = 0.0087  # rad (~0.5 deg), AHRS/compass
    position_precision_floor: float = 3.0  # m, civil GPS horizontal accuracy

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # One quadratic term per tracked output, summed, each normalised by a
    # documented minimum: the test configurations fly with zero turbulence,
    # so the achievable hold error is ~0 on every task (the shipped MPC holds
    # altitude to 0.1 m and the heading to 1e-3 rad, `scripts/measure_hold.py`)
    # and the sensor resolutions above are the scales. Altitude has a +-30 m
    # tolerance (a vertical-separation margin, provisional); heading and path
    # distance have none. No running cost on these tasks. Leaving the
    # envelope costs, per step, twice the altitude envelope's cost.
    reward_version: int = 2
    e_floor_altitude: float = 1.0  # m
    e_tol_altitude: float = 30.0  # m, provisional
    e_floor_heading: float = 0.0087  # rad
    e_floor_path: float = 3.0  # m
    tracking_exponent: float = 2.0
    failure_cost: float = 3.0e8
    #: Steps the plant is down after a trip before it restarts (a crash: no restart).
    restart_steps: int = NO_RESTART
    #: Tracking cost per step at the floor, in the reward's units; the NEA floor.
    #: One: the altitude term is zero inside its tolerance, so at the floor
    #: only the heading or path term costs 1 (the figure-8 has the path term
    #: alone).
    rho_floor_tracking: float = 1.0
    rho_floor: float = 1.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True
    # Figure-8: half-amplitude of the altitude twist (meters).  The curve
    # altitude is z_mean ± this value, so the two crossover passes differ
    # by 2× this.  200 m ≈ 660 ft — gentle enough for an A320 but enough
    # to require coordinated altitude+heading control.
    figure8_altitude_amplitude: float = 200.0
    # Random orientation of the lemniscate (radians).  ±15° by default.
    figure8_angle_range: Tuple[float, float] = (-0.26, 0.26)
    initial_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    #: How far from the commanded altitude the episode may start. The initial
    #: altitude used to be drawn independently of the target over the same
    #: 5 000 m band, so the median start was 1 590 m off its assigned level and
    #: the episode opened with several minutes of climb before the pattern
    #: could be flown at all. An aircraft handed a heading, a circle or a hold
    #: is at or near the level it was given. ``initial_altitude_range`` still
    #: bounds the result.
    initial_altitude_offset_range: Tuple[float, float] = (-600.0, 600.0)
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
    return log_scaled_reward(
        xp.abs(state.target_altitude - state.z),
        params.precision_floor,
        params.max_alt - params.min_alt,
        xp,
    )


def heading_reward(state, params, xp=jnp):
    """Heading tracking component, log-scaled over the full +/-pi range."""
    return log_scaled_reward(
        xp.abs(wrap_angle(state.psi - state.target_heading)),
        params.heading_precision_floor,
        jnp.pi,
        xp,
    )


def path_reward(dist, state, params, xp=jnp):
    """Path-proximity component, normalised by the radius.

    Using ``target_radius`` as the envelope keeps the reward independent of the
    size of the commanded path, and the aircraft starts *on* the path in both
    path-following tasks, so the ``dist >= radius`` floor is only reached by a
    controller that has already lost the shape entirely.
    """
    return log_scaled_reward(
        dist, params.position_precision_floor, state.target_radius, xp
    )


# ─── Heading task reward ────────────────────────────────


def compute_reward_heading_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: multiplicative altitude * heading, both log-scaled in the error.

    Mirrors the Plane (2D) altitude reward and extends it with a heading
    factor, so the task is "Plane + heading" rather than an additive
    simplification.  Multiplicative composition requires *both* objectives to
    be met.  Each factor is log-scaled (see ``log_scaled_reward``) so the
    reward keeps discriminating all the way down to sensor resolution instead
    of saturating a few hundred metres from the target.  Crash penalty matches
    Plane 2D.
    """
    reward = altitude_reward(state, params, xp) * heading_reward(state, params, xp)
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    return reward


# ─── Circle task reward ─────────────────────────────────


def compute_reward_heading(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_heading_v1(state, params, xp),
        R.total(compute_reward_terms_heading(state, params, xp), xp),
        xp,
    )


def distance_to_circle(state: PlaneState3D):
    """Signed distance from aircraft to the target circle (positive = outside)."""
    dx = state.x - state.target_x
    dy = state.y - state.target_y
    dist_to_center = jnp.sqrt(dx**2 + dy**2)
    return dist_to_center - state.target_radius


def distance_to_racetrack(state: PlaneState3D):
    """Distance from the aircraft to a racetrack holding pattern.

    A holding pattern is two straight legs joined by two 180 degree turns --
    what "holding" actually means in aviation, and the shape this library is
    named for. It is the union of two half-circles of radius ``target_radius``
    centred at the ends of a straight segment of length ``2 * target_radius *
    _RACETRACK_LEG``, oriented by ``target_heading``.

    Unlike the figure-8's, this distance is computed in closed form rather than
    searched over samples, so it has no resolution floor of its own -- check 11
    of the model review checklist exists because the figure-8's argmin metric
    quantised its own reward.
    """
    r = state.target_radius
    half_leg = r * _RACETRACK_LEG

    # Into pattern frame: origin at the centre, x along the straight legs.
    c, s_ = jnp.cos(state.target_heading), jnp.sin(state.target_heading)
    dx = state.x - state.target_x
    dy = state.y - state.target_y
    u = c * dx + s_ * dy
    v = -s_ * dx + c * dy

    # Straight legs where |u| <= half_leg, the turn circles beyond.
    on_leg = jnp.abs(jnp.abs(v) - r)
    cap_u = jnp.abs(u) - half_leg
    on_cap = jnp.abs(jnp.sqrt(cap_u**2 + v**2) - r)
    return jnp.where(jnp.abs(u) <= half_leg, on_leg, on_cap)


def racetrack_guidance(state: PlaneState3D):
    """Signed cross-track error and tangent heading for the holding pattern.

    Closed form, unlike the figure-8's, whose nearest point is an argmin over
    400 samples and therefore quantised -- check 11 of the model review
    checklist exists because that quantisation was larger than the expert's own
    tracking error. Here both come out of the geometry exactly.

    Returns ``(cross_track, tangent_heading, curvature)``. ``cross_track`` is
    positive outside the pattern, so a controller steers to drive it to zero.
    ``curvature`` is signed, zero on the straight legs and 1/r through the
    turns: without it a controller can only chase the tangent, and chasing a
    tangent lags a curved path by construction -- which is exactly how the
    circle expert used to fail, and why it now carries a coordinated-turn
    feedforward.
    """
    r = state.target_radius
    half_leg = r * _RACETRACK_LEG
    c, s_ = jnp.cos(state.target_heading), jnp.sin(state.target_heading)
    dx = state.x - state.target_x
    dy = state.y - state.target_y
    u = c * dx + s_ * dy
    v = -s_ * dx + c * dy

    on_leg = jnp.abs(u) <= half_leg
    side = jnp.sign(v) + (v == 0.0)

    # Nearest point on the pattern, and the tangent there. On a leg the pattern
    # runs parallel to +u on the far side and -u on the near one, so a circuit
    # goes up one and back down the other; on a cap it is a circle about the leg
    # end, turning the same way at both ends -- that is what makes it a
    # racetrack rather than a figure-8.
    leg_n = jnp.stack([u, side * r])
    leg_tan = state.target_heading + jnp.where(side > 0, 0.0, jnp.pi)

    cu = u - jnp.sign(u) * half_leg
    rho = jnp.sqrt(cu**2 + v**2) + 1e-6
    cap_n = jnp.stack([jnp.sign(u) * half_leg + r * cu / rho, r * v / rho])
    cap_tan = state.target_heading + jnp.arctan2(v, cu) - jnp.sign(u) * jnp.pi / 2.0

    nearest = jnp.where(on_leg, leg_n, cap_n)
    tangent = jnp.where(on_leg, leg_tan, cap_tan)

    # Signed in the tangent's own frame, so the sign means the same thing
    # everywhere: positive when the path lies to the aircraft's right. Taking it
    # from the geometry directly -- ``side * (|v| - r)`` -- reverses meaning
    # between the two legs, and a controller using it steers away from the path
    # on the return leg.
    local = state.target_heading + tangent * 0.0  # keep shapes aligned
    del local
    offset = jnp.stack([u, v]) - nearest
    tangent_local = tangent - state.target_heading
    normal = jnp.stack([-jnp.sin(tangent_local), jnp.cos(tangent_local)])
    cross = jnp.dot(offset, normal)

    # Straight legs have no curvature. Both caps turn the same way, so the sign
    # is constant rather than a function of which end the aircraft is at.
    curvature = jnp.where(on_leg, 0.0, -1.0 / r)
    return cross, jnp.arctan2(jnp.sin(tangent), jnp.cos(tangent)), curvature


def compute_reward_racetrack_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Altitude tracking * proximity to the holding pattern, both log-scaled."""
    alt_r = altitude_reward(state, params, xp)
    d = xp.abs(distance_to_racetrack(state))
    track_r = path_reward(d, state, params, xp)
    return alt_r * track_r


def compute_reward_racetrack(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_racetrack_v1(state, params, xp),
        R.total(compute_reward_terms_racetrack(state, params, xp), xp),
        xp,
    )


def compute_reward_circle_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: altitude tracking * proximity to the circle path, both log-scaled."""
    alt_r = altitude_reward(state, params, xp)
    d = xp.abs(distance_to_circle(state))
    circle_r = path_reward(d, state, params, xp)
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    return alt_r * circle_r


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


def compute_reward_circle(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_circle_v1(state, params, xp),
        R.total(compute_reward_terms_circle(state, params, xp), xp),
        xp,
    )


_N_CURVE_SAMPLES = 400

# Straight-leg half-length of the holding pattern, in turn radii. A standard
# civil hold is a one-minute leg, which at these speeds and bank limits is
# close to two turn radii each side.
_RACETRACK_LEG = 2.0


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

    # Sub-sample refinement.  The coarse argmin is quantised by the sample
    # spacing (~100 m for a 44 km lemniscate at 400 samples), so an aircraft
    # flying *exactly* along the commanded curve is reported up to half a
    # spacing off it -- an order of magnitude more than the tracking error the
    # reward is meant to resolve, which floors how precisely any controller can
    # be scored.  Projecting onto the two adjacent chords instead leaves only
    # the curve's sagitta over one segment, which is sub-metre here.
    pts = jnp.stack([curve_x, curve_y, curve_z], axis=-1)
    pos = jnp.array([state.x, state.y, state.z])
    anchor = pts[idx]

    def _project_onto_chord(other):
        chord = other - anchor
        t = jnp.clip(
            jnp.dot(pos - anchor, chord) / (jnp.dot(chord, chord) + 1e-12), 0.0, 1.0
        )
        return anchor + t * chord

    idx_fwd = (idx + 1) % _N_CURVE_SAMPLES
    idx_bwd = (idx - 1) % _N_CURVE_SAMPLES
    candidates = jnp.stack(
        [_project_onto_chord(pts[idx_fwd]), _project_onto_chord(pts[idx_bwd])]
    )
    delta = candidates[jnp.argmin(jnp.sum((candidates - pos) ** 2, axis=-1))] - pos

    nearest_dx, nearest_dy, nearest_dz = delta[0], delta[1], delta[2]
    dist = jnp.sqrt(jnp.sum(delta**2) + 1e-8)

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


def compute_reward_figure8_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: log-scaled 3D distance to the twisted lemniscate.

    Pure shape tracking — no moving reference, no shape backstop.  The 3D
    twist makes crossovers unambiguous (different altitudes), so the reward
    has a single global optimum: fly along the curve.
    """
    _, _, _, dist, _ = nearest_point_on_twisted_lemniscate(state, params)
    track_r = path_reward(dist, state, params, xp)
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    return track_r


# ─── Observation helpers ────────────────────────────────


def compute_reward_figure8(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_figure8_v1(state, params, xp),
        R.total(compute_reward_terms_figure8(state, params, xp), xp),
        xp,
    )


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


def get_obs_racetrack(state: PlaneState3D, xp=jnp):
    """Observation for the holding pattern (21 values).

    The circle's observation plus ``target_heading``, which orients the pattern
    and without which the straight legs are unobservable -- the aircraft would
    have to infer which way the hold lies from its own history.
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
            state.x - state.target_x,
            state.y - state.target_y,
            state.target_radius,
            state.target_heading,
            state.power,
            state.stick,
            state.aileron,
            # The guidance the pattern's geometry already determines, in closed
            # form: signed cross-track error and the tangent heading at the
            # nearest point. Provided rather than left to the controller because
            # it is exact here -- the figure-8 has to search for its nearest
            # point and pays a quantisation floor for it (check 11) -- and
            # because a policy that had to rediscover the pattern's shape from
            # position alone would be solving a different problem.
            *racetrack_guidance(state),
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
    integration_method: str = "rk4_2",
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
    # All-up mass for this step: fuel is a component of ``initial_mass``, so
    # burning it subtracts. Threaded through ``eff_params`` because the
    # dynamics read ``params.initial_mass``.
    mass_now = params.initial_mass - (params.initial_fuel_quantity - state.fuel)
    eff_params = params.replace(
        wind_x=total_wind_x,
        wind_y=total_wind_y,
        wind_z=total_wind_z,
        initial_mass=mass_now,
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
    # Fuel burned this step, charged against the thrust actually produced.
    burn = params.specific_fuel_consumption * 1e-3 * thrust * dt
    fuel = jnp.clip(state.fuel - burn, 0.0, params.initial_fuel_quantity)
    m = params.initial_mass - (params.initial_fuel_quantity - fuel)

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
        fuel=fuel,
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


def altitude_cost(state, params, xp=jnp):
    return R.tracking_cost(
        state.target_altitude - state.z,
        params.e_floor_altitude,
        params.e_tol_altitude,
        params.tracking_exponent,
        xp,
    )


def _downtime(terms, state, params, xp=jnp):
    terminated, _ = check_is_terminal_3d(state, params, xp)
    return R.with_downtime(
        terms, R.is_down(terminated, state, xp), params.failure_cost, xp
    )


def compute_reward_terms_heading(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    heading = R.tracking_cost(
        wrap_angle(state.psi - state.target_heading),
        params.e_floor_heading,
        0.0,
        params.tracking_exponent,
        xp,
    )
    return _downtime(
        {"tracking": altitude_cost(state, params, xp) + heading}, state, params, xp
    )


def compute_reward_terms_racetrack(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    path = R.tracking_cost(
        distance_to_racetrack(state),
        params.e_floor_path,
        0.0,
        params.tracking_exponent,
        xp,
    )
    return _downtime(
        {"tracking": altitude_cost(state, params, xp) + path}, state, params, xp
    )


def compute_reward_terms_circle(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    path = R.tracking_cost(
        distance_to_circle(state),
        params.e_floor_path,
        0.0,
        params.tracking_exponent,
        xp,
    )
    return _downtime(
        {"tracking": altitude_cost(state, params, xp) + path}, state, params, xp
    )


def compute_reward_terms_figure8(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    _, _, _, dist, _ = nearest_point_on_twisted_lemniscate(state, params)
    path = R.tracking_cost(dist, params.e_floor_path, 0.0, params.tracking_exponent, xp)
    return _downtime({"tracking": path}, state, params, xp)
