from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import (
    integrate_dynamics,
)
from target_gym.plane.dynamics import (
    advance_gust,
    compute_acceleration,
    compute_air_density_from_altitude,
    compute_alpha,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_next_power,
    compute_next_stick,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    compute_velocity_from_horizontal_and_vertical_speed,
    step_key,
    total_wind_2d,
)
from target_gym.utils import log_scaled_reward

DEBUG = False


@struct.dataclass
class EnvMetrics:
    drag: float
    lift: float
    S_x: float
    S_z: float
    C_x: float
    C_z: float
    F_x: float
    F_z: float


@struct.dataclass
class PlaneState(EnvState):
    x: float
    x_dot: float
    z: float
    z_dot: float
    theta: float
    theta_dot: float
    alpha: float
    gamma: float
    m: float
    power: float
    stick: float
    fuel: float
    target_altitude: float
    # Ornstein-Uhlenbeck turbulence gust (m/s), mean-reverting to 0.  Total wind
    # acting on the aircraft is params.wind_* + gust_*.  Default 0 => no gust.
    gust_x: float = 0.0
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
            compute_velocity_from_horizontal_and_vertical_speed(self.x_dot, self.z_dot),
            self.speed_of_sound,
        )


@struct.dataclass
class PlaneParams(EnvParams):
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

    # Finite-wing lift-curve slope, derived (PHYSICS.md 4):
    #   a = a0 / (1 + a0/(pi*AR*e))  with a0 = 2*pi/rad, AR = 34.1^2/122.6 = 9.48,
    #   e = 0.85  ->  5.034 /rad = 0.0879 /deg.
    # Was 0.04 /deg (-54%), which both inflated cruise trim AoA to 4.06 deg
    # (A320 flies ~2 deg) and disagreed with `k`, which correctly encodes the
    # same AR. With the derived slope, cl0 + cl_alpha*aoa_stall = 1.52 ~ CL_max,
    # so cl0/aoa_stall/CL_max/cl_alpha become mutually consistent.
    cl_alpha: float = 0.08786  # per deg
    cl0: float = 0.2  # zero-lift AoA
    cd0: float = 0.02  # zero-lift drag
    k: float = 0.045  # induced drag factor
    aoa_stall: float = 15.0  # deg
    CL_max: float = 1.5
    # Negative-stall limit. A cambered wing stalls asymmetrically: the negative
    # branch bottoms out near -10 deg AoA. Without this the corrected (steeper)
    # lift slope lets CL run away negative -- it was previously masked by the
    # too-shallow slope, not actually bounded.
    CL_min: float = -1.0
    # Degrees beyond `aoa_stall` at which the separation sigmoid is centred, so
    # that lift *peaks at* aoa_stall rather than being halved there.
    aoa_stall_width: float = 3.0
    # Wing aspect ratio, span^2 / S = 34.1^2 / 122.6 for the A320. Sourced
    # geometry (PHYSICS.md section 3), and it sets how much force a *finite*
    # wing develops once separated: a 2D flat plate reaches a normal-force
    # coefficient of 2, a finite one less, because the flow relieves around the
    # tips.
    aspect_ratio: float = 9.48
    # Drag-divergence Mach number. NOTE: this field was previously declared
    # twice (0.78 above the aero block, 0.80 here). Python keeps the last
    # definition, so 0.80 was always the effective value and the 0.78 was
    # dead -- it is the A320 *cruise* Mach, not the drag-divergence Mach,
    # which is where the confusion came from. Kept at 0.80 so behaviour is
    # unchanged; see PHYSICS.md for sourcing.
    M_crit: float = 0.80
    # Shock stall above M_crit (PHYSICS.md D3): the attainable lift falls
    # once shocks form, rather than rising with the Prandtl-Glauert factor.
    # k = 4.0 puts CL_max back to its low-speed value by M 0.9 and at 40 %
    # of it by M 0.95, which is the right order for lift divergence.
    k_shock_stall: float = 4.0
    shock_stall_floor: float = 0.25

    # Actuator lags, as first-order rates per second. TUNED -- not sourced.
    # The engine is deliberately far slower than the control surface: an
    # airliner spools up over seconds, the stick responds immediately, and it
    # is that separation the altitude controllers have to work around.
    # The altitude error below which "more precise" stops being meaningful, in
    # metres. It is a *resolution*, not a tolerance: the reward keeps paying for
    # every halving of the error down to this point, and only flattens beneath
    # it. 1 m is the order of a barometric altimeter's resolution, so tracking
    # tighter than this would be rewarding the controller for chasing noise.
    #
    # This is the parameter that decides how precise the benchmark asks a policy
    # to be, so it is deliberately a physical limit rather than a comfort band:
    # a band rewards reaching a tolerance and then stops caring, which cannot
    # show whether a learned policy holds altitude better than a PID.
    precision_floor: float = 1.0
    power_response_rate: float = 0.05
    stick_response_rate: float = 0.9
    k_drag: float = 5.0

    I: float = 9_000_000
    moment_arm_stabilizer: float = 15.0
    moment_arm_wings: float = 1.5
    stabilizer_surface: float = 27
    elevator_surface: float = 10

    max_steps_in_episode: int = 10_000
    min_alt: float = 0.0
    max_alt: float = 40_000.0 / 3.281
    # Look-ahead wind: apply the gust ALREADY stored in the state this step, so
    # the sensor that reads it reveals the disturbance *before* it acts. That
    # makes it genuine feedforward an agent can pre-cancel -- and a disturbance a
    # PID structurally cannot, which is the kind of gap this benchmark exists to
    # measure. With False (the default) the gust is advanced then applied, so an
    # ``observe_wind`` sensor only ever reports what has already hit the
    # aircraft, whose effect is in the velocities anyway.
    wind_lookahead: bool = False
    target_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    initial_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    initial_z_dot: float = 0.0
    initial_x_dot: float = 200.0
    initial_theta_dot: float = 0.0
    initial_theta: float = 0.0
    initial_power: float = 1.0
    initial_stick: float = 0.0

    # Steady mean wind (world frame, m/s).  Aerodynamics use the air-relative
    # velocity V_ground - (wind + gust).
    wind_x: float = 0.0
    wind_z: float = 0.0
    # Ornstein-Uhlenbeck turbulence on top of the mean wind: sigma is the gust
    # std (m/s), theta the mean-reversion rate (1/s; correlation time ~ 1/theta).
    # sigma = 0 (default) => no turbulence, wind is exactly the steady mean.
    turbulence_sigma: float = 0.0
    turbulence_theta: float = 0.2
    # Impulse gust mode. impulse_prob = 0 (default) => the OU turbulence above. When
    # impulse_prob > 0 the gust is instead a rare, memoryless "kick": each step a kick
    # arrives with per-step probability impulse_prob and jumps the gust by ~sigma*N,
    # then it decays at rate theta between kicks. Arrivals are white (no autocorrelation),
    # but the kick has temporal extent, so a lookahead sensor can anticipate it while
    # memory can only track its decay -- the impulsive-disturbance analog of a robot shove.
    impulse_prob: float = 0.0
    # Linear wind shear: the horizontal wind gains ``wind_shear_x`` m/s per metre
    # of altitude above ``shear_ref_alt`` (0 => no shear).  Makes the wind
    # altitude-dependent, so climbing/descending is itself a disturbance.
    wind_shear_x: float = 0.0
    shear_ref_alt: float = 0.0

    delta_t: float = 1.0


def check_mass_does_not_increase(old_mass, new_mass, xp=jnp):
    """Check that mass does not increase. Safe for JIT if wrapped in callback."""
    if jax is not None and xp is jnp:
        jax.debug.callback(
            lambda o, n: None if o >= n else AssertionError("Mass increased"),
            old_mass,
            new_mass,
        )
    else:
        assert old_mass >= new_mass


def check_is_terminal(state: PlaneState, params: PlaneParams, xp=jnp):
    """Return True if the episode should terminate."""
    terminated = xp.logical_or(state.z <= params.min_alt, state.z >= params.max_alt)
    truncated = state.time >= params.max_steps_in_episode

    # done = xp.logical_or(done_alt, done_steps)
    return terminated, truncated


def check_no_nan(x, id=None):
    """Assert that no NaNs are present in arrays, scalars, or PlaneState."""
    if isinstance(x, PlaneState):
        # Iterate over fields of the dataclass
        for name, value in x.__dict__.items():
            try:
                check_no_nan(value, id=f"{id}.{name}" if id else name)
            except AssertionError as e:
                raise AssertionError(str(e)) from None
    else:
        if jnp.isnan(x).any():
            raise AssertionError(f"NaN detected in {id}: {x}")


def compute_reward(state: PlaneState, params: PlaneParams, xp=jnp):
    """Return reward for a given state. Safe for JIT."""
    xp = jnp
    done_alt = xp.logical_or(state.z <= params.min_alt, state.z >= params.max_alt)
    # Log-scaled tracking: every halving of the error is worth the same, so
    # holding 1 m is rewarded over 2 m exactly as much as 100 m is over 200 m.
    # The altitude envelope only normalises the result into [0, 1]; unlike the
    # old reward it does not set the sensitivity, because the shape is
    # logarithmic and therefore scale-free.
    tracking = log_scaled_reward(
        xp.abs(state.target_altitude - state.z),
        params.precision_floor,
        params.max_alt - params.min_alt,
        xp,
    )
    return xp.where(done_alt, -1.0 * params.max_steps_in_episode, tracking)


def get_obs(state: PlaneState, xp=jnp):
    """Applies observation function to state."""
    return xp.stack(
        [
            state.x_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.gamma,
            state.target_altitude,
            state.power,
            state.stick,
        ]
    )


@partial(jax.jit, static_argnames=["min", "max"])
def clip_acceleration(a: jnp.ndarray, min: tuple, max: tuple):
    return jnp.clip(a, min=jnp.array(min), max=jnp.array(max))


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    power_requested: float,
    stick_requested: float,
    state: PlaneState,
    params: PlaneParams,
    integration_method: str = "rk4_1",
    key=None,
):
    """Compute next state and metrics using multiple sub-steps with jax.lax.scan.

    Wind is a physics-engine property: the total wind acting on the aircraft is
    the steady ``params.wind_*`` plus an Ornstein-Uhlenbeck turbulence gust
    (advanced with ``key`` when ``params.turbulence_sigma > 0``), and the
    aerodynamics/engine-Mach use the air-relative velocity while
    position/observations stay in the ground frame.  With ``wind = 0`` and
    ``turbulence_sigma = 0`` the behaviour is unchanged; ``key=None`` freezes the
    gust (so callers that don't model turbulence keep a constant wind).
    """
    dt = params.delta_t
    power = compute_next_power(
        power_requested, state.power, dt, params.power_response_rate
    )
    stick = compute_next_stick(
        stick_requested, state.stick, dt, params.stick_response_rate
    )

    # Total wind = steady mean + altitude shear + OU turbulence gust. `gust` is the
    # ADVANCED gust (stored for next step). With wind_lookahead we APPLY the gust
    # already in the state (which the sensor revealed before this action -> the
    # agent can pre-cancel it); otherwise we apply the advanced gust (post-hoc).
    gust = advance_gust(
        jnp.array([state.gust_x, state.gust_z]),
        params.turbulence_theta,
        params.turbulence_sigma,
        dt,
        step_key(key, state.time),
        params.impulse_prob,
    )
    # With look-ahead the gust already in the state is the one that acts now;
    # the freshly advanced one is stored for the next step.
    applied = jnp.where(
        params.wind_lookahead, jnp.array([state.gust_x, state.gust_z]), gust
    )
    total_wind_x, total_wind_z = total_wind_2d(state.z, applied[0], applied[1], params)
    # All-up mass for this step. ``initial_fuel_quantity`` is a component of
    # ``initial_mass``, not an addition, so burning fuel subtracts from it.
    mass_now = params.initial_mass - (params.initial_fuel_quantity - state.fuel)
    eff_params = params.replace(
        wind_x=total_wind_x, wind_z=total_wind_z, initial_mass=mass_now
    )

    # Engine ram/Mach effects depend on airspeed, not ground speed.
    air_speed = compute_velocity_from_horizontal_and_vertical_speed(
        state.x_dot - total_wind_x, state.z_dot - total_wind_z
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
    positions = jnp.array([state.x, state.z, state.theta])
    velocities = jnp.array([state.x_dot, state.z_dot, state.theta_dot])
    _compute_acceleration = partial(
        compute_acceleration,
        action=(thrust, stick),
        params=eff_params,
        clip=True,
        min_clip_boundaries=(-100, -100, -1.5),
        max_clip_boundaries=(100, 100, 1.5),
    )

    (x_dot, z_dot, theta_dot), (x, z, theta), metrics = integrate_dynamics(
        velocities=velocities,
        positions=positions,
        delta_t=dt,
        compute_acceleration=_compute_acceleration,
        method=integration_method,
    )

    alpha, gamma = compute_alpha(theta, x_dot, z_dot)
    # Fuel burned over the step. ``specific_fuel_consumption`` is thrust
    # specific: kg per kilonewton-second, so 17.5/1000 is 1.75e-5 kg/(N s),
    # the right order for a modern high-bypass turbofan. Burn is charged
    # against the thrust actually produced, which already carries the altitude
    # lapse and the Mach terms.
    burn = params.specific_fuel_consumption * 1e-3 * thrust * dt
    fuel = jnp.clip(state.fuel - burn, 0.0, params.initial_fuel_quantity)
    # Below the tanks running dry the engines cannot produce thrust; the mass
    # simply stops falling, and the aircraft is a glider from there.
    m = params.initial_mass - (params.initial_fuel_quantity - fuel)

    new_state = PlaneState(
        x=x,
        x_dot=x_dot,
        z=z,
        z_dot=z_dot,
        theta=theta,
        theta_dot=theta_dot,
        alpha=alpha,
        gamma=gamma,
        m=m,
        power=power,
        stick=stick,
        fuel=fuel,
        time=state.time + 1,
        target_altitude=state.target_altitude,
        gust_x=gust[0],
        gust_z=gust[1],
    )
    return new_state, metrics
