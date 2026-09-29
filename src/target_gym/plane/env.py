from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
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
    # The altitude sampled at reset. ``target_altitude`` is what is commanded
    # *now*, which differs once a moving pattern is selected.
    base_target_altitude: float = 0.0
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

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # Altitude: quadratic outside a +-``e_tol`` tolerance about the commanded
    # altitude (+-30 m, a defensible vertical-separation margin, provisional),
    # normalised by a documented minimum of 1 m (altimeter resolution): the
    # test configurations fly with zero turbulence, so the achievable hold
    # error is ~0 (the shipped PID holds 0.08 m, `scripts/measure_hold.py`).
    # Airspeed: the deviation from ``target_speed`` above the hold-phase
    # deviation of the better shipped controller (``c_hold``, m/s; 6.1 for the
    # altitude hold, 8.6 on the sinusoid, both PID -- the MPC ignores speed
    # and sits 50-65 m/s off) charged at weight 1, a stand-in for fuel. Flying
    # out of the altitude envelope costs, per step, twice the envelope.
    reward_version: int = 2
    # The MPC holds 0.84 m in the test turbulence (1.26 on the sinusoid, 4.55
    # on the ladder), below the 1 m barometric resolution: the instrument sets
    # the scale where the simulator's hold is finer than it.
    e_floor: float = 1.0  # m
    e_tol: float = (
        0.0  # m; a +-30 m band made the hold vacuous (both controllers held 0.1 m)
    )
    tracking_exponent: float = 2.0
    c_hold: float = (
        5.06  # m/s airspeed deviation while holding (PID; 8.15 sine, 5.23 ladder)
    )
    running_weight: float = 1.0
    failure_cost: float = 2.0 * (12192.0 / 1.0) ** 2
    #: Restart time priced into a trip (``reward.trip_cost``): a crash loses the
    #: sortie, 1 h of flight at 1 s steps (provisional).
    restart_steps: int = 3600
    #: Tracking cost per step at the floor, in the reward's units; the NEA floor.
    #: One: altitude at the floor costs 1 and the airspeed term is charged
    #: only above the hold-phase deviation.
    #: The NEA reference: the lowest per-seed hold cost the shipped MPC
    #: demonstrated, in the reward's units (the MPC's 0.84 m hold in 1 m units; per task in the registry).
    rho_floor_tracking: float = (0.84 / 1.0) ** 2
    rho_floor: float = (0.84 / 1.0) ** 2
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False
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
    # Commanded-altitude pattern. 0 = hold one altitude, which is what the
    # shipped baselines are measured against and therefore the default; the rest
    # make the setpoint move, which is a different and harder task -- a P
    # controller cannot hold a ramp with zero error at all, and a sinusoid
    # exposes the closed loop's bandwidth directly, since the amplitude ratio
    # and phase lag against frequency *are* its frequency response.
    #
    #   0  hold      a single altitude, sampled at reset
    #   1  steps     a staircase: discrete changes at fixed intervals
    #   2  ramp      a constant climb or descent rate
    #   3  sinusoid  a continuous oscillation about the sampled altitude
    #   4  chirp     a sinusoid whose frequency rises through the episode
    # Weight on an airspeed target, alongside the altitude one. Zero -- the
    # default -- leaves the task exactly as it was, tracking altitude alone.
    #
    # Above zero it becomes the energy-management problem the aircraft has
    # always physically had and never been scored on. It carries two actuators,
    # thrust and elevator, against one objective, so a controller has a spare
    # degree of freedom and can trade airspeed for altitude freely. Scoring both
    # removes it: thrust is finite, so climbing and accelerating compete, and
    # the exchange between them is precisely the phugoid.
    speed_weight: float = 0.0
    target_speed: float = 230.0  # m/s, the cruise the airspeed channel holds
    speed_precision_floor: float = 0.5  # m/s, air-data resolution
    speed_envelope: float = 60.0  # m/s, the error at which the term reaches zero
    target_pattern: int = 0
    target_amplitude: float = 800.0  # m, half-range of the moving patterns
    target_period: float = 240.0  # s, one cycle of the sinusoid
    target_steps: int = 4  # how many treads in the staircase
    target_chirp_octaves: float = 2.0  # frequency multiple across a chirp
    initial_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    #: How far from the commanded altitude the episode may start. The initial
    #: altitude used to be drawn independently of the target over the same
    #: 5 000 m band, so the expected gap was 1 667 m and could reach 5 000 m --
    #: minutes of open-loop climb at the aircraft's climb rate before any
    #: tracking began, and on a short clip that was the entire episode. Drawing
    #: it around the target instead keeps the sampling varied while making the
    #: episode about holding the altitude, which is what the reward scores.
    #: ``initial_altitude_range`` still bounds the result.
    initial_altitude_offset_range: Tuple[float, float] = (-600.0, 600.0)
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
    # 1.2 m/s per 1 s step is a 2 m/s stationary gust std at theta = 0.2:
    # light-to-moderate turbulence (provisional; see PHYSICS.md).
    turbulence_sigma: float = 1.2
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


def compute_reward_terms(state: PlaneState, params: PlaneParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    p = params
    speed = xp.sqrt(state.x_dot**2 + state.z_dot**2)
    terminated, _ = check_is_terminal(state, p, xp)
    terms = {
        "tracking": R.tracking_cost(
            state.target_altitude - state.z, p.e_floor, p.e_tol, p.tracking_exponent, xp
        ),
        "running": R.running_cost(
            xp.abs(speed - p.target_speed), p.c_hold, p.running_weight, xp
        ),
    }
    return R.with_trip(terms, terminated, R.trip_cost(p), xp)


def compute_reward_v1(state: PlaneState, params: PlaneParams, xp=jnp):
    """Log-scaled altitude tracking, optionally coupled to an airspeed hold.

    One over the commanded altitude when it is held exactly, decaying so that
    every halving of the error is worth the same. When ``speed_weight`` is
    above zero the airspeed term multiplies it, so the two must be satisfied
    together; at ``speed_weight = 0`` the factor is exactly 1.0 and the task is
    altitude alone. Safe for JIT.
    """
    xp = jnp
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
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    #
    # Airspeed, when it is being scored, multiplies rather than adds -- the
    # convention every environment here follows. Holding the speed while
    # abandoning the altitude earns nothing, which is the point: the two are
    # one objective the aircraft must satisfy together, not two it can pick
    # between. At speed_weight = 0 this is exactly 1.0 and the task is
    # unchanged.
    speed = xp.sqrt(state.x_dot**2 + state.z_dot**2)
    speed_term = log_scaled_reward(
        xp.abs(params.target_speed - speed),
        params.speed_precision_floor,
        params.speed_envelope,
        xp,
    )
    return tracking * (1.0 - params.speed_weight * (1.0 - speed_term))


def compute_reward(state: PlaneState, params: PlaneParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


def get_obs(state: PlaneState, params: PlaneParams = None, xp=jnp):
    """Applies observation function to state."""
    if params is None:
        raise ValueError(
            "get_obs needs the episode's params: target_speed is a parameter, "
            "so falling back to PlaneParams() would silently report the class "
            "default instead of what this episode is actually commanding."
        )
    params_target_speed = jnp.asarray(params.target_speed, dtype=jnp.float32)
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
            # Commanded airspeed, appended rather than inserted. Putting it
            # beside target_altitude would have shifted power and stick from
            # indices 7 and 8 to 8 and 9, silently breaking every consumer that
            # reads the observation positionally -- the PIDs do. It is present
            # whether or not it is scored, so the shape does not depend on a
            # reward weight.
            params_target_speed,
        ]
    )


@partial(jax.jit, static_argnames=["min", "max"])
def clip_acceleration(a: jnp.ndarray, min: tuple, max: tuple):
    return jnp.clip(a, min=jnp.array(min), max=jnp.array(max))


#: The levels the ``steps`` pattern walks, as fractions of ``target_amplitude``.
#: Eight of them, so the schedule does not repeat inside an episode, with
#: adjacent changes between 0.2 and 0.8 of the amplitude.
_STEP_LEVELS = jnp.array([0.0, 0.5, 0.1, -0.3, 0.3, -0.5, -0.1, 0.4])


def commanded_altitude(base: float, time, params: PlaneParams, xp=jnp):
    """The altitude being commanded at this step.

    ``base`` is the altitude sampled at reset; every moving pattern is written
    as an excursion about it, so the aircraft always starts being asked for
    something it can reach and the patterns stay inside the altitude envelope.

    Branch-free on purpose: all five are evaluated and one is selected, so the
    function traces to a single ``select`` and the pattern can live in params
    rather than forcing a separate environment class per shape.
    """
    t = xp.asarray(time, dtype=jnp.float32) * params.delta_t
    span = params.target_amplitude
    total = xp.maximum(params.max_steps_in_episode * params.delta_t, 1.0)

    hold = base
    tread = xp.floor(t / (total / params.target_steps))
    # A ladder of levels, not a square wave. Alternating between two altitudes
    # is a wave with a flat top, and at the excursion this used to run it read
    # as an aircraft being thrown between two extremes rather than an aircraft
    # being given new levels to hold. The ladder revisits neither level nor
    # direction on a two-tread cycle, so consecutive changes differ in size and
    # sign the way a real clearance sequence does, and the largest adjacent
    # change is 0.8 of the amplitude rather than 1.2.
    idx = xp.asarray(xp.mod(tread, float(len(_STEP_LEVELS))), dtype=jnp.int32)
    steps = base + span * _STEP_LEVELS[idx]
    ramp = base + span * (2.0 * t / total - 1.0)
    sine = base + span * xp.sin(2.0 * xp.pi * t / params.target_period)
    rate = 1.0 + (params.target_chirp_octaves - 1.0) * (t / total)
    chirp = base + span * xp.sin(2.0 * xp.pi * t * rate / params.target_period)

    options = xp.stack([hold * xp.ones_like(t), steps, ramp, sine, chirp])
    target = options[params.target_pattern]
    return xp.clip(target, params.min_alt + span * 0.2, params.max_alt - span * 0.2)


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    power_requested: float,
    stick_requested: float,
    state: PlaneState,
    params: PlaneParams,
    integration_method: str = "rk4_2",
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
        target_altitude=commanded_altitude(
            state.base_target_altitude, state.time + 1, params
        ),
        base_target_altitude=state.base_target_altitude,
        gust_x=gust[0],
        gust_z=gust[1],
    )
    return new_state, metrics
