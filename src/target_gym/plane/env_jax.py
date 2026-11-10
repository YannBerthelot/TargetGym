import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import actuate, canonical_reset, finish_step, measured_obs
from target_gym.expert_access import RegistryExperts
from target_gym.plane.dynamics import (
    LONGITUDINAL,
    PITCH_DEMAND_MAX,
    STICK_TRAVEL,
    compute_gamma,
    instruments,
    overspeed,
    read_angles,
    scheduled_airspeed,
    total_wind_2d,
    trim_level,
)
from target_gym.plane.env import (
    PlaneParams,
    PlaneState,
    airspeed,
    assigned_cas,
    check_is_terminal,
    commanded_altitude,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
)

# Both panels come from target_gym.render_aircraft, the kit every aircraft
# environment draws with. There used to be a second, older pygame scene in
# rendering.py, kept on the grounds that it was faster per frame; nothing
# imported it, and once the aircraft renderers were unified it was drawing in a
# style that matched nothing else shipped, so it is gone.
from target_gym.plane.rendering_console import _render
from target_gym.video import SavesVideo


class Airplane2D(
    RegistryExperts, SavesVideo, environment.Environment[PlaneState, PlaneParams]
):
    """
    JAX-compatible 2D airplane environment.

    Action (2,): [throttle, pitch demand], raw in [-1, 1], held for the step. A
    fixed inner loop in the plant flies the elevator to the demand (+-15 deg at
    full action); a policy does not touch the surfaces.
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "target_altitude",
        "base_target_altitude",
        "gust_x",
        "gust_z",
        "zone_start",
        "zone_end",
        "m",
        "fuel",
    )

    registry_name: str = "plane"

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400

    #: Each observation channel's index, so consumers need not count.
    obs_index = {
        name: i
        for i, name in enumerate(
            (
                "x_dot",
                "z",
                "z_dot",
                "theta",
                "theta_dot",
                "gamma",
                "target_altitude",
                "power",
                "stick",
                "target_speed",
                "airspeed",
                "load_factor",
            )
        )
    }
    obs_value_index: int = 1  # z (altitude)
    tracked_names: tuple = ("altitude (m)",)
    obs_target_index: int = 6  # target_altitude

    # PHYSICS.md section on readings; gamma follows from the velocities.
    readings: tuple = instruments(LONGITUDINAL)

    def __init__(self, integration_method: str = "rk4_1", observe_wind: bool = False):
        # observe_wind=False -> the wind is a hidden disturbance (POMDP);
        # observe_wind=True  -> wind (wind_x, wind_z) is appended to the obs
        #                       (a fully-observable baseline / feedforward case).
        self.observe_wind = observe_wind
        # Ten, not nine: the commanded airspeed is appended to the
        # observation whether or not it is being scored, so the shape does not
        # depend on a reward weight. Twelve with the wind sensor.
        # Twelve with airspeed and normal acceleration, the air data and IRS
        # readings an autopilot has (rule 16); fourteen with the wind sensor.
        self.obs_shape = (14,) if observe_wind else (12,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> PlaneParams:
        # The scored configuration of ``plane``, the altitude hold: 30 min of
        # cruise (ours: long enough for one turbulence zone of up to 30 min
        # with calm on either side when it is shorter [BEA447 Figure 24]).
        band = 65.0 * 0.3048  # +-65 ft [RVSM A.4.1.4]
        # Cruise levels FL290 to FL370 (Amendment 3): RVSM airspace, where the
        # band applies [RVSM], clear of the 39 100 ft maximum altitude [TCDS].
        levels = (29_000.0 * 0.3048, 37_000.0 * 0.3048)
        return PlaneParams(
            max_steps_in_episode=1800,
            target_altitude_range=levels,
            initial_altitude_range=levels,
            # 0.2 g peak to peak a minute at FL330 (derived, plane/PHYSICS.md).
            turbulence_sigma=0.598,
            e_floor=band,
            e_tol=band,
            speed_band_cost=True,
            # A hold inside both bands costs nothing, so the best hold does.
            rho_floor_tracking=0.0,
            rho_floor=0.0,
            # The ground is the one trip left (rule 14; REALITY.md finds no
            # other), and no source prices a crash: ours, an hour
            # (``restart_steps``) at the cost of 40 000 ft off the level.
            failure_cost=((12192.0 - band) / band) ** 2,
        )

    def assigned_cas(self, state, params=None):
        """The CAS assigned at ``state``'s step (``plane.env.assigned_cas``)."""
        params = self.default_params if params is None else params
        return assigned_cas(state.time, params)

    def commanded_level(self, state, params=None):
        """The level commanded at ``state``'s step."""
        params = self.default_params if params is None else params
        return commanded_altitude(state.base_target_altitude, state.time, params)

    def trim_action(self, state, params=None):
        """The action that holds ``state``'s throttle and pitch: on a reset
        state, the one that keeps the aircraft established."""
        params = self.default_params if params is None else params
        lever = (state.power - params.idle_power) / (1.0 - params.idle_power)
        return jnp.stack([2.0 * lever - 1.0, state.theta / PITCH_DEMAND_MAX])

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PlaneState,
        action: jnp.ndarray,
        params: PlaneParams = None,
    ):
        """
        Performs step transitions using JAX, returns observation, new state, reward, done, info
        """
        if params is None:
            params = self.default_params

        # Not every learner squashes, so a raw action past the box acts as the
        # bound rather than as a command the plant could never receive.
        action, state = actuate(self, state, jnp.asarray(action).reshape((2,)), params)
        power, pitch = jnp.clip(action, -1.0, 1.0)
        power = (power + 1) / 2  # map from [-1, 1] to [0, 1]

        new_state, metrics = compute_next_state(
            power,
            pitch * PITCH_DEMAND_MAX,
            state,
            params,
            integration_method=self.integration_method,
            key=key,
        )
        # A crash is part of the kernel: the step is charged the trip cost and
        # the flight restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        # The VMO/MMO warning [TCDS item 10]: reported, never a trip.
        warning = overspeed(airspeed(new_state, params), new_state.z, params)
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params, metrics=metrics, overspeed=warning
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    def is_terminated(self, state: PlaneState, params: PlaneParams) -> jax.Array:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params, xp=jnp)
        return terminated

    @canonical_reset
    def reset_env(self, key: chex.PRNGKey, params: PlaneParams = None):
        """
        Reset the environment using JAX random keys
        """
        if params is None:
            params = self.default_params
        key, altitude_key, target_key, zone_key = jax.random.split(key, 4)

        target_altitude = jax.random.uniform(
            target_key,
            minval=params.target_altitude_range[0],
            maxval=params.target_altitude_range[1],
        )
        # Established at the assigned level and speed, in trim (rule 17; the
        # cruise hold is flown established in ALT CRZ [BEA447]). The offset
        # is zero on the scored tasks; Extra variants may widen it.
        initial_z = jnp.clip(
            target_altitude
            + jax.random.uniform(
                altitude_key,
                minval=params.initial_altitude_offset_range[0],
                maxval=params.initial_altitude_offset_range[1],
            ),
            params.initial_altitude_range[0],
            params.initial_altitude_range[1],
        )
        air_speed = scheduled_airspeed(initial_z, params.target_cas, params.target_mach)
        # ``initial_fuel_quantity`` is a component of ``initial_mass``.
        initial_m = params.initial_mass
        theta, stick, power = trim_level(air_speed, initial_z, initial_m, params)
        x_dot = air_speed + params.wind_x
        onset_key, length_key = jax.random.split(zone_key)
        zone_start = jax.random.uniform(
            onset_key,
            minval=params.zone_onset_range[0],
            maxval=params.zone_onset_range[1],
        )
        zone_length = jax.random.uniform(
            length_key,
            minval=params.zone_duration_range[0],
            maxval=params.zone_duration_range[1],
        )

        state = PlaneState(
            x=0.0,
            x_dot=x_dot,
            z=initial_z,
            z_dot=0.0,
            theta=theta,
            theta_dot=0.0,
            alpha=theta,
            gamma=0.0,
            m=initial_m,
            power=power,
            stick=stick,
            fuel=params.initial_fuel_quantity,
            time=0,
            target_altitude=target_altitude,
            # What the moving patterns oscillate about; identical to
            # target_altitude while the pattern is 'hold'.
            base_target_altitude=target_altitude,
            pitch_integ=stick / STICK_TRAVEL,
            zone_start=zone_start,
            zone_end=zone_start + zone_length,
        )

        obs = self.get_obs(state, params)
        return obs, state

    @measured_obs
    def get_obs(self, state: PlaneState, params: PlaneParams = None):
        """
        Observation vector. When ``observe_wind`` is set, the wind components
        (wind_x, wind_z) are appended so the disturbance is fully observable.
        """
        gamma = compute_gamma(state.x_dot, state.z_dot)
        state = read_angles(state, params, gamma=gamma)
        obs = get_obs(state, params, xp=jnp)
        if self.observe_wind:
            # Realized wind = steady mean + altitude shear + current gust.
            wx, wz = total_wind_2d(state.z, state.gust_x, state.gust_z, params)
            obs = jnp.concatenate([obs, jnp.array([wx, wz])])
        return obs

    def render(self, screen, state: PlaneState, params: PlaneParams, frames, clock):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_plane(screen, state, params, frames, clock)
        return frames, screen, clock

    def action_space(self, params: PlaneParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: PlaneParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: PlaneParams) -> spaces.Box:
        """State space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(PlaneState.__dataclass_fields__), dtype=jnp.float32
        )


class PlaneEnergy(Airplane2D):
    """``plane_energy``: the 2D aircraft flying ATC's level and speed
    clearances, TECS's job (plane/REALITY.md, missions): altitude changes with
    the speed held, and speed changes at a constant level.

    From established cruise, every 300 s one clearance: a new level, in whole
    1 000 ft, or a new speed, in 10 kt (``plane.env.CLEARANCE_LEVELS_FT``,
    ``CLEARANCE_SPEEDS_KT``). The score is the job's bands: the speed inside
    +-10 kt [AIM 4-4-12] all along; the altitude inside +-65 ft [RVSM] of a
    level the aircraft closes at no less than 500 fpm from each clearance, and
    no more than 1 500 fpm within its last 1 000 ft [AIM 4-4-10]. Thrust is
    finite, so holding speed through a climb at the minimum rate is the
    speed-against-height trade TECS makes [TECS].

    Action (2,): [throttle, pitch demand], raw in [-1, 1], as ``Airplane2D``.
    """

    registry_name: str = "plane_energy"

    @property
    def default_params(self) -> PlaneParams:
        band = 65.0 * 0.3048  # +-65 ft [RVSM A.4.1.4]
        return PlaneParams(
            # Six 300 s clearances (``clearance_period``).
            max_steps_in_episode=1800,
            target_pattern=5,
            # The clearances span +2 000 to -1 000 ft about the start, so it
            # starts where they stay inside 3 000 to 8 000 m (ours).
            target_altitude_range=(3_400.0, 7_300.0),
            e_floor=band,
            e_tol=band,
            speed_band_cost=True,
            rho_floor_tracking=0.0,
            rho_floor=0.0,
            # As ``plane``: the ground is the one trip, priced by us.
            failure_cost=((12192.0 - band) / band) ** 2,
        )


class PlaneSine(Airplane2D):
    """``plane_sine``: the 2D aircraft tracking a sinusoidal altitude, a probe
    of the closed loop's bandwidth.

    Action (2,): [throttle, pitch demand], raw in [-1, 1], as ``Airplane2D``.
    """

    registry_name: str = "plane_sine"

    @property
    def default_params(self) -> PlaneParams:
        # The scored configuration. 480 steps is two periods of the 240 s
        # sinusoid: the first sheds the initial transient, the second is what
        # a bandwidth probe reads.
        #
        # The amplitude is set by the climb rate: amplitude A and period T
        # command a peak vertical rate of 2*pi*A/T. 300 m peaks at 7.9 m/s,
        # inside the aircraft's 14.6 m/s best sustained climb, so the response
        # is limited by the closed loop, not by the actuator saturating.
        #
        # ``c_hold`` is the PID's hold-phase airspeed deviation on the
        # sinusoid. The altitude floor is an earlier oracle's hold, kept
        # because the 1 m barometric resolution is within 1.5x of it; rho is
        # the current oracle's lowest per-seed hold, 0.5285 m, in floor units
        # (`scripts/measure_hold.py`).
        rho = (0.5285 / 1.26) ** 2
        return PlaneParams(
            max_steps_in_episode=480,
            target_pattern=3,
            target_amplitude=300.0,
            target_period=240.0,
            c_hold=8.15,
            e_floor=1.26,
            rho_floor_tracking=rho,
            rho_floor=rho,
            failure_cost=2.0 * (12192.0 / 1.26) ** 2,
        )
