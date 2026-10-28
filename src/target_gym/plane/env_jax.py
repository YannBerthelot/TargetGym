import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, finish_step
from target_gym.expert_access import RegistryExperts
from target_gym.plane.dynamics import total_wind_2d
from target_gym.plane.env import (
    PlaneParams,
    PlaneState,
    check_is_terminal,
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
from target_gym.utils import compute_norm_from_coordinates
from target_gym.video import SavesVideo


class Airplane2D(
    RegistryExperts, SavesVideo, environment.Environment[PlaneState, PlaneParams]
):
    """
    JAX-compatible 2D airplane environment.

    Action (2,): [power, stick], raw in [-1, 1]
    """

    registry_name: str = "plane"

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [x_dot, z, z_dot, theta, theta_dot, gamma, target_altitude, power, stick]
    obs_value_index: int = 1  # z (altitude)
    tracked_names: tuple = ("altitude (m)",)
    obs_target_index: int = 6  # target_altitude

    def __init__(self, integration_method: str = "rk4_2", observe_wind: bool = False):
        # observe_wind=False -> the wind is a hidden disturbance (POMDP);
        # observe_wind=True  -> wind (wind_x, wind_z) is appended to the obs
        #                       (a fully-observable baseline / feedforward case).
        self.observe_wind = observe_wind
        # Ten, not nine: the commanded airspeed is appended to the
        # observation whether or not it is being scored, so the shape does not
        # depend on a reward weight. Twelve with the wind sensor.
        self.obs_shape = (12,) if observe_wind else (10,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> PlaneParams:
        # The scored configuration of ``plane``, the altitude hold.
        return PlaneParams(max_steps_in_episode=280)

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

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

        power, stick = action
        power = (power + 1) / 2  # map from [-1, 1] to [0, 1]
        stick = jnp.deg2rad(stick * 15)  # radians

        new_state, metrics = compute_next_state(
            power,
            stick,
            state,
            params,
            integration_method=self.integration_method,
            key=key,
        )
        # A crash is part of the kernel: the step is charged the trip cost and
        # the flight restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params, metrics=metrics
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
        key, altitude_key, target_key = jax.random.split(key, 3)

        initial_x = 0.0
        target_altitude = jax.random.uniform(
            target_key,
            minval=params.target_altitude_range[0],
            maxval=params.target_altitude_range[1],
        )
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
        initial_z_dot = params.initial_z_dot
        initial_x_dot = params.initial_x_dot
        initial_theta = jnp.deg2rad(params.initial_theta)
        initial_gamma = jnp.arcsin(
            initial_z_dot
            / (
                compute_norm_from_coordinates(
                    jnp.array([initial_x_dot, initial_z_dot + 1e-6])
                )
            )
        )
        initial_alpha = initial_theta - initial_gamma
        # ``initial_fuel_quantity`` is a component of ``initial_mass``, not an
        # addition to it -- 73 500 kg is the all-up takeoff mass, of which
        # 19 088 kg is fuel (PHYSICS.md section 3). Adding them counted the
        # fuel twice, a 20-tonne error that was invisible while the dynamics
        # ignored ``state.m`` and used ``params.initial_mass`` directly.
        initial_m = params.initial_mass
        initial_power = params.initial_power
        initial_stick = jnp.deg2rad(params.initial_stick)
        initial_fuel = params.initial_fuel_quantity

        state = PlaneState(
            x=initial_x,
            x_dot=initial_x_dot,
            z=initial_z,
            z_dot=initial_z_dot,
            theta=initial_theta,
            theta_dot=jnp.deg2rad(params.initial_theta_dot),
            alpha=initial_alpha,
            gamma=initial_gamma,
            m=initial_m,
            power=initial_power,
            stick=initial_stick,
            fuel=initial_fuel,
            time=0,
            target_altitude=target_altitude,
            # What the moving patterns oscillate about; identical to
            # target_altitude while the pattern is 'hold'.
            base_target_altitude=target_altitude,
        )

        obs = self.get_obs(state, params)
        return obs, state

    def get_obs(self, state: PlaneState, params: PlaneParams = None):
        """
        Observation vector. When ``observe_wind`` is set, the wind components
        (wind_x, wind_z) are appended so the disturbance is fully observable.
        """
        obs = get_obs(state, params, xp=jnp)
        if self.observe_wind:
            if params is None:
                params = self.default_params
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
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(PlaneState.__class_params__), dtype=jnp.float32
        )


class PlaneEnergy(Airplane2D):
    """``plane_energy``: the 2D aircraft on an altitude staircase, scored on
    altitude *and* airspeed.

    With two actuators against one scored objective a controller could trade
    airspeed for altitude for free. Scoring both forces the trade,
    repeatedly. The pure staircase is ``PlaneParams(speed_weight=0.0)``.

    Action (2,): [power, stick], raw in [-1, 1]
    """

    registry_name: str = "plane_energy"

    @property
    def default_params(self) -> PlaneParams:
        # The scored configuration. ``c_hold`` is the PID's hold-phase
        # airspeed deviation on this task, and ``e_floor`` the oracle's
        # lowest per-seed altitude hold between ladder steps, 1.198 m
        # rounded; rho is its 1.18 m in floor units
        # (`scripts/measure_hold.py`).
        # 1600 steps, eight 200-step treads: twice the PID's settle after an
        # altitude step (about 90 steps), so every tread is half hold
        # (tests/test_episode_length.py).
        rho = (1.1773 / 1.20) ** 2
        return PlaneParams(
            max_steps_in_episode=1600,
            target_pattern=1,
            target_amplitude=900.0,
            target_steps=8,
            speed_weight=0.5,
            c_hold=5.23,
            e_floor=1.20,
            rho_floor_tracking=rho,
            rho_floor=rho,
            failure_cost=2.0 * (12192.0 / 1.20) ** 2,
        )


class PlaneSine(Airplane2D):
    """``plane_sine``: the 2D aircraft tracking a sinusoidal altitude, a probe
    of the closed loop's bandwidth.

    Action (2,): [power, stick], raw in [-1, 1]
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
