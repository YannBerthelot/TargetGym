"""
JAX-compatible 3D airplane environments.

Three task variants sharing the same 3D physics:
  - Plane3DHeading:  reach and follow a target heading at a target altitude
  - Plane3DCircle:   fly a circle of given radius at a target altitude
  - Plane3DFigureEight: fly a figure-8 (lemniscate) at a target altitude

Actions: throttle, pitch demand, bank demand, held for the step. A fixed inner
loop in the plant flies the surfaces to the demands; a policy does not touch them.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.base import actuate, canonical_reset, finish_step, measured_obs
from target_gym.expert_access import RegistryExperts
from target_gym.plane.dynamics import (
    BANK_DEMAND_MAX,
    LATERAL,
    LONGITUDINAL,
    PITCH_DEMAND_MAX,
    POSITION,
    STICK_TRAVEL,
    instruments,
    overspeed,
    total_wind_3d,
    trim_level,
    trim_stick,
)
from target_gym.plane3d.dynamics import read_angles_3d
from target_gym.plane3d.env import (
    _RACETRACK_LEG,
    PlaneParams3D,
    PlaneState3D,
    airspeed_3d,
    assigned_cas_3d,
    assigned_tas_3d,
    check_is_terminal_3d,
    compute_next_state_3d,
    compute_reward_circle,
    compute_reward_figure8,
    compute_reward_heading,
    compute_reward_racetrack,
    compute_reward_terms_circle,
    compute_reward_terms_figure8,
    compute_reward_terms_heading,
    compute_reward_terms_racetrack,
    get_obs_circle,
    get_obs_figure8,
    get_obs_heading,
    get_obs_racetrack,
    holding_turn_rate,
    wrap_angle,
)
from target_gym.plane3d.rendering import _render
from target_gym.utils import compute_norm_from_coordinates
from target_gym.video import SavesVideo


class _Airplane3DBase(
    RegistryExperts, SavesVideo, environment.Environment[PlaneState3D, PlaneParams3D]
):
    """
    Base class for all 3D airplane environments.

    Subclasses must set:
      - obs_shape
      - obs_value_index, obs_target_index
    and implement:
      - compute_reward(state, params)
      - _terms(state, params, tripped=None), the task's reward terms
      - _observe(state, params)
      - _reset_targets(key, params) -> dict of target fields
    """

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # PHYSICS.md section on readings; gamma and psi follow from the velocities.
    # The path tasks add the GNSS fix, which the heading task has no use for.
    readings: tuple = instruments(LONGITUDINAL + LATERAL + POSITION)

    def __init__(self, integration_method: str = "rk4_1", observe_wind: bool = False):
        # observe_wind=False -> wind is a hidden disturbance (POMDP);
        # observe_wind=True  -> (wind_x, wind_y, wind_z) appended to the obs
        #                       (fully-observable baseline).
        self.observe_wind = observe_wind
        self.positions_history_xz = []
        self.positions_history_xy = []
        self.integration_method = integration_method

    def _append_wind(self, obs, state, params):
        """Append the realized wind to *obs* when ``observe_wind`` is set."""
        if not self.observe_wind:
            return obs
        # Realized wind = steady mean + altitude shear + current gust.
        wx, wy, wz = total_wind_3d(
            state.z, state.gust_x, state.gust_y, state.gust_z, params
        )
        return jnp.concatenate([obs, jnp.array([wx, wy, wz])])

    @property
    def default_params(self) -> PlaneParams3D:
        return PlaneParams3D()

    def reward_terms(self, state, params, tripped=None):
        # ``tripped`` marks a measured copy, whose psi is the measured track.
        if tripped is not None:
            state = read_angles_3d(state, params)
        return self._terms(state, params, tripped=tripped)

    @measured_obs
    def get_obs(self, state: PlaneState3D, params: PlaneParams3D = None):
        state = read_angles_3d(state, params)
        return self._append_wind(self._observe(state, params), state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PlaneState3D,
        action: jnp.ndarray,
        params: PlaneParams3D = None,
    ):
        if params is None:
            params = self.default_params

        # Not every learner squashes, so a raw action past the box acts as the
        # bound rather than as a command the plant could never receive.
        action, state = actuate(self, state, jnp.asarray(action).reshape((3,)), params)
        power, pitch, bank = jnp.clip(action, -1.0, 1.0)
        power = (power + 1) / 2  # [-1, 1] -> [0, 1]

        new_state, metrics = compute_next_state_3d(
            power,
            pitch * PITCH_DEMAND_MAX,  # [-1, 1] -> +/-15 deg
            bank * BANK_DEMAND_MAX,  # [-1, 1] -> +/-45 deg
            state,
            params,
            integration_method=self.integration_method,
            key=key,
        )
        # A crash is part of the kernel: the step is charged the trip cost and
        # the flight restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        # The VMO/MMO warning [TCDS item 10], reported, never a trip when
        # amended.
        warning = overspeed(airspeed_3d(new_state, params), new_state.z, params)
        new_state = self._next_clearance(key, new_state, params)
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params, metrics=metrics, overspeed=warning
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    def _next_clearance(self, key, state, params):
        """The state after ATC's clearance for this step, if any (the heading
        task's vectors); the other tasks have none."""
        return state

    def is_terminated(self, state: PlaneState3D, params: PlaneParams3D) -> jax.Array:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal_3d(state, params, xp=jnp)
        return terminated

    def _reset_common(self, key: chex.PRNGKey, params: PlaneParams3D):
        """Compute shared initial state fields. Returns (remaining_key, base_kwargs)."""
        if params is None:
            params = self.default_params
        key, altitude_key, target_altitude_key = jax.random.split(key, 3)

        target_altitude = jax.random.uniform(
            target_altitude_key,
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
        initial_x_dot = params.initial_x_dot
        initial_y_dot = params.initial_y_dot
        initial_z_dot = params.initial_z_dot
        initial_theta = jnp.deg2rad(params.initial_theta)
        initial_phi = jnp.deg2rad(params.initial_phi)

        V_horiz = compute_norm_from_coordinates(
            jnp.array([initial_x_dot, initial_y_dot + 1e-6])
        )
        initial_gamma = jnp.arcsin(
            initial_z_dot
            / (
                compute_norm_from_coordinates(
                    jnp.array([V_horiz, initial_z_dot + 1e-6])
                )
            )
        )
        initial_alpha = initial_theta - initial_gamma
        initial_psi = jnp.arctan2(initial_y_dot, initial_x_dot)
        # The autopilot engages in trim; ``initial_stick`` is an offset from it.
        trim = trim_stick(V_horiz, initial_z, initial_theta, params)

        base_kwargs = dict(
            x=0.0,
            x_dot=initial_x_dot,
            y=0.0,
            y_dot=initial_y_dot,
            z=initial_z,
            z_dot=initial_z_dot,
            theta=initial_theta,
            theta_dot=jnp.deg2rad(params.initial_theta_dot),
            phi=initial_phi,
            phi_dot=jnp.deg2rad(params.initial_phi_dot),
            psi=initial_psi,
            alpha=initial_alpha,
            gamma=initial_gamma,
            m=params.initial_mass,  # fuel is a component of it, not an addition
            power=params.initial_power,
            stick=trim + jnp.deg2rad(params.initial_stick),
            aileron=jnp.deg2rad(params.initial_aileron),
            fuel=params.initial_fuel_quantity,
            time=0,
            target_altitude=target_altitude,
            pitch_integ=trim / STICK_TRAVEL,
        )
        return key, base_kwargs

    def assigned_cas(self, state, params=None):
        """The CAS assigned to ``state`` (``plane3d.env.assigned_cas_3d``)."""
        params = self.default_params if params is None else params
        return assigned_cas_3d(state.target_altitude, params)

    def trim_action(self, state, params=None):
        """The action that holds ``state``'s throttle and pitch, wings level."""
        params = self.default_params if params is None else params
        lever = (state.power - params.idle_power) / (1.0 - params.idle_power)
        return jnp.stack(
            [2.0 * lever - 1.0, state.theta / PITCH_DEMAND_MAX, jnp.zeros(())]
        )

    def _established(self, key, base_kwargs, params, heading):
        """``base_kwargs`` established (Amendment 2, rule 17): at the assigned
        level and speed, wings level on ``heading``, in trim, with the episode's
        turbulence zone drawn [BEA447 Figure 24]."""
        z = base_kwargs["target_altitude"]
        speed = assigned_tas_3d(z, z, params)
        theta, stick, power = trim_level(speed, z, params.initial_mass, params)
        onset_key, length_key = jax.random.split(key)
        start = jax.random.uniform(
            onset_key,
            minval=params.zone_onset_range[0],
            maxval=params.zone_onset_range[1],
        )
        length = jax.random.uniform(
            length_key,
            minval=params.zone_duration_range[0],
            maxval=params.zone_duration_range[1],
        )
        x_dot = speed * jnp.cos(heading) + params.wind_x
        y_dot = speed * jnp.sin(heading) + params.wind_y
        return dict(
            base_kwargs,
            z=z,
            x_dot=x_dot,
            y_dot=y_dot,
            z_dot=0.0,
            theta=theta,
            theta_dot=0.0,
            phi=0.0,
            phi_dot=0.0,
            # The track, worked out from the velocities as the step does.
            psi=jnp.arctan2(y_dot, x_dot),
            alpha=theta,
            gamma=0.0,
            power=power,
            stick=stick,
            aileron=0.0,
            pitch_integ=stick / STICK_TRAVEL,
            zone_start=start,
            zone_end=start + length,
        )

    def render(self, screen, state: PlaneState3D, params: PlaneParams3D, frames, clock):
        frames, screen, clock = self.render_plane(screen, state, params, frames, clock)
        return frames, screen, clock

    def action_space(self, params: PlaneParams3D | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0, -1.0]),
            high=jnp.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: PlaneParams3D) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: PlaneParams3D) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(PlaneState3D.__dataclass_fields__), dtype=jnp.float32
        )


#: Folded into a step's key for its heading clearance, apart from its gusts.
_CLEARANCE = 0x48444E47

# ─── Heading task ──────────────────────────────────────


class Plane3DHeading(_Airplane3DBase):
    """
    3D airplane: reach and follow a target heading at a target altitude.

    Observation (15,):
        [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
         gamma, psi, target_altitude, target_heading, power, stick, aileron]

    Action (3,): [throttle, pitch demand, bank demand] each in [-1, 1]
    """

    obs_value_index: int = 2  # z (altitude)
    tracked_names: tuple = ("altitude (m)",)
    obs_target_index: int = 10  # target_altitude
    obs_index = {
        n: i
        for i, n in enumerate(
            (
                "x_dot", "y_dot", "z", "z_dot", "theta", "theta_dot", "phi",
                "phi_dot", "gamma", "psi", "target_altitude", "target_heading",
                "power", "stick", "aileron", "target_speed", "airspeed",
                "load_factor",
            )
        )
    }  # fmt: skip
    task_type: str = "heading"
    registry_name: str = "plane3d_heading"
    readings: tuple = instruments(LONGITUDINAL + LATERAL)

    @property
    def default_params(self) -> PlaneParams3D:
        # The scored configuration (Amendment 2, plane3d/PHYSICS.md): heading
        # select from established cruise, 600 s (ours: the longest turn, 180
        # deg at a holding turn's rate, takes under 110 s, then a held heading
        # through the turbulence zone, which starts anywhere in the episode).
        band = 65.0 * 0.3048  # +-65 ft [RVSM A.4.1.4]
        ten = float(np.deg2rad(10.0))  # +-10 deg [ACS]
        return PlaneParams3D(
            max_steps_in_episode=600,
            heading_clearance_period=150.0,
            amended=True,
            turbulence_sigma=0.55,
            zone_onset_range=(0.0, 600.0),
            initial_altitude_offset_range=(0.0, 0.0),
            e_floor_altitude=band,
            e_tol_altitude=band,
            e_floor_heading=ten,
            e_tol_heading=ten,
            rho_floor_tracking=0.0,
            rho_floor=0.0,
            # As the 2D aircraft: the ground is the one trip, priced by us.
            failure_cost=((12192.0 - band) / band) ** 2,
        )

    def __init__(self, integration_method: str = "rk4_1", observe_wind: bool = False):
        super().__init__(integration_method, observe_wind)
        # 18 with the air data (Amendment 2, rule 16); 21 with the wind sensor.
        self.obs_shape = (21,) if observe_wind else (18,)

    def _next_clearance(self, key, state, params):
        """Every ``heading_clearance_period`` s a new heading over
        ``target_heading_range``; its size is the turn from the heading flown."""
        period = jnp.maximum(params.heading_clearance_period / params.delta_t, 1.0)
        due = (
            (params.heading_clearance_period > 0.0)
            & (state.time > 0)
            & (state.time < params.max_steps_in_episode)
            & (jnp.mod(state.time, period) == 0)
        )
        heading = jax.random.uniform(
            jax.random.fold_in(key, _CLEARANCE),
            minval=params.target_heading_range[0],
            maxval=params.target_heading_range[1],
        )
        return state.replace(
            target_heading=jnp.where(due, heading, state.target_heading),
            heading_change=jnp.where(
                due,
                jnp.abs(wrap_angle(heading - state.psi)),
                state.heading_change,
            ),
            clearance_time=jnp.where(due, state.time, state.clearance_time),
        )

    def compute_reward(self, state, params):
        return compute_reward_heading(state, params)

    def _terms(self, state, params, tripped=None):
        return compute_reward_terms_heading(state, params, tripped=tripped)

    def _observe(self, state, params):
        return get_obs_heading(state, xp=jnp, params=params)

    @canonical_reset
    def reset_env(self, key: chex.PRNGKey, params: PlaneParams3D = None):
        if params is None:
            params = self.default_params
        key, base_kwargs = self._reset_common(key, params)
        key, heading_key = jax.random.split(key)

        target_heading = jax.random.uniform(
            heading_key,
            minval=params.target_heading_range[0],
            maxval=params.target_heading_range[1],
        )
        # Amended: established on heading 0, cleared onto the target at once;
        # the capture is the job here (plane/REALITY.md, missions).
        key, zone_key = jax.random.split(key)
        established = self._established(zone_key, base_kwargs, params, 0.0)
        base_kwargs = jax.tree_util.tree_map(
            lambda a, b: jnp.where(params.amended, a, b), established,
            dict(base_kwargs, zone_start=0.0, zone_end=0.0),
        )  # fmt: skip
        state = PlaneState3D(
            **base_kwargs,
            target_heading=target_heading,
            target_x=0.0,
            target_y=0.0,
            target_radius=0.0,
            heading_change=jnp.abs(wrap_angle(target_heading - base_kwargs["psi"])),
        )
        return self.get_obs(state, params), state


# ─── Circle task ───────────────────────────────────────


class Plane3DRacetrack(_Airplane3DBase):
    """
    3D airplane: fly a racetrack holding pattern at a target altitude.

    A hold is two straight legs joined by two 180 degree turns, which is what
    "holding" means in aviation and the shape this library is named for. It
    exercises heading-hold on the legs and a sustained coordinated turn on the
    caps *in one task*, where Plane3DHeading and Plane3DCircle test those
    separately -- and the transition between them, twice a lap, is where a
    controller tuned for one regime shows what it costs in the other.

    The aircraft starts on a straight leg, tracking along it.

    Observation (21,):
        [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
         gamma, psi, target_altitude, rel_x, rel_y, target_radius,
         target_heading, power, stick, aileron,
         cross_track, tangent_heading, curvature]

    Action (3,): [throttle, pitch demand, bank demand] each in [-1, 1]
    """

    obs_value_index: int = 2
    # Declared, unlike the three tasks above it, only after its absence broke
    # everything that rolls an episode out: ``runners.rollout`` reads it to
    # know which observation carries the setpoint, so the tuner scored every
    # candidate as -inf and ``scripts/record_baselines.py`` could not have
    # recorded this environment at all.
    obs_target_index: int = 10
    tracked_names: tuple = ("altitude (m)",)
    # Without this the renderer's getattr falls back to "heading", and the
    # holding pattern is drawn as a straight dashed line -- a picture of a
    # different task than the one being scored.
    task_type: str = "racetrack"
    registry_name: str = "plane3d_racetrack"
    obs_index = {
        n: i
        for i, n in enumerate(
            (
                "x_dot", "y_dot", "z", "z_dot", "theta", "theta_dot", "phi",
                "phi_dot", "gamma", "psi", "target_altitude", "rel_x", "rel_y",
                "target_radius", "target_heading", "power", "stick", "aileron",
                "cross_track", "tangent_heading", "curvature", "target_half_leg",
                "target_speed", "airspeed", "load_factor",
            )
        )
    }  # fmt: skip

    @property
    def default_params(self) -> PlaneParams3D:
        # The scored configuration (Amendment 2, plane3d/PHYSICS.md): the
        # holding pattern from established on its inbound leg, 1 200 s (ours:
        # three laps of the slowest pattern, 2 x 90 s legs and two 96 s turns,
        # with the turbulence zone anywhere in it).
        band = 65.0 * 0.3048  # +-65 ft [RVSM A.4.1.4]
        ten = float(np.deg2rad(10.0))  # +-10 deg [ACS IR.III.B.S4]
        track = 0.75 * 1852.0  # 3/4 of the +-1 NM terminal CDI [ACS, AIM]
        return PlaneParams3D(
            max_steps_in_episode=1200,
            amended=True,
            turbulence_sigma=0.55,
            zone_onset_range=(0.0, 1200.0),
            initial_altitude_offset_range=(0.0, 0.0),
            assigned_speed_mode=1,
            e_floor_altitude=band,
            e_tol_altitude=band,
            e_floor_heading=ten,
            e_tol_heading=ten,
            e_floor_path=track,
            e_tol_path=track,
            rho_floor_tracking=0.0,
            rho_floor=0.0,
            failure_cost=((12192.0 - band) / band) ** 2,
        )

    def __init__(self, integration_method: str = "rk4_1", observe_wind: bool = False):
        super().__init__(integration_method, observe_wind)
        # 21: the circle's 18 plus target_heading and the two guidance values
        # the pattern's geometry determines exactly; 25 with the half leg and
        # the air data (Amendment 2); 28 with the wind sensor.
        self.obs_shape = (28,) if observe_wind else (25,)

    def compute_reward(self, state, params):
        return compute_reward_racetrack(state, params)

    def _terms(self, state, params, tripped=None):
        return compute_reward_terms_racetrack(state, params, tripped=tripped)

    def _observe(self, state, params):
        return get_obs_racetrack(state, xp=jnp, params=params)

    @canonical_reset
    def reset_env(self, key: chex.PRNGKey, params: PlaneParams3D = None):
        if params is None:
            params = self.default_params
        key, base_kwargs = self._reset_common(key, params)
        key, radius_key, heading_key, along_key = jax.random.split(key, 4)

        target_radius = jax.random.uniform(
            radius_key,
            minval=params.target_radius_range[0],
            maxval=params.target_radius_range[1],
        )
        # Amended: the hold's own geometry [IFH ch. 10]. Turns at its rate at
        # the assigned speed set the radius; the inbound leg's minute (1.5
        # above 14 000 ft) at that speed sets the leg, in still air.
        z = base_kwargs["target_altitude"]
        speed = assigned_tas_3d(z, z, params)
        hold_radius = speed / holding_turn_rate(speed, params)
        leg_time = jnp.where(
            z <= params.holding_split_altitude,
            params.holding_leg_low,
            params.holding_leg_high,
        )
        target_radius = jnp.where(params.amended, hold_radius, target_radius)
        half_leg = jnp.where(
            params.amended, 0.5 * speed * leg_time, target_radius * _RACETRACK_LEG
        )
        # Orientation of the pattern, and where along the near leg to start.
        pattern_heading = jax.random.uniform(heading_key, minval=0.0, maxval=2 * jnp.pi)
        along = jax.random.uniform(along_key, minval=-half_leg, maxval=half_leg)

        # Start on the leg at +radius abeam, heading along the pattern.
        c, s_ = jnp.cos(pattern_heading), jnp.sin(pattern_heading)
        start_x = along * c - target_radius * s_
        start_y = along * s_ + target_radius * c

        speed = compute_norm_from_coordinates(
            jnp.array([base_kwargs["x_dot"], base_kwargs["y_dot"] + 1e-6])
        )
        x_dot = speed * c
        y_dot = speed * s_

        base_kwargs.update(
            x=start_x,
            y=start_y,
            x_dot=x_dot,
            y_dot=y_dot,
            psi=jnp.arctan2(y_dot, x_dot),
        )
        # Amended: established on the leg at the hold's level and speed.
        key, zone_key = jax.random.split(key)
        established = self._established(
            zone_key, base_kwargs, params, jnp.arctan2(s_, c)
        )
        established.update(x=start_x, y=start_y)
        base_kwargs = jax.tree_util.tree_map(
            lambda a, b: jnp.where(params.amended, a, b), established,
            dict(base_kwargs, zone_start=0.0, zone_end=0.0),
        )  # fmt: skip
        state = PlaneState3D(
            **base_kwargs,
            target_x=0.0,
            target_y=0.0,
            target_radius=target_radius,
            target_heading=pattern_heading,
            target_half_leg=jnp.where(params.amended, half_leg, 0.0),
        )
        return self.get_obs(state, params), state


class Plane3DCircle(_Airplane3DBase):
    """
    3D airplane: fly a circle of given radius at a target altitude.

    The aircraft starts on the circle with heading tangent to it.

    Observation (17,):
        [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
         gamma, psi, target_altitude, rel_x, rel_y, target_radius,
         power, stick, aileron]

    Action (3,): [throttle, pitch demand, bank demand] each in [-1, 1]
    """

    obs_value_index: int = 2
    obs_target_index: int = 10
    task_type: str = "circle"
    registry_name: str = "plane3d_circle"

    @property
    def default_params(self) -> PlaneParams3D:
        # The scored configuration: the 1 m altimeter and 3 m GPS
        # resolutions. The oracle holds below both, and rho is its lowest
        # per-seed holds (0.6657 m, 0.4504 m; `scripts/measure_hold.py`) in
        # floor units.
        rho = (0.6657 / 1.0) ** 2 + (0.4504 / 3.0) ** 2
        return PlaneParams3D(
            max_steps_in_episode=300,
            e_floor_altitude=1.0,
            e_floor_path=3.0,
            rho_floor_tracking=rho,
            rho_floor=rho,
            failure_cost=2.0 * (12192.0 / 1.0) ** 2,
        )

    tracked_names: tuple = ("altitude (m)",)

    def __init__(self, integration_method: str = "rk4_1", observe_wind: bool = False):
        super().__init__(integration_method, observe_wind)
        self.obs_shape = (20,) if observe_wind else (17,)

    def compute_reward(self, state, params):
        return compute_reward_circle(state, params)

    def _terms(self, state, params, tripped=None):
        return compute_reward_terms_circle(state, params, tripped=tripped)

    def _observe(self, state, params):
        return get_obs_circle(state, xp=jnp)

    @canonical_reset
    def reset_env(self, key: chex.PRNGKey, params: PlaneParams3D = None):
        if params is None:
            params = self.default_params
        key, base_kwargs = self._reset_common(key, params)
        key, radius_key, angle_key = jax.random.split(key, 3)

        target_radius = jax.random.uniform(
            radius_key,
            minval=params.target_radius_range[0],
            maxval=params.target_radius_range[1],
        )
        # Random angle on the circle for starting position
        start_angle = jax.random.uniform(angle_key, minval=0.0, maxval=2 * jnp.pi)

        # Circle center at origin; aircraft starts on the circle
        target_x = 0.0
        target_y = 0.0
        start_x = target_x + target_radius * jnp.cos(start_angle)
        start_y = target_y + target_radius * jnp.sin(start_angle)

        # Tangent heading (CCW): perpendicular to radius vector
        tangent_heading = start_angle + jnp.pi / 2

        speed = compute_norm_from_coordinates(
            jnp.array([base_kwargs["x_dot"], base_kwargs["y_dot"] + 1e-6])
        )
        x_dot = speed * jnp.cos(tangent_heading)
        y_dot = speed * jnp.sin(tangent_heading)
        psi = jnp.arctan2(y_dot, x_dot)

        base_kwargs.update(
            x=start_x,
            y=start_y,
            x_dot=x_dot,
            y_dot=y_dot,
            psi=psi,
        )

        state = PlaneState3D(
            **base_kwargs,
            target_heading=0.0,
            target_x=target_x,
            target_y=target_y,
            target_radius=target_radius,
        )
        return self.get_obs(state, params), state


# ─── Figure-8 task ─────────────────────────────────────


class Plane3DFigureEight(_Airplane3DBase):
    """
    3D airplane: fly a figure-8 (lemniscate of Bernoulli) at a target altitude.

    The aircraft starts at the rightmost point of the right lobe.

    Observation (21,):
        [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
         gamma, psi, target_altitude, target_radius,
         nearest_dx, nearest_dy, nearest_dz, tangent_heading,
         power, stick, aileron, path_distance, target_path_distance(=0)]

    Action (3,): [throttle, pitch demand, bank demand] each in [-1, 1]
    """

    # The reward charges the distance to the curve, not the altitude: the
    # observed target_altitude is the curve's mean, which it swings 200 m about.
    obs_value_index: int = 19  # distance to the curve
    obs_target_index: int = 20  # constant 0
    task_type: str = "figure8"
    registry_name: str = "plane3d_figure8"

    @property
    def default_params(self) -> PlaneParams3D:
        # The scored configuration: the path term alone, its floor the 3 m
        # GPS resolution. The oracle holds below it, and rho is its lowest
        # per-seed hold (1.2871 m; `scripts/measure_hold.py`) in floor units.
        # The tips have radius a / 3, so at 25 deg of bank and 230 m/s a is
        # 34.7 km. A lap is 5.244 a, 183 km, so 1000 steps reach both tips at
        # the slowest speed the schedule flies and nearly close the lap. The
        # failure cost is twice the path's size, as it was at the old 10 km.
        rho = (1.2871 / 3.0) ** 2
        return PlaneParams3D(
            max_steps_in_episode=1000,
            target_radius_range=(34_000.0, 36_000.0),
            e_floor_path=3.0,
            rho_floor_tracking=rho,
            rho_floor=rho,
            failure_cost=2.0 * (70000.0 / 3.0) ** 2,
        )

    tracked_names: tuple = ("distance to path (m)",)

    def __init__(self, integration_method: str = "rk4_1", observe_wind: bool = False):
        super().__init__(integration_method, observe_wind)
        self.obs_shape = (24,) if observe_wind else (21,)

    def compute_reward(self, state, params):
        return compute_reward_figure8(state, params)

    def _terms(self, state, params, tripped=None):
        return compute_reward_terms_figure8(state, params, tripped=tripped)

    def _observe(self, state, params):
        return get_obs_figure8(state, params, xp=jnp)

    @canonical_reset
    def reset_env(self, key: chex.PRNGKey, params: PlaneParams3D = None):
        if params is None:
            params = self.default_params
        key, base_kwargs = self._reset_common(key, params)
        key, radius_key, angle_key = jax.random.split(key, 3)

        target_radius = jax.random.uniform(
            radius_key,
            minval=params.target_radius_range[0],
            maxval=params.target_radius_range[1],
        )

        # Random orientation of the lemniscate
        orientation = jax.random.uniform(
            angle_key,
            minval=params.figure8_angle_range[0],
            maxval=params.figure8_angle_range[1],
        )

        # Start at τ=0 on the twisted lemniscate: rightmost point of right
        # lobe at mean altitude.  Position and heading are rotated by
        # orientation.
        target_x = 0.0
        target_y = 0.0
        cos_o = jnp.cos(orientation)
        sin_o = jnp.sin(orientation)
        # At τ=0: base_x = radius, base_y = 0, z = z_mean
        start_x = target_x + target_radius * cos_o
        start_y = target_y + target_radius * sin_o

        # Tangent at τ=0 is (0, +1) in base frame → rotated
        tangent_heading = orientation + jnp.pi / 2

        speed = compute_norm_from_coordinates(
            jnp.array([base_kwargs["x_dot"], base_kwargs["y_dot"] + 1e-6])
        )
        x_dot = speed * jnp.cos(tangent_heading)
        y_dot = speed * jnp.sin(tangent_heading)
        psi = jnp.arctan2(y_dot, x_dot)

        base_kwargs.update(
            x=start_x,
            y=start_y,
            z=base_kwargs["target_altitude"],
            x_dot=x_dot,
            y_dot=y_dot,
            psi=psi,
        )

        state = PlaneState3D(
            **base_kwargs,
            target_heading=orientation,  # stores figure-8 orientation
            target_x=target_x,
            target_y=target_y,
            target_radius=target_radius,
        )
        return self.get_obs(state, params), state


# ─── Backward-compatible alias ─────────────────────────

Airplane3D = Plane3DHeading
