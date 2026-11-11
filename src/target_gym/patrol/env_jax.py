"""
JAX-compatible close-patrol (formation-keeping) environments.

Two task variants share the same two-aircraft physics and reward; they differ
only in what the follower observes:

  - :class:`PlanePatrol`            — full relative state (position + velocity
                                       of the lead in its body frame);
  - :class:`PlanePatrolRangeBearing`: partial, range + bearing to the lead only.

The lead is scripted (the 3D heading expert on a route of turns); the follower is
the learning agent.  See :mod:`.env` for the
transition, slot geometry, reward and observation functions.
"""

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import (
    ChannelReading,
    FieldReading,
    actuate,
    canonical_dtypes,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.expert_access import RegistryExperts
from target_gym.patrol.env import (
    N_LEAD_LEGS,
    PatrolParams,
    PatrolState,
    WakeSurfParams,
    check_is_terminal_patrol,
    compute_next_state_patrol,
    compute_reward_patrol,
    compute_reward_terms_patrol,
    compute_reward_terms_wake_surf,
    compute_reward_wake_surf,
    desired_slot_position,
    get_obs_full,
    get_obs_range_bearing,
    initial_wake,
    lead_cruise_speed,
    lead_leg,
    make_cruise_state,
)
from target_gym.patrol.rendering import _render
from target_gym.plane.dynamics import LATERAL, LONGITUDINAL, POSITION, instruments
from target_gym.plane.stream_events import (
    fuel_for,
    in_stream,
    sortie_key,
    sortie_mass,
)
from target_gym.plane3d.dynamics import read_angles_3d
from target_gym.plane3d.experts import Plane3DGuidanceState
from target_gym.video import SavesVideo

#: The follower's own instruments, and the lead's over the datalink. Their
#: velocities give both aircraft's gamma and psi (``read_angles_3d``).
_FOLLOWER = instruments(LONGITUDINAL + LATERAL + POSITION, "follower.")
_LEAD_VELOCITY = instruments(("x_dot", "y_dot", "z_dot"), "lead.")
_LEAD_FIX = instruments(POSITION, "lead.")
#: The relative altitude TCAS gets from a Mode S report: the lead's baro
#: altitude, coded in 25 ft steps. Rounding, not white noise: it is the one
#: altitude path behind ``lead.z`` and the elevation channel, whose scatter at
#: 250 m is ``25 ft / sqrt(12) / 250 m``, 0.0088 rad, falling as 1/range.
_LEAD_MODE_S_ALTITUDE = FieldReading("lead.z", sigma=0.2, step=25 * 0.3048)


def _read_angles(state: PatrolState, params: PatrolParams) -> PatrolState:
    """Both aircraft's gamma and psi, from their measured velocities."""
    return state.replace(
        follower=read_angles_3d(state.follower, params),
        lead=read_angles_3d(state.lead, params),
    )


class _PlanePatrolBase(
    RegistryExperts, SavesVideo, environment.Environment[PatrolState, PatrolParams]
):
    """Shared machinery for the close-patrol tasks.

    Subclasses set ``obs_shape``, ``obs_value_index``, ``obs_target_index`` and
    implement :meth:`get_obs`.
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "lead",
        "lead_memory",
        "slot_back",
        "slot_right",
        "slot_up",
        "lead_turn_rate",
        "lead_turn_schedule",
        "gust_history",
        "gust_own",
        "wake_history",
    )
    #: Runs as a continuing plant (``target_gym.stream``): sorties renew with a
    #: held mass, and the lead's route cycles on whole routes.
    STREAM_READY = True

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400
    task_type: str = "patrol"

    def __init__(self, integration_method: str = "rk4_1"):
        self.integration_method = integration_method
        self.positions_history_xz = []
        self.positions_history_xy = []
        # The scripted lead's memory at reset, in the types it comes back with.
        self._lead_memory = jax.tree.map(
            lambda x: jnp.asarray(x, jnp.float32), Plane3DGuidanceState()
        )

    @property
    def default_params(self) -> PatrolParams:
        # The scored configuration, for both observation variants.
        return PatrolParams(max_steps_in_episode=200)

    # -- to be provided by subclasses --------------------------------------
    def _observe(self, state: PatrolState) -> jnp.ndarray:
        raise NotImplementedError

    @measured_obs
    def get_obs(self, state: PatrolState, params: PatrolParams = None) -> jnp.ndarray:
        return self._observe(_read_angles(state, params))

    # -- core --------------------------------------------------------------
    def compute_reward(self, state, params):
        return compute_reward_patrol(state, params)

    def reward_terms(self, state, params, tripped=None):
        # ``tripped`` marks a measured copy, whose psi are the measured tracks.
        if tripped is not None:
            state = _read_angles(state, params)
        return compute_reward_terms_patrol(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PatrolState,
        action: jnp.ndarray,
        params: PatrolParams = None,
    ):
        if params is None:
            params = self.default_params

        action, state = actuate(self, state, action.reshape((3,)), params)
        new_state, metrics = compute_next_state_patrol(
            action,
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

    def is_terminated(self, state: PatrolState, params: PatrolParams) -> jax.Array:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal_patrol(state, params, xp=jnp)
        return terminated

    @canonical_reset
    def reset_env(self, key: chex.PRNGKey, params: PatrolParams = None):
        if params is None:
            params = self.default_params
        (
            key,
            alt_key,
            hdg_key,
            turn_key,
            back_key,
            right_key,
            up_key,
            noise_key,
        ) = jax.random.split(key, 8)

        # Lead cruise setpoints.
        lead_alt = jax.random.uniform(
            alt_key,
            minval=params.target_altitude_range[0],
            maxval=params.target_altitude_range[1],
        )
        lead_heading = jax.random.uniform(
            hdg_key,
            minval=params.target_heading_range[0],
            maxval=params.target_heading_range[1],
        )
        # One turn rate per leg of the route, drawn once and cycled.
        lead_turn_schedule = jax.random.uniform(
            turn_key,
            shape=(N_LEAD_LEGS,),
            minval=params.lead_turn_rate_range[0],
            maxval=params.lead_turn_rate_range[1],
        )
        lead_turn_rate = lead_turn_schedule[0]

        # Slot geometry (constant over the episode).
        slot_back = jax.random.uniform(
            back_key,
            minval=params.slot_back_range[0],
            maxval=params.slot_back_range[1],
        )
        slot_right = jax.random.uniform(
            right_key,
            minval=params.slot_right_range[0],
            maxval=params.slot_right_range[1],
        )
        # On either side of the lead, at random: the range is the offset's size.
        side = jnp.where(jax.random.bernoulli(jax.random.fold_in(right_key, 1)), 1, -1)
        slot_right = side * slot_right
        slot_up = jax.random.uniform(
            up_key, minval=params.slot_up_range[0], maxval=params.slot_up_range[1]
        )

        # In a stream a reset is a sortie: each aircraft draws its mass
        # (long-strata S1b). A window keeps the scored configuration's.
        lead_mass, follower_mass = (
            in_stream(
                params,
                sortie_mass(jax.random.fold_in(sortie_key(key), seat), params),
                params.initial_mass,
            )
            for seat in (0, 1)
        )
        speed = lead_cruise_speed(lead_alt, params)
        lead = make_cruise_state(
            x=0.0,
            y=0.0,
            z=lead_alt,
            psi=lead_heading,
            speed=speed,
            params=params,
            target_altitude=lead_alt,
            target_heading=lead_heading,
            **_loaded(lead_mass, params),
        )

        # Provisional state to compute the world-frame slot position.
        provisional = PatrolState(
            follower=lead,
            lead=lead,
            lead_memory=self._lead_memory,
            wake_history=initial_wake(lead, params),
            slot_back=slot_back,
            slot_right=slot_right,
            slot_up=slot_up,
            lead_turn_rate=lead_turn_rate,
            lead_turn_schedule=lead_turn_schedule,
            time=0,
        )
        slot_x, slot_y, slot_z = desired_slot_position(provisional)

        # Spawn the follower at the slot + isotropic noise, matching lead
        # velocity/heading so the episode starts solvable but untrimmed.
        noise = params.follower_spawn_noise * jax.random.normal(noise_key, (3,))
        follower = make_cruise_state(
            x=slot_x + noise[0],
            y=slot_y + noise[1],
            z=slot_z + noise[2],
            psi=lead_heading,
            speed=speed,
            params=params,
            target_altitude=lead_alt,
            target_heading=lead_heading,
            **_loaded(follower_mass, params),
        )

        state = provisional.replace(follower=follower)
        return self.get_obs(state, params), state

    def start_event(self, state: PatrolState, key, params: PatrolParams):
        """The stream's event hook (docs/proposals/long-strata.md, S1b): a new
        sortie, the pair in cruise on a fresh route and station, each aircraft
        at a fresh mass, in the air the last sortie left (its gusts) and on the
        same clocks. The patrol has no other event."""
        fresh = self.reset_env(key, params)[1]
        return fresh.replace(
            time=state.time,
            sensor_key=state.sensor_key,
            gust_history=state.gust_history,
            gust_own=state.gust_own,
            lead_turn_rate=fresh.lead_turn_schedule[lead_leg(state.time, params)],
            lead=fresh.lead.replace(time=state.lead.time),
            follower=fresh.follower.replace(time=state.follower.time),
        )

    def restart_plant(self, key: chex.PRNGKey, state: PatrolState, params):
        """After a trip the scripted lead flies on (``OUTSIDE_WORLD``), so the
        follower re-forms on it: at its slot plus the spawn noise, at the
        lead's heading and cruise speed, with its clock running on so the
        turbulence it meets is the one it would have met (long-strata S2)."""
        fresh = self.reset_env(key, params)[1]
        noise = params.follower_spawn_noise * jax.random.normal(
            jax.random.fold_in(key, 1), (3,)
        )
        slot_x, slot_y, slot_z = desired_slot_position(state)
        lead = state.lead
        follower = make_cruise_state(
            x=slot_x + noise[0],
            y=slot_y + noise[1],
            z=slot_z + noise[2],
            psi=lead.psi,
            speed=lead_cruise_speed(lead.target_altitude, params),
            params=params,
            target_altitude=state.follower.target_altitude,
            target_heading=state.follower.target_heading,
            # In a stream the sortie's mass, which the burn never lowers.
            mass=in_stream(params, state.follower.m, params.initial_mass),
            fuel=in_stream(params, state.follower.fuel, params.initial_fuel_quantity),
        ).replace(time=state.follower.time)
        return canonical_dtypes(fresh.replace(follower=follower))

    # -- rendering / video -------------------------------------------------
    def render(self, screen, state: PatrolState, params: PatrolParams, frames, clock):
        return self.render_plane(screen, state, params, frames, clock)

    # -- expert ------------------------------------------------------------

    # -- spaces ------------------------------------------------------------
    def action_space(self, params: PatrolParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0, -1.0]),
            high=jnp.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: PatrolParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: PatrolParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(PatrolState.__dataclass_fields__), dtype=jnp.float32
        )


class PlanePatrol(_PlanePatrolBase):
    """Close patrol with full relative-state observation (26 values).

    Observation layout — see :func:`target_gym.patrol.env.get_obs_full`.
    Action (3,): [throttle, pitch demand, bank demand], each in [-1, 1].
    """

    obs_value_index: int = 24  # slot_error
    tracked_names: tuple = ("slot error (m)",)
    obs_target_index: int = 25  # constant 0
    task_type: str = "patrol"
    registry_name: str = "patrol"

    # PHYSICS.md section on readings.
    readings: tuple = (
        *_FOLLOWER,
        *_LEAD_FIX,
        *instruments(("z",), "lead."),
        *_LEAD_VELOCITY,
    )

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (26,)

    def _observe(self, state: PatrolState) -> jnp.ndarray:
        return get_obs_full(state, xp=jnp)


class PlanePatrolRangeBearing(_PlanePatrolBase):
    """Close patrol with partial (range + bearing) observation (21 values).

    Observation layout: see :func:`target_gym.patrol.env.get_obs_range_bearing`.
    Action (3,): [throttle, pitch demand, bank demand], each in [-1, 1].
    """

    obs_value_index: int = 19  # measured range
    tracked_names: tuple = ("measured range (m)",)
    obs_target_index: int = 20  # commanded slot range
    task_type: str = "patrol"
    registry_name: str = "patrol_range_bearing"

    # PHYSICS.md section on readings. Provisional, pending a real sensor.
    # Elevation is no channel of its own: it inherits ``lead.z``. The lead's
    # other readings reach only the measured reward. The range has no reading:
    # the wingman sees the lead, and from vision knows how far it is (the
    # owner, 2026-11-10), in place of TCAS's 15 m, set for slots 150 to 300 m out.
    readings: tuple = (
        *_FOLLOWER,
        *_LEAD_FIX,
        _LEAD_MODE_S_ALTITUDE,
        *_LEAD_VELOCITY,
        ChannelReading(11, sigma=0.026),  # azimuth, rad
    )

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (21,)

    def _observe(self, state: PatrolState) -> jnp.ndarray:
        return get_obs_range_bearing(state, xp=jnp)

    def get_obs(self, state: PatrolState, params: PatrolParams = None) -> jnp.ndarray:
        # The tracked value is that one range reading, not a second instrument.
        obs = super().get_obs(state, params)
        return obs.at[19].set(obs[10])


class WakeSurf(PlanePatrol):
    """``wake_surf``: wake-energy retrieval. An A320 follower holds a station
    3 km behind an A320 lead, in the lead's upwash, as Airbus's fello'fly did
    with two A350s (plane3d/PHYSICS.md, wake surfing).

    The observation is the patrol's 26 values. The cost is the station box
    and the follower's operator cost, fuel and time by the cost index: inside
    the box, the less thrust the lead's upwash leaves the follower needing, the
    less it pays. The vortex core lies a span inboard of the station, and a
    vortex that rolls the follower past 60 deg is an upset, a trip.

    Action (3,): [throttle, pitch demand, bank demand], each in [-1, 1].
    """

    registry_name: str = "wake_surf"

    @property
    def default_params(self) -> WakeSurfParams:
        # The scored configuration: 30 min of cruise in trail (ours: one renewal
        # window of a long-haul pairing).
        return WakeSurfParams(max_steps_in_episode=1800)

    def compute_reward(self, state, params):
        return compute_reward_wake_surf(state, params)

    def reward_terms(self, state, params, tripped=None):
        if tripped is not None:
            state = _read_angles(state, params)
        return compute_reward_terms_wake_surf(state, params, tripped=tripped)


def _loaded(mass, params):
    """``make_cruise_state``'s mass and fuel for an aircraft of all-up ``mass``:
    a stream's sortie, or the scored configuration in a window."""
    fuel = in_stream(params, fuel_for(mass, params), params.initial_fuel_quantity)
    return dict(mass=mass, fuel=fuel)
