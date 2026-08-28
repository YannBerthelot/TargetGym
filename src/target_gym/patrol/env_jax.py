"""
JAX-compatible close-patrol (formation-keeping) environments.

Two task variants share the same two-aircraft physics and reward; they differ
only in what the follower observes:

  - :class:`PlanePatrol`            — full relative state (position + velocity
                                       of the lead in its body frame);
  - :class:`PlanePatrolBearingOnly` — partial: range + bearing to the lead only.

The lead is scripted (a tuned 3D heading autopilot flying straight-and-level or
a gentle orbit); the follower is the learning agent.  See :mod:`.env` for the
transition, slot geometry, reward and observation functions.
"""

from typing import Callable

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.experts.pid import make_plane3d_heading_pid
from target_gym.patrol.env import (
    PatrolParams,
    PatrolState,
    check_is_terminal_patrol,
    compute_next_state_patrol,
    compute_reward_patrol,
    desired_slot_position,
    get_obs_bearing_only,
    get_obs_full,
    make_cruise_state,
)
from target_gym.patrol.rendering import _render
from target_gym.utils import save_video


class _PlanePatrolBase(environment.Environment[PatrolState, PatrolParams]):
    """Shared machinery for the close-patrol tasks.

    Subclasses set ``obs_shape``, ``obs_value_index``, ``obs_target_index`` and
    implement :meth:`get_obs`.
    """

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400
    task_type: str = "patrol"

    def __init__(self, integration_method: str = "rk4_1"):
        self.integration_method = integration_method
        self.positions_history_xz = []
        self.positions_history_xy = []
        # Static gains for the scripted lead's heading autopilot.
        self._lead_pid_params, self._lead_pid_zero = make_plane3d_heading_pid()

    @property
    def default_params(self) -> PatrolParams:
        return PatrolParams()

    # -- to be provided by subclasses --------------------------------------
    def get_obs(self, state: PatrolState) -> jnp.ndarray:
        raise NotImplementedError

    # -- core --------------------------------------------------------------
    def compute_reward(self, state, params):
        return compute_reward_patrol(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PatrolState,
        action: jnp.ndarray,
        params: PatrolParams = None,
    ):
        if params is None:
            params = self.default_params

        new_state, metrics = compute_next_state_patrol(
            action,
            state,
            params,
            self._lead_pid_params,
            integration_method=self.integration_method,
            key=key,
        )
        reward = self.compute_reward(new_state, params)
        # gymnax >= 1.0 owns truncation: ``step_env`` reports natural
        # termination only, and the base ``Environment.step`` derives
        # ``truncated`` from ``state.time >= params.max_steps_in_episode``
        # -- the very condition ``check_is_terminal`` returns second.
        terminated, _ = check_is_terminal_patrol(new_state, params, xp=jnp)
        obs = self.get_obs(new_state)
        return (
            obs,
            new_state,
            reward,
            terminated,
            {"metrics": metrics, "last_state": new_state},
        )

    def is_terminated(self, state: PatrolState, params: PatrolParams) -> jax.Array:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal_patrol(state, params, xp=jnp)
        return terminated

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
        lead_turn_rate = jax.random.uniform(
            turn_key,
            minval=params.lead_turn_rate_range[0],
            maxval=params.lead_turn_rate_range[1],
        )

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
        slot_up = jax.random.uniform(
            up_key, minval=params.slot_up_range[0], maxval=params.slot_up_range[1]
        )

        speed = params.initial_x_dot
        lead = make_cruise_state(
            x=0.0,
            y=0.0,
            z=lead_alt,
            psi=lead_heading,
            speed=speed,
            params=params,
            target_altitude=lead_alt,
            target_heading=lead_heading,
        )

        # Provisional state to compute the world-frame slot position.
        provisional = PatrolState(
            follower=lead,
            lead=lead,
            lead_pid=self._lead_pid_zero,
            slot_back=slot_back,
            slot_right=slot_right,
            slot_up=slot_up,
            lead_turn_rate=lead_turn_rate,
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
        )

        state = provisional.replace(follower=follower)
        return self.get_obs(state), state

    # -- rendering / video -------------------------------------------------
    def render(self, screen, state: PatrolState, params: PatrolParams, frames, clock):
        return self.render_plane(screen, state, params, frames, clock)

    def save_video(
        self,
        select_action: Callable[[jnp.ndarray], jnp.ndarray],
        seed: int,
        params=None,
        folder="videos",
        episode_index=0,
        FPS=60,
        format="mp4",
        save_trajectory: bool = False,
    ):
        return save_video(
            self,
            select_action,
            folder,
            episode_index,
            FPS,
            params,
            seed=seed,
            format=format,
            save_trajectory=save_trajectory,
        )

    # -- expert ------------------------------------------------------------
    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_patrol_pid,
            patrol_pid_step,
        )

        params, zero_state = make_patrol_pid()
        return FunctionalExpertPolicy(params, zero_state, patrol_pid_step)

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
        return spaces.Box(-inf, inf, (1,), dtype=jnp.float32)


class PlanePatrol(_PlanePatrolBase):
    """Close patrol with full relative-state observation (26 values).

    Observation layout — see :func:`target_gym.patrol.env.get_obs_full`.
    Action (3,): [power, stick, aileron], each in [-1, 1].
    """

    obs_value_index: int = 24  # slot_error
    obs_target_index: int = 25  # constant 0
    task_type: str = "patrol"

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (26,)

    def get_obs(self, state: PatrolState) -> jnp.ndarray:
        return get_obs_full(state, xp=jnp)


class PlanePatrolBearingOnly(_PlanePatrolBase):
    """Close patrol with partial (range + bearing) observation (21 values).

    Observation layout — see :func:`target_gym.patrol.env.get_obs_bearing_only`.
    Action (3,): [power, stick, aileron], each in [-1, 1].
    """

    obs_value_index: int = 19  # measured range
    obs_target_index: int = 20  # commanded slot range
    task_type: str = "patrol"

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (21,)

    def get_obs(self, state: PatrolState) -> jnp.ndarray:
        return get_obs_bearing_only(state, xp=jnp)
