"""
JAX-compatible 3D airplane environments.

Three task variants sharing the same 3D physics:
  - Plane3DHeading:  reach and follow a target heading at a target altitude
  - Plane3DCircle:   fly a circle of given radius at a target altitude
  - Plane3DFigureEight: fly a figure-8 (lemniscate) at a target altitude

Actions: power (throttle), stick (elevator), aileron (roll control).
"""

from typing import Callable

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.plane3d.env import (
    PlaneParams3D,
    PlaneState3D,
    check_is_terminal_3d,
    compute_next_state_3d,
    compute_reward_circle,
    compute_reward_figure8,
    compute_reward_heading,
    get_obs_circle,
    get_obs_figure8,
    get_obs_heading,
)
from target_gym.plane3d.rendering import _render
from target_gym.utils import compute_norm_from_coordinates, save_video


class _Airplane3DBase(environment.Environment[PlaneState3D, PlaneParams3D]):
    """
    Base class for all 3D airplane environments.

    Subclasses must set:
      - obs_shape
      - obs_value_index, obs_target_index
    and implement:
      - compute_reward(state, params)
      - get_obs(state)
      - _reset_targets(key, params) -> dict of target fields
    """

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400

    def __init__(self, integration_method: str = "rk4_1"):
        self.positions_history_xz = []
        self.positions_history_xy = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> PlaneParams3D:
        return PlaneParams3D()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PlaneState3D,
        action: jnp.ndarray,
        params: PlaneParams3D = None,
    ):
        if params is None:
            params = self.default_params

        power, stick, aileron = action
        power = (power + 1) / 2  # [-1, 1] -> [0, 1]
        stick = jnp.deg2rad(stick * 15)  # [-1, 1] -> +/-15 deg
        aileron = jnp.deg2rad(aileron * 25)  # [-1, 1] -> +/-25 deg

        new_state, metrics = compute_next_state_3d(
            power,
            stick,
            aileron,
            state,
            params,
            integration_method=self.integration_method,
        )
        reward = self.compute_reward(new_state, params)
        terminated, truncated = check_is_terminal_3d(new_state, params, xp=jnp)
        done = terminated | truncated

        obs = self.get_obs(new_state, params)
        return (
            obs,
            new_state,
            reward,
            done,
            {"metrics": metrics, "last_state": new_state},
        )

    def is_terminal(self, state: PlaneState3D, params: PlaneParams3D) -> jax.Array:
        return check_is_terminal_3d(state, params, xp=jnp)

    def _reset_common(self, key: chex.PRNGKey, params: PlaneParams3D):
        """Compute shared initial state fields. Returns (remaining_key, base_kwargs)."""
        if params is None:
            params = self.default_params
        key, altitude_key, target_altitude_key = jax.random.split(key, 3)

        initial_z = jax.random.uniform(
            altitude_key,
            minval=params.initial_altitude_range[0],
            maxval=params.initial_altitude_range[1],
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

        target_altitude = jax.random.uniform(
            target_altitude_key,
            minval=params.target_altitude_range[0],
            maxval=params.target_altitude_range[1],
        )

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
            m=params.initial_mass + params.initial_fuel_quantity,
            power=params.initial_power,
            stick=jnp.deg2rad(params.initial_stick),
            aileron=jnp.deg2rad(params.initial_aileron),
            fuel=params.initial_fuel_quantity,
            time=0,
            target_altitude=target_altitude,
        )
        return key, base_kwargs

    def render(self, screen, state: PlaneState3D, params: PlaneParams3D, frames, clock):
        frames, screen, clock = self.render_plane(screen, state, params, frames, clock)
        return frames, screen, clock

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
            -inf, inf, len(PlaneState3D.__class_params__), dtype=jnp.float32
        )


# ─── Heading task ──────────────────────────────────────


class Plane3DHeading(_Airplane3DBase):
    """
    3D airplane: reach and follow a target heading at a target altitude.

    Observation (15,):
        [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
         gamma, psi, target_altitude, target_heading, power, stick, aileron]

    Action (3,): [power, stick, aileron] each in [-1, 1]
    """

    obs_value_index: int = 2  # z (altitude)
    obs_target_index: int = 10  # target_altitude
    task_type: str = "heading"

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (15,)

    def compute_reward(self, state, params):
        return compute_reward_heading(state, params)

    def get_obs(self, state: PlaneState3D, params: PlaneParams3D = None):
        return get_obs_heading(state, xp=jnp)

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

        state = PlaneState3D(
            **base_kwargs,
            target_heading=target_heading,
            target_x=0.0,
            target_y=0.0,
            target_radius=0.0,
        )
        return self.get_obs(state), state

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_plane3d_heading_pid,
            plane3d_heading_pid_step,
        )

        params, zero_state = make_plane3d_heading_pid()
        return FunctionalExpertPolicy(params, zero_state, plane3d_heading_pid_step)


# ─── Circle task ───────────────────────────────────────


class Plane3DCircle(_Airplane3DBase):
    """
    3D airplane: fly a circle of given radius at a target altitude.

    The aircraft starts on the circle with heading tangent to it.

    Observation (17,):
        [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
         gamma, psi, target_altitude, rel_x, rel_y, target_radius,
         power, stick, aileron]

    Action (3,): [power, stick, aileron] each in [-1, 1]
    """

    obs_value_index: int = 2
    obs_target_index: int = 10
    task_type: str = "circle"

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (17,)

    def compute_reward(self, state, params):
        return compute_reward_circle(state, params)

    def get_obs(self, state: PlaneState3D, params: PlaneParams3D = None):
        return get_obs_circle(state, xp=jnp)

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
        return self.get_obs(state), state

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_plane3d_circle_pid,
            plane3d_circle_pid_step,
        )

        params, zero_state = make_plane3d_circle_pid()
        return FunctionalExpertPolicy(params, zero_state, plane3d_circle_pid_step)


# ─── Figure-8 task ─────────────────────────────────────


class Plane3DFigureEight(_Airplane3DBase):
    """
    3D airplane: fly a figure-8 (lemniscate of Bernoulli) at a target altitude.

    The aircraft starts at the rightmost point of the right lobe.

    Observation (19,):
        [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
         gamma, psi, target_altitude, target_radius,
         nearest_dx, nearest_dy, nearest_dz, tangent_heading,
         power, stick, aileron]

    Action (3,): [power, stick, aileron] each in [-1, 1]
    """

    obs_value_index: int = 2
    obs_target_index: int = 10
    task_type: str = "figure8"

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (19,)

    def compute_reward(self, state, params):
        return compute_reward_figure8(state, params)

    def get_obs(self, state: PlaneState3D, params: PlaneParams3D = None):
        if params is None:
            params = self.default_params
        return get_obs_figure8(state, params, xp=jnp)

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
        return self.get_obs(state), state

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_plane3d_figure8_pid,
            plane3d_figure8_pid_step,
        )

        params, zero_state = make_plane3d_figure8_pid()
        return FunctionalExpertPolicy(params, zero_state, plane3d_figure8_pid_step)


# ─── Backward-compatible alias ─────────────────────────

Airplane3D = Plane3DHeading
