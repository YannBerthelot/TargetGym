import os
from typing import Callable, Tuple

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# Optional: jax imports (only used in jax env)
try:
    import chex
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
    chex = None

from plane_env.dynamics import (
    compute_acceleration,
    compute_air_density_from_altitude,
    compute_next_power,
    compute_next_stick,
    compute_speed_and_pos_from_acceleration,
    compute_thrust_output,
)
from plane_env.utils import compute_norm_from_coordinates

SPEED_OF_SOUND = 343.0


def get_env_classes(use_jax: bool = False) -> Tuple[type, type, type]:
    """
    Returns EnvState, EnvParams, EnvMetrics classes with either standard dataclass
    or flax.struct.dataclass depending on use_jax.
    """
    if use_jax:
        from flax import struct

        dataclass = struct.dataclass
        xp = jnp  # placeholder if you want to pass to properties
    else:
        from dataclasses import dataclass

        dataclass = dataclass
        xp = np

    @dataclass
    class EnvMetrics:
        drag: float
        lift: float
        S_x: float
        S_z: float
        C_x: float
        C_z: float
        F_x: float
        F_z: float

    @dataclass
    class EnvState:
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
        t: int
        target_altitude: float

        @property
        def rho(self):
            return compute_air_density_from_altitude(self.z)

        @property
        def speed_of_sound(self):
            h = self.z
            gamma_air = 1.4
            R = 287.0
            T0 = 288.15
            L = 0.0065
            T11 = 216.65
            T = xp.where(h <= 11000, T0 - L * h, T11)
            return xp.sqrt(gamma_air * R * T)

        @property
        def M(self):
            return (
                compute_norm_from_coordinates(xp.array([self.x_dot, self.z_dot]))
                / self.speed_of_sound
            )

    @dataclass
    class EnvParams:
        gravity: float = 9.81
        initial_mass: float = 73500.0
        thrust_output_at_sea_level: float = 240_000.0
        air_density_at_sea_level: float = 1.225
        frontal_surface: float = 12.6
        wings_surface: float = 122.6
        C_x0: float = 0.095
        C_z0: float = 0.9
        M_crit: float = 0.78
        initial_fuel_quantity: float = 23860 / 1.25
        specific_fuel_consumption: float = 17.5 / 1000
        delta_t: float = 0.5
        speed_of_sound: float = SPEED_OF_SOUND

        max_steps_in_episode: int = 1_000
        min_alt: float = 0.0
        max_alt: float = 40_000.0 / 3.281
        target_altitude_range: Tuple[float, float] = (3000.0, 5000.0)
        initial_altitude_range: Tuple[float, float] = (3000.0, 5000.0)
        initial_z_dot: float = 0.0
        initial_x_dot: float = 200.0
        initial_theta_dot: float = 0.0
        initial_theta: float = 0.0
        initial_power: float = 1.0
        initial_stick: float = 0.0

    return EnvState, EnvParams, EnvMetrics


EnvStateGymnasium, EnvParamsGymnasium, EnvMetricsGymnasium = get_env_classes(
    use_jax=False
)
EnvStateJAX, EnvParamsJAX, EnvMetricsJAX = get_env_classes(use_jax=True)
EnvState = EnvStateGymnasium | EnvStateJAX
EnvParams = EnvParamsGymnasium | EnvParamsJAX
EnvMetrics = EnvMetricsGymnasium | EnvMetricsJAX


def check_mass_does_not_increase(old_mass, new_mass, xp=np):
    """Check that mass does not increase. Safe for JIT if wrapped in callback."""
    if jax is not None and xp is jnp:
        jax.debug.callback(
            lambda o, n: None if o >= n else AssertionError("Mass increased"),
            old_mass,
            new_mass,
        )
    else:
        assert old_mass >= new_mass


def check_is_terminal(state: EnvState, params: EnvParams, xp=np):
    """Return True if the episode should terminate."""
    terminated = xp.logical_or(state.z < params.min_alt, state.z > params.max_alt)
    truncated = state.t >= params.max_steps_in_episode

    # done = xp.logical_or(done_alt, done_steps)
    return terminated, truncated


def compute_reward(state: EnvState, params: EnvParams, xp=np):
    """Return reward for a given state. Safe for JIT."""
    done_alt = xp.logical_or(state.z <= params.min_alt, state.z >= params.max_alt)
    max_alt_diff = params.max_alt - params.min_alt
    reward = xp.where(
        done_alt,
        -1.0 * params.max_steps_in_episode,
        ((max_alt_diff - xp.abs(state.target_altitude - state.z)) / max_alt_diff) ** 2,
    )
    return reward


def compute_next_state(
    power_requested: float,
    stick_requested: float,
    state: EnvState,
    params: EnvParams,
    xp=np,
):
    """Compute next state and metrics."""
    EnvState = EnvStateJAX if xp is jnp else EnvStateGymnasium
    power = compute_next_power(power_requested, state.power, params.delta_t)
    stick = compute_next_stick(stick_requested, state.stick, params.delta_t)

    thrust = compute_thrust_output(
        power=power,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
        rho=state.rho,
        M=state.M,
    )

    a_x, a_z, alpha_y, metrics = compute_acceleration(
        thrust=thrust,
        m=state.m,
        gravity=params.gravity,
        x_dot=state.x_dot,
        z_dot=state.z_dot,
        frontal_surface=params.frontal_surface,
        wings_surface=params.wings_surface,
        alpha=state.alpha,
        M=state.M,
        M_crit=params.M_crit,
        C_x0=params.C_x0,
        C_z0=params.C_z0,
        gamma=state.gamma,
        theta=state.theta,
        rho=state.rho,
        stick=stick,
    )

    x_dot, z_dot, theta_dot, x, z, theta = compute_speed_and_pos_from_acceleration(
        V_x=state.x_dot,
        V_z=state.z_dot,
        theta_dot=state.theta_dot,
        x=state.x,
        z=state.z,
        theta=state.theta,
        a_x=a_x,
        a_z=a_z,
        alpha_y=alpha_y,
        delta_t=params.delta_t,
    )

    t = state.t + 1
    gamma = xp.arcsin(
        z_dot / (compute_norm_from_coordinates(xp.array([x_dot, z_dot + 1e-6])))
    )
    alpha = theta - gamma
    m = params.initial_mass + state.fuel
    check_mass_does_not_increase(state.m, m, xp=xp)

    new_state = EnvState(
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
        fuel=state.fuel,
        t=t,
        target_altitude=state.target_altitude,
    )
    return new_state, metrics


def save_video(
    env,
    select_action: Callable,
    folder: str = "videos",
    episode_index: int = 0,
    FPS: int = 60,
    params=None,
    seed: int = None,
    format: str = "mp4",  # "mp4" or "gif"
):
    """
    Runs an episode using `select_action` and saves it as a video (mp4 or gif).
    Works for both JAX and Gymnasium environments.

    Arguments:
        env: the environment instance with methods `reset`, `step`, and `render`
        select_action: callable(obs) -> action
        folder: folder to save the video
        episode_index: index for the filename
        FPS: frames per second
        params: optional environment parameters
        seed: optional seed for environment reset
        format: output format, "mp4" or "gif"
    Returns:
        Path to the saved video.
    """
    if seed is not None:
        key = jax.random.PRNGKey(seed=seed)
        obs_state = (
            env.reset(seed=seed)
            if not hasattr(env, "default_params")
            else env.reset(key=key, params=env.default_params)
        )
    else:
        key = jax.random.PRNGKey(seed=42)
        obs_state = (
            env.reset()
            if not hasattr(env, "default_params")
            else env.reset(key=key, params=env.default_params)
        )

    if isinstance(obs_state, tuple) and len(obs_state) == 2:
        obs, state = obs_state
    else:
        obs = obs_state
        state = None

    done = False
    frames = []
    screen = None
    clock = None

    while not done:
        action = select_action(obs)
        step_result = (
            env.step(key, obs if state is None else state, action, params)
            if hasattr(env, "default_params")
            else env.step(state, action, params)
        )
        obs, state, reward, terminated, truncated, info = step_result
        done = terminated or truncated

        if hasattr(env, "render"):
            if hasattr(env, "default_params"):
                frames, screen, clock = env.render(
                    screen,
                    state,
                    params if params is not None else env.default_params,
                    frames,
                    clock,
                )
            else:
                frames.append(env.render())

    if len(frames) == 0:
        raise ValueError("No frames captured. Check that rendering is working.")

    os.makedirs(folder, exist_ok=True)
    video_path = os.path.join(folder, f"episode_{episode_index:03d}.{format}")

    frames_np = [np.asarray(frame).astype(np.uint8) for frame in frames]
    clip = ImageSequenceClip(frames_np, fps=FPS)

    if format == "mp4":
        clip.write_videofile(video_path, codec="libx264", audio=False)
    elif format == "gif":
        clip.write_gif(video_path, fps=30)
    else:
        raise ValueError("Unsupported format. Use 'mp4' or 'gif'.")

    print(f"Saved video to {video_path}")
    return video_path
