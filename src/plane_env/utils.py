import os
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flax.serialization import to_state_dict
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

EnvState = Any


def compute_norm_from_coordinates(coordinates: jnp.ndarray) -> float:
    """Compute the norm of a vector given its coordinates"""
    return jnp.linalg.norm(coordinates, axis=0)


def plot_curve(data, name, folder="figs"):
    fig, ax = plt.subplots()
    ax.plot(data)
    title = f"{name} vs time"
    plt.title(f"{name} vs time")
    plt.savefig(os.path.join(folder, title))
    plt.close()


def plot_features_from_trajectory(states: Sequence[EnvState], folder: str):
    for feature_name in states[0].__dataclass_fields__.keys():
        if "__dataclass_fields__" in dir(states[0].__dict__[feature_name]):
            plot_features_from_trajectory(
                [state.__dict__[feature_name] for state in states], folder
            )
        else:
            feature_values = [state.__dict__[feature_name] for state in states]
            plot_curve(feature_values, feature_name, folder=folder)


def list_to_array(list):
    cls = type(list[0])
    return cls(
        **{
            k: jnp.array([getattr(v, k) for v in list])
            for k in cls.__dataclass_fields__
        }
    )


def array_to_list(array):
    cls = type(array)
    size = len(getattr(array, cls._fields[0]))
    return [
        cls(**{k: v(getattr(array, k)[i]) for k, v in cls._field_types.items()})
        for i in range(size)
    ]


def convert_frames_from_gym_to_wandb(frames: list) -> np.ndarray:
    """Convert frames from gym format (time, width, height, channel) to wandb format (time, channel, height, width)"""
    return np.array(frames).swapaxes(1, 3).swapaxes(2, 3)


def save_video(
    env,
    select_action: Callable,
    folder: str = "videos",
    episode_index: int = 0,
    FPS: int = 60,
    params=None,
    seed: int = None,
    format: str = "mp4",  # "mp4" or "gif"
    save_trajectory: bool = False,
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
    rewards = 0
    states = []

    while not done:
        action = select_action(obs)
        step_result = (
            env.step(key, obs if state is None else state, action, params)
            if hasattr(env, "default_params")
            else env.step(state, action, params)
        )
        obs, state, reward, terminated, info = step_result
        states.append(to_state_dict(state))
        rewards += reward
        if params is None and hasattr(env, "default_params"):
            params = env.default_params
        truncated = state.t >= params.max_steps_in_episode
        done = terminated | truncated

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
    print(f"total rewards: {rewards}")
    if save_trajectory:
        pd.DataFrame(states).to_csv("trajectory.csv")
    return video_path


from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def integrate_state(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    accel_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dt: float,
    method: str = "rk_4",
    euler_substeps: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    General JAX-compatible integrator for positions and velocities with state-dependent acceleration.

    Args:
        pos: array of shape (n,), current positions
        vel: array of shape (n,), current velocities
        accel_fn: function(pos, vel) -> accelerations (shape (n,))
        dt: timestep
        method: 'euler_1', 'euler_10', 'rk1', 'rk4', 'rkN'
        euler_substeps: default substeps for Euler
        rk_steps: number of RK stages (for generalized RK)

    Returns:
        new_pos, new_vel
    """
    xp = jnp

    # ---------------------
    # Euler integration via lax.scan
    # ---------------------
    steps = int(method.split("_")[1])
    dt_sub = dt / steps
    if method.startswith("euler"):
        # parse number of substeps

        def euler_step(carry, _):
            pos_curr, vel_curr = carry
            a = accel_fn(pos_curr, vel_curr)
            vel_new = vel_curr + a * dt_sub
            pos_new = pos_curr + vel_new * dt_sub
            return (pos_new, vel_new), None

        (pos_final, vel_final), _ = jax.lax.scan(
            euler_step, (pos, vel), None, length=steps
        )
        return pos_final, vel_final

    # ---------------------
    # RK1 (simple Euler)
    # ---------------------
    elif method == "rk1":
        a = accel_fn(pos, vel)
        vel_new = vel + a * dt
        pos_new = pos + vel_new * dt
        return pos_new, vel_new

    # ---------------------
    # Generalized RK integration
    # ---------------------
    elif method.startswith("rk"):
        pos_stage = pos
        vel_stage = vel

        # For multiple RK stages, do repeated mid-point evaluations
        def rk_step(carry, _):
            pos_curr, vel_curr = carry

            k1_vel = accel_fn(pos_curr, vel_curr)
            k1_pos = vel_curr

            k2_vel = accel_fn(
                pos_curr + 0.5 * dt_sub * k1_pos, vel_curr + 0.5 * dt_sub * k1_vel
            )
            k2_pos = vel_curr + 0.5 * dt_sub * k1_vel

            k3_vel = accel_fn(
                pos_curr + 0.5 * dt_sub * k2_pos, vel_curr + 0.5 * dt_sub * k2_vel
            )
            k3_pos = vel_curr + 0.5 * dt_sub * k2_vel

            k4_vel = accel_fn(pos_curr + dt_sub * k3_pos, vel_curr + dt_sub * k3_vel)
            k4_pos = vel_curr + dt_sub * k3_vel

            vel_new = vel_curr + (dt_sub / 6.0) * (
                k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel
            )
            pos_new = pos_curr + (dt_sub / 6.0) * (
                k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos
            )
            return (pos_new, vel_new), None

        (pos_final, vel_final), _ = jax.lax.scan(
            rk_step, (pos_stage, vel_stage), None, length=steps
        )
        return pos_final, vel_final

    else:
        raise ValueError(f"Unknown integration method: {method}")
