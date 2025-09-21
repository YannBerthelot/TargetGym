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


def compute_episode_returns_vectorized(rewards: jnp.ndarray, dones: jnp.ndarray):
    """
    Compute per-episode cumulative reward, handling both terminated and truncated episodes.

    Args:
        rewards: (T,) float array of rewards per timestep
        dones:   (T,) int/boolean array, 1 if episode ends at that step

    Returns:
        episode_returns: (num_episodes,) array of total return per episode
    """
    # Cumulative rewards
    cumsum_rewards = jnp.cumsum(rewards)

    # Rewards at episode boundaries (where done=1)
    final_returns = cumsum_rewards[dones.astype(bool)]

    # Previous boundaries (prepend 0 for the very first episode)
    prev_final = jnp.concatenate([jnp.array([0.0]), final_returns[:-1]])

    # Proper per-episode returns
    episode_returns = final_returns - prev_final

    # --- Handle truncated last episode ---
    last_is_done = dones[-1] > 0
    if not last_is_done:
        # Add the return from last boundary (or start) up to end
        last_return = cumsum_rewards[-1] - (
            final_returns[-1] if final_returns.size > 0 else 0.0
        )
        episode_returns = jnp.concatenate([episode_returns, jnp.array([last_return])])

    return episode_returns


def run_n_steps(env, policy, params, n_steps=10_000, seed=0):
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)

    def step_fn(carry, _):
        obs, state, key = carry
        key, subkey = jax.random.split(key)

        action = policy(obs)
        obs, new_state, reward, done, _ = env.step_env(subkey, state, action, params)

        carry = (
            jax.lax.cond(
                done,
                lambda _: env.reset_env(key, params),
                lambda _: (obs, new_state),
                operand=None,
            )[
                0
            ],  # new obs
            jax.lax.cond(
                done,
                lambda _: env.reset_env(key, params),
                lambda _: (obs, new_state),
                operand=None,
            )[
                1
            ],  # new state
            key,
        )

        # If episode ended, mark this reward as last in episode
        ep_done = done.astype(jnp.float32)
        return carry, (reward, ep_done)

    # Scan for n_steps
    (_, _, _), (rewards, ep_dones) = jax.lax.scan(
        step_fn, (obs, state, key), None, n_steps
    )
    valid_returns = compute_episode_returns_vectorized(rewards, ep_dones)
    mean_return = jnp.mean(valid_returns)
    return mean_return


def convert_raw_action_to_range(raw_action, min_action, max_action):
    """
    Assuming the action is roughly in (-1,1), we rescale to it (min_action,max_action).
    """
    action = min_action + 0.5 * (jnp.clip(raw_action, -1, 1) + 1) * (
        max_action - min_action
    )
    return action
