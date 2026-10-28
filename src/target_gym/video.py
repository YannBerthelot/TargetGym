"""Recording an episode to a video: ``save_video`` and the ``SavesVideo`` mixin.

Out of ``utils.py`` and the environment packages on purpose: those are hashed
into the provenance fingerprints, and how an episode is filmed cannot change
what it scores. moviepy, which writes the file, is the ``[render]`` extra, so
it is imported only when a video is written.
"""

import inspect
import os
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import to_state_dict


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
        select_action: callable(obs) -> action, or callable(obs, state) -> action.
            Two-arg callables receive the full env state, which is needed by
            controllers (e.g. MPC) that depend on hidden fields like setpoint
            schedules or internal counters.
        folder: folder to save the video
        episode_index: index for the filename
        FPS: frames per second
        params: optional environment parameters
        seed: optional seed for environment reset
        format: output format, "mp4" or "gif"
    Returns:
        Path to the saved video.
    """
    # Detect whether the caller wants state threaded through.
    try:
        sig = inspect.signature(select_action)
        needs_state = (
            len(
                [
                    p
                    for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty
                ]
            )
            >= 2
        )
    except (TypeError, ValueError):
        needs_state = False
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

    # Use step_env (no auto-reset) when available to avoid gymnax's auto-reset
    # connecting terminal and initial states in the rendered frames (triangle artifact).
    use_step_env = hasattr(env, "step_env") and hasattr(env, "default_params")

    while not done:
        action = select_action(obs, state) if needs_state else select_action(obs)
        if use_step_env:
            action = jnp.asarray(action)
            obs, state, reward, terminated, info = env.step_env(
                key, state, action, params
            )
        elif hasattr(env, "default_params"):
            obs, state, reward, terminated, info = env.step(key, state, action, params)
        else:
            obs, state, reward, terminated, info = env.step(state, action, params)
        states.append(to_state_dict(state))
        rewards += reward
        if params is None and hasattr(env, "default_params"):
            params = env.default_params
        truncated = state.time >= params.max_steps_in_episode
        done = bool(terminated) | bool(truncated)

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

    # Imported here: moviepy is the ``[render]`` extra.
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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
        import pandas as pd

        pd.DataFrame(states).to_csv("trajectory.csv")
    return video_path


class SavesVideo:
    """``env.save_video(select_action, seed)``, the same on every environment.

    Runs one episode with ``select_action`` and writes it with
    :func:`save_video`; see there for the arguments.
    """

    def save_video(
        self,
        select_action: Callable,
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


def convert_frames_from_gym_to_wandb(frames: list) -> np.ndarray:
    """Convert frames from gym format (time, width, height, channel) to wandb format (time, channel, height, width)"""
    return np.array(frames).swapaxes(1, 3).swapaxes(2, 3)
