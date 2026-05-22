import inspect
import os
import pickle
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flax.serialization import to_state_dict
from matplotlib.backends.backend_agg import FigureCanvasAgg
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

EnvState = Any


def compute_norm_from_coordinates(coordinates: jnp.ndarray) -> float:
    """Compute the norm of a vector given its coordinates"""
    return jnp.linalg.norm(coordinates, axis=0)


def norm2(x, y) -> float:
    """Lean 2D norm: sqrt(x*x + y*y).

    Avoids the (abs, square, sum, sqrt) chain that ``jnp.linalg.norm`` lowers
    to via the general norm path.  Used in the plane hot loop.
    """
    return jnp.sqrt(x * x + y * y)


def norm3(x, y, z) -> float:
    """Lean 3D norm: sqrt(x*x + y*y + z*z)."""
    return jnp.sqrt(x * x + y * y + z * z)


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


# ---------------------------------------------------------------------------
# Headless episode runner + comparison GIF
# ---------------------------------------------------------------------------


def run_episode_headless(env, select_action, params, seed: int = 42):
    """
    Run one episode using step_env (no rendering, no gymnax auto-reset).
    Returns: (states, rewards) — one entry per timestep.
    """
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    states, rewards = [], []
    done = False
    while not done:
        action = jnp.asarray(select_action(obs))
        obs, state, reward, terminated, info = env.step_env(key, state, action, params)
        states.append(state)
        rewards.append(float(reward))
        truncated = bool(state.time >= params.max_steps_in_episode)
        done = bool(terminated) | truncated
    return states, rewards


def run_episode_headless_with_state(env, select_action, params, seed: int = 42):
    """
    Like run_episode_headless but select_action receives (obs, state).
    Used for MPC which needs the full JAX state for its rollout.
    """
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    states, rewards = [], []
    done = False
    while not done:
        action = jnp.asarray(select_action(obs, state))
        obs, state, reward, terminated, info = env.step_env(key, state, action, params)
        states.append(state)
        rewards.append(float(reward))
        truncated = bool(state.time >= params.max_steps_in_episode)
        done = bool(terminated) | truncated
    return states, rewards


def load_or_run_mpc_episode(
    cache_path: str, env, mpc_select_action, params, seed: int = 42
):
    """
    Load a cached MPC episode trajectory from disk, or run the MPC episode and
    save it.  The cache stores ``(states, rewards)`` as a pickle.

    This avoids re-running the expensive MPC optimisation on every GIF render.
    Delete the cache file to force a fresh run.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        _, rewards = result
        if np.all(np.isfinite(rewards)):
            return result
        print(f"Cached MPC episode at {cache_path} contains NaN/Inf — recomputing.")
        os.remove(cache_path)
    print(f"Running MPC episode (will be cached at {cache_path}) …")
    result = run_episode_headless_with_state(env, mpc_select_action, params, seed)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"MPC episode cached to {cache_path}")
    return result


def save_comparison_gif(
    env,
    const_select_action,
    pid_select_action,
    output_path: str,
    get_state_val,
    get_target_val,
    ylabel: str,
    const_label: str = "Constant action",
    pid_label: str = "PID",
    mpc_select_action=None,
    mpc_label: str = "MPC",
    mpc_cache_path: str = None,
    params=None,
    seed: int = 42,
    ylim=None,
    target_duration: float = 10.0,
    freeze_duration: float = 2.0,
):
    """
    Run constant-action, PID, and optionally MPC episodes headlessly, then render
    a comparison GIF:
      - Left panel  : state trajectories of all policies on the same axes (growing over time)
      - Right panel : live cumulative-reward horizontal bar chart

    ``const_select_action`` and ``pid_select_action`` receive ``obs`` only.
    ``mpc_select_action`` (optional) receives ``(obs, state)`` so the MPC can
    use the full JAX state for its internal rollout.

    If ``mpc_cache_path`` is provided the MPC episode is loaded from disk on
    subsequent calls instead of being re-computed (expensive).  Delete the file
    to force a fresh run.

    GIF duration is normalised to ``target_duration`` seconds regardless of
    episode length, with a ``freeze_duration``-second hold on the last frame.
    """
    if params is None:
        params = env.default_params

    states_c, rews_c = run_episode_headless(env, const_select_action, params, seed)
    states_p, rews_p = run_episode_headless(env, pid_select_action, params, seed)

    agents = [
        (states_c, rews_c, const_label, "steelblue"),
        (states_p, rews_p, pid_label, "darkorange"),
    ]

    if mpc_select_action is not None:
        if mpc_cache_path is not None:
            states_m, rews_m = load_or_run_mpc_episode(
                mpc_cache_path, env, mpc_select_action, params, seed
            )
        else:
            states_m, rews_m = run_episode_headless_with_state(
                env, mpc_select_action, params, seed
            )
        agents.append((states_m, rews_m, mpc_label, "seagreen"))

    n_total = max(len(s) for s, _, _, _ in agents)

    # Subsample to keep GIF ≤ 25 fps and fill target_duration
    max_fps = 25
    stride = max(1, int(np.ceil(n_total / (target_duration * max_fps))))
    indices = list(range(0, n_total, stride))
    fps = len(indices) / target_duration

    agent_vals = [
        np.array([float(get_state_val(s)) for s in states]) for states, *_ in agents
    ]
    target_val = (
        float(get_target_val(agents[0][0][0])) if get_target_val is not None else None
    )
    agent_cums = [np.cumsum(rews) for _, rews, _, _ in agents]
    max_rew = max(float(cum[-1]) for cum in agent_cums) * 1.1 or 1.0

    # y-axis range
    all_vals = np.concatenate(agent_vals)
    if ylim is None:
        pad = (all_vals.max() - all_vals.min()) * 0.15 or 0.1
        ylim = (float(all_vals.min()) - pad, float(all_vals.max()) + pad)

    labels = [label for _, _, label, _ in agents]
    colors_list = [color for _, _, _, color in agents]

    frames = []
    for t in indices:
        n_agents = len(agents)
        ts = [min(t + 1, len(agents[i][0])) for i in range(n_agents)]

        fig, (ax_t, ax_b) = plt.subplots(
            1,
            2,
            figsize=(12, 4.5),
            gridspec_kw={"width_ratios": [2, 1]},
        )

        # Trajectory panel
        for i, (vals, label, color) in enumerate(zip(agent_vals, labels, colors_list)):
            ax_t.plot(range(ts[i]), vals[: ts[i]], color=color, lw=2, label=label)
        if target_val is not None:
            ax_t.axhline(
                target_val, color="red", ls="--", lw=1.5, alpha=0.8, label="Target"
            )
        ax_t.set_xlim(0, n_total)
        ax_t.set_ylim(*ylim)
        ax_t.set_xlabel("Time step")
        ax_t.set_ylabel(ylabel)
        ax_t.legend(loc="best", fontsize=9)
        ax_t.grid(alpha=0.3)
        ax_t.set_title("State trajectory")

        # Cumulative reward bar chart
        cum_vals = [
            float(agent_cums[i][ts[i] - 1]) if ts[i] > 0 else 0.0
            for i in range(n_agents)
        ]
        bars = ax_b.barh(
            labels,
            cum_vals,
            color=colors_list,
            height=0.5,
        )
        ax_b.set_xlim(0, max_rew)
        ax_b.set_xlabel("Cumulative reward")
        ax_b.set_title("Cumulative Reward")
        ax_b.grid(alpha=0.3, axis="x")
        for bar, val in zip(bars, cum_vals):
            ax_b.text(
                min(bar.get_width() + max_rew * 0.02, max_rew * 0.92),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
            ..., :3
        ]
        plt.close(fig)
        frames.append(frame)

    # Freeze on last frame
    frames.extend([frames[-1]] * int(freeze_duration * fps))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ImageSequenceClip(frames, fps=fps).write_gif(output_path, fps=fps)
    print(f"Saved comparison GIF to {output_path}")
    return output_path


def save_comparison_figure(
    env,
    build_const_action,
    build_pid_action,
    build_mpc_action,
    output_path: str,
    get_state_val,
    get_target_val,
    ylabel: str,
    const_label: str = "Constant",
    pid_label: str = "PID",
    mpc_label: str = "MPC",
    mpc_cache_prefix: str = None,
    params=None,
    n_seeds: int = 20,
    ylim=None,
):
    """
    Multi-seed comparison figure: mean trajectory ± std (left) and
    mean cumulative reward ± std bar chart (right).

    Unlike ``save_comparison_gif`` this produces a static PNG/PDF and
    averages over ``n_seeds`` episodes for statistical robustness.

    Callbacks
    ---------
    build_const_action(seed) -> select_action(obs, state)
        Returns the constant-policy closure adapted to the seed's target.
    build_pid_action(seed) -> select_action(obs, state)
        Returns the PID closure (should reset the controller internally).
    build_mpc_action(seed) -> select_action(obs, state)
        Returns the MPC closure (should reset the controller internally).
    """
    if params is None:
        params = env.default_params

    n_steps = int(params.max_steps_in_episode)
    agents_cfg = [
        ("Constant", const_label, "steelblue", build_const_action, False),
        ("PID", pid_label, "darkorange", build_pid_action, False),
        ("MPC", mpc_label, "seagreen", build_mpc_action, True),
    ]

    # Collect per-seed trajectories and rewards
    all_trajs = {name: [] for name, *_ in agents_cfg}
    all_cum_rews = {name: [] for name, *_ in agents_cfg}

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
        per_seed = {}

        for name, _, _, build_fn, use_mpc_cache in agents_cfg:
            action_fn = build_fn(seed)
            if use_mpc_cache and mpc_cache_prefix is not None:
                cache_path = f"{mpc_cache_prefix}_seed{seed}.pkl"
                states, rews = load_or_run_mpc_episode(
                    cache_path, env, action_fn, params, seed
                )
            else:
                states, rews = run_episode_headless_with_state(
                    env, action_fn, params, seed
                )
            vals = np.array([float(get_state_val(s)) for s in states])
            # Pad to n_steps if episode terminated early
            if len(vals) < n_steps:
                vals = np.concatenate([vals, np.full(n_steps - len(vals), vals[-1])])
                rews = list(rews) + [0.0] * (n_steps - len(rews))
            all_trajs[name].append(vals[:n_steps])
            all_cum_rews[name].append(sum(rews))
            per_seed[name] = sum(rews)

        print("  ".join(f"{k}={v:.1f}" for k, v in per_seed.items()))

    # Compute statistics
    t = np.arange(n_steps)
    labels = [label for _, label, _, _, _ in agents_cfg]
    colors_list = [color for _, _, color, _, _ in agents_cfg]
    names = [name for name, *_ in agents_cfg]

    fig, (ax_t, ax_b) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Left: trajectory mean ± std
    for name, label, color in zip(names, labels, colors_list):
        arr = np.array(all_trajs[name])  # (n_seeds, n_steps)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax_t.plot(t, mean, color=color, lw=2, label=label)
        ax_t.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

    if get_target_val is not None:
        # Show mean target across seeds (might vary per seed)
        target_vals = []
        for seed in range(n_seeds):
            key = jax.random.PRNGKey(seed)
            _, init_state = env.reset_env(key, params)
            target_vals.append(float(get_target_val(init_state)))
        mean_target = np.mean(target_vals)
        std_target = np.std(target_vals)
        ax_t.axhline(
            mean_target, color="red", ls="--", lw=1.5, alpha=0.8, label="Target (mean)"
        )
        ax_t.axhspan(
            mean_target - std_target, mean_target + std_target, color="red", alpha=0.07
        )

    ax_t.set_xlim(0, n_steps)
    if ylim is not None:
        ax_t.set_ylim(*ylim)
    ax_t.set_xlabel("Time step")
    ax_t.set_ylabel(ylabel)
    ax_t.legend(loc="best", fontsize=9)
    ax_t.grid(alpha=0.3)
    ax_t.set_title(f"State trajectory (mean ± std, {n_seeds} seeds)")

    # Right: cumulative reward bar chart
    means = [np.mean(all_cum_rews[n]) for n in names]
    stds = [np.std(all_cum_rews[n]) for n in names]
    x = np.arange(len(labels))
    ax_b.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors_list,
        alpha=0.8,
        error_kw={"linewidth": 1.5},
    )
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        jitter = rng.uniform(-0.15, 0.15, size=n_seeds)
        ax_b.scatter(
            x[i] + jitter, all_cum_rews[name], color="black", s=20, alpha=0.5, zorder=3
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Cumulative reward")
    ax_b.set_title("Cumulative Reward")
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax_b.text(
            xi,
            m + s + max(stds) * 0.05,
            f"{m:.1f}±{s:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    base = output_path.rsplit(".", 1)[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(f"{base}.pdf")
    plt.savefig(f"{base}.png")
    plt.close()
    print(f"Saved comparison figure to {base}.png")
    return all_cum_rews


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


def load_or_build_interpolator(cache_path: str, build_fn: Callable):
    """Load a cached scipy interpolator from disk, or build and save it."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    interp = build_fn()
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(interp, f)
    return interp


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Return a truncated colormap from minval to maxval."""
    new_cmap = cm.colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
