import os
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def run_constant_policy_final_value(
    env,
    params,
    action: Union[float, Tuple[float, float]],
    state_attr: str,
    steps: int = 10_000,
    key_seed: int = 0,
):
    """
    Run a constant policy in a JAX environment and return the final value of a specified state attribute.
    Works safely with JAX traced arrays.
    """
    key = jax.random.PRNGKey(key_seed)
    init_obs, init_state = env.reset_env(key, params)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(
            key, state, action, params
        )
        truncated = new_state.time >= params.max_steps_in_episode
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, truncated)
        value = getattr(new_state, state_attr)
        last_value = (
            getattr(info["last_state"], state_attr) if "last_state" in info else value
        )
        return (key, state, done), (value, last_value, done)

    (_, final_state, done), (value_hist, last_value_hist, done_hist) = jax.lax.scan(
        step_fn, (key, init_state, False), None, length=steps
    )

    # Get index of first True in done_hist
    done_idx = jnp.argmax(done_hist)

    # Safely handle first step case using lax.cond
    final_value = jax.lax.cond(
        done_idx > 0,
        lambda idx: last_value_hist[idx - 1],
        lambda _: last_value_hist[-1],
        operand=done_idx,
    )

    return final_value


def run_input_grid(
    input_levels: jnp.ndarray,
    env,
    params,
    steps: int = 10_000,
    input_name: str = "input",
    second_input_levels: Optional[jnp.ndarray] = None,
    second_input_name: Optional[str] = None,
    state_attr: str = "velocity",
) -> Tuple[jnp.ndarray, pd.DataFrame]:
    """
    Runs a grid of constant inputs on an environment.

    Supports:
        - Single-input environments: CSTR, FirstOrder, etc.
        - Two-input environments: Plane

    Args:
        input_levels: 1D array of first input (throttle, T_c, power)
        env: JAX environment
        params: EnvParams
        steps: timesteps to run
        input_name: name for CSV column of first input
        second_input_levels: 1D array of second input (stick), optional
        second_input_name: CSV column name for second input, optional
        state_attr: which state attribute to track ("velocity", "T", "z", etc.)

    Returns:
        final_values: jnp array of final state_attr values
        df: pandas DataFrame with inputs and final values
    """
    if second_input_levels is None:
        # Single-input env
        def run_one_input(u):
            return run_constant_policy_final_value(
                env, params, action=u, state_attr=state_attr, steps=steps, key_seed=0
            )

        final_values = jax.vmap(run_one_input)(input_levels)
        df = pd.DataFrame({input_name: input_levels, "final_value": final_values})

    else:
        # Two-input env
        def run_one_first_input(u):
            return jax.vmap(
                lambda v: run_constant_policy_final_value(
                    env,
                    params,
                    action=(u, v),
                    state_attr=state_attr,
                    steps=steps,
                    key_seed=0,
                )
            )(second_input_levels)

        final_values = jax.vmap(run_one_first_input)(input_levels)

        # Flatten arrays for DataFrame
        df = pd.DataFrame(
            {
                input_name: jnp.repeat(input_levels, len(second_input_levels)),
                second_input_name: jnp.tile(second_input_levels, len(input_levels)),
                "final_value": final_values.flatten(),
            }
        )
    return final_values, df


# ---------------------------------------------------------------------------
# Shared video / GIF generation helpers
# ---------------------------------------------------------------------------


def generate_video(env, params, env_name: str, select_action, seed: int = 42):
    """Save a video + GIF for a given action policy."""
    from moviepy.video.io.VideoFileClip import VideoFileClip

    file = env.save_video(select_action, seed, params=params)
    os.makedirs(f"videos/{env_name}", exist_ok=True)
    video = VideoFileClip(file)
    gif_path = f"videos/{env_name}/output.gif"
    video.write_gif(gif_path, fps=30)
    return gif_path


def generate_pid_video(env, params, env_name: str, pid, seed: int = 42):
    """Save a PID-controlled video + GIF."""
    from moviepy.video.io.VideoFileClip import VideoFileClip

    def select_action(obs):
        return pid.step(obs)

    file = env.save_video(select_action, seed, params=params)
    os.makedirs(f"videos/{env_name}", exist_ok=True)
    video = VideoFileClip(file)
    gif_path = f"videos/{env_name}/pid_output.gif"
    video.write_gif(gif_path, fps=30)
    return gif_path


def generate_mpc_video(env, params, env_name: str, mpc, seed: int = 42):
    """Save an MPC-controlled video + GIF."""
    from moviepy.video.io.VideoFileClip import VideoFileClip

    def select_action(obs, state):
        return np.array([mpc.step(obs, state)])

    file = env.save_video(select_action, seed, params=params)
    os.makedirs(f"videos/{env_name}", exist_ok=True)
    video = VideoFileClip(file)
    gif_path = f"videos/{env_name}/mpc_output.gif"
    video.write_gif(gif_path, fps=30)
    return gif_path


# ---------------------------------------------------------------------------
# Shared constant-policy sweep + interpolator builder
# ---------------------------------------------------------------------------


def build_constant_sweep_interpolator(
    env,
    params,
    env_name: str,
    run_constant_fn: Callable,
    target_getter: Callable,
    action_range: Tuple[float, float] = (-1.0, 1.0),
    n_levels: int = 40,
    sweep_steps: Optional[int] = None,
):
    """Build an interpolator mapping target values -> best constant action.

    Args:
        run_constant_fn: (action, env, params, steps, seed) -> (final_val, trajectory)
        target_getter: (state) -> target_value
    """
    from scipy.interpolate import interp1d

    from target_gym.utils import load_or_build_interpolator

    action_levels = jnp.linspace(action_range[0], action_range[1], n_levels)
    steps = sweep_steps or int(params.max_steps_in_episode)

    def build_interp():
        final_vals = np.array(
            jax.vmap(
                lambda a: run_constant_fn(a, env, params, steps=steps, seed=0)[0]
                if callable(run_constant_fn)
                else run_constant_fn(a, env, params, steps=steps)
            )(action_levels)
        )
        a_np = np.array(action_levels)
        sort_idx = np.argsort(final_vals)
        return interp1d(
            final_vals[sort_idx],
            a_np[sort_idx],
            bounds_error=False,
            fill_value="extrapolate",
        )

    return load_or_build_interpolator(
        f"data/interpolators/{env_name}_comparison.pkl", build_interp
    )


# ---------------------------------------------------------------------------
# Shared comparison figure + GIF generation
# ---------------------------------------------------------------------------


def run_comparison(
    env,
    params,
    env_name: str,
    interpolator,
    pid,
    mpc,
    get_state_val: Callable,
    get_target_val: Callable,
    ylabel: str,
    action_clip: Tuple[float, float] = (-1.0, 1.0),
    n_seeds: int = 20,
    gif_seed: int = 0,
):
    """Run the comparison_gif mode: multi-seed figure + single-seed animated GIF.

    Shared across all SISO environments.
    """
    from target_gym.utils import save_comparison_figure, save_comparison_gif

    def build_const(seed):
        key = jax.random.PRNGKey(seed)
        _, st = env.reset_env(key, params)
        target = get_target_val(st)
        best_a = float(np.clip(interpolator(target), *action_clip))

        def action(_obs, _state=None, _a=best_a):
            return np.array([_a])

        return action

    def build_pid(seed):
        pid.reset()

        def action(obs, _state=None):
            return np.array([pid.step(obs)])

        return action

    def build_mpc(seed):
        mpc.reset()

        def action(obs, state):
            return np.array([mpc.step(obs, state)])

        return action

    os.makedirs(f"figures/{env_name}", exist_ok=True)
    save_comparison_figure(
        env=env,
        build_const_action=build_const,
        build_pid_action=build_pid,
        build_mpc_action=build_mpc,
        output_path=f"figures/{env_name}/comparison.png",
        get_state_val=get_state_val,
        get_target_val=get_target_val,
        ylabel=ylabel,
        const_label="Constant",
        pid_label="PID",
        mpc_cache_prefix=f"data/mpc_cache/{env_name}_mpc",
        params=params,
        n_seeds=n_seeds,
    )

    # Animated GIF for a single seed
    key = jax.random.PRNGKey(gif_seed)
    _, st = env.reset_env(key, params)
    best_a = float(np.clip(interpolator(get_target_val(st)), *action_clip))
    pid.reset()
    mpc.reset()
    os.makedirs(f"videos/{env_name}", exist_ok=True)
    save_comparison_gif(
        env=env,
        const_select_action=lambda _obs, _a=best_a: np.array([_a]),
        pid_select_action=lambda obs: np.array([pid.step(obs)]),
        mpc_select_action=lambda obs, state: np.array([mpc.step(obs, state)]),
        mpc_cache_path=f"data/mpc_cache/{env_name}_mpc_seed{gif_seed}.pkl",
        output_path=f"videos/{env_name}/comparison.gif",
        get_state_val=get_state_val,
        get_target_val=get_target_val,
        ylabel=ylabel,
        const_label=f"Constant a={best_a:.2f}",
        pid_label="PID",
        params=params,
        seed=gif_seed,
    )


def run_comparison_multi(
    env,
    params,
    env_name: str,
    interpolator,
    pid,
    mpc,
    get_state_val: Callable,
    get_target_val: Callable,
    ylabel: str,
    title_prefix: str,
    action_clip: Tuple[float, float] = (-1.0, 1.0),
    n_seeds: int = 20,
    plot: bool = True,
):
    """Run comparison_multi mode: bar chart of cumulative rewards across seeds.

    Shared across all SISO environments.
    """
    import matplotlib.pyplot as plt

    from target_gym.utils import load_or_run_mpc_episode, run_episode_headless_with_state

    cumulative_rewards = {"Constant": [], "PID": [], "MPC": []}

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
        key = jax.random.PRNGKey(seed)
        _, init_state = env.reset_env(key, params)
        best_a = float(np.clip(interpolator(get_target_val(init_state)), *action_clip))

        def const_action(_obs, _state=None, _a=best_a):
            return np.array([_a])

        def pid_action(obs, _state=None):
            return np.array([pid.step(obs)])

        def mpc_action(obs, state):
            return np.array([mpc.step(obs, state)])

        _, rews_c = run_episode_headless_with_state(env, const_action, params, seed)
        pid.reset()
        _, rews_p = run_episode_headless_with_state(env, pid_action, params, seed)
        mpc.reset()
        _, rews_m = load_or_run_mpc_episode(
            f"data/mpc_cache/{env_name}_mpc_seed{seed}.pkl",
            env,
            mpc_action,
            params,
            seed,
        )
        cumulative_rewards["Constant"].append(sum(rews_c))
        cumulative_rewards["PID"].append(sum(rews_p))
        cumulative_rewards["MPC"].append(sum(rews_m))
        print(
            f"const={sum(rews_c):.1f}  pid={sum(rews_p):.1f}  mpc={sum(rews_m):.1f}"
        )

    if plot:
        labels = ["Constant", "PID", "MPC"]
        colors_bar = ["steelblue", "darkorange", "seagreen"]
        means = [np.mean(cumulative_rewards[label]) for label in labels]
        stds = [np.std(cumulative_rewards[label]) for label in labels]
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(labels))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=6,
            color=colors_bar,
            alpha=0.8,
            error_kw={"linewidth": 1.5},
        )
        rng = np.random.default_rng(0)
        for i, label in enumerate(labels):
            jitter = rng.uniform(-0.15, 0.15, size=n_seeds)
            ax.scatter(
                x[i] + jitter,
                cumulative_rewards[label],
                color="black",
                s=20,
                alpha=0.5,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Cumulative reward")
        ax.set_title(
            f"{title_prefix}: cumulative reward over {n_seeds} seeds (mean +/- std)"
        )
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(
                xi,
                m + s + max(stds) * 0.05,
                f"{m:.1f}+/-{s:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        os.makedirs(f"figures/{env_name}", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"figures/{env_name}/comparison_multi.pdf")
        plt.savefig(f"figures/{env_name}/comparison_multi.png")
        plt.close()
    return cumulative_rewards
