import os
import time

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from target_gym import FirstOrderParams, FirstOrderSystem
from target_gym.experts.mpc import make_first_order_mpc
from target_gym.experts.pid import (
    make_first_order_pid,
    make_first_order_stateful_gs_pid,
    pid_step,
)
from target_gym.utils import (
    load_or_build_interpolator,
    load_or_run_mpc_episode,
    run_episode_headless_with_state,
    save_comparison_figure,
    save_comparison_gif,
    truncate_colormap,
)

env_name = "first_order"


def run_constant_policy(
    u: float,
    env: FirstOrderSystem,
    params: FirstOrderParams,
    steps: int = 200,
    seed: int = 0,
):
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(key, state, u, params)
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (state.x, done)

    (_, final_state, _), (x_history, _done_history) = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state.x, x_history


def run_pid_for_target(
    target_val,
    env: FirstOrderSystem,
    params: FirstOrderParams,
    pid_params,
    pid_state0,
    steps: int,
):
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    state = state.replace(target_x=target_val)

    def step_fn(carry, _):
        env_state, pid_state, done = carry
        obs = jnp.array([env_state.x, env_state.target_x])
        action, new_pid_state = pid_step(pid_params, pid_state, obs)
        _, new_env_state, _, new_done, _ = env.step_env(
            key, env_state, action[0], params
        )
        env_state = jax.lax.cond(
            done, lambda _: env_state, lambda _: new_env_state, operand=None
        )
        done = jnp.logical_or(done, new_done)
        return (env_state, new_pid_state, done), env_state.x

    _, x_history = jax.lax.scan(step_fn, (state, pid_state0, False), None, length=steps)
    return x_history


def run_mode(
    mode: str,
    u: float = 0.5,
    n_timesteps: int = 200,
    plot: bool = True,
    save: bool = True,
    resolution: int = 20,
    **kwargs,
):
    env = FirstOrderSystem(integration_method="rk4_1")
    n_seeds = kwargs.pop("n_seeds", 20)
    params = FirstOrderParams(**kwargs) if kwargs else env.default_params

    if mode == "2d":
        start_time = time.time()
        u_levels = jnp.linspace(-1.0, 1.0, (resolution * 2) + 1)

        _, trajectories = jax.vmap(
            lambda u_val: run_constant_policy(u_val, env, params, steps=n_timesteps)
        )(u_levels)

        elapsed = time.time() - start_time
        print(f"Ran in {elapsed:.3f}s ({elapsed / len(u_levels):.3f}s per run)")

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(vmin=u_levels.min(), vmax=u_levels.max())
            cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
            for i, traj in enumerate(trajectories):
                ax.plot(traj, color=cmap(norm(u_levels[i])))
                if i % 4 == 0:
                    ax.text(
                        x=len(traj) - 1,
                        y=float(traj[-1]),
                        s=f" {float(u_levels[i]):.2f} → {float(traj[-1]):.2f}",
                        color=cmap(norm(u_levels[i])),
                        fontsize=8,
                        va="center",
                        ha="left",
                    )
            # Mark the target (center of target_x_range)
            target_mid = 0.5 * (params.target_x_range[0] + params.target_x_range[1])
            ax.axhline(
                target_mid,
                color="red",
                ls="--",
                lw=1,
                alpha=0.7,
                label=f"target ≈ {target_mid:.2f}",
            )
            ax.axhline(params.x_max, color="gray", ls=":", lw=1, alpha=0.5)
            ax.axhline(params.x_min, color="gray", ls=":", lw=1, alpha=0.5)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            xmin, xmax = plt.xlim()
            plt.xlim(xmin, xmax + (xmax - xmin) * 0.12)
            ax.set_xlabel("Time step")
            ax.set_ylabel("x")
            ax.set_title("State trajectories for varying control inputs (u)")
            ax.legend(fontsize=8)
            os.makedirs(f"figures/{env_name}", exist_ok=True)
            plt.savefig(f"figures/{env_name}/u_trajectories.pdf")
            plt.savefig(f"figures/{env_name}/u_trajectories.png")
            plt.close()

    elif mode == "pid":
        pid_params, pid_state0 = make_first_order_pid()
        targets = jnp.linspace(
            params.target_x_range[0], params.target_x_range[1], resolution
        )

        start_time = time.time()
        trajectories = jax.vmap(
            lambda t: run_pid_for_target(
                t, env, params, pid_params, pid_state0, n_timesteps
            )
        )(targets)
        elapsed = time.time() - start_time
        print(f"PID ran in {elapsed:.3f}s")

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(vmin=float(targets[0]), vmax=float(targets[-1]))
            cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
            final_vals = [float(traj[-1]) for traj in trajectories]
            data_min, data_max = min(final_vals + [float(trajectories.min())]), max(
                final_vals + [float(trajectories.max())]
            )
            data_range = data_max - data_min or 0.01
            min_label_gap = data_range * 0.08
            last_label_y = -np.inf
            for i, traj in enumerate(trajectories):
                c = cmap(norm(float(targets[i])))
                ax.plot(traj, color=c, alpha=0.85)
                y_end = float(traj[-1])
                if y_end - last_label_y >= min_label_gap:
                    ax.text(
                        x=len(traj) - 1,
                        y=y_end,
                        s=f" {float(targets[i]):.2f} → {y_end:.2f}",
                        color=c,
                        fontsize=8,
                        va="center",
                        ha="left",
                    )
                    last_label_y = y_end
            margin = data_range * 0.1
            ax.set_ylim(data_min - margin, data_max + margin)
            xmin, xmax = plt.xlim()
            plt.xlim(xmin, xmax + (xmax - xmin) * 0.15)
            ax.set_xlabel("Time step")
            ax.set_ylabel("x")
            ax.set_title("PID response for varying setpoints")
            os.makedirs(f"figures/{env_name}", exist_ok=True)
            plt.savefig(f"figures/{env_name}/pid_response.pdf")
            plt.savefig(f"figures/{env_name}/pid_response.png")
            plt.close()

    elif mode == "video":
        seed = 42

        def select_action(_):
            return np.array([u])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/output.gif", fps=30)

    elif mode == "pid_video":
        seed = 42
        pid = make_first_order_stateful_gs_pid()

        def select_action(obs):
            return np.array([pid.step(obs)])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/pid_output.gif", fps=30)

    elif mode == "mpc_video":
        seed = 42
        mpc = make_first_order_mpc(env, params)
        mpc.reset()

        def select_action(obs, state):
            return np.array([mpc.step(obs, state)])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/mpc_output.gif", fps=30)

    elif mode == "comparison_gif":
        u_levels = jnp.linspace(-1.0, 1.0, 40)

        def build_interp():
            final_xs = np.array(
                jax.vmap(
                    lambda u_val: run_constant_policy(
                        u_val, env, params, steps=n_timesteps, seed=0
                    )[0]
                )(u_levels)
            )
            u_np = np.array(u_levels)
            sort_idx = np.argsort(final_xs)
            return interp1d(
                final_xs[sort_idx],
                u_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            f"data/interpolators/{env_name}_comparison.pkl", build_interp
        )
        pid = make_first_order_stateful_gs_pid()
        mpc = make_first_order_mpc(env, params)

        def build_const(seed):
            key = jax.random.PRNGKey(seed)
            _, st = env.reset_env(key, params)
            bu = float(np.clip(interpolator(float(st.target_x)), -1.0, 1.0))

            def action(_obs, _state=None, _u=bu):
                return np.array([_u])

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
            get_state_val=lambda s: float(s.x),
            get_target_val=lambda s: float(s.target_x),
            ylabel="x",
            const_label="Constant",
            pid_label="PID",
            mpc_cache_prefix=f"data/mpc_cache/{env_name}_mpc",
            params=params,
            n_seeds=n_seeds,
        )

        # Also produce animated GIF for seed 0
        gif_seed = 0
        key = jax.random.PRNGKey(gif_seed)
        _, st = env.reset_env(key, params)
        gif_u = float(np.clip(interpolator(float(st.target_x)), -1.0, 1.0))
        pid.reset()
        mpc.reset()
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        save_comparison_gif(
            env=env,
            const_select_action=lambda _obs: np.array([gif_u]),
            pid_select_action=lambda obs: np.array([pid.step(obs)]),
            mpc_select_action=lambda obs, state: np.array([mpc.step(obs, state)]),
            mpc_cache_path=f"data/mpc_cache/{env_name}_mpc_seed{gif_seed}.pkl",
            output_path=f"videos/{env_name}/comparison.gif",
            get_state_val=lambda s: float(s.x),
            get_target_val=lambda s: float(s.target_x),
            ylabel="x",
            const_label=f"Constant u={gif_u:.2f}",
            pid_label="PID",
            params=params,
            seed=gif_seed,
        )

    elif mode == "comparison_multi":
        u_levels = jnp.linspace(-1.0, 1.0, 40)

        def build_interp():
            final_xs = np.array(
                jax.vmap(
                    lambda u_val: run_constant_policy(
                        u_val, env, params, steps=n_timesteps, seed=0
                    )[0]
                )(u_levels)
            )
            u_np = np.array(u_levels)
            sort_idx = np.argsort(final_xs)
            return interp1d(
                final_xs[sort_idx],
                u_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            f"data/interpolators/{env_name}_comparison.pkl", build_interp
        )
        pid = make_first_order_stateful_gs_pid()
        mpc = make_first_order_mpc(env, params)
        cumulative_rewards = {"Constant": [], "PID": [], "MPC": []}

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
            key = jax.random.PRNGKey(seed)
            _, init_state = env.reset_env(key, params)
            best_u = float(np.clip(interpolator(float(init_state.target_x)), -1.0, 1.0))

            def const_action(_obs, _state=None, _u=best_u):
                return np.array([_u])

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
                f"FirstOrder: cumulative reward over {n_seeds} seeds (mean ± std)"
            )
            for xi, (m, s) in enumerate(zip(means, stds)):
                ax.text(
                    xi,
                    m + s + max(stds) * 0.05,
                    f"{m:.1f}±{s:.1f}",
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

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_videos():
    run_mode("video", u=0.5)
    run_mode("pid_video")
    run_mode("mpc_video")


def run_figures():
    n_steps = 200  # matches env default max_steps_in_episode=200
    run_mode("2d", n_timesteps=n_steps, resolution=20)
    run_mode("pid", n_timesteps=n_steps, resolution=10)
    run_mode("comparison_gif", u=0.5)


def run_all_modes():
    run_figures()
    run_videos()


if __name__ == "__main__":
    run_all_modes()
