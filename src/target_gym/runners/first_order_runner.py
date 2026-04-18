import os
import time

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from target_gym import FirstOrderParams, FirstOrderSystem
from target_gym.experts.mpc import make_first_order_mpc
from target_gym.experts.pid import (
    make_first_order_pid,
    make_first_order_stateful_gs_pid,
    pid_step,
)
from target_gym.runners.utils import (
    generate_mpc_video,
    generate_pid_video,
    generate_video,
    run_comparison,
    run_comparison_multi,
)
from target_gym.utils import truncate_colormap

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


def _get_interpolator(env, params, n_timesteps):
    from scipy.interpolate import interp1d

    from target_gym.utils import load_or_build_interpolator

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

    return load_or_build_interpolator(
        f"data/interpolators/{env_name}_comparison.pkl", build_interp
    )


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
                        s=f" {float(u_levels[i]):.2f} -> {float(traj[-1]):.2f}",
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
                label=f"target = {target_mid:.2f}",
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
                        s=f" {float(targets[i]):.2f} -> {y_end:.2f}",
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
        generate_video(env, params, env_name, lambda _: np.array([u]))

    elif mode == "pid_video":
        pid = make_first_order_stateful_gs_pid()
        generate_pid_video(env, params, env_name, pid)

    elif mode == "mpc_video":
        mpc = make_first_order_mpc(env, params)
        mpc.reset()
        generate_mpc_video(env, params, env_name, mpc)

    elif mode == "comparison_gif":
        interpolator = _get_interpolator(env, params, n_timesteps)
        pid = make_first_order_stateful_gs_pid()
        mpc = make_first_order_mpc(env, params)
        run_comparison(
            env, params, env_name, interpolator, pid, mpc,
            get_state_val=lambda s: float(s.x),
            get_target_val=lambda s: float(s.target_x),
            ylabel="x",
            n_seeds=n_seeds,
        )

    elif mode == "comparison_multi":
        interpolator = _get_interpolator(env, params, n_timesteps)
        pid = make_first_order_stateful_gs_pid()
        mpc = make_first_order_mpc(env, params)
        return run_comparison_multi(
            env, params, env_name, interpolator, pid, mpc,
            get_state_val=lambda s: float(s.x),
            get_target_val=lambda s: float(s.target_x),
            ylabel="x",
            title_prefix="FirstOrder",
            n_seeds=n_seeds,
            plot=plot,
        )

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
