"""Runner for the Reactor (point-kinetics + thermal feedback) environment.

Supports:
  - "2d":             sweep constant rod reactivities, plot neutron-density trajectories
  - "temp":           same sweep, plot the fuel-temperature trajectories (hidden state)
  - "pid":            sweep PID tracking for varying power setpoints
  - "video":          save a video/gif with a user-provided constant rod action
  - "pid_video":      save a video/gif with the tuned PID policy
  - "mpc_video":      save a video/gif with the do-mpc policy
  - "comparison_gif": constant vs PID vs MPC comparison (multi-seed figure + GIF)
"""

import os
import time

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from target_gym import Reactor, ReactorParams
from target_gym.experts.mpc import make_reactor_mpc
from target_gym.experts.pid import (
    make_reactor_pid,
    make_reactor_stateful_gs_pid,
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

env_name = "reactor"


def run_constant_policy(
    rho_raw: float,
    env: Reactor,
    params: ReactorParams,
    steps: int = 86400,
    seed: int = 0,
):
    """Roll out a constant raw rod action, return n and T_fuel histories."""
    key = jax.random.PRNGKey(seed)
    _, state = env.reset_env(key, params)

    def step_fn(carry, _):
        key, state, done = carry
        _, new_state, _, new_done, _ = env.step_env(key, state, rho_raw, params)
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (state.n, state.T_fuel, done)

    (_, final_state, _), (n_hist, T_fuel_hist, _) = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state, n_hist, T_fuel_hist


def run_constant_policy_final_n(
    rho_raw: float,
    env: Reactor,
    params: ReactorParams,
    steps: int,
    seed: int = 0,
):
    """Return the final neutron density for a constant raw rod action."""
    final_state, _, _ = run_constant_policy(
        rho_raw, env, params, steps=steps, seed=seed
    )
    return final_state.n


def run_pid_for_target(
    target_val,
    env: Reactor,
    params: ReactorParams,
    pid_params,
    pid_state0,
    steps: int,
):
    key = jax.random.PRNGKey(0)
    # Freeze the OU demand so this shows single-setpoint tracking.
    fixed_params = params.replace(demand_sigma=0.0, demand_theta=0.0)
    _, state = env.reset_env(key, fixed_params)
    state = state.replace(target_n=target_val)

    def step_fn(carry, _):
        env_state, pid_state, done = carry
        obs = env.get_obs(env_state, fixed_params)
        action, new_pid_state = pid_step(pid_params, pid_state, obs)
        _, new_env_state, _, new_done, _ = env.step_env(
            key, env_state, action[0], fixed_params
        )
        env_state = jax.lax.cond(
            done, lambda _: env_state, lambda _: new_env_state, operand=None
        )
        done = jnp.logical_or(done, new_done)
        return (env_state, new_pid_state, done), env_state.n

    _, n_history = jax.lax.scan(step_fn, (state, pid_state0, False), None, length=steps)
    return n_history


def _plot_sweep(
    hist,
    rho_levels,
    title,
    ylabel,
    save_name,
    t_seconds,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = colors.Normalize(vmin=float(rho_levels.min()), vmax=float(rho_levels.max()))
    cmap = truncate_colormap(cm.viridis, 0.0, 0.85)

    for i, traj in enumerate(hist):
        ax.plot(t_seconds, traj, color=cmap(norm(float(rho_levels[i]))))
        if (i % 3) == 0:
            ax.text(
                x=t_seconds[-1],
                y=float(traj[-1]),
                s=f" raw={float(rho_levels[i]):+.2f} → {float(traj[-1]):.2f}",
                color=cmap(norm(float(rho_levels[i]))),
                fontsize=8,
                va="center",
                ha="left",
            )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, xmax + (xmax - xmin) * 0.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    os.makedirs(f"figures/{env_name}", exist_ok=True)
    plt.savefig(f"figures/{env_name}/{save_name}.pdf")
    plt.savefig(f"figures/{env_name}/{save_name}.png")
    plt.close()


def run_mode(
    mode: str,
    rho_raw: float = 0.0,
    n_timesteps: int = 86400,
    plot: bool = True,
    save: bool = True,
    resolution: int = 10,
    **kwargs,
):
    env = Reactor(integration_method="tr_bdf2_2")
    n_seeds = kwargs.pop("n_seeds", 20)
    params = ReactorParams(**kwargs) if kwargs else env.default_params

    if mode in ("2d", "temp"):
        start_time = time.time()
        rho_levels = jnp.linspace(-1.0, 1.0, (resolution * 2) + 1)

        def run_vmapped(levels):
            return jax.vmap(
                lambda r: run_constant_policy(r, env, params, steps=n_timesteps)
            )(levels)

        _, n_hist, T_fuel_hist = run_vmapped(rho_levels)
        elapsed = time.time() - start_time
        print(
            f"Ran {len(rho_levels)} rollouts x {n_timesteps} steps "
            f"in {elapsed:.3f}s ({elapsed / len(rho_levels):.3f}s per run)"
        )

        if plot:
            t_seconds = np.arange(n_timesteps) * params.delta_t
            if mode == "2d":
                _plot_sweep(
                    n_hist,
                    rho_levels,
                    title="Neutron density trajectories for varying constant rod reactivity",
                    ylabel="n (normalised power)",
                    save_name="trajectories_n",
                    t_seconds=t_seconds,
                )
            else:  # "temp"
                _plot_sweep(
                    T_fuel_hist,
                    rho_levels,
                    title="Fuel temperature trajectories (hidden state)",
                    ylabel="T_fuel (K)",
                    save_name="trajectories_T_fuel",
                    t_seconds=t_seconds,
                )

    elif mode == "pid":
        pid_params, pid_state0 = make_reactor_pid()
        targets = jnp.linspace(
            params.target_n_range[0],
            params.target_n_range[1],
            resolution,
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
            t_seconds = np.arange(n_timesteps) * params.delta_t
            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(vmin=float(targets[0]), vmax=float(targets[-1]))
            cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
            for i, traj in enumerate(trajectories):
                c = cmap(norm(float(targets[i])))
                ax.plot(t_seconds, traj, color=c, alpha=0.9)
                ax.axhline(float(targets[i]), color=c, ls="--", lw=0.6, alpha=0.4)
                ax.text(
                    x=t_seconds[-1],
                    y=float(traj[-1]),
                    s=f" target={float(targets[i]):.2f} → {float(traj[-1]):.2f}",
                    color=c,
                    fontsize=8,
                    va="center",
                    ha="left",
                )
            xmin, xmax = plt.xlim()
            plt.xlim(xmin, xmax + (xmax - xmin) * 0.2)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("n (normalised power)")
            ax.set_title("PID response for varying power setpoints")
            os.makedirs(f"figures/{env_name}", exist_ok=True)
            plt.savefig(f"figures/{env_name}/pid_response.pdf")
            plt.savefig(f"figures/{env_name}/pid_response.png")
            plt.close()

    elif mode == "video":
        seed = 42

        def select_action(_):
            return np.array([rho_raw])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/output.gif", fps=30)

    elif mode == "pid_video":
        seed = 42
        pid = make_reactor_stateful_gs_pid()

        def select_action(obs):
            return np.array([pid.step(obs)])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/pid_output.gif", fps=30)

    elif mode == "mpc_video":
        seed = 42
        mpc = make_reactor_mpc(env, params)
        mpc.reset()

        def select_action(obs, state):
            return np.array([mpc.step(obs, state)])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/mpc_output.gif", fps=30)

    elif mode == "comparison_gif":
        # Interpolator: target_n -> best constant raw rod action.
        rho_levels = jnp.linspace(-1.0, 1.0, 40)
        sweep_steps = min(n_timesteps, 500)

        def build_interp():
            final_ns = np.array(
                jax.vmap(
                    lambda r: run_constant_policy_final_n(
                        r, env, params, steps=sweep_steps
                    )
                )(rho_levels)
            )
            r_np = np.array(rho_levels)
            sort_idx = np.argsort(final_ns)
            return interp1d(
                final_ns[sort_idx],
                r_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            f"data/interpolators/{env_name}_comparison.pkl", build_interp
        )
        pid = make_reactor_stateful_gs_pid()
        mpc = make_reactor_mpc(env, params)

        def build_const(seed):
            key = jax.random.PRNGKey(seed)
            _, st = env.reset_env(key, params)
            brho = float(np.clip(interpolator(float(st.target_n)), -1.0, 1.0))

            def action(_obs, _state=None, _r=brho):
                return np.array([_r])

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
            get_state_val=lambda s: float(s.n),
            get_target_val=lambda s: float(s.target_n),
            ylabel="n (normalised power)",
            const_label="Constant",
            pid_label="PID",
            mpc_cache_prefix=f"data/mpc_cache/{env_name}_mpc",
            params=params,
            n_seeds=n_seeds,
        )

        gif_seed = 0
        key = jax.random.PRNGKey(gif_seed)
        _, st = env.reset_env(key, params)
        gif_rho = float(np.clip(interpolator(float(st.target_n)), -1.0, 1.0))
        pid.reset()
        mpc.reset()
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        save_comparison_gif(
            env=env,
            const_select_action=lambda _obs: np.array([gif_rho]),
            pid_select_action=lambda obs: np.array([pid.step(obs)]),
            mpc_select_action=lambda obs, state: np.array([mpc.step(obs, state)]),
            mpc_cache_path=f"data/mpc_cache/{env_name}_mpc_seed{gif_seed}.pkl",
            output_path=f"videos/{env_name}/comparison.gif",
            get_state_val=lambda s: float(s.n),
            get_target_val=lambda s: float(s.target_n),
            ylabel="n (normalised power)",
            const_label=f"Constant rho_raw={gif_rho:+.2f}",
            pid_label="PID",
            params=params,
            seed=gif_seed,
        )

    elif mode == "comparison_multi":
        rho_levels = jnp.linspace(-1.0, 1.0, 40)
        sweep_steps = min(n_timesteps, 500)

        def build_interp():
            final_ns = np.array(
                jax.vmap(
                    lambda r: run_constant_policy_final_n(
                        r, env, params, steps=sweep_steps
                    )
                )(rho_levels)
            )
            r_np = np.array(rho_levels)
            sort_idx = np.argsort(final_ns)
            return interp1d(
                final_ns[sort_idx],
                r_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            f"data/interpolators/{env_name}_comparison.pkl", build_interp
        )
        pid = make_reactor_stateful_gs_pid()
        mpc = make_reactor_mpc(env, params)
        cumulative_rewards = {"Constant": [], "PID": [], "MPC": []}

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
            key = jax.random.PRNGKey(seed)
            _, init_state = env.reset_env(key, params)
            best_rho = float(
                np.clip(interpolator(float(init_state.target_n)), -1.0, 1.0)
            )

            def const_action(_obs, _state=None, _r=best_rho):
                return np.array([_r])

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
                f"Reactor: cumulative reward over {n_seeds} seeds (mean ± std)"
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
    run_mode("video", rho_raw=0.0)
    run_mode("pid_video")


def run_figures():
    n_steps = 86400  # matches env default max_steps_in_episode (24 hours @ dt=1.0s)
    run_mode("2d", n_timesteps=n_steps, resolution=8)
    run_mode("temp", n_timesteps=n_steps, resolution=8)
    run_mode("pid", n_timesteps=n_steps, resolution=8)
    run_mode("comparison_gif")


def run_all_modes():
    run_figures()
    run_videos()


if __name__ == "__main__":
    run_all_modes()
