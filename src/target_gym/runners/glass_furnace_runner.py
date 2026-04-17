"""Runner for the GlassFurnace environment.

Supports:
  - "2d":              sweep constant fuel flows, plot crown-temperature trajectories
  - "glass":           sweep constant fuel flows, plot hidden glass-temperature trajectories
  - "pid":             sweep PID tracking for varying crown setpoints
  - "video":           save a video/gif with a user-provided constant fuel action
  - "pid_video":       save a video/gif with the tuned PID policy
  - "mpc_video":       save a video/gif with the do-mpc policy
  - "comparison_gif":  constant vs PID vs MPC comparison (figure over n seeds + animated GIF)
  - "comparison_multi": bar chart of cumulative reward across seeds for constant / PID / MPC
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

from target_gym import GlassFurnace, GlassFurnaceParams
from target_gym.experts.mpc import make_glass_furnace_mpc
from target_gym.experts.pid import (
    make_glass_furnace_pid,
    make_glass_furnace_stateful_gs_pid,
    pid_step,
)
from target_gym.glass_furnace.env import N_SETPOINTS
from target_gym.utils import (
    load_or_build_interpolator,
    load_or_run_mpc_episode,
    run_episode_headless_with_state,
    save_comparison_figure,
    save_comparison_gif,
    truncate_colormap,
)

env_name = "glass_furnace"


def run_constant_policy(
    fuel_raw: float,
    env: GlassFurnace,
    params: GlassFurnaceParams,
    steps: int = 2000,
    seed: int = 0,
):
    """Roll out a constant fuel_raw action, return crown + glass temperature histories."""
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(
            key, state, fuel_raw, params
        )
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (
            state.T_crown,
            state.T_melt,
            state.T_work,
            done,
        )

    (_, final_state, _), (T_crown_hist, T_melt_hist, T_work_hist, _) = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state, T_crown_hist, T_melt_hist, T_work_hist


def run_constant_policy_final_T_crown(
    fuel_raw: float,
    env: GlassFurnace,
    params: GlassFurnaceParams,
    steps: int,
    seed: int = 0,
):
    """Return the final crown temperature for a constant raw fuel action."""
    key = jax.random.PRNGKey(seed)
    _, state = env.reset_env(key, params)

    def step_fn(carry, _):
        key, state, done = carry
        _, new_state, _, new_done, _ = env.step_env(key, state, fuel_raw, params)
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), None

    (_, final_state, _), _ = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state.T_crown


def run_pid_for_target(
    target_val,
    env: GlassFurnace,
    params: GlassFurnaceParams,
    pid_params,
    pid_state0,
    steps: int,
):
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    # Override the whole schedule with a constant target so this plot shows
    # pure single-setpoint PID tracking (no mid-episode jumps).
    state = state.replace(
        target_T_crown=target_val,
        target_schedule=jnp.full((N_SETPOINTS,), target_val),
    )

    def step_fn(carry, _):
        env_state, pid_state, done = carry
        # Match env.get_obs: fuel flow is exposed as % of fuel_max.
        fuel_pct = 100.0 * env_state.fuel_flow / params.fuel_max
        obs = jnp.array([env_state.T_crown, fuel_pct, env_state.target_T_crown])
        action, new_pid_state = pid_step(pid_params, pid_state, obs)
        _, new_env_state, _, new_done, _ = env.step_env(
            key, env_state, action[0], params
        )
        env_state = jax.lax.cond(
            done, lambda _: env_state, lambda _: new_env_state, operand=None
        )
        done = jnp.logical_or(done, new_done)
        return (env_state, new_pid_state, done), env_state.T_crown

    _, T_crown_history = jax.lax.scan(
        step_fn, (state, pid_state0, False), None, length=steps
    )
    return T_crown_history


def _plot_temperature_sweep(
    hist, fuel_levels, params, title, ylabel, save_name, t_hours
):
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = colors.Normalize(
        vmin=float(fuel_levels.min()), vmax=float(fuel_levels.max())
    )
    cmap = truncate_colormap(cm.viridis, 0.0, 0.85)

    for i, traj in enumerate(hist):
        ax.plot(t_hours, traj, color=cmap(norm(float(fuel_levels[i]))))
        if (i % 4) == 0:
            ax.text(
                x=t_hours[-1],
                y=float(traj[-1]),
                s=f" raw={float(fuel_levels[i]):.2f} → {float(traj[-1]):.0f}°C",
                color=cmap(norm(float(fuel_levels[i]))),
                fontsize=8,
                va="center",
                ha="left",
            )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, xmax + (xmax - xmin) * 0.15)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    os.makedirs(f"figures/{env_name}", exist_ok=True)
    plt.savefig(f"figures/{env_name}/{save_name}.pdf")
    plt.savefig(f"figures/{env_name}/{save_name}.png")
    plt.close()


def run_mode(
    mode: str,
    fuel_raw: float = 0.0,
    n_timesteps: int = 2880,
    plot: bool = True,
    save: bool = True,
    resolution: int = 10,
    **kwargs,
):
    env = GlassFurnace(integration_method="rk4_1")
    n_seeds = kwargs.pop("n_seeds", 20)
    params = GlassFurnaceParams(**kwargs) if kwargs else env.default_params

    if mode in ("2d", "glass"):
        start_time = time.time()
        fuel_levels = jnp.linspace(-1.0, 1.0, (resolution * 2) + 1)

        def run_vmapped(fuels):
            return jax.vmap(
                lambda f: run_constant_policy(f, env, params, steps=n_timesteps)
            )(fuels)

        _, T_crown_hist, T_melt_hist, T_work_hist = run_vmapped(fuel_levels)
        elapsed = time.time() - start_time
        print(
            f"Ran {len(fuel_levels)} rollouts x {n_timesteps} steps "
            f"in {elapsed:.3f}s ({elapsed / len(fuel_levels):.3f}s per run)"
        )

        if plot:
            t_hours = np.arange(n_timesteps) * params.delta_t / 3600.0

            if mode == "2d":
                _plot_temperature_sweep(
                    T_crown_hist,
                    fuel_levels,
                    params,
                    title="Crown temperature trajectories for varying constant fuel flows",
                    ylabel="T_crown (°C)",
                    save_name="trajectories_crown",
                    t_hours=t_hours,
                )
            else:  # "glass"
                _plot_temperature_sweep(
                    T_melt_hist,
                    fuel_levels,
                    params,
                    title="Melt-zone glass temperature trajectories (hidden state)",
                    ylabel="T_melt (°C)",
                    save_name="trajectories_melt",
                    t_hours=t_hours,
                )
                _plot_temperature_sweep(
                    T_work_hist,
                    fuel_levels,
                    params,
                    title="Working-end glass temperature trajectories (hidden state)",
                    ylabel="T_work (°C)",
                    save_name="trajectories_work",
                    t_hours=t_hours,
                )

    elif mode == "pid":
        pid_params, pid_state0 = make_glass_furnace_pid()
        targets = jnp.linspace(
            params.target_T_crown_range[0],
            params.target_T_crown_range[1],
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
            t_hours = np.arange(n_timesteps) * params.delta_t / 3600.0
            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(vmin=float(targets[0]), vmax=float(targets[-1]))
            cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
            for i, traj in enumerate(trajectories):
                c = cmap(norm(float(targets[i])))
                ax.plot(t_hours, traj, color=c, alpha=0.9)
                ax.axhline(float(targets[i]), color=c, ls="--", lw=0.6, alpha=0.4)
                ax.text(
                    x=t_hours[-1],
                    y=float(traj[-1]),
                    s=f" target={float(targets[i]):.0f}°C → {float(traj[-1]):.0f}°C",
                    color=c,
                    fontsize=8,
                    va="center",
                    ha="left",
                )
            xmin, xmax = plt.xlim()
            plt.xlim(xmin, xmax + (xmax - xmin) * 0.2)
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("T_crown (°C)")
            ax.set_title("PID response for varying crown-temperature setpoints")
            os.makedirs(f"figures/{env_name}", exist_ok=True)
            plt.savefig(f"figures/{env_name}/pid_response.pdf")
            plt.savefig(f"figures/{env_name}/pid_response.png")
            plt.close()

    elif mode == "video":
        seed = 42

        def select_action(_):
            return np.array([fuel_raw])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/output.gif", fps=30)

    elif mode == "pid_video":
        seed = 42
        pid = make_glass_furnace_stateful_gs_pid()

        def select_action(obs):
            return np.array([pid.step(obs)])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/pid_output.gif", fps=30)

    elif mode == "mpc_video":
        seed = 42
        mpc = make_glass_furnace_mpc(env, params)
        mpc.reset()

        # Two-arg closure — save_video detects this and threads ``state`` in so
        # the MPC can read the full setpoint schedule / step counter.
        def select_action(obs, state):
            return np.array([mpc.step(obs, state)])

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/mpc_output.gif", fps=30)

    elif mode == "comparison_gif":
        # Interpolator: target_T_crown -> best constant raw fuel action.
        fuel_levels = jnp.linspace(-1.0, 1.0, 40)
        sweep_steps = min(n_timesteps, 2000)

        def build_interp():
            final_temps = np.array(
                jax.vmap(
                    lambda f: run_constant_policy_final_T_crown(
                        f, env, params, steps=sweep_steps
                    )
                )(fuel_levels)
            )
            f_np = np.array(fuel_levels)
            sort_idx = np.argsort(final_temps)
            return interp1d(
                final_temps[sort_idx],
                f_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            f"data/interpolators/{env_name}_comparison.pkl", build_interp
        )
        pid = make_glass_furnace_stateful_gs_pid()
        mpc = make_glass_furnace_mpc(env, params)

        def build_const(seed):
            key = jax.random.PRNGKey(seed)
            _, st = env.reset_env(key, params)
            bfuel = float(np.clip(interpolator(float(st.target_T_crown)), -1.0, 1.0))

            def action(_obs, _state=None, _f=bfuel):
                return np.array([_f])

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
            get_state_val=lambda s: float(s.T_crown),
            get_target_val=lambda s: float(s.target_T_crown),
            ylabel="T_crown (°C)",
            const_label="Constant",
            pid_label="PID",
            mpc_cache_prefix=f"data/mpc_cache/{env_name}_mpc",
            params=params,
            n_seeds=n_seeds,
        )

        # Animated GIF for seed 0
        gif_seed = 0
        key = jax.random.PRNGKey(gif_seed)
        _, st = env.reset_env(key, params)
        gif_fuel = float(np.clip(interpolator(float(st.target_T_crown)), -1.0, 1.0))
        pid.reset()
        mpc.reset()
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        save_comparison_gif(
            env=env,
            const_select_action=lambda _obs: np.array([gif_fuel]),
            pid_select_action=lambda obs: np.array([pid.step(obs)]),
            mpc_select_action=lambda obs, state: np.array([mpc.step(obs, state)]),
            mpc_cache_path=f"data/mpc_cache/{env_name}_mpc_seed{gif_seed}.pkl",
            output_path=f"videos/{env_name}/comparison.gif",
            get_state_val=lambda s: float(s.T_crown),
            get_target_val=lambda s: float(s.target_T_crown),
            ylabel="T_crown (°C)",
            const_label=f"Constant fuel={gif_fuel:.2f}",
            pid_label="PID",
            params=params,
            seed=gif_seed,
        )

    elif mode == "comparison_multi":
        fuel_levels = jnp.linspace(-1.0, 1.0, 40)
        sweep_steps = min(n_timesteps, 2000)

        def build_interp():
            final_temps = np.array(
                jax.vmap(
                    lambda f: run_constant_policy_final_T_crown(
                        f, env, params, steps=sweep_steps
                    )
                )(fuel_levels)
            )
            f_np = np.array(fuel_levels)
            sort_idx = np.argsort(final_temps)
            return interp1d(
                final_temps[sort_idx],
                f_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            f"data/interpolators/{env_name}_comparison.pkl", build_interp
        )
        pid = make_glass_furnace_stateful_gs_pid()
        mpc = make_glass_furnace_mpc(env, params)
        cumulative_rewards = {"Constant": [], "PID": [], "MPC": []}

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
            key = jax.random.PRNGKey(seed)
            _, init_state = env.reset_env(key, params)
            best_fuel = float(
                np.clip(interpolator(float(init_state.target_T_crown)), -1.0, 1.0)
            )

            def const_action(_obs, _state=None, _f=best_fuel):
                return np.array([_f])

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
                f"GlassFurnace: cumulative reward over {n_seeds} seeds (mean ± std)"
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
    run_mode("video", fuel_raw=0.0)
    run_mode("pid_video", max_steps_in_episode=5760)
    run_mode("mpc_video", max_steps_in_episode=5760)


def run_figures():
    # Long-horizon sweep: 48 hours at dt=30s -> 5760 steps
    run_mode("2d", n_timesteps=5760, resolution=10)
    run_mode("glass", n_timesteps=5760, resolution=10)
    # PID / comparison use the full 48 h horizon (5760 steps) so the schedule's
    # piecewise setpoints are all exercised; crown dynamics settle in minutes
    # but the slower glass-zone advection plays out over hours.
    run_mode("pid", n_timesteps=5760, resolution=8, max_steps_in_episode=5760)
    run_mode("comparison_gif", max_steps_in_episode=5760)


def run_all_modes():
    run_figures()
    run_videos()


if __name__ == "__main__":
    run_all_modes()
