import os
import time

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from target_gym.experts.mpc import make_plane_mpc
from target_gym.experts.pid import (
    make_plane_pid,
    make_plane_stateful_gs_pid,
    mimo_pid_step,
)

# Import environment
from target_gym.plane.env_jax import Airplane2D, PlaneParams
from target_gym.utils import (
    load_or_build_interpolator,
    load_or_run_mpc_episode,
    run_episode_headless_with_state,
    save_comparison_figure,
    save_comparison_gif,
    truncate_colormap,
)


def run_constant_policy(
    power: float,
    stick: float,
    env: Airplane2D,
    params: PlaneParams,
    steps: int = 10_000,
):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = (power, stick)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(
            key, state, action, params
        )
        # Freeze state if already done
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (state.z, done)

    (_, final_state, done), (z_history, done_history) = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state.z, z_history * (1 - done_history)


def run_constant_policy_final_alt(
    power: float,
    stick: float,
    env: Airplane2D,
    params: PlaneParams,
    steps: int = 10_000,
):
    key = jax.random.PRNGKey(0)
    init_obs, init_state = env.reset_env(key, params)
    action = (power, stick)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(
            key, state, action, params
        )
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (new_state.z, info["last_state"].z, done)

    (_, final_state, done), (alt_hist, last_alt_hist, done_hist) = jax.lax.scan(
        step_fn, (key, init_state, False), None, length=steps
    )
    final_alt = last_alt_hist[
        jnp.argmax(done_hist) - 1
    ]  # the episode has already reset at done, so take the step before
    return final_alt


def run_power_stick_grid(
    power_levels, stick_levels, env, params, steps=10000, save_csv_path=None
):
    def run_one_power(power):
        return jax.vmap(
            lambda s: run_constant_policy_final_alt(power, s, env, params, steps)
        )(stick_levels)

    final_altitudes = jax.vmap(run_one_power)(power_levels)
    final_altitudes = jnp.maximum(final_altitudes, 0.0)
    df = pd.DataFrame(
        {
            "power": jnp.repeat(power_levels, len(stick_levels)),
            "stick": jnp.tile(stick_levels, len(power_levels)),
            "altitude": final_altitudes.flatten(),
        }
    )
    if save_csv_path is not None:
        os.makedirs("/".join(save_csv_path.split("/")[:-1]), exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"Saved grid results to {save_csv_path}")

    return final_altitudes, df


def build_power_interpolator_from_df(df, stick=0.0):
    tol = 1e-6
    df_stick = df[np.abs(df["stick"] - stick) < tol]

    if df_stick.empty:
        raise ValueError(f"No data found for stick={stick}.")

    df_stick = df_stick.sort_values("altitude")
    altitudes = df_stick["altitude"].to_numpy()
    powers = df_stick["power"].to_numpy()

    if not (np.all(np.diff(altitudes) >= 0) or np.all(np.diff(altitudes) <= 0)):
        raise ValueError(
            f"Altitude not monotonic for stick={stick}, interpolation ambiguous."
        )

    interpolator = interp1d(
        altitudes,
        powers,
        bounds_error=False,
        fill_value=np.nan,
        kind="linear",
    )
    return interpolator


def get_interpolator(stick: float = 0.0):
    df = run_mode(
        "3d", n_timesteps=20_000, max_alt=20_000, plot=False, save=False, resolution=20
    )
    return build_power_interpolator_from_df(df, stick=stick)


def last_zero_array_arg(arrs):
    """
    Given an array of arrays, return the index of the last array
    whose last element equals 0. Return -1 if none.
    """
    arrs = np.asarray(arrs)
    mask = arrs[:, -1] == 0
    idxs = np.where(mask)[0]
    return idxs[-1] if idxs.size > 0 else -1


def run_pid_for_target(
    target_val, env: Airplane2D, params: PlaneParams, pid_params, pid_state0, steps: int
):
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    state = state.replace(target_altitude=target_val)

    def step_fn(carry, _):
        env_state, pid_state, done = carry
        obs = env.get_obs(env_state)
        action, new_pid_state = mimo_pid_step(pid_params, pid_state, obs)
        _, new_env_state, _, new_done, _ = env.step_env(key, env_state, action, params)
        env_state = jax.lax.cond(
            done, lambda _: env_state, lambda _: new_env_state, operand=None
        )
        done = jnp.logical_or(done, new_done)
        return (env_state, new_pid_state, done), env_state.z * 3.28084  # m → ft

    _, z_history = jax.lax.scan(step_fn, (state, pid_state0, False), None, length=steps)
    return z_history


def run_mode(
    mode: str,
    power=1.0,
    stick=0.0,
    n_timesteps=10_000,
    plot: bool = True,
    save: bool = True,
    show: bool = False,
    resolution: int = 20,
    **kwargs,
):
    env = Airplane2D(integration_method="rk4_1")
    n_seeds = kwargs.pop("n_seeds", 20)
    if kwargs:
        params = PlaneParams(**kwargs)
    else:
        params = env.default_params

    if mode == "2d":
        start_time = time.time()
        power_levels = jnp.linspace(-1.0, 1.0, (resolution * 2) + 1)
        stick_level = jnp.array(stick)

        def run_vmapped(powers):
            return jax.vmap(
                lambda p: run_constant_policy(
                    p, stick_level, env, params, steps=n_timesteps
                )
            )(powers)

        final_alts, trajectories = run_vmapped(power_levels)
        final_alts = jnp.maximum(final_alts, 0.0) * 3.28084
        trajectories *= 3.28084
        elapsed = time.time() - start_time
        print(f"Ran in {elapsed:.3f}s ({elapsed / len(power_levels):.3f}s per run)")
        if plot:

            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(vmin=power_levels.min(), vmax=power_levels.max())
            cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
            idx_zero = last_zero_array_arg(trajectories)
            for i, traj in enumerate(trajectories):
                ax.plot(traj, color=cmap(norm(power_levels[i])))
                if ((i % 4) == 0 and traj[-1] > 0) or i == idx_zero:
                    ax.text(
                        x=len(traj) - 1,
                        y=traj[-1],
                        s=f" {float((power_levels[i]+1)/2):.2f} - {float(traj[-1]):.0f}ft",  # format the throttle value
                        color=cmap(norm(power_levels[i])),
                        fontsize=8,
                        va="center",
                        ha="left",
                    )

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # fig.colorbar(sm, ax=ax).set_label("Power level")
            xmin, xmax = plt.xlim()
            plt.xlim(xmin, xmax + (xmax - xmin) * 0.09)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Altitude (ft)")
            ax.set_title("Altitude trajectories for varying power levels")
            os.makedirs("figures/plane", exist_ok=True)
            plt.savefig("figures/plane/power_trajectories.pdf")
            plt.savefig("figures/plane/power_trajectories.png")

    elif mode == "3d":
        power_levels = jnp.linspace(-1.0, 1.0, resolution + 1)
        stick_levels = jnp.linspace(-1.0, 1.0, resolution + 1)
        final_alts, df = run_power_stick_grid(
            power_levels,
            stick_levels,
            env,
            params,
            steps=n_timesteps,
            save_csv_path="data/plane/power_stick_altitudes.csv" if save else None,
        )
        if plot:
            P, S = jnp.meshgrid(power_levels, stick_levels * 15, indexing="ij")
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface((P + 1) / 2, S, final_alts * 3.28084, cmap="viridis")
            fig.colorbar(
                surf, ax=ax, shrink=0.5, aspect=10, label="Final Altitude (ft)"
            )
            ax.set_xlabel("Power")
            ax.set_ylabel("Stick position")
            ax.set_zlabel("Final Altitude (ft)")
            ax.set_title("Final altitude vs Power and Stick")
            ax.view_init(elev=30, azim=200)
            fig = plt.gcf()
            os.makedirs("figures/plane", exist_ok=True)
            fig.savefig("figures/plane/3d_altitude.pdf")
            fig.savefig("figures/plane/3d_altitude.png")
            if show:
                plt.show()
        return df

    elif mode == "pid":
        pid_params, pid_state0 = make_plane_pid()
        targets = jnp.linspace(
            params.target_altitude_range[0],
            params.target_altitude_range[1],
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
            targets_ft = targets * 3.28084
            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(
                vmin=float(targets_ft[0]), vmax=float(targets_ft[-1])
            )
            cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
            for i, traj in enumerate(trajectories):
                c = cmap(norm(float(targets_ft[i])))
                ax.plot(traj, color=c, alpha=0.85)
                ax.text(
                    x=len(traj) - 1,
                    y=float(traj[-1]),
                    s=f" {float(targets_ft[i]):.0f} → {float(traj[-1]):.0f}ft",
                    color=c,
                    fontsize=8,
                    va="center",
                    ha="left",
                )
            xmin, xmax = plt.xlim()
            plt.xlim(xmin, xmax + (xmax - xmin) * 0.15)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Altitude (ft)")
            ax.set_title("PID response for varying altitude setpoints")
            os.makedirs("figures/plane", exist_ok=True)
            plt.savefig("figures/plane/pid_response.pdf")
            plt.savefig("figures/plane/pid_response.png")
            plt.close()

    elif mode == "video":
        seed = 42

        def select_action(_):
            return (power, stick)

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane", exist_ok=True)
        video.write_gif("videos/plane/output.gif", fps=30)

    elif mode == "pid_video":
        seed = 42
        pid = make_plane_stateful_gs_pid()

        def select_action(obs):
            action = pid.step(obs)  # [power, stick]
            return (float(action[0]), float(action[1]))

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane", exist_ok=True)
        video.write_gif("videos/plane/pid_output.gif", fps=30)

    elif mode == "mpc_video":
        seed = 42
        mpc = make_plane_mpc(env, params)
        mpc.reset()

        def select_action(obs, state):
            return mpc.step(obs, state)

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane", exist_ok=True)
        video.write_gif("videos/plane/mpc_output.gif", fps=30)

    elif mode == "comparison_gif":
        # Sweep power with neutral stick — altitude equilibrium is power-controlled.
        power_levels = jnp.linspace(-1.0, 1.0, 40)
        fixed_stick = 0.0

        def build_interp():
            final_alts = np.array(
                jax.vmap(
                    lambda p: run_constant_policy_final_alt(
                        p, fixed_stick, env, params, steps=n_timesteps
                    )
                )(power_levels)
            )
            p_np = np.array(power_levels)
            sort_idx = np.argsort(final_alts)
            return interp1d(
                final_alts[sort_idx],
                p_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            "data/interpolators/plane_comparison.pkl", build_interp
        )
        pid = make_plane_stateful_gs_pid()
        mpc = make_plane_mpc(env, params)

        def build_const(seed):
            key = jax.random.PRNGKey(seed)
            _, st = env.reset_env(key, params)
            bp = float(np.clip(interpolator(float(st.target_altitude)), -1.0, 1.0))

            def action(_obs, _state=None, _bp=bp):
                return np.array([_bp, fixed_stick])

            return action

        def build_pid(seed):
            pid.reset()

            def action(obs, _state=None):
                a = pid.step(obs)
                return np.array([float(a[0]), float(a[1])])

            return action

        def build_mpc(seed):
            mpc.reset()

            def action(obs, state):
                return mpc.step(obs, state)

            return action

        os.makedirs("figures/plane", exist_ok=True)
        save_comparison_figure(
            env=env,
            build_const_action=build_const,
            build_pid_action=build_pid,
            build_mpc_action=build_mpc,
            output_path="figures/plane/comparison.png",
            get_state_val=lambda s: float(s.z) * 3.28084,
            get_target_val=lambda s: float(s.target_altitude) * 3.28084,
            ylabel="Altitude (ft)",
            const_label="Constant",
            pid_label="PID",
            mpc_cache_prefix="data/mpc_cache/plane_mpc",
            params=params,
            n_seeds=n_seeds,
        )

        # Also produce animated GIF for seed 0
        gif_seed = 0
        key = jax.random.PRNGKey(gif_seed)
        _, st = env.reset_env(key, params)
        gif_p = float(np.clip(interpolator(float(st.target_altitude)), -1.0, 1.0))
        pid.reset()
        mpc.reset()
        os.makedirs("videos/plane", exist_ok=True)
        save_comparison_gif(
            env=env,
            const_select_action=lambda _obs: np.array([gif_p, fixed_stick]),
            pid_select_action=lambda obs: (
                lambda a: np.array([float(a[0]), float(a[1])])
            )(pid.step(obs)),
            mpc_select_action=lambda obs, state: mpc.step(obs, state),
            mpc_cache_path=f"data/mpc_cache/plane_mpc_seed{gif_seed}.pkl",
            output_path="videos/plane/comparison.gif",
            get_state_val=lambda s: float(s.z) * 3.28084,
            get_target_val=lambda s: float(s.target_altitude) * 3.28084,
            ylabel="Altitude (ft)",
            const_label=f"Constant p={gif_p:.2f}",
            pid_label="PID",
            params=params,
            seed=gif_seed,
        )

    elif mode == "comparison_multi":
        fixed_stick = 0.0
        power_levels = jnp.linspace(-1.0, 1.0, 40)

        def build_interp():
            final_alts = np.array(
                jax.vmap(
                    lambda p: run_constant_policy_final_alt(
                        p, fixed_stick, env, params, steps=n_timesteps
                    )
                )(power_levels)
            )
            p_np = np.array(power_levels)
            sort_idx = np.argsort(final_alts)
            return interp1d(
                final_alts[sort_idx],
                p_np[sort_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )

        interpolator = load_or_build_interpolator(
            "data/interpolators/plane_comparison.pkl", build_interp
        )

        pid = make_plane_stateful_gs_pid()
        mpc = make_plane_mpc(env, params)

        cumulative_rewards = {"Constant": [], "PID": [], "MPC": []}

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)

            # Determine best constant power for this seed's target altitude
            key = jax.random.PRNGKey(seed)
            _, init_state = env.reset_env(key, params)
            target_alt = float(init_state.target_altitude)
            best_power = float(np.clip(interpolator(target_alt), -1.0, 1.0))

            def const_action(_obs, _state=None):
                return np.array([best_power, fixed_stick])

            def pid_action(obs, _state=None):
                action = pid.step(obs)
                return np.array([float(action[0]), float(action[1])])

            def mpc_action(obs, state):
                return mpc.step(obs, state)

            _, rews_c = run_episode_headless_with_state(env, const_action, params, seed)
            pid.reset()
            _, rews_p = run_episode_headless_with_state(env, pid_action, params, seed)
            mpc.reset()
            cache_path = f"data/mpc_cache/plane_mpc_seed{seed}.pkl"
            _, rews_m = load_or_run_mpc_episode(
                cache_path, env, mpc_action, params, seed
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
            # Overlay individual points
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
            ax.set_title(f"Plane: cumulative reward over {n_seeds} seeds (mean ± std)")
            for xi, (m, s) in enumerate(zip(means, stds)):
                ax.text(
                    xi,
                    m + s + max(stds) * 0.05,
                    f"{m:.1f}±{s:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            os.makedirs("figures/plane", exist_ok=True)
            plt.tight_layout()
            plt.savefig("figures/plane/comparison_multi.pdf")
            plt.savefig("figures/plane/comparison_multi.png")
            if show:
                plt.show()
            plt.close()

        return cumulative_rewards

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_videos():
    run_mode("video", power=0.5, stick=0)
    run_mode("pid_video")
    run_mode("mpc_video")


def run_figures(show: bool = False):
    run_mode("2d", n_timesteps=5000)
    run_mode("pid", n_timesteps=5000, resolution=10)
    run_mode("comparison_gif")
    run_mode("3d", n_timesteps=20_000, max_alt=20_000, resolution=40, show=show)


def run_all_modes(show: bool = False):
    run_figures(show=show)
    run_videos()


if __name__ == "__main__":
    run_all_modes(show=True)
