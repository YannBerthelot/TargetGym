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

# Import environment
from target_gym import CSTR, CSTRParams
from target_gym.utils import truncate_colormap

env_name = "cstr"


def run_constant_policy(
    throttle: float, env: CSTR, params: CSTRParams, steps: int = 1_000
):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = throttle

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step(key, state, action, params)
        # Freeze state if already done
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (state.T, done)

    (_, final_state, _), (velocity_history, done_history) = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state.T, (velocity_history * (1 - done_history))[:-1]


def run_constant_policy_final_alt(
    power: float, stick: float, env: CSTR, params: CSTRParams, steps: int = 10_000
):
    key = jax.random.PRNGKey(0)
    init_obs, init_state = env.reset_env(key, params)
    action = (power, stick)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step(key, state, action, params)
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (
            new_state.T,
            info["last_state"].T,
            done,
        )

    (_, final_state, done), (velocity_hist, last_velocity_hist, done_hist) = (
        jax.lax.scan(step_fn, (key, init_state, False), None, length=steps)
    )
    final_velocity = last_velocity_hist[
        jnp.argmax(done_hist) - 1
    ]  # the episode has already reset at done, so take the step before
    return final_velocity


def build_power_interpolator_from_df(df, stick=0.0):
    raise NotImplementedError
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
    raise NotImplementedError
    df = run_mode(
        "3d", n_timesteps=20_000, max_alt=20_000, plot=False, save=False, resolution=20
    )
    return build_power_interpolator_from_df(df, stick=stick)


def run_mode(
    mode: str,
    throttle=1.0,
    n_timesteps=10_000,
    plot: bool = True,
    save: bool = True,
    resolution: int = 20,
    **kwargs,
):
    env = CSTR(integration_method="rk4_1")
    if kwargs is not None:
        params = CSTRParams(**kwargs)
    else:
        params = env.default_params

    if mode == "2d":
        start_time = time.time()
        throttle_levels = jnp.linspace(0.0, 1.0, (resolution * 2) + 1)

        def run_vmapped(powers):
            return jax.vmap(
                lambda t: run_constant_policy(t, env, params, steps=n_timesteps)
            )(powers)

        final_velocities, trajectories = run_vmapped(throttle_levels)
        # final_alts = jnp.maximum(final_velocities, 0.0)
        elapsed = time.time() - start_time
        print(f"Ran in {elapsed:.3f}s ({elapsed / len(throttle_levels):.3f}s per run)")
        if plot:

            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(
                vmin=throttle_levels.min(), vmax=throttle_levels.max()
            )
            cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
            for i, traj in enumerate(trajectories):
                ax.plot(traj, color=cmap(norm(throttle_levels[i])))
                if (i % 4) == 0:
                    ax.text(
                        x=len(traj) - 1,
                        y=traj[-1],
                        s=f" {float(throttle_levels[i]):.2f} - {float(traj[-1]):.2f}K",  # format the throttle value
                        color=cmap(norm(throttle_levels[i])),
                        fontsize=8,
                        va="center",
                        ha="left",
                    )

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)

            sm.set_array([])
            xmin, xmax = plt.xlim()
            plt.xlim(xmin, xmax + (xmax - xmin) * 0.08)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Temperature (K)")
            ax.set_title(
                "Temperature trajectories for varying cooling temperature levels"
            )
            os.makedirs(f"figures/{env_name}", exist_ok=True)
            plt.savefig(f"figures/{env_name}/trajectories.pdf")
            plt.savefig(f"figures/{env_name}/trajectories.png")

    elif mode == "video":
        seed = 42

        def select_action(_):
            return np.random.uniform(-1, 1)

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs(f"videos/{env_name}", exist_ok=True)
        video.write_gif(f"videos/{env_name}/output.gif", fps=30)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_all_modes():
    n_steps = 10_000
    run_mode(
        "2d", n_timesteps=n_steps, max_steps_in_episode=n_steps, resolution=20
    )  # or "2d" or "video"
    run_mode("video", throttle=0.15, max_steps_in_episode=1_000)


if __name__ == "__main__":
    run_all_modes()
