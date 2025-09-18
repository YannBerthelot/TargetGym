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
from target_gym import Bike, BikeParams


def run_constant_policy(
    torque: float,
    displacement: float,
    env: Bike,
    params: BikeParams,
    steps: int = 10_000,
):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = (torque, displacement)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step(key, state, action, params)
        # Freeze state if already done
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (state.distance_from_start, done)

    (_, final_state, done), (d_history, done_history) = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state.distance_from_start, d_history * (1 - done_history)


def run_constant_policy_final_alt(
    torque: float,
    displacement: float,
    env: Bike,
    params: BikeParams,
    steps: int = 10_000,
):
    key = jax.random.PRNGKey(0)
    init_obs, init_state = env.reset_env(key, params)
    action = (torque, displacement)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step(key, state, action, params)
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (
            new_state.distance_from_start,
            info["last_state"].distance_from_start,
            done,
        )

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
            "torque": jnp.repeat(power_levels, len(stick_levels)),
            "displacement": jnp.tile(stick_levels, len(power_levels)),
            "distance": final_altitudes.flatten(),
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


def run_mode(
    mode: str,
    power=1.0,
    stick=0.0,
    n_timesteps=10_000,
    plot: bool = True,
    save: bool = True,
    resolution: int = 20,
    **kwargs,
):
    env = Bike(integration_method="rk4_1")
    if kwargs is not None:
        params = BikeParams(**kwargs)
    else:
        params = env.default_params

    if mode == "2d":
        start_time = time.time()
        torques = jnp.linspace(-1.0, 1.0, (resolution * 2) + 1)
        displacements = jnp.array(stick)

        def run_vmapped(powers):
            return jax.vmap(
                lambda p: run_constant_policy(
                    p, displacements, env, params, steps=n_timesteps
                )
            )(powers)

        _, trajectories = run_vmapped(torques)
        elapsed = time.time() - start_time
        print(f"Ran in {elapsed:.3f}s ({elapsed / len(torques):.3f}s per run)")
        if plot:

            fig, ax = plt.subplots(figsize=(10, 6))
            norm = colors.Normalize(vmin=torques.min(), vmax=torques.max())
            cmap = cm.viridis
            # import pdb

            # pdb.set_trace()
            # for i, traj in enumerate(trajectories):
            #     ax.plot(traj, color=cmap(norm(torques[i])))
            ax.plot(torques, trajectories.max(axis=1))

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # fig.colorbar(sm, ax=ax).set_label("Power level")
            ax.set_xlabel("Torque")
            ax.set_ylabel("Distance (m)")
            ax.set_title("Distance from start for varying torque")
            os.makedirs("figures/bike", exist_ok=True)
            plt.savefig("figures/bike/power_trajectories.pdf")
            plt.savefig("figures/bike/power_trajectories.png")

    elif mode == "3d":
        torques = jnp.linspace(-1.0, 1.0, resolution + 1)
        displacements = jnp.linspace(-1.0, 1.0, resolution + 1)
        final_alts, df = run_power_stick_grid(
            torques,
            displacements,
            env,
            params,
            steps=n_timesteps,
            save_csv_path="data/bike/power_stick_altitudes.csv" if save else None,
        )
        if plot:
            P, S = jnp.meshgrid(torques, displacements, indexing="ij")
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(P, S, final_alts, cmap="viridis")
            fig.colorbar(
                surf, ax=ax, shrink=0.5, aspect=10, label="Distance from start (m)"
            )
            ax.set_xlabel("Torque")
            ax.set_ylabel("Displacement")
            ax.set_zlabel("Distance from start (m)")
            ax.set_title("Final Distance vs Torque and Displacement")
            ax.view_init(elev=15, azim=100)
            fig = plt.gcf()
            os.makedirs("figures/bike", exist_ok=True)
            fig.savefig("figures/bike/3d_altitude.pdf")
            plt.show()
        return df

    elif mode == "video":
        seed = 42

        def select_action(_):
            return (power, stick)

        file = env.save_video(select_action, seed, params=params)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/bike", exist_ok=True)
        video.write_gif("videos/bike/output.gif", fps=30)

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    run_mode("2d", n_timesteps=100)  # or "2d" or "video"
    run_mode("video", power=0.5, stick=0, max_steps_in_episode=100)
    run_mode("3d", n_timesteps=100, resolution=40)
