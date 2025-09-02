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
from plane_env.env_jax import Airplane2D, EnvParams


def run_constant_policy(
    power: float, stick: float, env: Airplane2D, params: EnvParams, steps: int = 10000
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
        return (key, state, done), state.z

    (_, final_state, done), z_history = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state.z, z_history


def run_constant_policy_final_alt(
    power: float, stick: float, env: Airplane2D, params: EnvParams, steps: int = 10000
):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = (power, stick)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(
            key, state, action, params
        )
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), None

    (_, final_state, done), _ = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    return final_state.z


def run_power_stick_grid(
    power_levels, stick_levels, env, params, steps=10000, save_csv_path=None
):
    def run_one_power(power):
        return jax.vmap(
            lambda s: run_constant_policy_final_alt(power, s, env, params, steps)
        )(stick_levels)

    final_altitudes = jax.vmap(run_one_power)(power_levels)
    final_altitudes = jnp.maximum(final_altitudes, 0.0)

    if save_csv_path is not None:
        df = pd.DataFrame(
            {
                "power": jnp.repeat(power_levels, len(stick_levels)),
                "stick": jnp.tile(stick_levels, len(power_levels)),
                "altitude": final_altitudes.flatten(),
            }
        )
        df.to_csv(save_csv_path, index=False)
        print(f"Saved grid results to {save_csv_path}")

    return final_altitudes


def build_power_interpolator_from_csv(csv_path, stick=0.0):
    df = pd.read_csv(csv_path)
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


def run_mode(mode: str, power=1.0, stick=0.0):
    env = Airplane2D()
    params = env.default_params

    if mode == "2d":
        start_time = time.time()
        power_levels = jnp.linspace(0.0, 1.0, 11)
        stick_level = stick

        def run_vmapped(powers):
            return jax.vmap(lambda p: run_constant_policy(p, stick_level, env, params))(
                powers
            )

        final_alts, trajectories = run_vmapped(power_levels)
        final_alts = jnp.maximum(final_alts, 0.0)
        elapsed = time.time() - start_time
        print(f"Ran in {elapsed:.3f}s ({elapsed / len(power_levels):.3f}s per run)")

        fig, ax = plt.subplots(figsize=(10, 6))
        norm = colors.Normalize(vmin=power_levels.min(), vmax=power_levels.max())
        cmap = cm.viridis
        for i, traj in enumerate(trajectories):
            ax.plot(traj, color=cmap(norm(power_levels[i])))

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax).set_label("Power level")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Altitude (m)")
        ax.set_title("Altitude trajectories for varying power")
        plt.show()

    elif mode == "3d":
        power_levels = jnp.linspace(0.0, 1.0, 21)
        stick_levels = jnp.linspace(-1.0, 1.0, 21)
        final_alts = run_power_stick_grid(
            power_levels,
            stick_levels,
            env,
            params,
            steps=20000,
            save_csv_path="power_stick_altitudes.csv",
        )

        P, S = jnp.meshgrid(power_levels, stick_levels * 15, indexing="ij")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(P, S, final_alts, cmap="viridis")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Final Altitude (m)")
        ax.set_xlabel("Power")
        ax.set_ylabel("Stick position")
        ax.set_zlabel("Final Altitude (m)")
        ax.set_title("Final altitude vs Power and Stick")
        plt.show()

    elif mode == "video":
        key = jax.random.PRNGKey(42)

        def select_action(_):
            return (power, stick)

        file = env.save_video(select_action, key)
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        video.write_gif("videos/output.gif", fps=30)

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    run_mode("video")  # or "2d" or "video"
