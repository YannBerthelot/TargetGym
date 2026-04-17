"""
Runner for the 3D airplane environment.

Generates videos and trajectory figures showing altitude + heading tracking.
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

from target_gym.experts.mpc import make_plane3d_mpc
from target_gym.experts.pid import (
    make_plane3d_circle_stateful_pid,
    make_plane3d_figure8_stateful_pid,
    make_plane3d_heading_stateful_pid,
)
from target_gym.plane3d.env import PlaneParams3D
from target_gym.plane3d.env_jax import Plane3DCircle, Plane3DFigureEight, Plane3DHeading
from target_gym.utils import (
    load_or_build_interpolator,
    save_comparison_figure,
    save_comparison_gif,
    truncate_colormap,
)


def run_constant_policy_3d(
    power: float,
    stick: float,
    aileron: float,
    env: Plane3DHeading,
    params: PlaneParams3D,
    steps: int = 5_000,
):
    """Run a constant-action episode and return (z_history, psi_history, done_history)."""
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = jnp.array([power, stick, aileron])

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(
            key, state, action, params
        )
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, new_done)
        return (key, state, done), (state.z, state.psi, done)

    (_, final_state, _), (z_hist, psi_hist, done_hist) = jax.lax.scan(
        step_fn,
        (key, state, False),
        None,
        length=steps,
    )
    return z_hist, psi_hist, done_hist


def run_const_final_alt_3d(
    power: float,
    stick: float,
    aileron: float,
    env: Plane3DHeading,
    params: PlaneParams3D,
    steps: int = 2_000,
):
    """Run a constant action and return the final (last non-terminated) altitude.

    Used to build the stick→altitude interpolator for the comparison baseline.
    Mirrors plane_runner.run_constant_policy_final_alt.
    """
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = jnp.array([power, stick, aileron])

    def step_fn(carry, _):
        k, st, done = carry
        _, new_st, _, new_done, info = env.step_env(k, st, action, params)
        st = jax.lax.cond(done, lambda _: st, lambda _: new_st, operand=None)
        done = jnp.logical_or(done, new_done)
        return (k, st, done), (new_st.z, info["last_state"].z, done)

    (_, _, _), (alt_hist, last_alt_hist, done_hist) = jax.lax.scan(
        step_fn, (key, state, False), None, length=steps
    )
    # Take the altitude at the step before termination (auto-reset gymnax behaviour);
    # if the episode never terminates, take the last value.
    idx = jnp.argmax(done_hist) - 1
    final_alt = jnp.where(done_hist.any(), last_alt_hist[idx], alt_hist[-1])
    return jnp.maximum(final_alt, 0.0)


class HeadingPolicy:
    """
    Track a target heading and altitude.
    Uses P-controllers on altitude, pitch, and heading (via bank).
    """

    def __init__(
        self,
        power: float = 0.8,
        kp_alt: float = 0.002,
        kp_pitch: float = -2.0,
        kp_heading: float = 0.5,
        kp_bank: float = -0.15,
        max_bank_deg: float = 20.0,
    ):
        self.power = power
        self.kp_alt = kp_alt
        self.kp_pitch = kp_pitch
        self.kp_heading = kp_heading
        self.kp_bank = kp_bank
        self.max_bank_rad = np.deg2rad(max_bank_deg)

    def reset(self):
        pass

    def __call__(self, obs):
        z = float(obs[2])
        theta = float(obs[4])
        phi = float(obs[6])
        psi = float(obs[9])
        target_alt = float(obs[10])
        target_heading = float(obs[11])

        # Altitude-hold
        alt_err = target_alt - z
        desired_theta = np.clip(self.kp_alt * alt_err, -0.15, 0.15)
        stick = np.clip(self.kp_pitch * (theta - desired_theta), -1.0, 1.0)

        # Heading tracking: compute heading error, wrap to [-pi, pi]
        heading_err = np.arctan2(
            np.sin(target_heading - psi), np.cos(target_heading - psi)
        )
        # Desired bank proportional to heading error
        desired_phi = np.clip(
            self.kp_heading * heading_err,
            -self.max_bank_rad,
            self.max_bank_rad,
        )
        bank_err = phi - desired_phi
        aileron = np.clip(self.kp_bank * bank_err, -1.0, 1.0)

        return jnp.array([self.power, stick, aileron])


class FigureEightPolicy:
    """
    Policy that draws a figure-8 on the ground track.

    Uses a sinusoidal *target bank angle* and closes the loop on both
    bank (via aileron) and altitude (via elevator/stick).

    obs layout: [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
                 gamma, psi, target_altitude, target_heading, power, stick, aileron]
    """

    def __init__(
        self,
        power: float = 0.8,
        max_bank_deg: float = 15.0,
        period: float = 400.0,
        kp_alt: float = 0.002,
        kp_pitch: float = -2.0,
        kp_bank: float = -0.15,
    ):
        self.power = power
        self.max_bank_rad = np.deg2rad(max_bank_deg)
        self.period = period
        self.kp_alt = kp_alt
        self.kp_pitch = kp_pitch
        self.kp_bank = kp_bank
        self.t = 0

    def reset(self):
        self.t = 0

    def __call__(self, obs):
        z = float(obs[2])
        theta = float(obs[4])
        phi = float(obs[6])
        target_alt = float(obs[10])

        # Altitude-hold: desired pitch from alt error, then stick from pitch error
        alt_err = target_alt - z
        desired_theta = np.clip(self.kp_alt * alt_err, -0.15, 0.15)
        stick = np.clip(self.kp_pitch * (theta - desired_theta), -1.0, 1.0)

        # Sinusoidal target bank angle for figure-8
        phase = 2.0 * np.pi * self.t / self.period
        desired_phi = self.max_bank_rad * np.sin(phase)
        bank_err = phi - desired_phi
        aileron = np.clip(self.kp_bank * bank_err, -1.0, 1.0)

        self.t += 1
        return jnp.array([self.power, stick, aileron])


class CirclePolicy:
    """
    Gentle constant-bank circle with altitude-hold.
    Closes the loop on bank angle to hold a steady turn.
    """

    def __init__(
        self,
        power: float = 0.8,
        target_bank_deg: float = 10.0,
        kp_alt: float = 0.002,
        kp_pitch: float = -2.0,
        kp_bank: float = -0.15,
    ):
        self.power = power
        self.target_bank_rad = np.deg2rad(target_bank_deg)
        self.kp_alt = kp_alt
        self.kp_pitch = kp_pitch
        self.kp_bank = kp_bank

    def reset(self):
        pass

    def __call__(self, obs):
        z = float(obs[2])
        theta = float(obs[4])
        phi = float(obs[6])
        target_alt = float(obs[10])

        alt_err = target_alt - z
        desired_theta = np.clip(self.kp_alt * alt_err, -0.15, 0.15)
        stick = np.clip(self.kp_pitch * (theta - desired_theta), -1.0, 1.0)

        bank_err = phi - self.target_bank_rad
        aileron = np.clip(self.kp_bank * bank_err, -1.0, 1.0)

        return jnp.array([self.power, stick, aileron])


# ──────────────────────────────────────────────────────────────────────────
# Comparison helpers (constant / PID / MPC)
# ──────────────────────────────────────────────────────────────────────────

# Task → (env_class, pid_factory, label) mapping for the comparison modes.
# All three tasks share the same step_env and the same action layout so we
# can reuse make_plane3d_mpc unchanged.
_TASK_REGISTRY = {
    "heading": (Plane3DHeading, make_plane3d_heading_stateful_pid, "Heading"),
    "circle": (Plane3DCircle, make_plane3d_circle_stateful_pid, "Circle"),
    "figure8": (Plane3DFigureEight, make_plane3d_figure8_stateful_pid, "Figure-8"),
}


def _run_comparison_for_task(
    task: str,
    params: PlaneParams3D,
    n_seeds: int,
    sweep_steps: int,
    cruise_power: float = 0.2,  # raw [-1, 1] action → 0.6 throttle (matches PID)
    aileron_const: float = 0.0,
):
    """
    Build the constant/PID/MPC comparison figure + GIF for one 3D plane task.

    The constant baseline is ``[cruise_power, best_stick(target_alt), aileron_const]``
    where ``best_stick`` comes from a stick→final-altitude interpolator built once
    per task and cached on disk.  Stick reaches an altitude equilibrium at fixed
    power=0.6 and aileron=0; for the circle/figure-8 tasks the constant baseline
    is therefore expected to ignore the lateral target entirely — that's the
    point of the comparison.

    Trajectory panel plots altitude (ft) vs target altitude — the only metric
    shared across all three tasks.  The cumulative-reward bar chart on the right
    captures the task-specific tracking quality (heading / circle / figure-8).
    """
    env_cls, pid_factory, _ = _TASK_REGISTRY[task]
    env = env_cls(integration_method="rk4_1")

    # 1. Stick → final-altitude interpolator (cached) for the constant baseline.
    stick_levels = jnp.linspace(-1.0, 1.0, 40)

    def build_interp():
        final_alts = np.array(
            jax.vmap(
                lambda s: run_const_final_alt_3d(
                    cruise_power,
                    s,
                    aileron_const,
                    env,
                    params,
                    steps=sweep_steps,
                )
            )(stick_levels)
        )
        s_np = np.array(stick_levels)
        sort_idx = np.argsort(final_alts)
        return interp1d(
            final_alts[sort_idx],
            s_np[sort_idx],
            bounds_error=False,
            fill_value="extrapolate",
        )

    interpolator = load_or_build_interpolator(
        f"data/interpolators/plane3d_{task}_comparison.pkl",
        build_interp,
    )

    pid = pid_factory()
    mpc = make_plane3d_mpc(env, params)

    def build_const(seed):
        key = jax.random.PRNGKey(seed)
        _, st = env.reset_env(key, params)
        bs = float(np.clip(interpolator(float(st.target_altitude)), -1.0, 1.0))

        def action(_obs, _state=None, _bs=bs):
            return np.array([cruise_power, _bs, aileron_const])

        return action

    def build_pid(seed):
        pid.reset()

        def action(obs, _state=None):
            a = pid.step(obs)
            return np.array([float(a[0]), float(a[1]), float(a[2])])

        return action

    def build_mpc(seed):
        mpc.reset()

        def action(obs, state):
            return mpc.step(obs, state)

        return action

    os.makedirs("figures/plane3d", exist_ok=True)
    save_comparison_figure(
        env=env,
        build_const_action=build_const,
        build_pid_action=build_pid,
        build_mpc_action=build_mpc,
        output_path=f"figures/plane3d/comparison_{task}.png",
        get_state_val=lambda s: float(s.z) * 3.28084,
        get_target_val=lambda s: float(s.target_altitude) * 3.28084,
        ylabel="Altitude (ft)",
        const_label="Constant",
        pid_label="PID",
        mpc_cache_prefix=f"data/mpc_cache/plane3d_{task}_mpc",
        params=params,
        n_seeds=n_seeds,
    )

    # GIF for seed 0
    gif_seed = 0
    key = jax.random.PRNGKey(gif_seed)
    _, st = env.reset_env(key, params)
    gif_stick = float(np.clip(interpolator(float(st.target_altitude)), -1.0, 1.0))
    pid.reset()
    mpc.reset()
    os.makedirs("videos/plane3d", exist_ok=True)
    save_comparison_gif(
        env=env,
        const_select_action=lambda _obs: np.array(
            [cruise_power, gif_stick, aileron_const]
        ),
        pid_select_action=lambda obs: (
            lambda a: np.array([float(a[0]), float(a[1]), float(a[2])])
        )(pid.step(obs)),
        mpc_select_action=lambda obs, state: mpc.step(obs, state),
        mpc_cache_path=f"data/mpc_cache/plane3d_{task}_mpc_seed{gif_seed}.pkl",
        output_path=f"videos/plane3d/comparison_{task}.gif",
        get_state_val=lambda s: float(s.z) * 3.28084,
        get_target_val=lambda s: float(s.target_altitude) * 3.28084,
        ylabel="Altitude (ft)",
        const_label=f"Constant stick={gif_stick:.2f}",
        pid_label="PID",
        params=params,
        seed=gif_seed,
    )


def run_mode(
    mode: str,
    power: float = 0.8,
    stick: float = 0.0,
    aileron: float = 0.3,
    n_timesteps: int = 5_000,
    plot: bool = True,
    save: bool = True,
    show: bool = False,
    resolution: int = 10,
    **kwargs,
):
    n_seeds = kwargs.pop("n_seeds", 20)
    sweep_steps = kwargs.pop("sweep_steps", 2_000)

    if mode in (
        "trajectories",
        "video_heading",
        "pid_video_heading",
        "comparison_heading",
        "mpc_video_heading",
    ):
        env = Plane3DHeading(integration_method="rk4_1")
    elif mode in (
        "video_figure8",
        "pid_video_figure8",
        "comparison_figure8",
        "mpc_video_figure8",
    ):
        env = Plane3DFigureEight(integration_method="rk4_1")
    elif mode in (
        "video_circle",
        "pid_video_circle",
        "comparison_circle",
        "mpc_video_circle",
    ):
        env = Plane3DCircle(integration_method="rk4_1")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if kwargs:
        params = PlaneParams3D(**kwargs)
    else:
        params = env.default_params

    if mode == "trajectories":
        # Vary aileron input, keep power and stick fixed — show how banking
        # affects altitude and heading.
        aileron_levels = jnp.linspace(-0.8, 0.8, resolution * 2 + 1)

        start_time = time.time()
        z_all, psi_all, done_all = jax.vmap(
            lambda a: run_constant_policy_3d(
                power, stick, a, env, params, steps=n_timesteps
            )
        )(aileron_levels)
        elapsed = time.time() - start_time
        print(f"Ran {len(aileron_levels)} episodes in {elapsed:.3f}s")

        if plot:
            fig, (ax_alt, ax_hdg) = plt.subplots(1, 2, figsize=(14, 5))

            norm = colors.Normalize(
                vmin=float(aileron_levels.min()), vmax=float(aileron_levels.max())
            )
            cmap = truncate_colormap(cm.coolwarm, 0.1, 0.9)

            for i, ail in enumerate(aileron_levels):
                mask = 1 - done_all[i]
                c = cmap(norm(float(ail)))
                ax_alt.plot(z_all[i] * mask * 3.28084, color=c, alpha=0.8)
                ax_hdg.plot(np.rad2deg(psi_all[i]) * mask, color=c, alpha=0.8)

            ax_alt.set_xlabel("Time step")
            ax_alt.set_ylabel("Altitude (ft)")
            ax_alt.set_title("Altitude trajectories (varying aileron)")
            ax_alt.axhline(
                float(params.target_altitude_range[0]) * 3.28084,
                color="red",
                ls="--",
                lw=1,
                alpha=0.6,
                label="Target range",
            )
            ax_alt.axhline(
                float(params.target_altitude_range[1]) * 3.28084,
                color="red",
                ls="--",
                lw=1,
                alpha=0.6,
            )
            ax_alt.legend(loc="best", fontsize=8)

            ax_hdg.set_xlabel("Time step")
            ax_hdg.set_ylabel("Heading (deg)")
            ax_hdg.set_title("Heading trajectories (varying aileron)")

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=[ax_alt, ax_hdg], label="Aileron input", shrink=0.8)

            os.makedirs("figures/plane3d", exist_ok=True)
            plt.tight_layout()
            plt.savefig("figures/plane3d/aileron_trajectories.pdf")
            plt.savefig("figures/plane3d/aileron_trajectories.png")
            if show:
                plt.show()
            plt.close()

    elif mode == "video_figure8":
        seed = 42
        policy = FigureEightPolicy(power=0.8, max_bank_deg=22.0, period=600.0)

        file = env.save_video(policy, seed, params=params, format="mp4")
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane3d", exist_ok=True)
        video.write_gif("videos/plane3d/figure8.gif", fps=30)

    elif mode == "video_circle":
        seed = 42
        policy = CirclePolicy(power=0.8, target_bank_deg=22.0)

        file = env.save_video(policy, seed, params=params, format="mp4")
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane3d", exist_ok=True)
        video.write_gif("videos/plane3d/circle.gif", fps=30)

    elif mode == "video_heading":
        seed = 42
        policy = HeadingPolicy(power=0.8, max_bank_deg=20.0)

        file = env.save_video(policy, seed, params=params, format="mp4")
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane3d", exist_ok=True)
        video.write_gif("videos/plane3d/heading.gif", fps=30)

    elif mode == "comparison_heading":
        _run_comparison_for_task("heading", params, n_seeds, sweep_steps)

    elif mode == "comparison_circle":
        _run_comparison_for_task("circle", params, n_seeds, sweep_steps)

    elif mode == "comparison_figure8":
        _run_comparison_for_task("figure8", params, n_seeds, sweep_steps)

    elif mode in ("pid_video_heading", "pid_video_circle", "pid_video_figure8"):
        task = mode.replace("pid_video_", "")
        seed = 42
        _, pid_factory, _ = _TASK_REGISTRY[task]
        pid = pid_factory()
        pid.reset()

        def select_action(obs, _state=None):
            return pid.step(obs)

        file = env.save_video(select_action, seed, params=params, format="mp4")
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane3d", exist_ok=True)
        video.write_gif(f"videos/plane3d/pid_{task}.gif", fps=30)

    elif mode in ("mpc_video_heading", "mpc_video_circle", "mpc_video_figure8"):
        task = mode.replace("mpc_video_", "")
        seed = 42
        mpc = make_plane3d_mpc(env, params)
        mpc.reset()

        def select_action(obs, state):
            return mpc.step(obs, state)

        file = env.save_video(select_action, seed, params=params, format="mp4")
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video = VideoFileClip(file)
        os.makedirs("videos/plane3d", exist_ok=True)
        video.write_gif(f"videos/plane3d/mpc_{task}.gif", fps=30)


def run_videos():
    run_mode(
        "video_heading",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_heading_range=(0.8, 0.8),
        max_steps_in_episode=600,
    )
    run_mode(
        "video_figure8",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_radius_range=(10000, 10000),
        max_steps_in_episode=600,
    )
    run_mode(
        "video_circle",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_radius_range=(10000, 10000),
        max_steps_in_episode=600,
    )
    run_mode(
        "pid_video_heading",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_heading_range=(0.8, 0.8),
        max_steps_in_episode=600,
    )
    run_mode(
        "pid_video_figure8",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_radius_range=(10000, 10000),
        max_steps_in_episode=600,
    )
    run_mode(
        "pid_video_circle",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_radius_range=(10000, 10000),
        max_steps_in_episode=600,
    )


def run_figures(show: bool = False):
    run_mode(
        "trajectories",
        power=0.8,
        stick=0.0,
        n_timesteps=2_000,
        resolution=8,
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        max_steps_in_episode=2_000,
    )
    # Constant / PID / MPC comparison for each 3D task.
    # Episode length matches run_videos so the comparison aligns with the GIFs.
    run_mode(
        "comparison_heading",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_heading_range=(0.8, 0.8),
        max_steps_in_episode=600,
        n_seeds=10,
        sweep_steps=600,
    )
    run_mode(
        "comparison_circle",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_radius_range=(10000, 10000),
        max_steps_in_episode=600,
        n_seeds=10,
        sweep_steps=600,
    )
    run_mode(
        "comparison_figure8",
        target_altitude_range=(5000, 5000),
        initial_altitude_range=(5000, 5000),
        target_radius_range=(10000, 10000),
        max_steps_in_episode=600,
        n_seeds=10,
        sweep_steps=600,
    )


def run_all_modes(show: bool = False):
    run_figures(show=show)
    run_videos()


if __name__ == "__main__":
    run_all_modes(show=True)
