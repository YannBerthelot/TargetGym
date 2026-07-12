"""
Runner for the close-patrol (formation-keeping) environment.

Produces:
  - a trajectory figure: slot-tracking error over time for the pursuit-guidance
    PID vs a naive constant-action follower, plus the ground track of both
    aircraft under the PID;
  - a PID formation video (GIF).

Self-contained: no MPC / interpolator machinery (the target here is a moving
slot, not a stick->altitude equilibrium), so this runner is much simpler than
the plane runners.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from target_gym.experts.pid import make_patrol_pid, patrol_pid_step
from target_gym.patrol.env import desired_slot_position, slot_error
from target_gym.patrol.env_jax import PlanePatrol

# A gently-orbiting lead and a fixed trailing slot make a legible demo.
_DEMO_PARAMS = dict(
    max_steps_in_episode=600,
    slot_back_range=(220.0, 220.0),
    slot_right_range=(120.0, 120.0),
    slot_up_range=(0.0, 0.0),
    lead_turn_rate_range=(0.0022, 0.0022),
    follower_spawn_noise=40.0,
)


def _rollout_pid(env, params, seed, steps):
    """Scan the pursuit-guidance PID; return per-step slot error and both tracks."""
    pid_params, pid0 = make_patrol_pid()
    obs, state = env.reset_env(jax.random.PRNGKey(seed), params)

    def step_fn(carry, _):
        obs, state, pid = carry
        action, pid = patrol_pid_step(pid_params, pid, obs)
        obs, new_state, r, done, _ = env.step_env(
            jax.random.PRNGKey(seed), state, action, params
        )
        sx, sy, _ = desired_slot_position(new_state)
        out = (
            slot_error(new_state),
            new_state.follower.x,
            new_state.follower.y,
            new_state.lead.x,
            new_state.lead.y,
            sx,
            sy,
        )
        return (obs, new_state, pid), out

    _, out = jax.lax.scan(step_fn, (obs, state, pid0), None, length=steps)
    return out


def _rollout_constant(env, params, seed, steps, action):
    """Scan a constant action; return per-step slot error."""
    obs, state = env.reset_env(jax.random.PRNGKey(seed), params)
    action = jnp.asarray(action)

    def step_fn(carry, _):
        obs, state = carry
        obs, new_state, r, done, _ = env.step_env(
            jax.random.PRNGKey(seed), state, action, params
        )
        return (obs, new_state), slot_error(new_state)

    _, errs = jax.lax.scan(step_fn, (obs, state), None, length=steps)
    return errs


def run_figures(show: bool = False):
    env = PlanePatrol(integration_method="rk4_1")
    params = env.default_params.replace(**_DEMO_PARAMS)
    steps = params.max_steps_in_episode

    err_pid, fx, fy, lx, ly, sx, sy = jax.vmap(
        lambda s: _rollout_pid(env, params, s, steps)
    )(jnp.arange(6))
    # Naive baseline: hold cruise throttle, wings level, neutral stick.
    err_const = jax.vmap(
        lambda s: _rollout_constant(env, params, s, steps, [0.2, 0.0, 0.0])
    )(jnp.arange(6))

    fig, (ax_err, ax_trk) = plt.subplots(1, 2, figsize=(14, 5))

    t = np.arange(steps)
    ep = np.array(err_pid)
    ec = np.array(err_const)
    ax_err.plot(t, ep.mean(0), color="C0", label="Pursuit PID")
    ax_err.fill_between(t, ep.min(0), ep.max(0), color="C0", alpha=0.2)
    ax_err.plot(t, ec.mean(0), color="C3", label="Constant action")
    ax_err.fill_between(t, ec.min(0), ec.max(0), color="C3", alpha=0.2)
    ax_err.axhline(
        float(params.slot_tolerance),
        color="k",
        ls="--",
        lw=1,
        alpha=0.6,
        label="Slot tolerance",
    )
    ax_err.set_xlabel("Time step")
    ax_err.set_ylabel("Slot error (m)")
    ax_err.set_ylim(0, min(1500, float(np.percentile(ec, 99))))
    ax_err.set_title("Formation tracking error (6 seeds)")
    ax_err.legend(loc="best", fontsize=9)

    # Ground track for seed 0 (convert to km for readability).
    ax_trk.plot(np.array(lx[0]) / 1e3, np.array(ly[0]) / 1e3, color="C0", label="Lead")
    ax_trk.plot(
        np.array(fx[0]) / 1e3, np.array(fy[0]) / 1e3, color="C1", label="Follower"
    )
    ax_trk.plot(
        np.array(sx[0]) / 1e3,
        np.array(sy[0]) / 1e3,
        color="C2",
        ls=":",
        label="Slot",
    )
    ax_trk.set_xlabel("x (km)")
    ax_trk.set_ylabel("y (km)")
    ax_trk.set_aspect("equal", adjustable="datalim")
    ax_trk.set_title("Ground track (seed 0)")
    ax_trk.legend(loc="best", fontsize=9)

    os.makedirs("figures/patrol", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/patrol/trajectories.png")
    plt.savefig("figures/patrol/trajectories.pdf")
    if show:
        plt.show()
    plt.close()


def run_videos():
    env = PlanePatrol(integration_method="rk4_1")
    params = env.default_params.replace(**_DEMO_PARAMS)

    pid_params, pid0 = make_patrol_pid()
    box = {"pid": pid0}

    def select_action(obs):
        a, box["pid"] = patrol_pid_step(pid_params, box["pid"], jnp.asarray(obs))
        return a

    os.makedirs("videos/patrol", exist_ok=True)
    # NB: save_video resets with env.default_params (not ``params``), so the
    # seed selects the slot geometry.  Seed 15 draws a clean combat-spread
    # (wingman ~250 m back, ~120 m to the side, co-altitude) that reads well.
    file = env.save_video(
        select_action, seed=15, params=params, folder="videos/patrol", format="mp4"
    )
    from moviepy.video.io.VideoFileClip import VideoFileClip

    VideoFileClip(file).write_gif("videos/patrol/pid_formation.gif", fps=30)


def run_formation_video(num_wingmen: int = 4, seed: int = 7, steps: int = 420):
    """Render an expert-driven N-plane formation (1 lead + K wingmen) to a GIF.

    The multi-agent env uses a dict action interface, so this drives every
    agent by hand (heading autopilot for the lead — commanded onto a slowly
    rotating heading so the formation banks through a gentle turn — and the
    pursuit PID for each wingman) and renders each step.
    """
    from PIL import Image

    from target_gym.experts.pid import (
        make_plane3d_heading_pid,
        plane3d_heading_pid_step,
    )
    from target_gym.patrol.marl import LEAD, PlanePatrolMARL, wingman_name

    env = PlanePatrolMARL(num_wingmen=num_wingmen)
    params = env.default_params.replace(max_steps_in_episode=steps)
    fpp, fp0 = make_patrol_pid()
    lpp, lp0 = make_plane3d_heading_pid()

    obs, state = env.reset(jax.random.PRNGKey(seed), params)
    lp, wps = lp0, [fp0] * num_wingmen
    h0 = float(obs[LEAD][11])
    screen = clock = None
    frames = []
    for t in range(steps):
        lead_obs = obs[LEAD][:15].at[11].set(h0 + 0.0011 * t)  # gentle rotating turn
        la, lp = plane3d_heading_pid_step(lpp, lp, lead_obs)
        acts = {LEAD: la}
        new_wps = []
        for i in range(num_wingmen):
            wa, nwp = patrol_pid_step(fpp, wps[i], obs[wingman_name(i)])
            acts[wingman_name(i)] = wa
            new_wps.append(nwp)
        wps = new_wps
        obs, state, r, d, info = env.step_env(
            jax.random.PRNGKey(seed), state, acts, params
        )
        frames, screen, clock = env.render(screen, state, params, frames, clock)

    # Downscale + halve the frame rate so the full turn fits in a light GIF.
    small = []
    for k, fr in enumerate(frames):
        if k % 2:
            continue
        im = Image.fromarray(fr.astype("uint8"))
        im = im.resize((int(im.width * 0.6), int(im.height * 0.6)), Image.LANCZOS)
        small.append(im)
    os.makedirs("videos/patrol", exist_ok=True)
    out = f"videos/patrol/formation_{num_wingmen + 1}planes_short.gif"
    small[0].save(
        out, save_all=True, append_images=small[1:], duration=66, loop=0, optimize=True
    )
    print(f"Saved {out} ({len(small)} frames)")


def run_all_modes(show: bool = False):
    run_figures(show=show)
    run_videos()
    run_formation_video(num_wingmen=4)


if __name__ == "__main__":
    run_all_modes(show=True)
