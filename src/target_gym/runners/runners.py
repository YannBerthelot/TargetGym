"""Figures and videos for every registered environment.

One runner, driven by :data:`target_gym.registry.REGISTRY`; nothing here names
an environment. Three pieces of the environment interface make that possible:

``obs_value_index`` / ``obs_target_index``
    Class attributes on every environment giving the observation slots that
    hold the tracked variable and its setpoint (a tuple for multi-loop plants).
``action_space(params)``
    Supplies the bounds a constant-action sweep should span.
``EnvSpec.make_pid`` / ``make_mpc``
    The baselines, already registered.

Setpoints are varied by *seed*: every environment samples its own target on
reset, so a handful of seeds gives a spread from its own distribution.

Usage
-----
    python -m target_gym.runners.runners                      # everything
    python -m target_gym.runners.runners --env cstr reactor
    python -m target_gym.runners.runners --only figures
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.utils import truncate_colormap  # noqa: E402

FIGURE_DIR = "figures"
VIDEO_DIR = "videos"


# ---------------------------------------------------------------------------
# Generic rollout machinery
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _jitted_step(env):
    """One jitted ``step_env`` per environment, reused across rollouts, so the
    executable compiles once per process instead of once per rollout."""
    return jax.jit(env.step_env)


def _wants_state(policy: Callable) -> bool:
    """True when *policy* takes ``(obs, state)`` rather than ``(obs,)``.

    Decided from the signature: catching a TypeError would also swallow one
    raised inside a single-argument policy.
    """
    try:
        return len(inspect.signature(policy).parameters) >= 2
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return False


def _as_tuple(index) -> tuple[int, ...]:
    """Normalise an observation index that may be a scalar or a tuple."""
    return tuple(index) if isinstance(index, (tuple, list)) else (int(index),)


def _action_bounds(env, params) -> tuple[np.ndarray, np.ndarray]:
    """Low and high action bounds, broadcast to the action shape."""
    space = env.action_space(params)
    shape = space.shape if space.shape else (1,)
    return (
        np.broadcast_to(np.asarray(space.low, dtype=float), shape).copy(),
        np.broadcast_to(np.asarray(space.high, dtype=float), shape).copy(),
    )


def rollout(spec, params, policy: Callable, seed: int = 0, return_trips: bool = False):
    """Run one episode, returning tracked values, targets and rewards.

    A trip never ends the window (``base.failure_kernel``): the step is
    charged the trip cost and the plant restarts. With ``return_trips`` the
    number of trips in the window is returned as a fourth value.

    ``policy`` is called as ``policy(obs)`` or, when it accepts two arguments,
    ``policy(obs, state)`` -- the MPC baselines need the full state.

    Returns
    -------
    values : ``(T, n_tracked)`` array of the tracked variable(s)
    targets : ``(T, n_tracked)`` array of their setpoint(s)
    rewards : ``(T,)`` array
    """
    env = spec.make_env()
    value_idx = _as_tuple(env.obs_value_index)
    target_idx = _as_tuple(env.obs_target_index)

    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    step = _jitted_step(env)

    values, targets, rewards = [], [], []
    trips = 0
    # Loop on the environment's own clock, so a plant cannot be scored past
    # its time limit.
    while int(state.time) < int(params.max_steps_in_episode):
        obs_np = np.asarray(obs)
        values.append(obs_np[list(value_idx)])
        targets.append(obs_np[list(target_idx)])
        # Hand the policy the NumPy view: given a JAX array, every operation in
        # a host PID is a separate un-jitted dispatch, several times the step's
        # own cost.
        action = policy(obs_np, state) if _wants_state(policy) else policy(obs_np)
        obs, state, reward, terminated, info = step(
            key, state, jnp.atleast_1d(jnp.asarray(action)), params
        )
        rewards.append(float(reward))
        trips += int(bool(info.get("tripped", False)))
        if bool(terminated):
            break
    if return_trips:
        return np.array(values), np.array(targets), np.array(rewards), trips
    return np.array(values), np.array(targets), np.array(rewards)


def baseline_policy(spec, kind: str, params=None) -> Callable | None:
    """A shipped baseline as one uniform ``(obs, state) -> action`` callable.

    Both kinds take ``(obs, state)``, so one evaluation loop serves either,
    and a learned policy written to the same shape drops in beside them. **The
    PID reads only ``obs``**, as a plant controller does. **The MPC reads only
    ``state``**, the true simulator state: it is a full-state upper bound, not
    a peer.

    ``kind`` is ``"pid"`` or ``"mpc"``. Returns ``None`` when the environment
    does not ship that baseline; every registered environment ships both.
    """
    if kind == "pid":
        if not spec.has_pid:
            return None
        pid = spec.make_pid()
        if hasattr(pid, "reset"):
            pid.reset()
        call = pid if callable(pid) else pid.step
        return lambda obs, state=None: np.atleast_1d(call(obs))

    if kind == "mpc":
        if not spec.has_mpc:
            return None
        params = spec.make_test_params() if params is None else params
        # The planner predicts the mean disturbance: ``plan_params`` zeroes
        # ``spec.noise_fields``, as the batched recorder does.
        from target_gym.experts.mpc import plan_params

        mpc = spec.make_mpc(spec.make_env(), plan_params(spec, params))
        mpc.reset()
        return lambda obs, state: np.atleast_1d(mpc.step(obs, state))

    raise ValueError(f"kind must be 'pid' or 'mpc', not {kind!r}")


def rollout_mpc_batch(spec, params, n_seeds: int):
    """Run ``n_seeds`` MPC episodes at once, vmapped over the seed axis.

    Only for ``GradientMPC``, whose ``_optimize(actions_init, state)`` is a
    pure function and vmaps as it stands; the CasADi planners call IPOPT,
    outside JAX, and are parallelised by process instead.

    Returns ``(values, targets, rewards, trips)``: the first three with a
    leading seed axis, matching :func:`rollout`, and the trips per seed
    (``info["tripped"]``; a trip never ends the window).
    """
    env = spec.make_env()
    from target_gym.experts.mpc import plan_params

    mpc = spec.make_mpc(env, plan_params(spec, params))
    value_idx = _as_tuple(env.obs_value_index)
    target_idx = _as_tuple(env.obs_target_index)
    n_steps = int(params.max_steps_in_episode)

    keys = jnp.stack([jax.random.PRNGKey(s) for s in range(n_seeds)])
    obs, state = jax.jit(jax.vmap(env.reset_env, in_axes=(0, None)))(keys, params)
    actions = jnp.zeros((n_seeds, mpc.horizon, mpc.action_dim))

    optimize = jax.jit(jax.vmap(mpc._optimize, in_axes=(0, 0)))
    step = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))

    values, targets, rewards = [], [], []
    trips = jnp.zeros((n_seeds,), dtype=jnp.int32)
    for _ in range(n_steps):
        values.append(obs[:, list(value_idx)])
        targets.append(obs[:, list(target_idx)])
        # Shift the warm start by one and repeat the last action, exactly as
        # ``GradientMPC.step`` does for a single episode.
        actions = optimize(
            jnp.concatenate([actions[:, 1:], actions[:, -1:]], axis=1), state
        )
        u = actions[:, 0]
        if mpc.action_dim == 1:
            u = u[:, 0]
        obs, state, reward, _terminated, info = step(keys, state, u, params)
        rewards.append(reward)
        trips = trips + jnp.asarray(info["tripped"], dtype=jnp.int32)

    return (
        np.asarray(jnp.stack(values, axis=1)),
        np.asarray(jnp.stack(targets, axis=1)),
        np.asarray(jnp.stack(rewards, axis=1)),
        np.asarray(trips),
    )


def constant_policy(value, env, params) -> Callable:
    """A policy holding *value*, expressed as a fraction of the action range.

    ``value`` runs -1 to 1 and is mapped onto each dimension's own bounds, so
    one sweep specification is meaningful across environments whose actions
    are voltages, valve fractions or degrees of elevator.
    """
    low, high = _action_bounds(env, params)
    mid, half = (high + low) / 2.0, (high - low) / 2.0
    action = mid + float(value) * half
    return lambda _obs: jnp.asarray(action)


def pid_policy(spec) -> Callable | None:
    """The registered PID baseline, reset and ready."""
    if not spec.has_pid:
        return None
    pid = spec.make_pid()
    if hasattr(pid, "reset"):
        pid.reset()
    # Some factories return an object exposing ``step``, others a callable.
    call = pid if callable(pid) else pid.step
    return lambda obs: call(obs)


def mpc_policy(spec, env, params) -> Callable | None:
    """The registered MPC baseline, reset and ready."""
    from target_gym.experts.mpc import plan_params

    if not spec.has_mpc:
        return None
    mpc = spec.make_mpc(env, plan_params(spec, params))
    mpc.reset()

    def policy(obs, state):
        return np.atleast_1d(mpc.step(obs, state))

    # Expose the planner, so a caller can read its solver health afterwards.
    # mypy does not model attributes on function objects, hence the ignore.
    policy.controller = mpc  # type: ignore[attr-defined]
    return policy


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _tracked_labels(env, n: int) -> list[str]:
    """Axis labels for the tracked channels, if the environment names them."""
    names = getattr(env, "tracked_names", None)
    if names and len(names) == n:
        return list(names)
    return [f"tracked {i}" for i in range(n)] if n > 1 else ["tracked value"]


def figure_sweep(name: str, params=None, resolution: int = 9, plot: bool = True):
    """Constant-action sweep: what the plant does open-loop, across its range."""
    spec = REGISTRY[name]
    env = spec.make_env()
    params = params or _media_params(spec)
    levels = np.linspace(-1.0, 1.0, resolution)

    runs = [rollout(spec, params, constant_policy(u, env, params))[0] for u in levels]
    if not plot:
        return runs

    n_ch = runs[0].shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 4 * n_ch), squeeze=False)
    cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    for ch in range(n_ch):
        ax = axes[ch][0]
        for u, values in zip(levels, runs):
            ax.plot(values[:, ch], color=cmap(norm(u)), lw=1.2)
        ax.set_xlabel("Time step")
        ax.set_ylabel(_tracked_labels(env, n_ch)[ch])
    axes[0][0].set_title(f"{name}: open-loop response across the action range")
    fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes.ravel().tolist(),
        label="action (fraction of range)",
    )
    _save(fig, f"{FIGURE_DIR}/{name}/sweep")
    return runs


def figure_pid(name: str, params=None, n_seeds: int = 6, plot: bool = True):
    """Closed-loop PID response, one trace per sampled setpoint."""
    spec = REGISTRY[name]
    if not spec.has_pid:
        return None
    env = spec.make_env()
    params = params or _media_params(spec)

    policy = pid_policy(spec)
    assert policy is not None  # guarded by has_pid above
    runs = [rollout(spec, params, policy, seed=s) for s in range(n_seeds)]
    if not plot:
        return runs

    n_ch = runs[0][0].shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 4 * n_ch), squeeze=False)
    cmap = truncate_colormap(cm.viridis, 0.0, 0.85)
    for ch in range(n_ch):
        ax = axes[ch][0]
        for i, (values, targets, _) in enumerate(runs):
            c = cmap(i / max(len(runs) - 1, 1))
            ax.plot(values[:, ch], color=c, lw=1.2)
            # The setpoint is what the trace is trying to reach; drawing it
            # dashed in the same colour is what makes tracking error legible.
            ax.plot(targets[:, ch], color=c, lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Time step")
        ax.set_ylabel(_tracked_labels(env, n_ch)[ch])
    axes[0][0].set_title(f"{name}: PID tracking across {n_seeds} sampled setpoints")
    _save(fig, f"{FIGURE_DIR}/{name}/pid_response")
    return runs


#: Playback rate for the gallery clips. Ten, not the 30 the GIF was written at.
#: These are heavily time-lapsed already -- a 3D aircraft episode is 800 s of
#: flight -- so the constraint is legibility, not smoothness.
GIF_FPS = 10


_MEDIA_MIN_STEPS = 600
_MEDIA_MAX_STEPS = 1200

#: Environments whose motion is meant to be read as motion, and the clip's
#: playback speed for them.
#:
#: For these the clip shows the *opening* of an episode at a fixed speed-up
#: rather than a whole episode time-lapsed. An aircraft episode is 800 s of
#: flight; squeezed into a 200-frame clip at 10 fps it played at forty times
#: real time, which reads as an aerobatic display rather than an airliner on
#: 8 km lobes. Ten seconds of flight per second of playback is fast enough to
#: show a full turn and slow enough that the attitude changes are legible.
#:
#: The process environments deliberately keep the full-episode time-lapse.
#: Their subject is a setpoint change playing out over twenty minutes or six
#: hours, and the first ten seconds of one says nothing at all. The point of a
#: gallery clip is to convey the task, and for those the task *is* the whole
#: episode.
_MEDIA_REALTIME_GROUPS = ("plane", "patrol", "formation")
MEDIA_SPEEDUP = 10.0
MEDIA_SECONDS = 10.0


def _media_params(spec):
    """Parameters for a figure or a video: the scored configuration
    (``spec.make_test_params()``), with the episode lengthened to something
    worth watching."""
    params = spec.make_test_params()
    steps = int(params.max_steps_in_episode)
    default_steps = int(spec.params_cls().max_steps_in_episode)
    # Long enough to watch, short enough to read: the longer of the scored
    # and default lengths, then bounded (the aircraft default to 10,000).
    target = int(np.clip(max(steps, default_steps), _MEDIA_MIN_STEPS, _MEDIA_MAX_STEPS))
    if target != steps:
        params = params.replace(max_steps_in_episode=target)
    return params


def _clip_params(spec):
    """Parameters for a gallery clip, which is not the same as for a figure.

    A figure wants the whole episode: it is a record of what the controller did
    from start to finish. A clip wants whatever length reads best as a moving
    picture, and for the environments in ``_MEDIA_REALTIME_GROUPS`` that is a
    short opening at a fixed speed rather than the episode compressed to fit.
    """
    params = _media_params(spec)
    if not spec.name.startswith(_MEDIA_REALTIME_GROUPS):
        return params
    # One rendered frame per simulated step, so the speed-up is delta_t times
    # the frame rate and the length is however many steps fill MEDIA_SECONDS of
    # playback. The renderer's own frame_stride comes out at ``stride`` for
    # this length, which is what makes the arithmetic hold.
    dt = float(getattr(params, "delta_t", 1.0))
    stride = max(1, round(MEDIA_SPEEDUP / (dt * GIF_FPS)))
    steps = int(round(MEDIA_SECONDS * GIF_FPS * stride))

    # A task whose setpoint moves needs the clip to contain a change, or it
    # shows an aircraft holding a level and says nothing about what is being
    # asked of it. Ten times real time covers 100 s; a level lasts 300 s. So
    # the clip is lengthened to two of them and the speed-up rises to suit --
    # the aircraft is pinned mid-panel in these views and has no attitude to
    # misread, which is what made a fast time-lapse unwatchable on the 3D tasks.
    pattern = int(getattr(params, "target_pattern", 0))
    overrides = {}
    if pattern == 1:
        # The ladder's tread is the episode divided by ``target_steps``, so
        # simply shortening the episode would compress the schedule and show a
        # faster sequence of levels than the environment ever asks for. Scaling
        # ``target_steps`` with the clip keeps each tread the length it really
        # has, and the clip is then a window onto the task rather than a
        # different one.
        spec_params = spec.make_test_params()
        tread = float(spec_params.max_steps_in_episode) / max(
            float(getattr(spec_params, "target_steps", 1.0)), 1.0
        )
        steps = max(steps, int(round(2.0 * tread)))
        overrides["target_steps"] = max(1, int(round(steps / tread)))
    elif pattern in (3, 4):
        # The sinusoid and the chirp are written against ``target_period`` in
        # seconds, so they keep their shape whatever the episode length.
        period = float(getattr(params, "target_period", 0.0))
        if period > 0:
            steps = max(steps, int(round(2.0 * period / dt)))

    return params.replace(max_steps_in_episode=steps, **overrides)


def figure_comparison(name: str, params=None, n_seeds: int = 5, plot: bool = True):
    """Cumulative return of the best constant action, the PID and the MPC."""
    spec = REGISTRY[name]
    env = spec.make_env()
    params = params or _media_params(spec)

    # Bracketing constants rather than a fine sweep: the point of the bar is
    # that the baselines beat open loop, not to find the optimal constant.
    constants = (-0.5, 0.0, 0.5)
    scores: dict[str, list[float]] = {"Constant": [], "PID": [], "MPC": []}
    for seed in range(n_seeds):
        best = max(
            float(rollout(spec, params, constant_policy(c, env, params), seed)[2].sum())
            for c in constants
        )
        scores["Constant"].append(best)
        pid = pid_policy(spec)
        if pid is not None:
            scores["PID"].append(float(rollout(spec, params, pid, seed)[2].sum()))
        mpc = mpc_policy(spec, env, params)
        if mpc is not None:
            scores["MPC"].append(float(rollout(spec, params, mpc, seed)[2].sum()))
    scores = {k: v for k, v in scores.items() if v}
    if not plot:
        return scores

    labels = list(scores)
    means = [float(np.mean(scores[k])) for k in labels]
    stds = [float(np.std(scores[k])) for k in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=["steelblue", "darkorange", "seagreen"][: len(labels)],
        alpha=0.85,
    )
    rng = np.random.default_rng(0)
    for i, k in enumerate(labels):
        ax.scatter(
            x[i] + rng.uniform(-0.15, 0.15, len(scores[k])),
            scores[k],
            color="black",
            s=20,
            alpha=0.5,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cumulative reward")
    ax.set_title(f"{name}: cumulative reward over {n_seeds} seeds (mean ± std)")
    _save(fig, f"{FIGURE_DIR}/{name}/comparison")
    return scores


def _save(fig, stem: str) -> None:
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


def retime_gif(path: str, fps: int = GIF_FPS) -> str:
    """Rewrite a GIF's frame delays in place, leaving the frames alone.

    Slowing playback is free, where adding frames grows the file. Done here
    rather than in ``utils.save_video`` because ``utils`` is in the
    fingerprints, and editing it would stale every recorded baseline.
    """
    from PIL import Image, ImageSequence

    with Image.open(path) as im:
        frames = [f.copy() for f in ImageSequence.Iterator(im)]
    if not frames:
        return path
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / max(fps, 1))),
        loop=0,
        optimize=True,
    )
    return path


def video(name: str, params=None, seed: int = 0) -> str | None:
    """Render one PID episode to ``videos/<name>/pid_output.gif``."""
    spec = REGISTRY[name]
    policy = pid_policy(spec)
    if policy is None:
        return None
    env = spec.make_env()
    params = params or _clip_params(spec)
    folder = f"{VIDEO_DIR}/{name}"
    os.makedirs(folder, exist_ok=True)
    # FPS=30, the rate ``utils.save_video`` writes at; at the default 60,
    # moviepy would drop every other frame.
    written = env.save_video(
        policy, seed, params=params, folder=folder, format="gif", FPS=30
    )
    # save_video names its output episode_000.gif; the gallery and
    # scripts/shorten_gifs.py both expect pid_output.gif.
    final = os.path.join(folder, "pid_output.gif")
    os.replace(written, final)
    return retime_gif(final)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_figures(envs: Sequence[str] | None = None) -> None:
    for name in tqdm(list(envs or REGISTRY), desc="figures"):
        tqdm.write(f"\n── {name} ──")
        figure_sweep(name)
        figure_pid(name)
        figure_comparison(name)


def run_videos(envs: Sequence[str] | None = None) -> None:
    for name in tqdm(list(envs or REGISTRY), desc="videos"):
        tqdm.write(f"\n── {name} ──")
        video(name)


def run_all(envs: Sequence[str] | None = None) -> None:
    run_figures(envs)
    run_videos(envs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate figures and videos for target-gym environments."
    )
    parser.add_argument(
        "--env",
        nargs="*",
        choices=list(REGISTRY),
        default=None,
        metavar="ENV",
        help="environments to run (default: all)",
    )
    parser.add_argument(
        "--only",
        choices=["videos", "figures"],
        default=None,
        help="run only videos or only figures (default: both)",
    )
    args = parser.parse_args()
    {"videos": run_videos, "figures": run_figures}.get(args.only, run_all)(args.env)


if __name__ == "__main__":
    main()
