"""Figures and videos for every registered environment.

One runner, driven by :data:`target_gym.registry.REGISTRY`. It replaced seven
per-environment modules that shared an identical six-function skeleton and
between them covered eight environments; this covers all of them, because
nothing here names an environment.

Three pieces of the environment interface make that possible:

``obs_value_index`` / ``obs_target_index``
    Class attributes on every environment giving the observation slots that
    hold the tracked variable and its setpoint (a tuple of slots for the
    multi-loop plants). Reading the pair back out of the observation is what
    lets a plot label itself without knowing which plant it is looking at.
``action_space(params)``
    Supplies the bounds a constant-action sweep should span.
``EnvSpec.make_pid`` / ``make_mpc``
    The baselines, already registered.

Setpoints are varied by *seed* rather than by writing into a named state
field: every environment samples its own target on reset, so a handful of
seeds gives a spread drawn from the distribution the environment actually
defines, and no per-environment field name is needed.

Usage
-----
    python -m target_gym.runners.runners                      # everything
    python -m target_gym.runners.runners --env cstr reactor
    python -m target_gym.runners.runners --only figures
"""

from __future__ import annotations

import argparse
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


def _wants_state(policy: Callable) -> bool:
    """True when *policy* takes ``(obs, state)`` rather than ``(obs,)``.

    Decided from the signature rather than by calling and catching TypeError,
    which would also swallow a TypeError raised legitimately inside a
    single-argument policy and silently call it with the wrong arity.
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


def rollout(spec, params, policy: Callable, seed: int = 0):
    """Run one episode, returning tracked values, targets and rewards.

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
    step = jax.jit(env.step_env)

    values, targets, rewards = [], [], []
    for _ in range(int(params.max_steps_in_episode)):
        obs_np = np.asarray(obs)
        values.append(obs_np[list(value_idx)])
        targets.append(obs_np[list(target_idx)])
        # Hand the policy the NumPy view, which this loop has already paid for.
        # The stateful PIDs are plain Python doing scalar arithmetic; given a JAX
        # array every operation -- the indexing, the clip, the anti-windup
        # ``where`` -- is a separate un-jitted dispatch, and the controller ends
        # up costing several times the environment step it is controlling.
        action = policy(obs_np, state) if _wants_state(policy) else policy(obs_np)
        obs, state, reward, terminated, _ = step(
            key, state, jnp.atleast_1d(jnp.asarray(action)), params
        )
        rewards.append(float(reward))
        if bool(terminated):
            break
    return np.array(values), np.array(targets), np.array(rewards)


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
    if not spec.has_mpc:
        return None
    mpc = spec.make_mpc(env, params)
    mpc.reset()
    return lambda obs, state: np.atleast_1d(mpc.step(obs, state))


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
    params = params or spec.params_cls()
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
    params = params or spec.params_cls()

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


def figure_comparison(name: str, params=None, n_seeds: int = 5, plot: bool = True):
    """Cumulative return of the best constant action, the PID and the MPC."""
    spec = REGISTRY[name]
    env = spec.make_env()
    params = params or spec.params_cls()

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


def video(name: str, params=None, seed: int = 0) -> str | None:
    """Render one PID episode to ``videos/<name>/pid_output.gif``."""
    spec = REGISTRY[name]
    policy = pid_policy(spec)
    if policy is None:
        return None
    env = spec.make_env()
    params = params or spec.params_cls()
    folder = f"{VIDEO_DIR}/{name}"
    os.makedirs(folder, exist_ok=True)
    written = env.save_video(policy, seed, params=params, folder=folder, format="gif")
    # save_video names its output episode_000.gif; the gallery and
    # scripts/shorten_gifs.py both expect pid_output.gif.
    final = os.path.join(folder, "pid_output.gif")
    os.replace(written, final)
    return final


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
