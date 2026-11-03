"""Smoke checks on the installed package, one per install (RELEASING.md §2, step 4).

The release's package gate (``.github/workflows/package.yml``) installs the
built wheel in a fresh environment outside the checkout and runs this with
that environment's Python, once per install:

    python tests/release_smoke.py core     # target-gym alone
    python tests/release_smoke.py render   # target-gym[render]

It is a script, not a pytest module, so the suite never collects it.

Contract (the public API in ``target_gym/__init__.py`` and ``api.py``, the
extras in ``pyproject.toml``, and the CHANGELOG's promises for them):

core
    Importing ``target_gym`` loads neither extra. ``list_envs()`` names every
    task, and on each one the expert flies a whole scored episode within the
    action bounds and returns, on seed 0, the PID return recorded for it,
    which also shows the wheel carries the per-task record files.
    ``make_gymnasium`` and ``FormationMARL`` step. ``make_oracle`` builds
    every oracle, and each plans a finite action.
render
    With the extra, an aircraft and a process plant each write a video.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

# Before JAX, pygame and matplotlib are imported: the gate's runners have no
# GPU and no display.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax  # noqa: E402
import numpy as np  # noqa: E402

import target_gym as tg  # noqa: E402

EXTRA_MODULES = ("pygame", "moviepy")

# The recorded PID returns were taken by the host controller on GitHub's Linux
# runners. The jitted expert matches the host one to float32 rounding, and the
# same commit on runners with different CPUs has moved a return by 0.12%
# (plane3d_racetrack, 12.661 to 12.676, on 2026-10-29), so 1% separates that
# drift from a wheel whose physics, gains or records differ from the tree's.
RETURN_RTOL = 1e-2


def _expert_episode(name):
    """Seed 0's scored episode, flown by the expert in one jitted scan."""
    from target_gym.benchmark import episode_keys

    env, params = tg.make(name)
    expert = tg.make_expert(name)
    space = env.action_space(params)
    reset_keys, step_keys = episode_keys(
        jax.random.PRNGKey(0), 1, int(params.max_steps_in_episode)
    )

    def step(carry, step_key):
        obs, env_state, memory = carry
        action, memory = expert.step(memory, obs)
        obs, env_state, reward, _, _ = env.step_env(step_key, env_state, action, params)
        return (obs, env_state, memory), (action, reward)

    @jax.jit
    def episode():
        obs, env_state = env.reset_env(reset_keys[0], params)
        return jax.lax.scan(step, (obs, env_state, expert.reset()), step_keys[:, 0])

    _, (actions, rewards) = episode()
    actions, rewards = np.asarray(actions), np.asarray(rewards)
    assert np.isfinite(rewards).all(), f"{name}: a reward is not finite"
    low, high = np.asarray(space.low), np.asarray(space.high)
    assert (actions >= low - 1e-6).all() and (
        actions <= high + 1e-6
    ).all(), f"{name}: the expert left the action bounds"
    return float(rewards.sum())


def check_core():
    loaded = [m for m in EXTRA_MODULES if m in sys.modules]
    assert not loaded, f"importing target_gym loaded {loaded}"

    from target_gym.provenance import load_recorded_baselines

    recorded = load_recorded_baselines()
    names = tg.list_envs()
    assert names, "list_envs() is empty"
    for name in names:
        expected = recorded[name]["pid_returns"][0]
        got = _expert_episode(name)
        assert np.isclose(
            got, expected, rtol=RETURN_RTOL
        ), f"{name}: the expert returns {got:.6g} on seed 0, recorded {expected:.6g}"
        print(f"{name}: expert episode {got:.6g} (recorded {expected:.6g})")

    gym_env = tg.make_gymnasium("plane")
    obs, _ = gym_env.reset(seed=0)
    assert gym_env.observation_space.contains(obs)
    obs, reward, terminated, truncated, _ = gym_env.step(gym_env.action_space.sample())
    assert np.isfinite(reward) and gym_env.observation_space.contains(obs)

    marl = tg.FormationMARL()
    _, marl_params = tg.make("formation")
    key = jax.random.PRNGKey(0)
    obs, state = marl.reset(key, marl_params)
    actions = {agent: marl.action_space(agent).sample(key) for agent in marl.agents}
    obs, state, rewards, dones, _ = marl.step(key, state, actions, marl_params)
    assert set(rewards) == set(marl.agents) and "__all__" in dones

    from target_gym.registry import REGISTRY

    for name in names:
        if not REGISTRY[name].has_mpc:
            continue
        oracle = tg.make_oracle(name)
        env, params = tg.make(name)
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        action = np.asarray(oracle.step(obs, state))
        assert np.isfinite(action).all(), f"{name}: the oracle planned {action}"
        print(f"{name}: oracle planned {action}")


def check_render():
    with tempfile.TemporaryDirectory() as folder:
        for name in ("plane", "cstr"):
            env, params = tg.make(name, max_steps_in_episode=10)
            path = env.save_video(
                tg.make_expert(name),
                seed=0,
                folder=folder,
                episode_index=0,
                params=params,
                format="gif",
            )
            size = pathlib.Path(path).stat().st_size
            assert size > 0, f"{name}: the video at {path} is empty"
            print(f"{name}: wrote {size} bytes")


CHECKS = {"core": check_core, "render": check_render}

if __name__ == "__main__":
    CHECKS[sys.argv[1]]()
    print(f"{sys.argv[1]}: ok")
