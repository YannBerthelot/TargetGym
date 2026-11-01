"""Smoke check on the installed 0.6 package (RELEASING.md §2, step 4).

The release's package gate (``.github/workflows/package.yml``) installs the
built wheel in a fresh environment outside the checkout and runs this with
that environment's Python:

    python tests/release_smoke.py core

It is a script, not a pytest module, so the suite never collects it. This is
the ``release/0.6`` copy, written against 0.6's API (``REGISTRY``, ``make_pid``).

Contract (``docs/getting-started.md`` and ``target_gym/registry.py`` in 0.6):
every registered task builds from its spec, resets, and its shipped PID, when
it has one, flies the first 50 steps of a scored episode with finite rewards;
the Gymnasium plane wrapper steps.
"""

from __future__ import annotations

import os
import sys

# Before JAX and pygame are imported: the gate's runners have no GPU and no
# display.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax  # noqa: E402
import numpy as np  # noqa: E402

import target_gym  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

STEPS = 50


def check_core():
    assert REGISTRY, "the registry is empty"
    for name, spec in REGISTRY.items():
        env, params = spec.make_env(), spec.make_test_params()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        pid = spec.make_pid()
        if pid is not None:
            pid.reset()
        rewards = []
        for _ in range(STEPS):
            key, step_key = jax.random.split(key)
            if pid is not None:
                action = pid(np.asarray(obs))
            else:
                action = env.action_space(params).sample(step_key)
            obs, state, reward, _, _, _ = env.step(step_key, state, action, params)
            rewards.append(float(reward))
        assert np.isfinite(rewards).all(), f"{name}: a reward is not finite"
        print(f"{name}: {STEPS} steps, return {sum(rewards):.6g}")

    gym_env = target_gym.GymnasiumPlane()
    obs, _ = gym_env.reset(seed=0)
    obs, reward, terminated, truncated, _ = gym_env.step(gym_env.action_space.sample())
    assert np.isfinite(reward) and gym_env.observation_space.contains(obs)


CHECKS = {"core": check_core}

if __name__ == "__main__":
    CHECKS[sys.argv[1]]()
    print(f"{sys.argv[1]}: ok")
