"""Throughput benchmark for the registered environments.

Reports steps per second for a vmapped, jitted, scanned rollout -- the way an
RL loop actually drives these environments, not one Python-level step at a
time.

Usage
-----
    python -m target_gym.benchmark_speed                 # every environment
    python -m target_gym.benchmark_speed --envs hvac cstr
    python -m target_gym.benchmark_speed --batch 512 --steps 2000 --markdown
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp


def benchmark_env(env, params, steps: int = 1000, batch_size: int = 256) -> float:
    """Steps per second for ``batch_size`` environments run ``steps`` deep.

    Uses ``step_env`` rather than ``step``: the latter adds gymnax's auto-reset,
    which is real work an RL loop does but is not the environment's own cost,
    and it returns six values rather than five.
    """
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size)
    _, state = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)

    action_shape = env.action_space(params).shape
    actions = jnp.zeros((batch_size,) + tuple(action_shape))

    def rollout(state):
        def step_fn(carry, _):
            new_state = jax.vmap(env.step_env, in_axes=(None, 0, 0, None))(
                key, carry, actions, params
            )[1]
            return new_state, None

        return jax.lax.scan(step_fn, state, None, length=steps)[0]

    jitted = jax.jit(rollout)
    jax.block_until_ready(jitted(state))  # warm up / compile

    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(state))
        dt = time.perf_counter() - t0
        best = max(best, steps * batch_size / dt)
    return best


def main() -> None:
    from target_gym.registry import REGISTRY, all_specs

    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=[s.name for s in all_specs()])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--markdown", action="store_true", help="emit a README cell")
    args = ap.parse_args()

    results = {}
    for name in args.envs:
        spec = REGISTRY[name]
        env = spec.make_env()
        params = spec.params_cls().replace(max_steps_in_episode=args.steps + 10)
        try:
            rate = benchmark_env(env, params, steps=args.steps, batch_size=args.batch)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {name:22s} FAILED: {exc}")
            continue
        results[name] = rate
        print(f"  {name:22s} {rate / 1e6:8.2f} M steps/s")

    if args.markdown:
        print("\nREADME cells:")
        for name, rate in results.items():
            print(f"  {name:22s} ~{rate / 1e6:.2f}M")


if __name__ == "__main__":
    main()
