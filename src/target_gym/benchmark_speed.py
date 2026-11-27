"""Throughput benchmark for the registered environments.

Reports steps per second for a vmapped, jitted, scanned rollout, the way an RL
loop drives these environments: gymnax's ``step`` with its auto-reset, a key of
its own for every environment and step, random actions inside the action space,
and everything ``step`` returns kept (outside review M4, 2026-11-27). A run that
kept only the state, with one key and zero actions, let the compiler drop the
observation and the reward, and overstated the rate 6 to 45 times.

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


def draws(env, params, key, steps: int, batch_size: int):
    """``(keys, actions)``: a key for every step and environment, ``(steps,
    batch_size, 2)``, and an action drawn uniformly inside the action space for
    each, ``(steps, batch_size, *action_shape)``."""
    space = env.action_space(params)
    shape = tuple(space.shape)
    k_steps, k_actions = jax.random.split(key)
    keys = jax.random.split(k_steps, steps * batch_size).reshape(steps, batch_size, -1)
    low = jnp.broadcast_to(jnp.asarray(space.low, jnp.float32), shape)
    high = jnp.broadcast_to(jnp.asarray(space.high, jnp.float32), shape)
    actions = jax.random.uniform(
        k_actions, (steps, batch_size) + shape, minval=low, maxval=high
    )
    return keys, actions


def rollout(env, params, state, keys, actions):
    """``(state, totals)``: gymnax's ``step`` on every environment of ``state``,
    one step per row of ``keys`` and ``actions`` (``draws``), and the sums of
    everything it returned: the observations, rewards, terminations,
    truncations and every array of ``info``."""
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def body(carry, xs):
        state, totals = carry
        k, a = xs
        obs, state, reward, terminated, truncated, info = step(k, state, a, params)
        seen = {
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": jax.tree.leaves(info),
        }
        totals = {
            name: totals[name] + sum(jnp.sum(x) for x in jax.tree.leaves(v))
            for name, v in seen.items()
        }
        return (state, totals), None

    zero = jnp.zeros((), jnp.float32)
    totals = dict.fromkeys(("obs", "reward", "terminated", "truncated", "info"), zero)
    (state, totals), _ = jax.lax.scan(body, (state, totals), (keys, actions))
    return state, totals


def benchmark_env(env, params, steps: int = 1000, batch_size: int = 256) -> float:
    """Steps per second for ``batch_size`` environments run ``steps`` deep, the
    best of three timed runs of :func:`rollout` after one that compiles it."""
    key = jax.random.PRNGKey(0)
    k_reset, k_run = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, batch_size)
    _, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)
    keys, actions = draws(env, params, k_run, steps, batch_size)

    jitted = jax.jit(lambda s, k, a: rollout(env, params, s, k, a))
    jax.block_until_ready(jitted(state, keys, actions))  # warm up / compile

    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(state, keys, actions))
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
        # The scored configuration, its episodes ending and resetting as they do
        # in training.
        params = spec.make_test_params()
        try:
            rate = benchmark_env(env, params, steps=args.steps, batch_size=args.batch)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {name:22s} FAILED: {exc}", flush=True)
            continue
        results[name] = rate
        # Four decimals: the slowest plants step tens of thousands a second.
        print(f"  {name:22s} {rate / 1e6:9.4f} M steps/s", flush=True)

    if args.markdown:
        print("\nREADME cells:")
        for name, rate in results.items():
            print(f"  {name:22s} ~{rate / 1e6:.3g}M")


if __name__ == "__main__":
    main()
