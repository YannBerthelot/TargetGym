"""
Benchmark steps/sec across all TargetGym envs and capture trajectory hashes
for correctness verification across optimization passes.

Usage:
    python scripts/bench_envs.py --tag before --out bench_before.json
    # ...apply optimizations...
    python scripts/bench_envs.py --tag after  --out bench_after.json
    python scripts/bench_envs.py --diff bench_before.json bench_after.json
"""

import argparse
import hashlib
import json
import os
import time

# Pin to CPU for reproducible timing (the README's published numbers are CPU).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import (
    CSTR,
    CSTRParams,
    FirstOrderParams,
    FirstOrderSystem,
    FourTank,
    FourTankParams,
    GlassFurnace,
    GlassFurnaceParams,
    Plane,
    Plane3DCircle,
    Plane3DFigureEight,
    Plane3DHeading,
    PlaneParams,
    PlaneParams3D,
    Reactor,
    ReactorParams,
)


# Each env spec: (name, env, params, action_dim).
ENVS = [
    ("FirstOrder", FirstOrderSystem(), FirstOrderParams(max_steps_in_episode=200), 1),
    ("CSTR", CSTR(), CSTRParams(max_steps_in_episode=100), 1),
    ("FourTank", FourTank(), FourTankParams(max_steps_in_episode=500), 2),
    ("GlassFurnace", GlassFurnace(), GlassFurnaceParams(max_steps_in_episode=2_000), 1),
    ("Reactor", Reactor(), ReactorParams(max_steps_in_episode=2_000), 1),
    ("Plane", Plane(), PlaneParams(max_steps_in_episode=10_000), 2),
    (
        "Plane3DHeading",
        Plane3DHeading(),
        PlaneParams3D(max_steps_in_episode=10_000),
        3,
    ),
    ("Plane3DCircle", Plane3DCircle(), PlaneParams3D(max_steps_in_episode=10_000), 3),
    (
        "Plane3DFigureEight",
        Plane3DFigureEight(),
        PlaneParams3D(max_steps_in_episode=10_000),
        3,
    ),
]


def build_rollout(env, action_dim: int, n_steps: int, batch_size: int):
    """Return a JIT-compiled function that runs n_steps of the env in parallel."""

    def rollout(key, params):
        keys = jax.random.split(key, batch_size)
        obs, state = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
        actions = jnp.zeros((batch_size, action_dim), dtype=jnp.float32)

        def step_fn(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            step_keys = jax.random.split(subkey, batch_size)
            obs, new_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_keys, state, actions, params)
            return (new_state, key), reward

        (final_state, _), rewards = jax.lax.scan(
            step_fn, (state, key), xs=None, length=n_steps
        )
        return final_state, rewards

    return jax.jit(rollout)


def array_hash(x) -> str:
    """Stable hash of a jax/numpy array (rounded to 6 decimals to be JIT-stable)."""
    a = np.asarray(x).astype(np.float64)
    a = np.round(a, 6)
    return hashlib.sha1(a.tobytes()).hexdigest()


def array_summary(x):
    """Summary stats for tolerance-based comparison across runs."""
    a = np.asarray(x).astype(np.float64)
    return {
        "shape": list(a.shape),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "sum": float(a.sum()),
    }


def bench_env(name, env, params, action_dim, n_steps, batch_size):
    print(f"  building rollout ...", flush=True)
    rollout = build_rollout(env, action_dim, n_steps, batch_size)
    key = jax.random.PRNGKey(0)

    # Warm up (JIT compile)
    print(f"  warmup ...", flush=True)
    t0 = time.perf_counter()
    state, rewards = rollout(key, params)
    rewards.block_until_ready()
    t_warm = time.perf_counter() - t0

    # Timed runs.  Two issues bias quick benches:
    #   (1) XLA caching → bimodal timings (one lucky fast run); min-of-N
    #       picks that outlier and reports misleading numbers.
    #   (2) System noise (other processes, thermals) → upper-tail outliers.
    # We measure for a fixed wall-clock budget: 3 seconds per env or at least
    # 5 trials, whichever takes longer. Then take the lower-quartile time,
    # which discards lucky cache-hits and the slow tail.
    target_wall_s = 3.0
    min_trials, max_trials = 5, 25
    times = []
    elapsed = 0.0
    while len(times) < min_trials or (
        elapsed < target_wall_s and len(times) < max_trials
    ):
        t0 = time.perf_counter()
        state, rewards = rollout(key, params)
        rewards.block_until_ready()
        dt = time.perf_counter() - t0
        times.append(dt)
        elapsed += dt
    times_sorted = sorted(times)
    best = times_sorted[len(times_sorted) // 4]  # lower-quartile estimator
    total_steps = n_steps * batch_size
    sps = total_steps / best

    # Trajectory hash + summary stats for correctness check
    h = array_hash(rewards)
    summary = array_summary(rewards)
    print(
        f"  {name:18s}  warm={t_warm:7.2f}s  best={best:7.3f}s  n={len(times):2d}  "
        f"steps/s={sps:11,.0f}  hash={h[:12]}  sum={summary['sum']:.6e}",
        flush=True,
    )
    return {
        "name": name,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "warmup_s": t_warm,
        "best_s": best,
        "all_s": times,
        "steps_per_s": sps,
        "rewards_hash": h,
        "rewards_summary": summary,
    }


def run_all(n_steps: int, batch_size: int):
    results = []
    for name, env, params, action_dim in ENVS:
        print(f"== {name} ==", flush=True)
        results.append(bench_env(name, env, params, action_dim, n_steps, batch_size))
    return results


def diff_two(a_path, b_path):
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    a_idx = {r["name"]: r for r in a["results"]}
    b_idx = {r["name"]: r for r in b["results"]}
    print(
        f"\n{'env':18s}  {'before':>14s}  {'after':>14s}  "
        f"{'speedup':>8s}  match  rel_sum_diff"
    )
    print("-" * 80)
    for name in a_idx:
        ar = a_idx[name]
        br = b_idx.get(name)
        if br is None:
            print(f"{name:18s}  (missing in after)")
            continue
        speed_a = ar["steps_per_s"]
        speed_b = br["steps_per_s"]
        ratio = speed_b / speed_a if speed_a else float("nan")
        hash_match = ar["rewards_hash"] == br["rewards_hash"]
        sa = ar.get("rewards_summary")
        sb = br.get("rewards_summary")
        rel_diff = float("nan")
        tol_ok = "?"
        if sa and sb:
            denom = max(abs(sa["sum"]), 1e-12)
            rel_diff = abs(sa["sum"] - sb["sum"]) / denom
            mean_diff = abs(sa["mean"] - sb["mean"])
            std_diff = abs(sa["std"] - sb["std"])
            # Tolerate 1e-4 relative drift in summary stats (FP rounding from
            # changing **10 vs float_power, or scan-vs-direct call ordering).
            tol_ok = (
                "OK"
                if (rel_diff < 1e-4 and mean_diff < 1e-5 and std_diff < 1e-5)
                else "DRIFT"
            )
        flag = "EXACT" if hash_match else tol_ok
        print(
            f"{name:18s}  {speed_a:>14,.0f}  {speed_b:>14,.0f}  "
            f"{ratio:>7.2f}x  {flag:>5s}  {rel_diff:.2e}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--diff", nargs=2, default=None, help="Print speedup table from two json files."
    )
    args = p.parse_args()

    if args.diff:
        diff_two(*args.diff)
        return

    print(
        f"Running bench tag={args.tag}  n_steps={args.n_steps}  "
        f"batch_size={args.batch_size}  device={jax.devices()[0]}"
    )
    results = run_all(args.n_steps, args.batch_size)
    out = {
        "tag": args.tag,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "device": str(jax.devices()[0]),
        "results": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
