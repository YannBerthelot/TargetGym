"""Measure the shared oracle solver on glass_furnace's planner problem (G7).

Run with ``uv run python scripts/measure_oracle_solver.py [--warm N]
[--no-limits]``. It prints one JSON line: compile times, peak memory,
iterations, KKT errors, per-solve times, and the per-episode time they imply.
A measurement, not an oracle (docs/proposals/oracle-solver.md, decision G7):
nothing here is shipped or recorded.

The planner problem, built with ``target_gym.oracles.shooting``:

- model: ``compute_next_state`` on the planner's params (``plan_params``),
  in float64, with the trip and reset kernel dropped (T3);
- input: the raw firing in [-1, 1], bounded, with the clip removed from the
  planner's model only (T1);
- cost: tracking as (e / e_floor)^2, exact since e_tol = 0 and p = 2 (T2);
  the running cost max(fuel - c_hold, 0) through epigraph rows on every
  visited state (T2);
- limits: the crown, melt and working-end bands on every visited state,
  elastic at the trip cost per degree (T3; ``--no-limits`` drops them);
- the two reachable batch clips stay as they are (smoothing them is T5);
- horizon: the probe's provisional T_c = 13 steps and 62 stages (assumed;
  the horizon protocol has not run).

One cold solve from training seed 100's reset, then ``--warm`` warm solves,
each one T_c later on the plant (the env with its test params) after the
plan's first move was held over that T_c.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from target_gym import registry
from target_gym.experts.mpc import _from_x, _to_x, plan_params
from target_gym.glass_furnace import env as glass
from target_gym.oracles.shooting import Hinge, Limit, Shooting, StageModel
from target_gym.reward import trip_cost

T_C, STAGES, SEED = 13, 62, 100  # provisional (oracle-solver.md, section 9)


def _float64(tree):
    def cast(a):
        a = jnp.asarray(a)
        if jnp.issubdtype(a.dtype, jnp.floating):
            return a.astype(jnp.float64)
        return a.astype(jnp.int64) if a.dtype == jnp.int32 else a

    return jax.tree_util.tree_map(cast, tree)


@contextlib.contextmanager
def unclipped_actions():
    """T1 for the planner's model: the raw-action map without its clip, while
    the planner is traced. The plant keeps its clip."""
    clipped = glass.convert_raw_action_to_range

    def unclipped(raw, min_action, max_action):
        return min_action + 0.5 * (raw + 1.0) * (max_action - min_action)

    glass.convert_raw_action_to_range = unclipped
    try:
        yield
    finally:
        glass.convert_raw_action_to_range = clipped


def planner_model(env, plan, limits: bool) -> StageModel:
    p = plan
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0
    key = jax.random.PRNGKey(0)  # the pull noise is zeroed by plan_params

    def step(data, x, u, t):
        st = _from_x(x, data).replace(time=data.time + t)
        new, _ = glass.compute_next_state.__wrapped__(
            u[0], st, p, key, integration_method=env.integration_method
        )
        cost = ((new.target_T_crown - new.T_crown) / p.e_floor) ** 2
        y = [p.running_weight * (new.fuel_flow - p.c_hold) / p.c_hold]
        if limits:
            y += [new.T_crown, new.T_melt, new.T_work]
        return _to_x(new), cost, jnp.stack(y)

    price = float(trip_cost(p))
    bands = (
        Limit(p.T_crown_min, p.T_crown_max, price),
        Limit(p.T_glass_min, p.T_glass_max, price),
        Limit(p.T_glass_min, p.T_glass_max, price),
    )
    return StageModel(
        step=step, block=T_C, lo=(-1.0,), hi=(1.0,), hinges=(Hinge(weight=1.0),),
        limits=bands if limits else (),
    )  # fmt: skip


def _rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss * (1 if sys.platform == "darwin" else 1024) / 1e9, 2)


def main(warm: int, limits: bool) -> dict:
    spec = registry.get("glass_furnace")
    env, params = spec.make_env(), spec.make_test_params()
    with enable_x64():
        params = _float64(params)
        plan_p = _float64(plan_params(spec, params))
        _, state = env.reset_env(jax.random.PRNGKey(SEED), params)
        state = _float64(state)
        planner = Shooting(
            planner_model(env, plan_p, limits), STAGES, _to_x(state).size
        )
        u_hold = jnp.zeros(1)  # reset holds the nominal firing, raw 0
        plant = jax.jit(env.step_env)
        out = {"task": "glass_furnace", "T_c": T_C, "stages": STAGES,
               "limits": limits, "n": planner.n}  # fmt: skip

        def compile_(*args):
            t = time.perf_counter()
            with unclipped_actions():
                fn = planner.solve.lower(*args).compile()
            return fn, round(time.perf_counter() - t, 1)

        cold, out["compile_cold_s"] = compile_(state, _to_x(state), u_hold)
        out["rss_after_cold_compile_gb"] = _rss_gb()
        rows, plan, fn = [], None, cold
        for i in range(warm + 1):
            args = (state, _to_x(state), u_hold) + ((plan,) if plan else ())
            if i == 1:
                fn, out["compile_warm_s"] = compile_(*args)
            t = time.perf_counter()
            plan = fn(*args)
            jax.block_until_ready(plan.z)
            c, dt = plan.certificate, time.perf_counter() - t
            rows.append({
                "s": round(dt, 2), "iterations": int(c.iterations),
                "s_per_iteration": round(dt / max(int(c.iterations), 1), 3),
                "error": float(c.error), "capped": bool(c.capped),
                "failed": bool(c.failed), "violation": float(c.violation),
                "shift": float(c.shift),
            })  # fmt: skip
            u = planner.moves(plan)[0]
            for j in range(T_C):
                key = jax.random.fold_in(jax.random.PRNGKey(SEED), i * T_C + j)
                state = _float64(plant(key, state, u, params)[1])
        out["solves"] = rows
        warm_s = [r["s"] for r in rows[1:]] or [rows[0]["s"]]
        out["episode_s"] = round(
            int(params.max_steps_in_episode) / T_C * np.mean(warm_s)
        )
        out["peak_rss_gb"] = _rss_gb()
        out["load_average"] = [round(v, 1) for v in os.getloadavg()]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--warm", type=int, default=3, help="warm solves after the cold")
    ap.add_argument("--no-limits", action="store_true", help="drop the trip bands")
    a = ap.parse_args()
    print(json.dumps(main(a.warm, not a.no_limits)))
