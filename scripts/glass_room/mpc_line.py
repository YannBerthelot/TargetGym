"""Glass room study: the MPC line, what the true state gives without foresight. The
oracle without a seed as the library ships it (``glass_furnace/oracle.py``: it reads
the true state and plans on the walks' mean path with the events known) on one scenario
and seed of the ladder's suite, run through ``benchmark.run_policy`` at sensor noise 1
as the ceiling's plans are replayed. At 7143497 that oracle planned once a day on hard
limits, which is the report's MPC line; it now plans every knot and prices the chance
of a lost step. Beside its cost: the steps under a mark, and how far the fining share
and the canal came out from each plan's own forecast over the steps it was played.
``FINING_BACKOFF`` (share) and ``CANAL_BACKOFF`` (K) replace the oracle's own.

Run: PYTHONPATH=src python scripts/glass_room/mpc_line.py SCENARIO SEED OUT_DIR
         [MAX_ITER [FINING_BACKOFF [CANAL_BACKOFF]]]
"""

import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import lib as L  # noqa: E402
from glass_slow_oracle import ITER_CAP, under_a_mark  # noqa: E402

from target_gym.benchmark import run_policy, task_info  # noqa: E402
from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.glass_furnace import oracle as O  # noqa: E402

FIELDS = ("status", "iterations", "capped", "kkt", "violation", "how", "cost", "lost_steps")  # fmt: skip


def reading(name):
    return lambda s, p, tripped: L.P.sensors(s.plant, p.grid, p.plant)[name]


class Line:
    """The oracle as a ``benchmark.Policy``, keeping the step and state of each plan."""

    oracle = True

    def __init__(self, mpc, params):
        self.mpc, self.made, self.under = mpc, [], 0
        self.seen = jax.jit(lambda s: under_a_mark(s, params, False))

    def reset(self, key, obs, task):
        self.mpc.reset()

    def act(self, carry, obs, t, state=None):
        s = jax.tree.map(lambda x: x[0], state)
        self.under += bool(self.seen(s))
        n = len(self.mpc.plans)
        a = self.mpc.step(None, s)
        if len(self.mpc.plans) > n:
            self.made.append((t, s))
            print(f"planned at step {t}, {self.under} steps under a mark so far", flush=True)  # fmt: skip
            jax.clear_caches()  # each plan compiled its own horizon
        return np.full((1, 1), a), carry


def main(make=O.GlassOracle, tag=""):
    """One episode of the oracle ``make(env, params)`` builds; ``tag`` names its files."""
    name, seed, out = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else ITER_CAP
    tag = f"{name}_{seed}{tag}"
    if len(sys.argv) > 5:
        O.FINING_BACKOFF = float(sys.argv[5])
        tag += f"_b{O.FINING_BACKOFF:g}"
    if len(sys.argv) > 6:
        O.CANAL_BACKOFF = float(sys.argv[6])
        tag += f"_c{O.CANAL_BACKOFF:g}"
    solve = O.solve

    def logged(*a, **k):  # the oracle's solve, capped as the ceiling's and logged
        t0 = time.time()
        plan = solve(
            *a, max_iter=cap, **k,
            log=lambda i, z, f, r: print(f"it {i} {f:.6f} {r:.2e} {time.time() - t0:.0f}s", flush=True),
        )  # fmt: skip
        row = {f: getattr(plan, f) for f in FIELDS} | {"seconds": time.time() - t0}
        print("plan", json.dumps(row), flush=True)
        return plan

    O.solve = logged
    p = L.scenario_params(name)
    n = int(p.max_steps_in_episode)
    line = Line(make(L.ENV, plan_params(L.SPEC, p)), p)
    t0 = time.time()
    run = run_policy(
        line, L.ENV, p, task_info(L.SPEC.name, L.ENV, p, 1), seeds=[seed],
        record={"under": under_a_mark, "fining": reading("fining"), "canal": reading("canal")},
    )  # fmt: skip
    seconds = time.time() - t0
    rec = {k: v[0] for k, v in run["records"].items()}
    under, glass = rec["under"] > 0, run["terms"]["tracking"][0]
    # each plan's forecast over the steps it was played: the mean path from its state
    mean_p = plan_params(L.SPEC, p).replace(walk_sd=(0.0,) * len(L.E.WALKS))
    forecast = jax.jit(lambda s, k, v: O.rollout(L.ENV, mean_p, s, k, v)[2:4])
    plans, off = [], {"fining": [], "canal": []}
    ends = [t for t, _ in line.made[1:]] + [n]
    for (t, s), end, plan in zip(line.made, ends, line.mpc.plans):
        keys = jax.random.split(jax.random.PRNGKey(0), n - t)[: end - t]
        told = forecast(s, keys, jnp.asarray(plan.valve[: end - t]))
        row = {"step": t} | {f: getattr(plan, f) for f in FIELDS}
        row["under"] = int(under[t:end].sum())
        for k, v in zip(off, told):
            off[k].append(rec[k][t:end] - np.asarray(v))
            row[f"{k}_low"] = float(rec[k][t:end].min())
            row[f"{k}_off_low"] = float(off[k][-1].min())
            row[f"{k}_off_sd"] = float(off[k][-1].std())
        plans.append(row)
    row = {
        "scenario": name, "seed": seed, "max_iter": cap, "seconds": seconds,
        "fining_backoff": O.FINING_BACKOFF, "canal_backoff": O.CANAL_BACKOFF,
        "cost": float(-run["reward"][0].mean()),
        "fuel": float(run["terms"]["running"][0].mean()),
        "glass": float(glass.mean()),
        "glass_under_a_mark": float(glass[under].sum() / n),
        "lost": int(under.sum()), "trips": int(run["tripped"][0].sum()),
        "fining_mean": float(rec["fining"].mean()), "fining_low": float(rec["fining"].min()),
        "plans_kept": getattr(line.mpc, "kept", 0),
        "plans": plans,
    }  # fmt: skip
    np.savez(
        out / f"{tag}.npz", fining=rec["fining"], canal=rec["canal"], under=under,
        valve=np.concatenate([q.valve[: e - t] for (t, _), e, q in
                              zip(line.made, ends, line.mpc.plans)]),
        **{f"{k}_off": np.concatenate(v) for k, v in off.items()},
    )  # fmt: skip
    (out / f"{tag}.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k != "plans"}), flush=True)


if __name__ == "__main__":
    main()
