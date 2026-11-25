"""Glass room study: the ceiling, or the library's planner without foresight, on one
member of the family, over the scored week from the member's start (``family.py``).
Both are told the member: the ceiling is the problem solved, and the planner is what
the true state and the plant's numbers give without foresight. Neither is what an agent
on the readings has.

``ceiling``: the oracle's plan with the week foreseen, from the design valve and again
from that plan, each replayed on the env's own step; the cheaper one that keeps its
marks stands. ``planner``: the oracle without a seed, stepped on the true state.

Run: PYTHONPATH=src python scripts/glass_room/family_line.py ceiling|planner BAND SEED
         FAMILY_DIR OUT_DIR [MAX_ITER]
"""

import json
import os
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import family as F  # noqa: E402
import lib as L  # noqa: E402
from mpc_line import FIELDS  # noqa: E402

from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.glass_furnace import oracle as O  # noqa: E402

E, P = L.E, L.P
KNOT = 4 * L.HOUR


def main():
    kind, band, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    family, out = pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5])
    cap = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    p = F.member(band)
    pp = plan_params(L.SPEC, p)
    safe = json.load(open(family / "family.json"))["safe"]
    state0, _ = F.load(family / f"member_{band:+g}.npz", band, safe)
    st, _, keys = (jax.tree.map(lambda a: a[0], v) for v in F.starts(state0, p, [seed]))
    n = int(os.environ.get("STEPS", keys.shape[0]))  # fewer, for a smoke run
    solve, t0 = O.solve, time.time()

    def logged(*a, **k):
        t1 = time.time()
        k.setdefault("max_iter", cap)
        plan = solve(*a, **k, log=lambda i, z, f, r: print(f"it {i} {f:.6f} {r:.2e} {time.time() - t1:.0f}s", flush=True))  # fmt: skip
        print("plan", json.dumps({f: getattr(plan, f) for f in FIELDS}), flush=True)
        return plan

    O.solve = logged

    def step(s, x):
        k, u = x
        _, s1, r, _, _ = L.ENV.step_env(k, s, E.to_action(u, p), p)
        read = P.sensors(s1.plant, p.grid, p.plant)
        fuel = E.compute_reward_terms(s1, p)["running"]
        return s1, (-r, fuel, read["fining"], read["canal"])

    replay = jax.jit(lambda valve: jax.lax.scan(step, st, (keys, valve))[1])
    one = jax.jit(lambda s, k, u: step(s, (k, u)))

    def read(cost, fuel, fining, canal):
        cost, fuel, fining, canal = (np.asarray(v, float) for v in (cost, fuel, fining, canal))  # fmt: skip
        under = (fining < float(p.fining_pass)) | (canal < float(p.canal_target))
        return dict(
            cost=float(cost.mean()), fuel=float(fuel.mean()),
            glass=float((cost - fuel).mean()), under=int(under.sum()),
            fining_mean=float(fining.mean()), fining_low=float(fining.min()),
        )  # fmt: skip

    row = dict(kind=kind, band=band, seed=seed, table=safe, max_iter=cap)
    if kind == "ceiling":
        first = O.solve(L.ENV, pp, st, keys, KNOT)
        again = O.solve(L.ENV, pp, st, keys, KNOT, start=first.knots)
        runs = [read(*replay(jnp.asarray(q.valve))) for q in (first, again)]
        kept = min(range(2), key=lambda i: (runs[i]["under"] > 0, runs[i]["cost"]))
        row |= runs[kept] | dict(
            kept=("first", "restart")[kept], starts=runs,
            plans=[{f: getattr(q, f) for f in FIELDS} for q in (first, again)],
            knots=np.asarray((first, again)[kept].knots).tolist(),
        )  # fmt: skip
    else:
        mpc, state, trace = O.GlassOracle(L.ENV, pp), st, []
        for t in range(n):
            made = len(mpc.plans)
            a = mpc.step(None, state)
            state, got = one(state, keys[t], jnp.asarray(E.to_fuel(a, p), jnp.float32))
            trace.append([float(v) for v in got])
            if len(mpc.plans) > made:
                print(f"planned at step {t}", flush=True)
                if len(mpc.plans) % 6 == 0:
                    jax.clear_caches()  # each plan compiled its own horizon
        row |= read(*np.array(trace).T) | dict(
            plans_kept=mpc.kept,
            plans=[{f: getattr(q, f) for f in FIELDS} for q in mpc.plans],
        )  # fmt: skip
    row["seconds"] = time.time() - t0
    (out / f"{kind}_{band:+g}_{seed}.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k not in ("plans", "starts", "knots")}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
