"""Glass room study: the planner without foresight on the readings alone. The library's
oracle without a seed is handed the true state. Here the same planner is handed the
averaged state of a particle filter (``state_filter.py``), which is shown the readings
and the valve asked and nothing else, and at each plan it prices the spread of that
filter's own particles on sampled paths of the walks, in place of the spread from one
known state. The policy is run as the benchmark runs any agent: ``oracle`` is False, so
the runner never passes it the state.

One scenario and seed, through ``benchmark.run_policy`` at sensor noise 1. The filter
infers no event: the normal scenario only, until it does.

Ours: 512 particles (``PARTICLES``), 64 of them for the spread at each plan, and the
filter's own tuning (``state_filter.py``).

Run: PYTHONPATH=src python scripts/glass_room/readings_line.py SCENARIO SEED OUT_DIR [MAX_ITER]
"""

import json
import os
import pathlib
import sys
import time

os.environ.setdefault("PARTICLES", "512")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import lib as L  # noqa: E402
from glass_slow_oracle import ITER_CAP, under_a_mark  # noqa: E402
from mpc_line import FIELDS, reading  # noqa: E402
from state_filter import OWN, Filter, N  # noqa: E402

from target_gym.benchmark import run_policy, task_info  # noqa: E402
from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.glass_furnace import oracle as O  # noqa: E402

E = L.E
PRICED = 64  # particles the spread is read from at each plan


class OnReadings(O.GlassOracle):
    """The library's planner, shown the filter's averaged state in place of the true
    one, and pricing at each plan the spread of the filter's particles. ``carry`` is
    the filter's, set by the policy before each step."""

    def __init__(self, env, params, filt):
        super().__init__(env, params)
        self.filt, self.carry, lead = filt, None, self.lead
        mean_p = params.replace(walk_sd=(0.0,) * len(E.WALKS))
        told_keys = jax.random.split(jax.random.PRNGKey(0), lead)

        def spread(some, state, key):
            valve = jnp.full(lead, state.fuel)
            told = O._scan(env, mean_p, state, told_keys, valve)[1][2]
            keys = jax.vmap(lambda k: jax.random.split(k, lead))(
                jax.random.split(key, PRICED)
            )
            path = lambda s, ks: O._scan(env, params, s, ks, valve)[1][2]  # noqa: E731
            came = jax.vmap(path)(some, keys)
            return jnp.sqrt(jnp.mean((came - told) ** 2, axis=0))

        self.price = jax.jit(spread)

    def _without_foresight(self, state, t, n):
        key = jax.random.fold_in(self.carry[2], n)
        some = self.filt.shown(self.carry, key, PRICED)
        self._spread = np.asarray(self.price(some, state, jax.random.fold_in(key, 1)))
        return super()._without_foresight(state, t, n)


class Line:
    """The planner as a ``benchmark.Policy`` that is shown readings only."""

    oracle = False

    def __init__(self, planner, filt, params):
        self.planner, self.filt, self.p = planner, filt, params
        self.asked, self.fewest, self.spreads = None, np.inf, []

    def reset(self, key, obs, task):
        self.planner.reset()
        self.planner.carry = self.filt.start(OWN)

    def act(self, carry, obs, t):
        if t > 0:  # the reading that came after the valve last asked
            y = jnp.asarray(obs[0], jnp.float32)
            self.planner.carry, count = self.filt.step(
                self.planner.carry, self.asked, y
            )
            self.fewest = min(self.fewest, float(count))
        n = len(self.planner.plans)
        a = self.planner.step(None, self.filt.averaged(self.planner.carry))
        self.asked = jnp.asarray(E.to_fuel(a, self.p), jnp.float32)
        if len(self.planner.plans) > n:
            self.spreads.append(float(self.planner._spread[-1]))
            print(f"planned at step {t}, spread priced at its last lead {self.spreads[-1]:.5f}", flush=True)  # fmt: skip
            if len(self.planner.plans) % 6 == 0:
                jax.clear_caches()  # each plan compiled its own horizon
        return np.full((1, 1), a), carry


def main():
    name, seed, out = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else ITER_CAP
    solve = O.solve

    def logged(*a, **k):  # the oracle's solve, capped and logged
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
    filt = Filter(p, N)
    line = Line(OnReadings(L.ENV, plan_params(L.SPEC, p), filt), filt, p)
    t0 = time.time()
    run = run_policy(
        line, L.ENV, p, task_info(L.SPEC.name, L.ENV, p, 1), seeds=[seed],
        record={"under": under_a_mark, "fining": reading("fining"), "canal": reading("canal")},
    )  # fmt: skip
    rec = {k: v[0] for k, v in run["records"].items()}
    under, glass = rec["under"] > 0, run["terms"]["tracking"][0]
    plans = line.planner.plans
    row = {
        "scenario": name, "seed": seed, "max_iter": cap, "particles": N,
        "seconds": time.time() - t0,
        "cost": float(-run["reward"][0].mean()),
        "fuel": float(run["terms"]["running"][0].mean()),
        "glass": float(glass.mean()),
        "glass_under_a_mark": float(glass[under].sum() / n),
        "lost": int(under.sum()), "trips": int(run["tripped"][0].sum()),
        "fining_mean": float(rec["fining"].mean()), "fining_low": float(rec["fining"].min()),
        "plans_kept": line.planner.kept, "fewest_particles": line.fewest,
        "spread_priced": line.spreads,
        "plans": [{f: getattr(q, f) for f in FIELDS} for q in plans],
    }  # fmt: skip
    tag = f"{name}_{seed}_readings"
    np.savez(out / f"{tag}.npz", fining=rec["fining"], canal=rec["canal"], under=under,
             action=np.asarray(run["applied"][0]).ravel())  # fmt: skip
    (out / f"{tag}.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k not in ("plans", "spread_priced")}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
