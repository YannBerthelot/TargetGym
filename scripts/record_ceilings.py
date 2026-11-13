"""Measure a charter task's ceiling (charter Words "Ceiling", J3) on its current
plant and write ``data/oracle_ceilings/<task>.json`` stamped with that plant.

    python scripts/record_ceilings.py solve TASK SCENARIO SEED OUT_DIR
    python scripts/record_ceilings.py write TASK OUT_DIR

``solve`` plans one episode with the true model, the true state and the episode's
future, on the seed's own keys (``benchmark.episode_keys``), and replays the plan
through ``benchmark.run_policy`` at sensor noise 1 with the ladder row's score
parameters, as the J3 lines are scored. Only the replayed cost is reported.
``write`` gathers every solved seed into the ceilings file, which
``scripts/record_charter.py --parts ceiling`` then reads. The glass furnace has its
own script, ``scripts/glass_slow_oracle.py``.

The protocols are the ones the ceilings were first measured with (2026-11-08 to
2026-11-10), whose drivers were not kept:

- process (cstr, distillation, compressor_surge, unstable_cstr, ph_neutralization): ``pc_gym.foresight``
  L-BFGS-B from the reset's hold (the reference's first move held) and the kept
  reference's moves, the cheaper untripped plan kept; compressor_surge also from their
  midpoint, each start restarted twice from its own end plan.
- shooting (boiler_drum, hvac, wind_turbine): the shared interior-point solver
  (``oracles.shooting``) on the oracle's own model, one plan over the episode, block 1,
  through K7's width rungs (``data/oracle_widths``), limits backed off by the task's
  ``ceiling_backoffs`` (hvac: its slab edge through 1, 0.1, 0.01 K, a capped seed
  re-solved through 1, 0.3, 0.1, 0.03, 0.01 K at cap 1500); the wind turbine re-solved with the tower's rainflow turning
  points frozen until they stop changing (``OUTER`` solves at most).
- battery: ``energy.battery.oracle.ceiling_search``.
- reactor: ``scripts/reactor_ipopt.py``'s whole-day NLP from the simulated-state
  start (G8 allows it in the ceiling script only), the load's wobble read off a
  rollout on the seed's keys.
"""

from __future__ import annotations

import datetime
import importlib
import json
import pathlib
import platform
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from target_gym import charter, scenarios  # noqa: E402
from target_gym.benchmark import (
    episode_keys,
    episodes_of,
    run_policy,
    task_info,
)  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

PROCESS = {"cstr": 3000, "distillation": 3000, "compressor_surge": 2000,
           "unstable_cstr": 300, "ph_neutralization": 2000}  # fmt: skip
SHOOTING = {"boiler_drum": "boiler_model", "hvac": "hvac_model",
            "wind_turbine": "wind_model"}  # fmt: skip
OUTER, CAP = 8, 500
#: Width rungs the first ceilings used where the widths row has no site (hvac's slab
#: edge, in K), and the finer re-solve with its cap for a seed a rung capped.
RUNGS = {"hvac": ([1.0], [0.1], [0.01])}
RETRY = {"hvac": (([1.0], [0.3], [0.1], [0.03], [0.01]), 1500)}
NOISE = 1.0


def setup(task, scenario, seed):
    import reference_ladder as L

    spec = REGISTRY[task]
    row = L.Row(0, None) if task in charter.EXPERT_REFERENCE else L.ROWS[task]()
    s = {x.name: x for x in scenarios.suite(spec)}[scenario]
    params = scenarios.params_for(spec, s, sensor_noise_scale=NOISE, **row.score)
    env = spec.make_env()
    T = int(params.max_steps_in_episode)
    resets, steps = episode_keys(jax.random.PRNGKey(seed), 1, T)
    _, st = env.reset_env(resets[0], params)
    package = type(env).__module__.rsplit(".", 1)[0]
    oracle = importlib.import_module(package + ".oracle")
    return L, spec, row, env, params, st, steps[:, 0], T, oracle


def replay(task, env, params, seed, policy) -> dict:
    (ep,) = episodes_of(
        run_policy(policy, env, params, task_info(task, env, params, 1), seeds=[seed])
    )
    return {"cost": float(ep.cost.mean()), "trips": int(np.sum(ep.failed))}


class StatePlan:
    """Fixed moves per step, mapped to the action by ``action(state, u)`` on the true
    state where the model has one (an oracle row, so ``run_policy`` passes the state).
    """

    name, oracle = "ceiling", True

    def __init__(self, moves, action=None):
        self.moves = np.asarray(moves, float)[None]
        self.f = None if action is None else jax.jit(jax.vmap(action))

    def reset(self, key, obs, task):
        return None

    def act(self, carry, obs, t, state=None):
        u = self.moves[:, t]
        if self.f is not None:
            u = np.asarray(self.f(state, jnp.asarray(u, jnp.float32)), float)
        return np.clip(u, -1.0, 1.0), carry


def solve_process(task, scenario, seed):
    from target_gym.pc_gym import foresight as F

    L, spec, row, env, params, st, keys, T, O = setup(task, scenario, seed)
    kept = charter.load(task)["reference"]["kept"]
    ref = L.Reference(kept, L.build(task, row)[kept])
    rollout = run_policy(
        ref, env, params, task_info(task, env, params, 1), seeds=[seed]
    )
    moves = np.asarray(rollout["action"][0], float)
    starts = {"hold": np.tile(moves[:1], (T, 1)), "reference": moves}
    restarts = 0
    if task == "compressor_surge":
        starts["mid"] = 0.5 * (starts["hold"] + moves)
        restarts = 2
    window = getattr(O, "CEILING_WINDOW", None)
    runs = []
    for name, init in starts.items():
        plan = init
        for k in range(restarts + 1):
            plan, cert = F.solve(env, params, seed, O.ceiling_margins, O.CEILING_TERMS,
                                 plan, maxiter=PROCESS[task], window=window)  # fmt: skip
            tag = name + (f"+restart{k}" if k else "")
            runs.append((tag, plan, cert))
            print(tag, json.dumps({k: cert[k] for k in ("status", "planned_cost",
                  "tripped", "capped", "fell_back")}), flush=True)  # fmt: skip
    ok = [r for r in runs if not r[2]["tripped"]] or runs
    tag, plan, cert = min(ok, key=lambda r: r[2]["planned_cost"])
    feedback = getattr(O, "CEILING_FEEDBACK", None)
    if feedback is None:
        policy = F.PlanPolicy(plan[None])
    else:
        path = F.tracking(env, params, seed, plan, feedback)
        policy = F.TrackingPolicy(plan[None], [path], feedback[0])
    out = replay(task, env, params, seed, policy)
    solver = {k: cert[k] for k in ("status", "capped", "fell_back", "iterations",
                                   "projected_gradient", "min_margin")}  # fmt: skip
    return out | {
        "planned": cert["planned_cost"],
        "start": tag,
        "solver": solver,
        "starts": {t: c["planned_cost"] for t, _, c in runs},
    }


def _rungs(task):
    if task in RUNGS:
        return [np.array(w, float) for w in RUNGS[task]]
    from target_gym.oracles.controller import widths_row

    sites = list(widths_row(task)["sites"].values())
    if not sites:
        return [np.zeros(0)]
    w = np.array([s["width"] for s in sites], float)
    scale = np.array([s["scale"] for s in sites], float)
    n = int(np.max(np.round(np.log10(scale / w))))
    return [np.maximum(scale * 10.0**-r, w) for r in range(1, n + 1)]


def _cert(plan) -> dict:
    c = plan.certificate
    return {"error": float(c.error), "iterations": int(c.iterations),
            "capped": bool(c.capped), "failed": bool(c.failed),
            "violation": float(c.violation)}  # fmt: skip


def solve_shooting(task, scenario, seed):
    from target_gym.oracles.controller import enable_x64, float64, widths_row
    from target_gym.oracles.shooting import Shooting, StageModel

    L, spec, row, env, params, st, keys, T, O = setup(task, scenario, seed)
    build = getattr(O, SHOOTING[task])
    kw = {"rest": False} if task == "wind_turbine" else {}
    tm32 = build(env, params, **kw)
    base32 = O.ceiling_data(env, params, st, keys)
    rungs, rows = _rungs(task), []
    with enable_x64():
        p64 = float64(params)
        tm = build(env, p64, **kw)
        if hasattr(O, "ceiling_backoffs"):
            back = list(O.ceiling_backoffs(env, p64))
        else:
            back = widths_row(task).get("limit_backoffs") or []
        back = back + [0.0] * (len(tm.limits) - len(back))
        rows_fn = tm.move_rows
        stage = StageModel(
            step=lambda d, x, u, t: tm.step(d, x, u, t)[:3], block=1,
            lo=(-1.0,) * tm.nu, hi=(1.0,) * tm.nu, hinges=tm.hinges,
            limits=tuple(q._replace(backoff=float(b)) for q, b in zip(tm.limits, back)),
            move_rows=None if rows_fn is None else lambda x, u: rows_fn(x, u, 1),
            move_scale=None if tm.move_scale is None else tm.move_scale(1), tie=0.0,
        )  # fmt: skip
        shoot = Shooting(stage, T, tm.nx, max_iter=CAP)
        st64, base = float64(st), float64(base32)
        x0, hold = tm.state_x(st64), tm.hold(st64)
        frozen, plan, best = None, None, None
        for k in range(OUTER + 1 if task == "wind_turbine" else 1):
            data = base if frozen is None else (*base, frozen)
            certs = []
            for w in rungs if plan is None else rungs[-1:]:
                d = {"task": data, "widths": jnp.asarray(w), "end": T}
                plan = (shoot.solve(d, x0, hold) if plan is None
                        else shoot.resolve(d, x0, hold, plan))  # fmt: skip
                jax.block_until_ready(plan.z)
                certs.append(_cert(plan))
            if task in RETRY and (certs[-1]["capped"] or certs[-1]["failed"]):
                finer, cap = RETRY[task]
                again = Shooting(stage, T, tm.nx, max_iter=cap)
                plan = None
                for w in finer:
                    d = {"task": data, "widths": jnp.asarray(w), "end": T}
                    plan = (again.solve(d, x0, hold) if plan is None
                            else again.resolve(d, x0, hold, plan))  # fmt: skip
                    jax.block_until_ready(plan.z)
                    certs.append(_cert(plan) | {"retry": True})
            us = np.asarray(shoot.moves(plan))[:T]
            with enable_x64(False):
                out = replay(task, env, params, seed, StatePlan(us, tm32.action))
            r = out | {"outer": k, "certificates": certs}
            new = None
            if task == "wind_turbine":
                new = _turns(O, tm, data, x0, us, T, p64, st64)
                r["changed"] = (
                    None if frozen is None else int(np.sum(frozen[0] != new[0]))
                )
            print(json.dumps(r), flush=True)
            rows.append(r)
            if not r["trips"] and (best is None or r["cost"] < best["cost"]):
                best = r
            if r.get("changed") == 0:
                break
            frozen = new
    keep = best or rows[-1]
    last = keep["certificates"][-1]
    return {
        "cost": keep["cost"],
        "trips": keep["trips"],
        "outer": rows,
        "solver": {"converged": not (last["failed"] or last["capped"]), **last},
    }


def _turns(O, tm, data, x0, us, T, p64, st64):
    """The plan's tower turning points (``oracle.turning_points``), from the model's
    base moment along the plan."""
    from target_gym.energy.wind_turbine.env import tower_base_moment

    d = {"task": data, "widths": jnp.zeros(len(_rungs("wind_turbine")[-1])), "end": T}

    def f(x, inp):
        u, t = inp
        x1 = tm.step(d, x, u, t)[0]
        return x1, tower_base_moment(x1[3], p64)

    moments = jax.lax.scan(f, x0, (jnp.asarray(us), jnp.arange(T)))[1]
    return O.turning_points(float(st64.moment), float(st64.moment_turn),
                            float(st64.moment_dir), np.asarray(moments))  # fmt: skip


def solve_battery(task, scenario, seed):
    L, spec, row, env, params, st, keys, T, O = setup(task, scenario, seed)
    res = O.ceiling_search(env, params, st, keys)
    B = jnp.asarray(res["baselines_raw"], jnp.float32)
    periods = jnp.asarray(res["periods"])

    class Baselines(StatePlan):
        def act(self, carry, obs, t, state=None):
            a = jax.vmap(lambda s: O.raw_action(B, periods, s.time, params))(state)
            return np.clip(np.asarray(a, float), -1.0, 1.0), carry

    out = replay(task, env, params, seed, Baselines(np.zeros((T, 1))))
    keep = ("converged", "final_step", "evaluations", "cost", "baselines_raw")
    return out | {"planned": res["cost"] / T,
                  "solver": {k: res[k] for k in keep}}  # fmt: skip


def solve_reactor(task, scenario, seed):
    import reactor_ipopt as R

    from target_gym.oracles.controller import enable_x64, float64
    from target_gym.pc_gym.foresight import PlanPolicy

    L, spec, row, env, params, st, keys, T, O = setup(task, scenario, seed)

    def wobble(s, k):
        before = s.load_wobble
        return env.step_env(k, s, env_hold, params)[1], before

    env_hold = jnp.zeros(env.action_space(params).shape, jnp.float32)
    path = jax.lax.scan(wobble, st, keys)[1]
    rungs = [(R.SCALE * 0.1, 0.1), (R.WIDTH, 0.01), (R.WIDTH, 0.001)]
    certs = []
    with enable_x64():
        q, st64 = float64(params), float64(st)
        model, shoot = R.build(env, q, T, 1, R_CAP)
        data = {"task": model.data(st64), "end": T, "path": jnp.asarray(path, float)}
        x0, hold = model.state_x(st64), model.hold(st64)
        plan = None
        for width, boron in rungs:
            data["widths"], data["boron_width"] = jnp.asarray([width]), jnp.asarray(
                boron
            )
            t0 = time.time()
            if plan is None:
                plan = shoot.solve(data, x0, hold, start=jnp.tile(hold, (shoot.N, 1)))
            else:
                plan = shoot.resolve(data, x0, hold, plan)
            jax.block_until_ready(plan.z)
            certs.append(_cert(plan) | {"seconds": time.time() - t0})
            print(json.dumps(certs[-1]), flush=True)
        moves = np.asarray(shoot.moves(plan))[:T]
    out = replay(task, env, params, seed, PlanPolicy(moves[None]))
    last = certs[-1]
    return out | {
        "certificates": certs,
        "solver": {"converged": not (last["failed"] or last["capped"]), **last},
    }


R_CAP = 600


def solve(task, scenario, seed, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if task in PROCESS:
        res = solve_process(task, scenario, seed)
    elif task in SHOOTING:
        res = solve_shooting(task, scenario, seed)
    elif task == "battery":
        res = solve_battery(task, scenario, seed)
    elif task == "reactor":
        res = solve_reactor(task, scenario, seed)
    else:
        raise SystemExit(f"{task}: no ceiling protocol here")
    res = {"task": task, "scenario": scenario, "seed": seed,
           "seconds": time.time() - t0} | res  # fmt: skip
    (out / f"{task}__{scenario}__{seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("scenario", "seed", "cost", "trips")}))


def write(task, out_dir):
    from target_gym.provenance import environment_fingerprint

    spec = REGISTRY[task]
    weights = scenarios.weights(spec)
    got = {}
    for f in pathlib.Path(out_dir).glob(f"{task}__*.json"):
        r = json.loads(f.read_text())
        got.setdefault(r["scenario"], []).append(r)
    rows = {}
    for s in weights:
        runs = sorted(got.get(s, []), key=lambda r: r["seed"])
        if not runs:
            raise SystemExit(f"{task}: no solve for {s}")
        sol = [r["solver"] for r in runs]
        if task in PROCESS:
            solver = {"status": [x["status"] for x in sol],
                      "capped": sum(bool(x["capped"]) for x in sol),
                      "fell_back": sum(bool(x["fell_back"]) for x in sol)}  # fmt: skip
        else:
            solver = {"converged": sum(bool(x["converged"]) for x in sol)}
        rows[s] = {
            "weight": weights[s], "seeds": [r["seed"] for r in runs],
            "ceiling_costs": [r["cost"] for r in runs],
            "ceiling_trips": sum(r["trips"] for r in runs),
            "ceiling_mean": float(np.mean([r["cost"] for r in runs])),
            "solver": solver, "per_seed": [{k: v for k, v in r.items()
                                            if k not in ("outer", "starts")}
                                           for r in runs],
        }  # fmt: skip
    data = {
        "task": task,
        "method": _method(task),
        "J_ceiling": sum(weights[s] * rows[s]["ceiling_mean"] for s in weights),
        "scenarios": rows,
        "generated": datetime.date.today().isoformat(),
        "cpu": platform.node(),
    }
    p = charter.write_ceiling(task, data, environment_fingerprint(spec))
    print(f"{task}: J_ceiling {data['J_ceiling']:.6g}, wrote {p}")


def _method(task) -> str:
    for line in (__doc__ or "").split("\n- ")[1:]:
        head = line.split(":")[0]
        if (
            task in head
            or (task in PROCESS and head.startswith("process"))
            or (task in SHOOTING and head.startswith("shooting"))
        ):
            return " ".join(line.split())
    return ""


if __name__ == "__main__":
    if sys.argv[1] == "solve":
        solve(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif sys.argv[1] == "write":
        write(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
