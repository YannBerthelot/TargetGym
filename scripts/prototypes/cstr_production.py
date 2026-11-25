"""The registered cstr task (version 8) measured as its charter record will be,
until the charter's own scripts take it over. Nothing here is recorded.

    python scripts/prototypes/cstr_production.py OUT_DIR build    # ladder, set points
    python scripts/prototypes/cstr_production.py OUT_DIR judge COMMIT   # J2's runs
    python scripts/prototypes/cstr_production.py OUT_DIR measure  # reference, ceiling
    python scripts/prototypes/cstr_production.py OUT_DIR sweep    # B7's draws

The plant, its score and its scenarios are the package's (``pc_gym/cstr``); this
was their prototype until the owner's decision of 2026-11-24. The reference is the
ladder's valve position rung on the row ``scripts/ladder_rows/cstr.py``. Set
points are placed on a fixed grid by the owner's rule (2026-11-23): the cheapest
at which no seed of the normal day leaves the band, a seed no set point saves not
deciding. The ladder's own placement (PR #232) replaces ``place`` once merged.

The ceiling is ``pc_gym.foresight``'s on the package's margins
(``cstr/oracle.py``). It starts from the reset's hold and from the reference's
moves, restarts from its own end plan, and is never above the reference: its
moves are a candidate too.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from target_gym import scenarios as suites  # noqa: E402
from target_gym.benchmark import episodes_of, run_policy, task_info  # noqa: E402
from target_gym.pc_gym import foresight as F  # noqa: E402
from target_gym.pc_gym.cstr.env import CSTRParams, made  # noqa: E402
from target_gym.pc_gym.cstr.oracle import CEILING_TERMS, ceiling_margins  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

SPEC = REGISTRY["cstr"]


def hold(o, xp):
    """The raw action holding both valves where the observation shows them."""
    return xp.stack([o[2] / 50.0 - 1.0, o[3] / 50.0 - 1.0])


SEEDS = tuple(range(100, 110))
#: Solves per start, each from the last one's end plan. Two settle every scenario
#: but the catalyst decay, whose starts ended up to 46 % apart; six brought them
#: within 12 % (1.5 % at the median) and its ceiling 2.4 % lower.
SOLVES = {"catalyst_loss": 6}
RECORD = {
    "made": lambda s, p, t: made(s, p),
    "T": lambda s, p, t: s.T,
    "off": lambda s, p, t: jnp.abs(s.T - s.target_T) > p.T_band,
    "Fw": lambda s, p, t: s.F_w,
    "q": lambda s, p, t: s.q,
}
#: The placement grid, fixed before any run: the temperature set point in tenths
#: of a kelvin under the band's top (the score's resolution) down to its centre,
#: and the valve's set point from [P1]'s 95 % toward its top and down to 90 %.
GRID_VALVE = (0.99, 0.98, 0.95, 0.9)
#: The project's own numbers and the ranges B7 draws them from (cstr/PHYSICS.md,
#: "Production prototype"). The feed's floor runs down to S1's 0 [S1, Table 1].
OURS = {
    "T_band": (0.5, 2.0),
    "q_min": (0.0, 2.4),
    "setpoint_ramp": (30.0, 120.0),
    "feed_rise": (0.03, 0.10),
    "cooling_loss": (2.0, 8.0),
    "normal_weight": (0.5, 0.9),
}


def run(env, p, policy, seeds):
    """``policy`` on ``seeds``: a summary, its moves and the true traces."""
    task = task_info("cstr", env, p, len(seeds))
    ro = run_policy(policy, env, p, task, seeds=seeds, record=RECORD)
    eps, rec = episodes_of(ro), ro["records"]
    summary = {
        "cost": [float(e.cost.mean()) for e in eps],
        "trips": [bool(np.any(e.failed)) for e in eps],
        "on_spec": (rec["made"] * (1 - rec["off"])).mean(1).tolist(),
        "off_spec": rec["off"].mean(1).tolist(),
        "T_max": rec["T"].max(1).tolist(),
        "valve": ((rec["Fw"].mean(1) / 60 - 40) / 96).tolist(),
        "feed_gps": (rec["q"].mean(1) * 1000 / 60).tolist(),
    }
    return summary, np.asarray(ro["action"], float), rec


def show(name, label, r):
    print(
        f"{name:14s} {label:28s} cost {np.mean(r['cost']):.5f}"
        f" on-spec {np.mean(r['on_spec']):.3f} mol/min off {np.mean(r['off_spec']):.3f}"
        f" Tmax {max(r['T_max']) - 273.15:.2f} C trips {sum(r['trips'])}",
        flush=True,
    )


def ladder():
    import reference_ladder as RL

    return RL


def gains(RL, test=None):
    """The ladder's row for cstr, built: the PI's and the valve position loop's
    SIMC gains, from its bump tests. ``test`` changes the plant under them."""
    row = RL.ROWS["cstr"]()
    row.test = {**row.test, **(test or {})}
    kept, build = {}, RL.ladder_controller
    RL.ladder_controller = lambda rung, *a: kept.update({rung.name: rung}) or build(
        rung, *a
    )
    try:
        RL.build("cstr", row)
    finally:
        RL.ladder_controller = build
    rung = kept["valve position"]
    pi, vp = rung.loops[0], rung.valve_positions[0]
    return {"water": [pi.kc, pi.ti, pi.alpha], "valve": [vp.kc, vp.ti]}


def reference(RL, p, g, b, valve):
    """The valve position rung on the stored gains, the temperature held ``b``
    under the band's top and the water valve at ``valve`` of its span."""
    from target_gym.experts.ladder import Pi, Rung, ValvePosition, ladder_controller

    band = float(p.T_band)
    position = ValvePosition(
        lambda o, xp: o[2] / 50.0 - 1.0, 2 * valve - 1, 1, *g["valve"]
    )
    rung = Rung(
        "valve position",
        (Pi(0, *g["water"]),),
        np.array([[1.0], [0.0]]),
        (),
        (),
        (position,),
    )
    law = ladder_controller(
        rung, lambda o, xp: xp.stack([o[0]]), lambda o, xp: xp.stack([o[4] + band - b]),
        lambda o, xp: xp.zeros(2), float(p.delta_t), hold,
    )  # fmt: skip
    return RL.Reference("reference", law)


def grid_b(p):
    """Tenths of a kelvin under the band's top, down to its centre."""
    return tuple(
        round(0.1 * k, 1) for k in range(1, int(float(p.T_band) / 0.1 + 1e-6) + 1)
    )


def place(RL, env, p, g, seeds=SEEDS):
    """``(b, valve)`` by the owner's rule: on the fixed grid, the cheapest setting
    at which no deciding seed of the normal day leaves the band or trips. A seed
    that does so at every setting does not decide, and is returned by name."""
    cost, exits = {}, {}
    for valve in GRID_VALVE:
        for b in grid_b(p):
            r, _, _ = run(env, p, reference(RL, p, g, b, valve), seeds)
            cost[b, valve] = float(np.mean(r["cost"]))
            exits[b, valve] = (np.array(r["off_spec"]) > 0) | np.array(r["trips"])
    left_out = np.all(list(exits.values()), axis=0)
    stays = [k for k in cost if not exits[k][~left_out].any()]
    best = min(stays, key=cost.get) if stays else None
    return best, [seeds[i] for i in np.flatnonzero(left_out)], cost


def scenarios(p, weights=None, **sizes):
    """``{name: (params, weight)}`` over the registered suite, weights summing
    to 1; ``sizes`` replace an event's size where a scenario sets it."""
    shares = suites.weights(SPEC)
    out = {}
    for s in suites.suite(SPEC):
        over = {k: sizes.get(k, v) for k, v in s.overrides.items()}
        out[s.name] = (p.replace(**over), (weights or shares)[s.name])
    return out


class Constant:
    """A null: both valves held at one raw value."""

    oracle = False

    def __init__(self, raw):
        self.raw, self.name = raw, f"constant {(raw + 1) / 2:g}"

    def reset(self, key, obs, task):
        self.n = task.n_episodes

    def act(self, carry, obs, t):
        return np.full((self.n, 2), self.raw), carry


def nulls(RL):
    return [Constant(-1.0), Constant(0.0), Constant(1.0), RL.HoldReset()]


def load(out):
    ref = json.loads((out / "production_reference.json").read_text())
    return ref["gains"], ref["b"], ref["valve"]


def build(out):
    RL, env, p = ladder(), SPEC.make_env(), CSTRParams(sensor_noise_scale=1.0)
    g = gains(RL)
    best, left_out, cost = place(RL, env, p, g)
    record = {"gains": g, "b": best[0], "valve": best[1], "left_out": left_out}
    record["grid"] = {f"{b}/{v}": c for (b, v), c in cost.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "production_reference.json").write_text(
        json.dumps(record, indent=1, default=float)
    )
    print(
        json.dumps({k: v for k, v in record.items() if k != "grid"}, default=float),
        flush=True,
    )


def judge(out):
    """J2's runs: the reference and the nulls on the hash seeds of ``COMMIT``."""
    from target_gym import charter

    RL, env, base = ladder(), SPEC.make_env(), CSTRParams(sensor_noise_scale=1.0)
    g, b, valve = load(out)
    weights = suites.weights(SPEC)
    seeds = charter.j2_seeds(sys.argv[3], weights)
    policies = {"reference": lambda p: reference(RL, p, g, b, valve)}
    policies |= {n.name: (lambda p, n=n: n) for n in nulls(RL)}
    costs, trips = {k: {} for k in policies}, 0
    for name, (p, _) in scenarios(base).items():
        for label, make in policies.items():
            policy = make(p)
            if hasattr(policy, "bind"):
                policy.bind(env, p, seeds[name])
            r, _, _ = run(env, p, policy, tuple(seeds[name]))
            costs[label][name] = r["cost"]
            trips += sum(r["trips"]) if label == "reference" else 0
    n = sum(len(v) for v in seeds.values())
    wins = {
        k: charter.null_wins(costs["reference"], v)
        for k, v in costs.items()
        if k != "reference"
    }
    result = {
        "commit": sys.argv[3],
        "seeds": seeds,
        "n": n,
        "reference_trips": trips,
        "wins": wins,
        "costs": costs,
    }
    (out / "production_j2.json").write_text(json.dumps(result, indent=1, default=float))
    print(
        json.dumps({k: result[k] for k in ("n", "reference_trips", "wins")}), flush=True
    )


def ceiling(env, p, name, seeds, moves, cost, solves):
    """Each seed's plan: L-BFGS-B from the reference's first move held and from
    its moves, restarted from its own end plan, the reference's moves a candidate."""
    plans, certs = [], []
    for i, seed in enumerate(seeds):
        first = moves[i]
        options = [(cost[i], first, "reference", 0.0)]
        for start, init in (
            ("hold", np.tile(first[:1], (len(first), 1))),
            ("moves", first),
        ):
            plan = init
            for _ in range(solves):
                plan, cert = F.solve(
                    env, p, seed, ceiling_margins, CEILING_TERMS, plan, maxiter=3000
                )
            if not cert["tripped"]:
                options.append(
                    (cert["planned_cost"], plan, start, cert["projected_gradient"])
                )
        solved = sorted(o[0] for o in options[1:])
        best = min(options, key=lambda o: o[0])
        plans.append(best[1])
        gap = (
            (solved[-1] - solved[0]) / max(solved[0], 1e-9) if len(solved) > 1 else None
        )
        certs.append(
            {"start": best[2], "starts_gap": gap, "projected_gradient": best[3]}
        )
        print(
            f"  {name} seed {seed}: {best[0]:.5f} from {best[2]}, gap {gap}", flush=True
        )
    row, _, _ = run(env, p, F.PlanPolicy(np.stack(plans)), seeds)
    return row | {"certs": certs}


def measure(out, seeds=SEEDS):
    """The placed reference and the ceiling on every scenario, per seed."""
    RL, env, base = ladder(), SPEC.make_env(), CSTRParams(sensor_noise_scale=1.0)
    g, b, valve = load(out)
    result = {}
    for name, (p, weight) in scenarios(base).items():
        ref, moves, _ = run(env, p, reference(RL, p, g, b, valve), seeds)
        top = ceiling(env, p, name, seeds, moves, ref["cost"], SOLVES.get(name, 2))
        result[name] = {"weight": weight, "reference": ref, "ceiling": top}
        show(name, "reference", ref)
        show(name, "ceiling", top)
        (out / "production_measure.json").write_text(
            json.dumps(result, indent=1, default=float)
        )
    print("done", flush=True)


def sweep(out):
    """B7: the project's own numbers drawn jointly, 8 Latin-hypercube draws on
    fixed seeds. On each the ladder is rebuilt and placed, and the reference must
    cost less than every null on the median over seeds; the ceiling is solved on
    the two draws at the ends of the band, the number the room moves with most."""
    from scipy.stats import qmc

    RL, env = ladder(), SPEC.make_env()
    unit = qmc.LatinHypercube(d=len(OURS), seed=0).random(8)
    draws = [
        {k: lo + u * (hi - lo) for u, (k, (lo, hi)) in zip(row, OURS.items())}
        for row in unit
    ]
    bands = [d["T_band"] for d in draws]
    ends, result = {int(np.argmin(bands)), int(np.argmax(bands))}, []
    for i, d in enumerate(draws):
        plant = {k: d[k] for k in ("T_band", "q_min", "setpoint_ramp")}
        p = CSTRParams(sensor_noise_scale=1.0, **plant)
        g = gains(RL, plant)
        best, left_out, _ = place(RL, env, p, g)
        other = (1.0 - d["normal_weight"]) / 0.3
        weights = {
            k: (d["normal_weight"] if k == "normal" else w * other)
            for k, w in suites.weights(SPEC).items()
        }
        row = {"draw": d, "placed": best, "left_out": left_out, "J": {}, "trips": 0}
        if best is not None:
            suite = scenarios(
                p, weights, feed_rise=d["feed_rise"], cooling_loss=d["cooling_loss"]
            )
            J = {n.name: np.zeros(len(SEEDS)) for n in nulls(RL)} | {
                "reference": np.zeros(len(SEEDS))
            }
            top = np.zeros(len(SEEDS))
            for name, (q, w) in suite.items():
                ref, moves, _ = run(env, q, reference(RL, q, g, *best), SEEDS)
                J["reference"] += w * np.array(ref["cost"])
                row["trips"] += sum(ref["trips"])
                for null in nulls(RL):
                    if hasattr(null, "bind"):
                        null.bind(env, q, SEEDS)
                    J[null.name] += w * np.array(run(env, q, null, SEEDS)[0]["cost"])
                if i in ends:
                    top += w * np.array(
                        ceiling(env, q, name, SEEDS[:5], moves, ref["cost"], 2)["cost"]
                        + [0.0] * 5
                    )
            row["J"] = {k: float(np.median(v)) for k, v in J.items()}
            row["beats_nulls"] = all(
                row["J"]["reference"] < v
                for k, v in row["J"].items()
                if k != "reference"
            )
            if i in ends:
                row["ceiling_J"] = float(np.mean(top[:5]))
                row["reference_J_same_seeds"] = float(np.mean(J["reference"][:5]))
                row["ceiling_beats_reference"] = (
                    row["ceiling_J"] < row["reference_J_same_seeds"]
                )
        row["holds"] = bool(
            best is not None
            and row.get("beats_nulls")
            and row.get("ceiling_beats_reference", True)
        )
        result.append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k != "J"}
                | {"J_ref": row["J"].get("reference")},
                default=float,
            ),
            flush=True,
        )
        (out / "production_sweep.json").write_text(
            json.dumps(result, indent=1, default=float)
        )
    print("sweep holds on every draw:", all(r["holds"] for r in result), flush=True)


if __name__ == "__main__":
    {"build": build, "judge": judge, "measure": measure, "sweep": sweep}[sys.argv[2]](
        pathlib.Path(sys.argv[1])
    )
