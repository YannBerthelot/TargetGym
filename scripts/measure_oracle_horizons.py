"""Measure each oracle's planning horizon by Cooper's DMC recipe.

Run with ``uv run python scripts/measure_oracle_horizons.py [--envs ...]``.
It runs docs/proposals/oracle-horizon-protocol.md, with Amendments 1 and 2, on the
14 process and industrial tasks and writes one file per task,
``src/target_gym/data/oracle_horizons/<task>.json``, a compact summary that
ships in the wheel, and ``evidence/oracle_horizons/<task>.json`` at the
repository root, every test, retest and fit, in git but not in the wheel
(H13, H19). The 10
aircraft tasks are skipped: their tests fly through ArduPlane 4.0.9's default
loops (decision D6), which PR D builds.

Per training seed 100-109 (``target_gym.identification``):

1. the shipped expert runs one episode from the seed's reset; the moves then
   hold the mean of its last 20 commands (the glass furnace: its last
   reversal cycle), the steady operation of amended H8;
2. each MV is stepped alone by its C2 step, away from the nearer limit, and
   every CV of H6 is read against the unstepped run for one episode, up to
   the first trip of either run (H8, H9);
3. both structures are fitted and judged (H14), failed tests are retested
   (H15, H18), and each pair-seed ends valid, insignificant or invalid under
   its CV's class (H16, H17, H19).

Only valid fits enter T_c and P (``target_gym.oracles.horizon``). A task with
none is non-compliant: no protocol horizon is written, and it keeps its
current oracle (Amendment 2, N3). unstable_cstr is measured
on its stabilised plant (section 4.4): the cascade's outer loop in manual,
the inner PD closed, its setpoint stepped 0.5 K down as in the expert test.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import jax.numpy as jnp
import numpy as np

from target_gym import records
from target_gym.identification import TRAIN_SEEDS, Plant, evidence, identify
from target_gym.oracles.horizon import (
    episode_samples,
    sample_time,
    task_horizon,
    task_ramp,
)
from target_gym.registry import REGISTRY

EVIDENCE = pathlib.Path(__file__).resolve().parent.parent / "evidence"
#: What the package keeps of a task's horizon (H13); the rest is evidence.
SUMMARY = ("protocol", "compliant", "T_c_steps", "N", "M", "ramp", "P")
SUMMARY += ("T_c_set_by", "P_set_by", "ramp_set_by", "counts", "cpu", "generated")
#: H21: a task's tracked CV with a target range, and each rate-limited MV's
#: slower rate, in raw action units per second as the steps and gains are. The
#: wind pitch joins with C2's rebuild; the reactor's rod has no compliant horizon.
RATE_LIMITS = {
    "compressor_surge": (
        "dp",
        lambda p: p.p_ref_range[1] - p.p_ref_range[0],
        lambda p: {
            "speed": 2 * p.drive_rate / (p.N_max - p.N_min),
            "recycle": 2 * min(p.valve_open_rate, p.valve_close_rate),
        },
    ),
    "wind_turbine": (
        "P",
        lambda p: (p.target_power_range[1] - p.target_power_range[0]) / 1.0e6,
        lambda p: {"pitch": 2 * p.pitch_rate_max / (p.pitch_max - p.pitch_min)},
    ),
}


def write(name: str, row: dict) -> None:
    """The compact summary in the package and the full evidence beside the
    repository's sources (H13)."""
    evidence = EVIDENCE / "oracle_horizons" / f"{name}.json"
    summary = {k: row[k] for k in SUMMARY if k in row}
    summary["classes"] = {c: v["class"] for c, v in row["classes"].items()}
    summary["evidence"] = evidence.relative_to(EVIDENCE.parent).as_posix()
    records.write("oracle_horizons", name, summary, compact=True)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


def _s(field, lo=None, hi=None):
    """A CV read from the state, with its trip limits as params fields."""
    return (lambda obs, state, p: getattr(state, field)), lo, hi


def _kiln_lime(obs, state, p):
    from target_gym.cement_kiln.env import discharge_lime

    return discharge_lime(state)


def _omega(factor):
    from target_gym.energy.wind_turbine.env import omega_rated

    return lambda p: getattr(p, factor) * omega_rated(p)


_H = {f"h{i}": _s(f"h{i}", "h_min", "h_max") for i in range(1, 5)}
_GLASS = {c: _s(c, "T_glass_min", "T_glass_max") for c in ("T_melt", "T_work")}

# Per task: the MVs with their C2 steps in raw units (existing steps stay, a
# new one is 0.1, 5 % of [-1, 1]); the CVs of H6 (tracked outputs, then every
# variable check_is_terminal trips on) with their trip limits; and, on the
# multi-input tasks, the MV the shipped expert holds each CV with (H18).
TASKS = {
    "first_order": ({"u": 0.1}, {"x": _s("x", "x_min", "x_max")}, {}),
    "cstr": ({"coolant": 0.1}, {"C_a": _s("C_a")}, {}),
    "four_tank": ({"pump1": 0.1, "pump2": 0.1}, _H, {"h1": "pump1", "h2": "pump2"}),
    # +0.1 mL/s on the 10 to 22 mL/s base flow.
    "ph_neutralization": ({"base": 2 * 0.1 / 12.0}, {"pH": _s("pH")}, {}),
    "distillation": (
        {"L": 0.1, "V": 0.1},
        {
            "yD": ((lambda o, s, p: s.x[-1]), "yD_floor", None),
            "xB": ((lambda o, s, p: s.x[0]), None, "xB_ceiling"),
        },
        {"yD": "L", "xB": "V"},
    ),
    "glass_furnace": (
        {"fuel": 0.05},
        {"T_crown": _s("T_crown", "T_crown_min", "T_crown_max"), **_GLASS},
        {},
    ),
    "reactor": (
        {"rod": 0.1},
        {
            "n": _s("n", "n_min", "n_max"),
            "T_fuel": _s("T_fuel", "T_fuel_min", "T_fuel_max"),
            "T_coolant": _s("T_coolant", "T_coolant_min", "T_coolant_max"),
        },
        {},
    ),
    "hvac": ({"heating": 0.1}, {"T_air": _s("T_air", "T_air_min", "T_air_max")}, {}),
    "cement_kiln": (
        {"fuel": 0.1, "speed": 0.1},
        {
            "lime": (_kiln_lime, None, "lime_max"),
            "T_bz": ((lambda o, s, p: jnp.max(s.T_solid)), "T_bz_min", "T_bz_max"),
        },
        {"lime": "fuel", "T_bz": "fuel"},
    ),
    # Firing +0.02 raw; feedwater +2.0 kg/s on 0 to 200 kg/s.
    "boiler_drum": (
        {"fuel": 0.02, "feedwater": 2 * 2.0 / 200.0},
        {
            "level": ((lambda o, s, p: s.level), lambda p: -p.level_trip, "level_trip"),
            "pressure": _s("pressure", "pressure_min", "pressure_max"),
        },
        {"level": "feedwater", "pressure": "fuel"},
    ),
    "wind_turbine": (
        {"pitch": 0.1, "torque": 0.1},
        {
            "P": ((lambda o, s, p: o[3]), None, None),
            "omega": _s(
                "omega", _omega("underspeed_factor"), _omega("overspeed_factor")
            ),
        },
        {"P": "torque", "omega": "pitch"},
    ),
    "battery": (
        {"power": 0.1},
        {
            "P": ((lambda o, s, p: o[3]), None, None),
            "soc": _s("soc", "soc_min", "soc_max"),
            "V_cell": ((lambda o, s, p: o[1]), "V_cell_min", "V_cell_max"),
            "T_cell": _s("T_cell", None, "T_max"),
        },
        {},
    ),
    # Speed +1.75 % of rated on 70 to 105 %; recycle +0.05 on 0 to 1.
    "compressor_surge": (
        {"speed": 2 * 0.0175 / 0.35, "recycle": 2 * 0.05 / 1.0},
        {"dp": _s("dp"), "phi_min": _s("phi_min", lambda p: 2 * p.W, None)},
        {"dp": "speed", "phi_min": "recycle"},
    ),
    "unstable_cstr": (
        {"T_sp": -0.5},
        {"C_a": _s("C_a", lambda p: 0.0, None), "T": _s("T", "T_valid_min", "T_trip")},
        {},
    ),
}


def _stabilised(p):
    """The unstable_cstr base loop: the expert's inner PD on T, its setpoint
    held at the mean of the cascade's last 20 outputs and stepped as given."""
    from target_gym.pc_gym.unstable_cstr import experts as X

    g, dt = X.load_gains(), float(p.delta_t)

    def actuate(u, y, T_prev):
        T = y[1]  # the CV "T"
        T_prev = T if T_prev is None else T_prev
        T_c = g["Tc_bias"] + g["Kp_T"] * (u[0] - T) - g["Kd_T"] * (T - T_prev) / dt
        return [float(X.raw_from_coolant(np.clip(T_c, p.T_c_min, p.T_c_max), p))], T

    return {
        "actuate": actuate,
        "start": lambda acts, mems: np.asarray(mems.T_sp_prev)[:, -20:].mean(1),
        "mv_range": (-math.inf, math.inf),
        "direction": lambda u, size: size,
    }


def _clean(x, dt):
    """JSON-safe, to 6 significant digits (every decision is taken before):
    non-finite numbers as strings; a delay or lag at its lower bound written
    "< dt" (10.3)."""
    if isinstance(x, dict):
        low = {"theta", "tau", "tau2"}
        return {
            k: (
                "< dt"
                if k in low and isinstance(v, float) and v <= 1e-6 * dt
                else _clean(v, dt)
            )
            for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [_clean(v, dt) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.6g}") if math.isfinite(x) else str(float(x))
    return x


def _test_record(run, a, mvs, cvs):
    loops = [
        {"cv": cvs[lp.cv], "mv": mvs[lp.mv], "kc": lp.kc, "n_pre": lp.n_pre}
        for lp in run.loops
    ]
    trips = {
        k: {"index": i, "cv": cvs[c] if c is not None else None}
        for k, (i, c) in run.trips.items()
    }
    head = {"step": run.step, "window": run.window, "loops": loops, "trips": trips}
    return {**head, "samples": a["samples"], **a["fits"]}


def measure(name: str) -> dict:
    spec = REGISTRY[name]
    p = spec.make_test_params()
    steps, cv_defs, expert = TASKS[name]
    mvs, cvs = list(steps), list(cv_defs)

    def limit(b):
        return None if b is None else float(b(p) if callable(b) else getattr(p, b))

    extra = _stabilised(p) if name == "unstable_cstr" else {}
    if name == "glass_furnace":
        extra["window"] = int(round(float(p.reversal_period) / float(p.delta_t)))
    plant = Plant(
        spec,
        p,
        [f for f, _, _ in cv_defs.values()],
        [(limit(lo), limit(hi)) for _, lo, hi in cv_defs.values()],
        expert_mv={cvs.index(c): mvs.index(m) for c, m in expert.items()},
        **extra,
    )
    pairs, classes = identify(plant, list(steps.values()))
    dt, n_ep = plant.dt, plant.n_ep
    key = {(j, i, s): (mvs[j], cvs[i], TRAIN_SEEDS[s]) for j, i, s in pairs}
    valid = {key[k]: r["fit"] for k, r in pairs.items() if r["label"] == "valid"}
    tc, rows, top = task_horizon(valid, dt)
    at = ("mv", "cv", "seed")
    horizon = {}
    if top is not None:  # else non-compliant: no protocol horizon (N3)
        (N, M, P), n_samples = rows[top], episode_samples(n_ep, tc)
        fastest = min(valid, key=lambda k: sample_time(valid[k]))
        ramp, ramp_key = 0, None
        if name in RATE_LIMITS:
            # A gain needs a settled fit, not a quiet one: V1, the noise test
            # alone failing, leaves it usable (H21).
            usable = {
                key[k]: r["fit"]
                for k, r in pairs.items()
                if r.get("fit") and r["label"] != "insignificant"
                and set(r.get("failed", [])) <= {"V1"}
            }  # fmt: skip
            cv, span, rates = RATE_LIMITS[name]
            ramp, ramp_key = task_ramp(usable, cv, float(span(p)), rates(p), tc * dt)
        P += ramp
        horizon = {
            "episode_samples": n_samples,
            "T_c_steps": tc,
            "T_c_seconds": tc * dt,
            "T_c_set_by": dict(zip(at, fastest)),
            "N": N,
            "M": M,
            "ramp": ramp,
            "P": P,
            "P_steps": P * tc,
            "P_set_by": dict(zip(at, top)),
            **({"ramp_set_by": dict(zip(at, ramp_key))} if ramp_key else {}),
            "plans_to_episode_end": P >= n_samples,
        }
    labels = [r["label"] for r in pairs.values()]
    zero = sorted(
        f"{m} -> {c}"
        for m in mvs
        for c in cvs
        if all(
            r["label"] == "insignificant"
            for k, r in pairs.items()
            if key[k][:2] == (m, c)
        )
    )
    row = {
        "protocol": "docs/proposals/oracle-horizon-protocol.md, Amendments 1 to 3",
        "dt": dt,
        "episode_steps": n_ep,
        "compliant": bool(horizon),
        "status": (
            "compliant"
            if horizon
            else "non-compliant: no valid pair, so no protocol horizon; the task "
            "keeps its current oracle, and C1 and C2 leave it as it is (N3)"
        ),
        **horizon,
        "counts": {k: labels.count(k) for k in ("valid", "invalid", "insignificant")},
        "retests": sum(len(r["tests"]) - 1 for (j, i, s), r in pairs.items() if i == 0),
        "zero_models": zero,
        "cvs_without_valid_pair": [c for c in cvs if not any(k[1] == c for k in valid)],
        "classes": {cvs[i]: c for i, c in classes.items()},
        "test": {
            "seeds": list(TRAIN_SEEDS),
            "settle_steps": n_ep,
            "window_steps": n_ep,
            "steps": steps,
            "hold_window": extra.get("window", 20),
            "noise": "residual of each fitted model over its fit window (N1)",
        },
        "pairs": [
            {
                **dict(zip(at, key[k])),
                "label": r["label"],
                "failed": r.get("failed", []),
                "kept": r["kept"],
                "evidence": evidence(r["tests"][r["kept"]][1]),
                "fit": r.get("fit"),
                **dict(zip(("N", "M", "P"), rows.get(key[k], (None,) * 3))),
                "tests": [_test_record(run, a, mvs, cvs) for run, a in r["tests"]],
            }
            for k, r in pairs.items()
        ],
    }
    return _clean(row, dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--envs", nargs="*", default=list(TASKS))
    args = ap.parse_args()
    skipped = sorted(set(REGISTRY) - set(TASKS))
    print(
        f"skipped, {', '.join(skipped)}: their tests fly through ArduPlane "
        "4.0.9's default loops (decision D6), which PR D builds"
    )
    for name in (n for n in args.envs if n in TASKS):
        row = measure(name)
        write(name, row := records.stamped(row))
        if row["compliant"]:
            end = ", to the end" if row["plans_to_episode_end"] else ""
            what = (
                f"T_c {row['T_c_steps']}, P {row['P']} ({row['P_steps']} of "
                f"{row['episode_steps']} steps{end}), set by {row['P_set_by']}"
            )
        else:
            what = "NON-COMPLIANT, keeps its current oracle"
        classes = {c: v["class"] for c, v in row["classes"].items()}
        print(
            f"{name:18s} {what}; {row['counts']}, retests {row['retests']}; "
            f"classes {classes}"
        )


if __name__ == "__main__":
    main()
