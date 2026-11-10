"""Fit each task's null policies and record how they fare against its expert.

Run with ``uv run python scripts/record_credibility.py [--envs NAME ...]``.
Writes two rows per task (``target_gym.null_policies``):

``src/target_gym/data/null_policies/<task>.json``
    The table, fitted to where the shipped expert settles on the training
    seeds, and the I-only gains, from the bump tests in ``evidence/``, which
    are outside the wheel.
``src/target_gym/data/credibility/<task>.json``
    Each null's result on the held-out seeds, the sourced figure and any
    accepted exception. The benchmark reads its task list from these.

Both move with the expert, so the CI recorder runs this after it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from target_gym import records  # noqa: E402
from target_gym.benchmark import shipped_policy, task_info  # noqa: E402
from target_gym.identification import TRAIN_SEEDS  # noqa: E402
from target_gym.null_policies import (  # noqa: E402
    Table,
    credibility_row,
    gate,
    run_on_seeds,
)
from target_gym.provenance import expert_fingerprint  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

EVIDENCE = ROOT / "evidence" / "oracle_horizons"

#: Per task, the recorded pair (action index, name, tracked name) that holds each
#: tracked output, as the horizon protocol pairs them (H18). A task with no
#: recorded bump test on a raw action has no I-only null: the aircraft have none
#: yet, and ``unstable_cstr``'s goes through the expert's inner loop.
PAIRS = {
    "first_order": ((0, "u", "x"),),
    "cstr": ((0, "water", "T"),),
    "ph_neutralization": ((0, "base", "pH"),),
    "glass_furnace": ((0, "fuel", "T_crown"),),
    "reactor": ((0, "rod", "T_avg"),),
    "hvac": ((0, "heating", "T_air"),),
    "battery": ((0, "power", "P"),),
    "four_tank": ((0, "pump1", "h1"), (1, "pump2", "h2")),
    "distillation": ((0, "L", "yD"), (1, "V", "xB")),
    "boiler_drum": ((1, "feedwater", "level"), (0, "fuel", "pressure")),
    "cement_kiln": ((0, "fuel", "lime"),),
    "compressor_surge": ((0, "speed", "dp"),),
    "wind_turbine": ((1, "torque", "P"),),
}


def fit_table(targets, actions) -> Table:
    """A least-squares quadratic in the targets, scaled per output. A target that
    never moves leaves the mean action."""
    mean, scale = targets.mean(axis=0), targets.std(axis=0)
    empty = Table(mean, np.where(scale > 0.0, scale, 1.0), np.zeros(0))
    coef = np.linalg.lstsq(empty.features(targets), actions, rcond=1e-6)[0]
    return empty._replace(coef=coef)


def simc_i_gain(k: float, tau: float, theta: float) -> float:
    """The I-only gain per step of a first-order lag with delay, both in steps.
    SIMC's rule for a pure delay is ``1 / (k (tau_c + theta))``. The lag counts as
    delay, the half rule's cautious limit, and ``tau_c`` equals that delay."""
    return 1.0 / (2.0 * k * (tau + theta))


def i_gains(name: str, n_act: int, n_out: int, step: float) -> np.ndarray | None:
    """``(n_act, n_out)`` gains per step from the recorded bump tests, or None
    with no usable test. A pair needs a valid first-order fit on most seeds: an
    invalid fit reads something else, and a pure integrator around an integrator
    cannot settle. ``step`` is the control step, in the fits' time units."""
    if name not in PAIRS:
        return None
    rows = json.loads((EVIDENCE / f"{name}.json").read_text())
    rows = next(v for k, v in rows.items() if k in ("analyses", "pairs", "tests"))
    gains = np.zeros((n_act, n_out))
    for out, (act, mv, cv) in enumerate(PAIRS[name]):
        fits = [
            r["fit"]
            for r in rows
            if (r["mv"], r["cv"], r["label"]) == (mv, cv, "valid")
            and r["fit"]["model"] == "fopdt"
        ]
        if len(fits) * 2 <= len(TRAIN_SEEDS):
            continue
        delay = [0.0 if isinstance(f["theta"], str) else f["theta"] for f in fits]
        gains[act, out] = simc_i_gain(
            statistics.median(f["k"] for f in fits),
            statistics.median(f["tau"] for f in fits) / step,
            statistics.median(delay) / step,
        )
    return gains if gains.any() else None


def settled_points(rollout: dict) -> tuple[np.ndarray, np.ndarray]:
    """(targets, actions): the mean of each over the last quarter of every
    target cycle, where the controller has settled if it ever does. A cycle
    with a trip in it is left out: after the trip the plant is a restart, and
    its tail would pair the target with a re-settling action."""
    targets, actions = [], []
    tripped = np.asarray(rollout["tripped"], bool)
    for i in range(len(rollout["length"])):
        cycle = np.flatnonzero(rollout["target_change"][i])
        cuts = [0, *cycle, int(rollout["length"][i])]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a or tripped[i, a:b].any():  # set on the first step, or tripped
                continue
            tail = slice(b - max((b - a) // 4, 1), b)
            targets.append(rollout["target"][i, tail].mean(axis=0))
            actions.append(rollout["action"][i, tail].mean(axis=0))
    return np.array(targets), np.array(actions)


def fitted_row(name: str) -> dict:
    """``name``'s table and gains, from the expert on the training seeds."""
    spec = REGISTRY[name]
    params = spec.make_test_params()
    info = task_info(name, spec.make_env(), params, 1)
    table = fit_table(
        *settled_points(run_on_seeds(shipped_policy("pid"), name, TRAIN_SEEDS))
    )
    env = spec.make_env()
    step = getattr(env, "control_period", 1) * float(params.delta_t)
    gains = i_gains(name, info.n_act, len(info.target_index), step)
    return {
        "expert_fingerprint": expert_fingerprint(spec),
        "table": {k: np.asarray(v).tolist() for k, v in table._asdict().items()},
        "i_gains": None if gains is None else gains.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=sorted(REGISTRY))
    args = ap.parse_args(argv)
    for name in args.envs:
        records.write("null_policies", name, records.stamped(fitted_row(name)))
        row = credibility_row(name, gate(name))
        row["expert_fingerprint"] = expert_fingerprint(REGISTRY[name])
        records.write("credibility", name, records.stamped(row))
        print(name, "null wins:", row["null_wins"] or "none", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
