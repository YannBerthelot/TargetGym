"""A task's charter record: what docs/proposals/task-charter.md (J2 to J4) says a
task's judgement holds, one JSON per task in ``data/charter/<task>.json``.

``scripts/record_charter.py`` fills the parts; this module holds the schema, the
pure rules (J2's seeds and verdict, J3's lines, J4's anchor), validation and
staleness. It is in no fingerprint (``target_gym.provenance`` hashes the env
packages, the shared physics, ``experts/pid.py``, ``experts/mpc.py`` and
``oracles/``), so editing it never stales a record.

A record has one block per part, each with its own ``fingerprint``:

- ``reference`` (J1): the kept rung or the shipped expert, and its J3 ``line``.
- ``nulls`` (J2's nulls on the ladder seeds): one ``line`` per null; J3's best
  null is the cheapest on ``J``.
- ``mpc`` (J3): the task's oracle, an MPC with the true model, or "not measured".
- ``ceiling`` (J3): the foresight ceiling from ``data/oracle_ceilings``, or "not
  measured".
- ``j2``: commit, hash seeds, reference trips, each null's share of seeds beating
  the reference, the sweep (B7) and ``accepted``.
- ``anchor`` (J4): the reference on the fixed scoring seeds, per scenario and on J.

A ``line`` is ``{"scenarios": {s: {"costs", "mean", "trips"}}, "J", "trips",
"converged"}`` with ``J = sum_s w_s * mean_s``; ``converged`` is None for a
closed-loop run with no solver. Any line may instead be ``NOT_MEASURED``.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

PKG = pathlib.Path(__file__).resolve().parent
REPO = PKG.parents[1]
DATA = PKG / "data" / "charter"
LOG = DATA / "log.jsonl"
SCRIPTS = REPO / "scripts"
ROWS_DIR = SCRIPTS / "ladder_rows"

PARTS = ("reference", "nulls", "mpc", "ceiling", "j2", "anchor")
NOT_MEASURED = "not measured"
#: J2: 30 seeds, or this many per scenario if that is more; every scenario this many.
J2_TOTAL, J2_PER_SCENARIO = 30, 5
#: J2: the reference may trip on at most one seed in every this many.
TRIPS_PER = 30
#: J2's hash seeds are drawn at or above this, apart from the ladder's 100-109 and
#: the scoring seeds.
J2_SEED_FLOOR = 1000
#: J4's scoring seeds, the same in every scenario, set at the tag.
ANCHOR_SEEDS = tuple(range(200, 210))
#: Rows in ladder_rows that are test fixtures, not tasks.
FIXTURES = ("first_order",)
#: Tasks whose reference is the shipped expert (reference_ladder ``--expert``).
EXPERT_REFERENCE = ("reactor",)
#: Tasks whose oracle may not be run, with the reason J3 prints.
MPC_NOT_MEASURED = {
    "ph_neutralization": "oracle not rebuilt for the job score (oracle.py, ph_stage)"
}
#: Keys of a data file that only describe its run.
_RUN_KEYS = frozenset({"generated", "cpu", "seconds"})


def charter_tasks(rows_dir: pathlib.Path = ROWS_DIR) -> list[str]:
    """Every task with a ladder row, less the fixtures, plus the expert-reference
    tasks."""
    rows = {p.stem for p in rows_dir.glob("*.py") if not p.stem.startswith("_")}
    return sorted((rows - set(FIXTURES)) | set(EXPERT_REFERENCE))


# --- J2 ---------------------------------------------------------------------


def seed_counts(weights: dict) -> dict:
    """J2's seeds per scenario: ``max(30, 5 * n)`` in all, ``5`` to each scenario,
    the rest split by weight by largest remainder, a tie to the heavier scenario,
    then by name."""
    n = len(weights)
    total = max(J2_TOTAL, J2_PER_SCENARIO * n)
    rest = total - J2_PER_SCENARIO * n
    w = sum(weights.values())
    share = {s: rest * v / w for s, v in weights.items()}
    counts = {s: J2_PER_SCENARIO + math.floor(x) for s, x in share.items()}
    left = total - sum(counts.values())
    frac = {s: share[s] - math.floor(share[s]) for s in share}
    order = sorted(weights, key=lambda s: (-frac[s], -weights[s], s))
    for s in order[:left]:
        counts[s] += 1
    return counts


def _hash_int(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def j2_seeds(commit: str, weights: dict) -> dict:
    """``{scenario: seeds}`` from a hash of ``commit``: deterministic per commit,
    distinct within a scenario, all at or above ``J2_SEED_FLOOR``."""
    out = {}
    for s, k in seed_counts(weights).items():
        seeds, i = [], 0
        while len(seeds) < k:
            x = J2_SEED_FLOOR + _hash_int(commit, s, i) % (2**31 - J2_SEED_FLOOR)
            if x not in seeds:
                seeds.append(x)
            i += 1
        out[s] = seeds
    return out


def null_wins(reference: dict, null: dict) -> int:
    """Seeds, over every scenario, where ``null`` costs strictly less than the
    reference on the same seed. Both are ``{scenario: costs}``."""
    return sum(
        sum(b < a for a, b in zip(reference[s], null[s], strict=True))
        for s in reference
    )


def verdict(n_seeds: int, reference_trips: int, wins: dict, sweep) -> dict:
    """J2's acceptance: the reference trips on at most one seed in every 30, no
    null beats it on more than half the seeds, and the sweep holds (``sweep`` is
    True, False or ``NOT_MEASURED``, which cannot accept)."""
    reasons = []
    if reference_trips * TRIPS_PER > n_seeds:
        reasons.append(f"reference trips on {reference_trips} of {n_seeds} seeds")
    for k, w in sorted(wins.items()):
        if w > n_seeds / 2:
            reasons.append(f"null {k!r} beats the reference on {w} of {n_seeds}")
    if sweep is not True:
        reasons.append("sweep not measured" if sweep == NOT_MEASURED else "sweep fails")
    return {"accepted": not reasons, "reasons": reasons}


# --- J3 and J4 --------------------------------------------------------------


def line(scenarios: dict, weights: dict, converged=None) -> dict:
    """A J3 line from ``{scenario: {"costs": [...], "trips": int}}``."""
    per = {
        s: {
            "costs": [float(c) for c in v["costs"]],
            "mean": sum(v["costs"]) / len(v["costs"]),
            "trips": int(v["trips"]),
        }
        for s, v in scenarios.items()
    }
    return {
        "scenarios": per,
        "J": sum(weights[s] * per[s]["mean"] for s in per),
        "trips": sum(v["trips"] for v in per.values()),
        "converged": converged,
    }


def best_null(lines: dict):
    """The cheapest null on ``J``, or ``NOT_MEASURED`` when none ran."""
    run = {k: v for k, v in lines.items() if v != NOT_MEASURED}
    return min(run.items(), key=lambda kv: kv[1]["J"]) if run else NOT_MEASURED


def j3(record: dict) -> dict:
    """J3's four lines from a record."""
    nulls = record.get("nulls", {}).get("lines", {})
    best = best_null(nulls)
    return {
        "best_null": best if best == NOT_MEASURED else {"name": best[0], **best[1]},
        "reference": record.get("reference", {}).get("line", NOT_MEASURED),
        "mpc": record.get("mpc", {}).get("line", NOT_MEASURED),
        "ceiling": record.get("ceiling", {}).get("line", NOT_MEASURED),
    }


def ceiling_line(stored: dict, weights: dict):
    """A ``data/oracle_ceilings`` file as a J3 line, or ``NOT_MEASURED`` where it
    holds no per-seed costs. Converged: every seed's solver status 0 or 2 with no
    cap or fall-back (foresight), or every seed converged (shooting)."""
    scenarios, converged = {}, []
    for s, v in stored.get("scenarios", {}).items():
        costs = v.get("ceiling_costs", v.get("ceiling_per_seed"))
        if s not in weights or not isinstance(costs, list):
            return NOT_MEASURED
        trips = v.get("ceiling_trips", v.get("replay_trips"))
        solver = v.get("solver") if isinstance(v.get("solver"), dict) else {}
        if "status" in solver:
            ok = all(x in (0, 2) for x in solver["status"])
            converged.append(
                ok and not solver.get("capped") and not solver.get("fell_back")
            )
        elif "converged" in solver:
            converged.append(solver["converged"] == len(costs))
        else:
            converged.append(False)
        scenarios[s] = {"costs": costs, "trips": trips if trips is not None else 0}
    if set(scenarios) != set(weights):
        return NOT_MEASURED
    return line(scenarios, weights, converged=all(converged))


# --- schema -----------------------------------------------------------------

_REQUIRED = {
    "reference": ("kind", "kept", "line"),
    "nulls": ("lines", "broken"),
    "mpc": ("line",),
    "ceiling": ("line", "source"),
    "j2": ("commit", "seeds", "reference_trips", "null_wins", "null_share")
    + ("sweep", "accepted", "reasons"),
    "anchor": ("seeds", "line"),
}


def _line_errors(where: str, ln, weights: dict) -> list[str]:
    if ln == NOT_MEASURED:
        return []
    if not isinstance(ln, dict) or set(ln.get("scenarios", {})) != set(weights):
        return [f"{where}: not a line over the task's scenarios"]
    errs = []
    for s, v in ln["scenarios"].items():
        if not v["costs"] or not math.isclose(
            v["mean"], sum(v["costs"]) / len(v["costs"]), rel_tol=1e-9, abs_tol=1e-12
        ):
            errs.append(f"{where}.{s}: mean is not the mean of its costs")
    J = sum(weights[s] * v["mean"] for s, v in ln["scenarios"].items())
    if not math.isclose(ln["J"], J, rel_tol=1e-9, abs_tol=1e-12):
        errs.append(f"{where}: J is not the weighted sum of the scenario means")
    if ln["trips"] != sum(v["trips"] for v in ln["scenarios"].values()):
        errs.append(f"{where}: trips is not the scenarios' sum")
    if ln.get("converged") not in (True, False, None):
        errs.append(f"{where}: converged must be true, false or null")
    return errs


def validate(record: dict) -> list[str]:
    """Every way ``record`` breaks the schema; empty when it holds."""
    errs = [f"missing {k!r}" for k in ("task", "version", "weights") if k not in record]
    if errs:
        return errs
    weights = record["weights"]
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9):
        errs.append("weights do not sum to 1")
    for part, keys in _REQUIRED.items():
        block = record.get(part)
        if block is None:
            continue
        errs += [
            f"{part}: missing {k!r}" for k in ("fingerprint", *keys) if k not in block
        ]
    if errs:
        return errs
    for part in ("reference", "mpc", "ceiling", "anchor"):
        if part in record:
            errs += _line_errors(part, record[part]["line"], weights)
    for k, ln in record.get("nulls", {}).get("lines", {}).items():
        errs += _line_errors(f"nulls.{k}", ln, weights)
    if "j2" in record:
        errs += _j2_errors(record["j2"], weights, record.get("anchor"))
    return errs


def _j2_errors(j2: dict, weights: dict, anchor) -> list[str]:
    errs = []
    if j2["seeds"] != j2_seeds(j2["commit"], weights):
        errs.append("j2: seeds are not the commit's hash seeds")
    n = sum(len(v) for v in j2["seeds"].values())
    if verdict(n, j2["reference_trips"], j2["null_wins"], j2["sweep"]) != {
        "accepted": j2["accepted"],
        "reasons": j2["reasons"],
    }:
        errs.append("j2: accepted does not follow from its numbers")
    if anchor and any(set(v) & set(anchor["seeds"]) for v in j2["seeds"].values()):
        errs.append("j2: seeds overlap the scoring seeds")
    return errs


def stale_parts(record: dict | None, current: dict) -> list[str]:
    """The parts of ``record`` missing or whose fingerprint differs from
    ``current["fingerprints"]``; every part when the env version stamp moved or
    there is no record."""
    if not record or record.get("version") != current["version"]:
        return list(PARTS)
    fps = current["fingerprints"]
    return [p for p in PARTS if record.get(p, {}).get("fingerprint") != fps[p]]


# --- storage ----------------------------------------------------------------


def path(task: str, root: pathlib.Path = DATA) -> pathlib.Path:
    return root / f"{task}.json"


def load(task: str, root: pathlib.Path = DATA) -> dict | None:
    p = path(task, root)
    return json.loads(p.read_text()) if p.exists() else None


def write(task: str, record: dict, root: pathlib.Path = DATA) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    p = path(task, root)
    p.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return p


def append_log(entry: dict, log: pathlib.Path = LOG) -> None:
    """J2's append-only log: one JSON line per run, never rewritten."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


# --- fingerprints -----------------------------------------------------------


def _digest(*items) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:16]


def _file_digest(p: pathlib.Path) -> str:
    if not p.is_file():
        return "absent"
    row = json.loads(p.read_text())
    if isinstance(row, dict):
        row = {k: v for k, v in row.items() if k not in _RUN_KEYS}
    return _digest(row)


def current(task: str) -> dict:
    """The env version stamp and each part's fingerprint as the tree stands."""
    from target_gym import provenance as P
    from target_gym.registry import REGISTRY

    spec = REGISTRY[task]
    stamp = P.environment_fingerprint(spec)
    code = P._digest_paths
    if task in EXPERT_REFERENCE:
        ref = _digest(stamp, "expert", P.expert_fingerprint(spec))
    else:
        ladder = [ROWS_DIR / f"{task}.py", SCRIPTS / "reference_ladder.py"]
        ladder += [PKG / "experts" / f for f in ("ladder.py", "pid.py")]
        ref = _digest(stamp, "ladder", code(ladder + [PKG / "identification.py"]))
    nulls = _digest(
        stamp,
        code([PKG / "null_policies.py", SCRIPTS / "reference_ladder.py"]),
        _file_digest(PKG / "data" / "null_policies" / f"{task}.json"),
    )
    oracle = P.oracle_fingerprint(spec) if spec.make_mpc is not None else "none"
    mpc = _digest(stamp, oracle, MPC_NOT_MEASURED.get(task))
    ceiling = _digest(
        stamp, oracle, _file_digest(PKG / "data" / "oracle_ceilings" / f"{task}.json")
    )
    package = type(spec.make_env()).__module__.rsplit(".", 1)[0].split(".")
    sweep = _digest(
        code([SCRIPTS / "family_sweep.py"]),
        _file_digest(REPO / "src" / pathlib.Path(*package) / "ours.json"),
    )
    j2 = _digest(ref, nulls, sweep, J2_TOTAL, J2_PER_SCENARIO, TRIPS_PER)
    fps = {"reference": ref, "nulls": nulls, "mpc": mpc, "ceiling": ceiling}
    fps |= {"j2": j2, "anchor": _digest(ref, ANCHOR_SEEDS)}
    return {
        "version": {"env": f"{task}-v{spec.version}", "stamp": stamp},
        "fingerprints": fps,
    }


def stale_report(tasks=None) -> dict:
    """``{task: stale parts}`` over the charter tasks, the current ones left out."""
    report = {t: stale_parts(load(t), current(t)) for t in tasks or charter_tasks()}
    return {t: parts for t, parts in report.items() if parts}
