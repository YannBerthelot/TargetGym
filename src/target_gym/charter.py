"""A task's charter record: what docs/proposals/task-charter.md (J2 to J4) says a
task's judgement holds, one JSON per task in ``data/charter/<task>.json``.

``scripts/record_charter.py`` fills the parts; this module holds the schema, the
pure rules (J2's seeds and verdict, J3's lines, J4's anchor and tie tolerance),
validation and staleness. It is in no fingerprint (``target_gym.provenance``
hashes the env packages, the shared physics, ``experts/pid.py``,
``experts/mpc.py`` and ``oracles/``), so editing it never stales a record.

A record has one block per part, each with its own ``fingerprint``:

- ``reference`` (J1): the kept rung or the shipped expert, and its J3 ``line``.
- ``nulls`` (J2's nulls on the ladder seeds): one ``line`` per null; J3's best
  null is the cheapest on ``J``.
- ``mpc`` (J3): the task's oracle, an MPC with the true model, or "not measured".
- ``ceiling`` (J3): the foresight ceiling from ``data/oracle_ceilings``, or "not
  measured", also when the file was measured on another plant: each file carries
  the ``stamp`` (``provenance.environment_fingerprint``) of the plant it ran on,
  and one that differs from the plant's stamp leaves the part stale.
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
#: The foresight ceilings (J3), one file per task, each stamped with its plant.
CEILINGS = PKG / "data" / "oracle_ceilings"
LOG = DATA / "log.jsonl"
SCRIPTS = REPO / "scripts"
#: The reference ladder's rows, one module per task (``target_gym.reference.rows``).
ROWS_DIR = PKG / "reference" / "rows"

PARTS = ("reference", "nulls", "mpc", "ceiling", "j2", "anchor")
NOT_MEASURED = "not measured"
#: J2: 30 seeds, or this many per scenario if that is more; every scenario this many.
J2_TOTAL, J2_PER_SCENARIO = 30, 5
#: J2: the reference may trip on at most one seed in every this many.
TRIPS_PER = 30
#: J2's hash seeds are drawn at or above this, apart from the ladder's 100-109 and
#: the scoring seeds.
J2_SEED_FLOOR = 1000
#: J4's scoring seeds, the same in every scenario, set at the tag, read once by the
#: one final scoring run (owner, 2026-11-27, "Fixed seeds, run-level bootstrap": no
#: stop rule adds seeds after a look).
ANCHOR_SEEDS = tuple(range(200, 210))
#: The validation block, the same in every scenario: checkpoints and
#: hyperparameters are chosen here, never on the scoring seeds (the review's M6).
VALIDATION_SEEDS = tuple(range(300, 350))
#: Each scored task's noise floor, one file per task (``noise_floor``).
NOISE = DATA / "noise_floor"
#: The tie tolerance where no floor is measured: float32 rounding over an episode.
DEFAULT_TIE = 1e-3
#: Rows that are test fixtures, not tasks.
FIXTURES = ("first_order",)
#: Tasks whose reference is the shipped expert (reference_ladder ``--expert``).
EXPERT_REFERENCE = ("reactor",)
#: Tasks whose oracle may not be run, with the reason J3 prints. pH's oracle was
#: rebuilt for the job score on the shared solver (#199), so it is measured.
MPC_NOT_MEASURED: dict[str, str] = {
    "distillation": (
        "the oracle is being rebuilt in its own PR (owner, 2026-11-13); its "
        "MPC line is recorded when that lands"
    ),
}
#: Keys of a data file that only describe its run.
_RUN_KEYS = frozenset({"generated", "cpu", "seconds"})

#: The flagship (charter Words and J5): the scored suite, the owner's choice of
#: 2026-11-24. This is the one place its membership is written. The results,
#: the headline, the radar and the records held current all follow it, and a
#: task outside it ships registered, not scored. A task joins by the bar the
#: owner set the same day: a review score of 8 or more, a room whose interval
#: is above 1 from a converged ceiling, J2 and the sweep (B7) passed, and its
#: plant, reference and tests merged. cstr left it for 0.8.0 (owner, picker
#: 2026-11-27, "Registered, not scored").
FLAGSHIP = ("reactor", "hvac", "compressor_surge", "ph_neutralization")

#: What each scored task's ceiling reads that its policy does not observe (owner,
#: picker 2026-11-25, "Report foresight, name it plainly"): the ceiling stays the
#: room's top, and its page says so in these words. Each list is taken from the
#: ceiling's code: ``pc_gym.foresight`` plans on the seed's own keys from the
#: true state, so it knows every draw ahead; hvac's ``ceiling_data`` reads the
#: weather's path off the episode's keys; the reactor's ceiling
#: (``scripts/reactor_ipopt.py``, ``reactor.oracle`` with a seed) starts from the
#: simulated state and reads the load's wobble the key draws. What is hidden is
#: the env's state and draws against its observation table. Kept here, outside
#: every fingerprint, so a wording change stales no record.
CEILING_READS = {
    # reactor/env.py: Xe_hat, I_hat and their tilts in the state, boron in the
    # core (obs 7 is the boronmeter, 20 min late), load_wobble keyed on demand_key.
    "reactor": (
        "the core's xenon and iodine, top and bottom",
        "the boron in the core now, where the policy reads it 20 min late",
        "the load's frequency-control wobble ahead",
    ),
    # hvac/env.py: weather_dev, an OU draw per step; the policy reads the
    # outdoor temperature now and its next-24 h mean.
    "hvac": ("the outdoor temperature ahead, hour by hour, past its 24 h mean",),
    # compressor_surge/env.py: line_shift (the machine's surge line off the map),
    # demand_levels drawn per block, demand_dev an OU draw per step.
    "compressor_surge": (
        "the machine's true surge line, up to 5 % off its map",
        "the consumers' demand levels ahead and their random part",
    ),
    # ph_neutralization/env.py: q2, an OU draw per step; acid_scale and the
    # influent upsets drawn by start_upset (the flow now is metered, obs 6).
    "ph_neutralization": (
        "the acid strength",
        "the buffer flow",
        "when the influent's acid and flow upsets come and how large",
    ),
}

SCORED, UNDER_REVIEW, EXTRA, FIXTURE = "scored", "under review", "extra", "fixture"


def status(name: str) -> str:
    """Where task ``name`` stands, derived and never stored: ``"scored"`` in
    the flagship, ``"extra"`` with a reason in ``EnvSpec.extra``, ``"fixture"``
    in ``FIXTURES``, and ``"under review"`` otherwise: registered and runnable,
    its charter lines not measured. ``KeyError`` for an unregistered name."""
    from target_gym.registry import get

    spec = get(name)
    if name in FLAGSHIP:
        return SCORED
    if spec.extra is not None:
        return EXTRA
    return FIXTURE if name in FIXTURES else UNDER_REVIEW


def charter_tasks(rows_dir: pathlib.Path = ROWS_DIR) -> list[str]:
    """The candidates the recorder can run: every task with a ladder row, less
    the fixtures and the Extra tasks, plus the expert-reference tasks. Only
    the flagship's records are held current (owner's decisions, 2026-11-15
    for Extra and 2026-11-24 for every task outside the flagship)."""
    from target_gym.registry import REGISTRY

    if not rows_dir.is_dir():
        raise FileNotFoundError(f"{rows_dir} is absent: no ladder rows to read")
    rows = {p.stem for p in rows_dir.glob("*.py") if not p.stem.startswith("_")}
    extra = {name for name, spec in REGISTRY.items() if spec.extra}
    return sorted((rows - set(FIXTURES) - extra) | set(EXPERT_REFERENCE))


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


#: The shared gate rules (owner, 2026-11-27, "Shared rules, then ship"). Nulls: a
#: null that beats the reference by under ``NULL_MARGIN`` of the reference's cost is
#: "not held", reported, not a failure; a win of ``NULL_MARGIN`` or more fails.
NULL_MARGIN = 0.01
#: Ceiling: converged when the kept plans had no capped or failed solve, the two
#: best foresight starts' J agree within ``CEILING_AGREE`` of the kept J, and, where
#: the task has a floor, the kept J is within ``CEILING_NEAR`` above it.
CEILING_AGREE, CEILING_NEAR = 0.01, 0.03


def null_edge(reference: float, null: float) -> str:
    """``"held"`` where the reference costs less than ``null``, ``"not held"``
    where ``null`` beats it by under ``NULL_MARGIN`` of its cost (a tie
    included), else ``"fails"``."""
    if reference < null:
        return "held"
    return "not held" if reference - null < NULL_MARGIN * reference else "fails"


def _edges(reference: dict, null: dict) -> list[str]:
    return [
        null_edge(a, b)
        for s in reference
        for a, b in zip(reference[s], null[s], strict=True)
    ]


def null_wins(reference: dict, null: dict) -> int:
    """Seeds, over every scenario, where ``null`` beats the reference on the same
    seed by ``NULL_MARGIN`` or more. Both are ``{scenario: costs}``."""
    return _edges(reference, null).count("fails")


def null_not_held(reference: dict, null: dict) -> int:
    """Seeds where ``null`` beats the reference by under ``NULL_MARGIN``."""
    return _edges(reference, null).count("not held")


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


def draw_outcome(draw: dict) -> tuple[bool, list[str]]:
    """One sweep draw under the null margin: whether the property holds, and the
    scenarios held only within ``NULL_MARGIN``. A ladder draw (``family_sweep.ladder``)
    holds when a rung was kept, a null ran, and no null's median beats the kept
    rung's by the margin in any scenario; a reactor draw (``family_sweep.reactor``)
    when its recipe ran, the plant rests, and no scenario's null median beats the
    reference's by it. A draw without its medians keeps its own verdict."""
    if draw.get("tune_refused"):
        return False, []
    if "rungs" in draw:
        kept, nulls = draw.get("kept"), draw.get("nulls") or {}
        if kept is None or not nulls:
            return False, []
        ref = draw["rungs"][kept]
        pairs = [(s, ref[s], n[s]) for n in nulls.values() for s in n]
    elif draw.get("scenarios"):
        sc = draw["scenarios"].items()
        pairs = [(s, v["reference_median"], v["null_median"]) for s, v in sc]
        if draw.get("at_rest_cost", 0.0) >= 1.0:
            return False, []
    else:
        return bool(draw["holds"]), []
    edges = [(s, null_edge(r, n)) for s, r, n in pairs]
    held = all(e != "fails" for _, e in edges)
    return held, sorted({s for s, e in edges if e == "not held"})


def sweep_result(draws):
    """B7's outcome for J2: ``NOT_MEASURED`` without a sweep (``draws`` None),
    else whether the property held on every draw (``draw_outcome``)."""
    if draws is None:
        return NOT_MEASURED
    return all(draw_outcome(d)[0] for d in draws)


def sweep_not_held(draws) -> list[str]:
    """The scenarios some draw holds only within ``NULL_MARGIN``."""
    return sorted({s for d in draws or () for s in draw_outcome(d)[1]})


def stop_rule(starts: dict, floor=None, agree=None, near=None) -> bool:
    """The ceiling's agreement and floor tests: ``starts`` maps each foresight
    start to the J its plan reaches. True when the two best agree within
    ``agree`` of the kept (cheapest) J and, where ``floor`` is given, the kept J
    is within ``near`` above it (a plan under it is near it). With fewer than two
    starts the agreement is ``NOT_MEASURED``, and so is the verdict: the ceiling
    is not converged."""
    agree = CEILING_AGREE if agree is None else agree
    near = CEILING_NEAR if near is None else near
    costs = sorted(float(c) for c in (starts or {}).values())
    if len(costs) < 2:
        return NOT_MEASURED
    kept = costs[0]
    agrees = costs[1] - kept <= agree * kept
    return agrees and (floor is None or kept <= (1.0 + near) * float(floor))


def best_known(ceiling, mpc, weights: dict):
    """The ceiling as the cheapest plan known per seed: each seed's cheaper
    untripped run of the foresight ``ceiling`` line and the ``mpc`` line, the
    foresight on a tie, with ``picked`` naming each seed's source. The MPC line
    joins only where it ran the same seeds in every scenario, and a seed of it
    only where its line records that run converged (no capped or failed solve,
    ``converged`` per seed); convergence stays the foresight's (the MPC line is
    not in the agreement test)."""
    if ceiling == NOT_MEASURED:
        return NOT_MEASURED
    same = mpc != NOT_MEASURED and all(
        len(mpc["scenarios"].get(s, {}).get("costs", ())) == len(v["costs"])
        for s, v in ceiling["scenarios"].items()
    )
    picked, per = {}, {}
    for s, v in ceiling["scenarios"].items():
        rows = []
        for i, c in enumerate(v["costs"]):
            best = (c, "foresight")
            m = mpc["scenarios"][s] if same else {}
            ok = (m.get("converged") or [False] * len(v["costs"]))[i] is True
            if ok:
                # a line keeps one trip count per scenario: untripped only when 0
                if not m["trips"] and (bool(v["trips"]) or m["costs"][i] < c):
                    best = (m["costs"][i], "mpc")
            rows.append(best)
        picked[s] = [src for _, src in rows]
        trips = v["trips"] if any(p == "foresight" for p in picked[s]) else 0
        per[s] = {"costs": [c for c, _ in rows], "trips": trips}
    return line(per, weights, converged=ceiling.get("converged")) | {"picked": picked}


def stale_artefacts(null_table: dict | None, expert_fp: str) -> list[str]:
    """The stored artefacts a gate reads that fail their own staleness check and
    must be rebuilt by their recipe first: the null table
    (``data/null_policies``), fitted to an expert whose fingerprint has moved."""
    current = (null_table or {}).get("expert_fingerprint") == expert_fp
    return [] if current else ["null_policies"]


# --- J3 and J4 --------------------------------------------------------------


def line_seeds(spec, weights: dict, stored: dict | None) -> dict:
    """``{scenario: seeds}`` the J3 lines run on: the task's evaluation seeds
    (``evaluation.Evaluation``), and its extra seeds in a scenario where the ceilings
    file ``stored`` solved them. The seed rule (``evaluation.line_seeds``) is applied
    where the ceiling is solved (``scripts/record_ceilings.py owed``), so every line
    is paired with the ceiling seed by seed."""
    from target_gym import evaluation

    ev = evaluation.of(spec)
    out = {}
    for s in weights:
        row = ((stored or {}).get("scenarios") or {}).get(s) or {}
        ran = set(row.get("seeds") or ())
        out[s] = ev.seeds + (ev.extra_seeds if ran & set(ev.extra_seeds) else ())
    return out


def line(scenarios: dict, weights: dict, converged=None) -> dict:
    """A J3 line from ``{scenario: {"costs": [...], "trips": int}}``."""
    per = {
        s: {
            "costs": [float(c) for c in v["costs"]],
            "mean": sum(v["costs"]) / len(v["costs"]),
            "trips": int(v["trips"]),
        }
        # each seed's run converged, where the source records it (best_known)
        | ({"converged": [bool(x) for x in v["converged"]]} if "converged" in v else {})
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


#: Room where the ceiling's cost is within its measurement noise of 0 (charter, Words).
UNBOUNDED = "unbounded"


def room(reference, ceiling, weights: dict, noise: dict | None = None):
    """Room (charter, Words and J3): the reference's cost over the ceiling's, per
    scenario and on ``J``, from two J3 lines, or ``NOT_MEASURED`` without both.

    ``noise`` is each scenario's ceiling measurement noise in cost per step, such
    as what its solver's starts end apart; a scenario it leaves out has none.
    Where the ceiling is within it of 0 the room is ``UNBOUNDED``, and on ``J``
    against the weighted noise. Each scenario also carries its ``gap``, ``w_s``
    times the reference's mean less the ceiling's; they sum to ``J``'s. A scenario
    whose cost no controller can avoid shows as a large cost on both lines and a
    small gap, which a ratio of the two ``J`` alone would hide.
    """
    if NOT_MEASURED in (reference, ceiling):
        return NOT_MEASURED
    noise = noise or {}

    def ratio(cost, floor, within):
        return UNBOUNDED if floor <= within else cost / floor

    per = {}
    for s, w in weights.items():
        r, c = (ln["scenarios"][s]["mean"] for ln in (reference, ceiling))
        per[s] = {"room": ratio(r, c, noise.get(s, 0.0)), "gap": w * (r - c)}
    within = sum(w * noise.get(s, 0.0) for s, w in weights.items())
    return {
        "scenarios": per,
        "J": ratio(reference["J"], ceiling["J"], within),
        "gap": reference["J"] - ceiling["J"],
    }


def ceiling_line(stored: dict, weights: dict, agreement: bool = True):
    """A ``data/oracle_ceilings`` file as a J3 line, or ``NOT_MEASURED`` where it
    holds no per-seed costs. Converged: every seed's solver status 0 or 2 with no
    cap or fall-back (foresight), or every seed converged (shooting), and, with
    ``agreement``, the file's stop rule recorded as converged (the shared rule,
    2026-11-27: not recorded is not measured, so not converged). A file that
    records ``per_seed_converged`` carries it on each scenario, which
    ``best_known`` reads for an MPC line."""
    scenarios, converged = {}, []
    for s, v in stored.get("scenarios", {}).items():
        costs = v.get("ceiling_costs", v.get("ceiling_per_seed"))
        if s not in weights or not isinstance(costs, list):
            return NOT_MEASURED
        trips = v.get("ceiling_trips", v.get("replay_trips"))
        solver = v.get("solver") if isinstance(v.get("solver"), dict) else {}
        each = solver.get("per_seed_converged")
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
        if isinstance(each, list):
            scenarios[s]["converged"] = each
    if set(scenarios) != set(weights):
        return NOT_MEASURED
    if agreement:  # the stop rule's agreement and floor: not recorded, not measured
        converged.append((stored.get("stop_rule") or {}).get("converged") is True)
    return line(scenarios, weights, converged=all(converged))


def ceiling_part(stored: dict | None, stamp: str, weights: dict) -> dict:
    """The record's ``ceiling`` part from a ceilings file: ``NOT_MEASURED`` with
    no file, or with one measured on another plant than ``stamp`` (or that
    carries no stamp), else the file as a J3 line."""
    if stored is None:
        return {"line": NOT_MEASURED, "why": "no ceiling file"}
    if stored.get("stamp") != stamp:
        on = stored.get("stamp") or "an unstamped plant"
        return {"line": NOT_MEASURED, "why": f"measured on another plant ({on})"}
    return {"line": ceiling_line(stored, weights)}


def load_ceiling(task: str, root: pathlib.Path = CEILINGS) -> dict | None:
    p = root / f"{task}.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_ceiling(
    task: str, data: dict, stamp: str, root: pathlib.Path = CEILINGS
) -> pathlib.Path:
    """Write a ceilings file stamped with the plant it ran on: ``stamp`` is
    ``provenance.environment_fingerprint`` of the task's spec."""
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{task}.json"
    p.write_text(json.dumps({**data, "stamp": stamp}, indent=1) + "\n")
    return p


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
    there is no record. The ceiling is also stale when its file was measured on
    another plant (``current["ceiling_stamp"]``, present when a file exists)."""
    if not record or record.get("version") != current["version"]:
        return list(PARTS)
    fps = current["fingerprints"]
    moved = {"ceiling"} if _ceiling_moved(current) else set()
    return [
        p for p in PARTS if p in moved or record.get(p, {}).get("fingerprint") != fps[p]
    ]


def _ceiling_moved(current: dict) -> bool:
    return (
        "ceiling_stamp" in current
        and current["ceiling_stamp"] != current["version"]["stamp"]
    )


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
    from target_gym.scoring_runs import PROTOCOLS

    spec = REGISTRY[task]
    code = P._digest_paths
    # A task scored on stream runs: the runner and its protocol are part of every
    # line's measurement. Others keep their fingerprints as they were.
    runs = (code([PKG / "scoring_runs.py"]),) if task in PROTOCOLS else ()
    stamp = P.environment_fingerprint(spec)
    if task in EXPERT_REFERENCE:
        ref = _digest(stamp, "expert", P.expert_fingerprint(spec), *runs)
    else:
        ladder = [ROWS_DIR / f"{task}.py", SCRIPTS / "reference_ladder.py"]
        ladder += [PKG / "reference" / f for f in ("row.py", "frozen.py")]
        ladder += [PKG / "experts" / f for f in ("ladder.py", "pid.py")]
        ref = _digest(
            stamp, "ladder", code(ladder + [PKG / "identification.py"]), *runs
        )
    nulls = _digest(
        stamp,
        code([PKG / "null_policies.py", SCRIPTS / "reference_ladder.py"]),
        _file_digest(PKG / "data" / "null_policies" / f"{task}.json"),
        *runs,
    )
    oracle = P.oracle_fingerprint(spec) if spec.make_mpc is not None else "none"
    mpc = _digest(stamp, oracle, MPC_NOT_MEASURED.get(task), *runs)
    ceiling = _digest(stamp, oracle, _file_digest(CEILINGS / f"{task}.json"), *runs)
    package = type(spec.make_env()).__module__.rsplit(".", 1)[0].split(".")
    sweep = _digest(
        code([SCRIPTS / "family_sweep.py"]),
        _file_digest(REPO / "src" / pathlib.Path(*package) / "ours.json"),
    )
    j2 = _digest(ref, nulls, sweep, J2_TOTAL, J2_PER_SCENARIO, TRIPS_PER)
    fps = {"reference": ref, "nulls": nulls, "mpc": mpc, "ceiling": ceiling}
    # the anchor's line records the actuators' travel and reversals (review M8)
    fps |= {"j2": j2, "anchor": _digest(ref, ANCHOR_SEEDS, "travel")}
    out = {
        "version": {"env": f"{task}-v{spec.version}", "stamp": stamp},
        "fingerprints": fps,
    }
    stored = load_ceiling(task)
    if stored is not None:
        out["ceiling_stamp"] = stored.get("stamp")
    if spec.evaluation is not None:
        # The lines run on the seeds the ceiling solved (``line_seeds``): a ceiling
        # that adds a scenario's extra seeds stales them. A task that declares no
        # evaluation runs on fixed seeds, and its fingerprints stay as they were.
        from target_gym import scenarios

        weights = scenarios.weights(spec)
        seeds = _digest(sorted(line_seeds(spec, weights, stored).items()))
        for part in ("reference", "nulls", "mpc"):
            fps[part] = _digest(fps[part], seeds)
    return out


def stale_report(tasks=FLAGSHIP) -> dict:
    """``{task: stale parts}`` over the flagship, whose records are the ones
    held current, or over ``tasks``; the current ones are left out."""
    report = {t: stale_parts(load(t), current(t)) for t in tasks}
    return {t: parts for t, parts in report.items() if parts}


# --- J4's tie tolerance ------------------------------------------------------


def noise_floor(task: str, root: pathlib.Path = NOISE):
    """``task``'s noise floor (the outside review's H3, owner 2026-11-27), or
    ``NOT_MEASURED``: one controller run through two code paths on the scoring seeds
    of every scenario (``scripts/record_charter.py --noise-floor``). ``per_scenario``
    holds each seed's relative gap ``|x / b - 1|`` and ``floor`` the largest; the
    file carries the plant's ``stamp`` and the controller's ``expert_fingerprint``."""
    p = root / f"{task}.json"
    return json.loads(p.read_text()) if p.exists() else NOT_MEASURED


def tie_tolerance(task: str) -> float:
    """The relative tolerance within which a seed is a tie, never a win (J4's
    probability of improvement, ``scoring``'s win rates): ``task``'s noise floor,
    or ``DEFAULT_TIE`` where none is measured."""
    floor = noise_floor(task)
    return DEFAULT_TIE if floor == NOT_MEASURED else float(floor["floor"])
