"""Fill or refresh parts of a task's charter record (docs/proposals/task-charter.md,
J1 to J4; schema and rules in ``target_gym.charter``).

    uv run python scripts/record_charter.py --tasks cstr [--parts j2,mpc] [--force]

Parts: ``reference`` (J1: the ladder's ``climb``, or the shipped expert on the
expert-reference tasks), ``nulls`` (each of ``null_set`` over the ladder seeds),
``mpc`` (the task's oracle, ``shipped_policy("mpc")``), ``ceiling`` (read from
``data/oracle_ceilings``, which ``pc_gym.foresight`` and the tasks' ceiling scripts
write through ``charter.write_ceiling``; a file measured on another plant reads
"not measured"), ``j2`` (the kept reference and every null on the commit's hash
seeds, and the family sweep, ``scripts/family_sweep.py``, on the task's
``ours.json``) and ``anchor`` (J4: the reference on the
scoring seeds). J3's lines run on the ladder seeds 100 to 109, as the ceilings do.
``nulls``, ``j2`` and ``anchor`` read the recorded reference.

The J3 lines (reference, nulls, mpc) run on ``charter.line_seeds``: the task's
evaluation seeds (``evaluation.Evaluation``), plus the extra seeds of a scenario whose
ceiling solved them. On a task that declares an evaluation, record its ceilings
first (``scripts/record_ceilings.py``, ``owed`` lists the seeds the seed rule asks
for), then these lines, so every line is paired with the ceiling seed by seed.

A task with a scoring-run protocol (``target_gym.scoring_runs``: the reactor) runs
every part on its stream pinned to each kept scenario, with the twins' weight
folded, instead of one-day windows; its ceiling part also names the scenarios
whose room interval contains 1, which take the protocol's extra seeds (the stop
rule), and its reference and ceiling are then run again on them.

The shared gate rules (owner, 2026-11-27): the ceiling part is the cheapest plan
known per seed, the MPC line's run against the ceilings file's foresight
(``charter.best_known``; the foresight line stays on record), its convergence the
foresight's stop rule; J2 counts a null's win only at ``charter.NULL_MARGIN`` or more
and reports the rest as not held, in the seeds and in the sweep; and ``nulls`` or
``j2`` first refits a stale null table (``refresh_artefacts``).

A part whose fingerprint is current is skipped unless ``--force``. Each part run
is written at once and appended to ``data/charter/log.jsonl``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from target_gym import charter  # noqa: E402

NM = charter.NOT_MEASURED


def git_commit() -> tuple[str, bool]:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)

    status = git("status", "--porcelain", "--untracked-files=no").stdout
    return git("rev-parse", "HEAD").stdout.strip(), bool(status.strip())


class Task:
    """What every part of one task's run shares: the ladder module, its row, the
    scenario weights and the record so far."""

    def __init__(self, name: str, record: dict):
        import reference_ladder as L

        from target_gym import scenarios
        from target_gym import scoring_runs as S
        from target_gym.registry import REGISTRY

        self.L, self.name, self.record, self.S = L, name, record, S
        self.spec = REGISTRY[name]
        self.protocol = name in S.PROTOCOLS
        self.mpc_pieces: list[str] = []
        self.weights = (
            S.weights(name) if self.protocol else scenarios.weights(self.spec)
        )
        self.expert = name in charter.EXPERT_REFERENCE
        self.row = L.Row(0, None) if self.expert else L.ROWS[name]()
        self.seeds = charter.line_seeds(
            self.spec, self.weights, charter.load_ceiling(name)
        )

    def line(self, result: dict, converged=None) -> dict:
        return charter.line(result["scenarios"], self.weights, converged)

    def kept(self) -> str:
        ref = self.record.get("reference")
        if not ref or not ref.get("kept"):
            raise SystemExit(f"{self.name}: record a reference with a kept rung first")
        return ref["kept"]

    def policy(self):
        from target_gym.benchmark import shipped_policy

        kept = self.kept()
        if self.expert:
            return "pid" if self.protocol else shipped_policy("pid")
        return self.L.Reference(kept, self.L.build(self.name, self.row)[kept])

    def on_seeds(self, result: dict) -> dict:
        """``result``, a run on the ladder's seeds, cut to each scenario's line
        seeds (all of them among the ladder's)."""
        out = {}
        for s, v in result["scenarios"].items():
            at = [self.L.SEEDS.index(x) for x in self.seeds[s]]
            tripped = [v["tripped"][i] for i in at]
            out[s] = {"costs": [v["costs"][i] for i in at], "trips": sum(tripped)}
        return {"scenarios": out}

    def nulls(self, seeds=None) -> tuple[dict, dict]:
        runs, broken = {}, {}
        for null in self.L.null_set(self.name):
            try:
                if self.protocol:
                    # On the stream the reset is the stream's own: its held move.
                    held = "null" if null.name == "hold reset" else null
                    runs[null.name] = self.suite(held, seeds)
                    continue
                runs[null.name] = self.L.run_suite(
                    self.name, self.row, null, seeds=seeds
                )
            except Exception as err:  # a stale table's shape, as check_nulls says
                broken[null.name] = f"{type(err).__name__}: {err}"
        return runs, broken

    def seeds_for(self, scenario: str) -> list[int]:
        """A protocol scenario's seeds: the first block, and the extra one where the
        recorded ceiling part found its room interval containing 1."""
        extended = scenario in self.record.get("ceiling", {}).get("extended", [])
        return self.S.seeds(self.name, extended=extended)

    def suite(self, policy, seeds=None) -> dict:
        """``run_suite``'s result on the protocol's runs: ``policy`` a
        ``scoring_runs`` kind or a ``benchmark`` policy; ``seeds`` one list for
        every scenario, ``{scenario: seeds}``, or each scenario's own."""
        per = {}
        for s in self.weights:
            own = seeds.get(s) if isinstance(seeds, dict) else seeds
            out = self.S.run(self.name, s, policy, own or self.seeds_for(s))
            per[s] = {"costs": out["costs"], "trips": sum(out["trips"])}
        counts = {len(v["costs"]) for v in per.values()}
        costs = None
        if len(counts) == 1:
            costs = [
                sum(self.weights[s] * per[s]["costs"][i] for s in per)
                for i in range(counts.pop())
            ]
        return {
            "costs": costs,
            "trips": sum(v["trips"] for v in per.values()),
            "scenarios": per,
        }


def extended(reference, ceiling, weights: dict, more_seeds) -> list[str]:
    """The stop rule on two J3 lines: the scenarios, run on their first seeds only,
    whose room interval contains 1 (``scoring_runs.needs_more``)."""
    if NM in (reference, ceiling):
        return []
    out = []
    for s in weights:
        r, c = (ln["scenarios"][s]["costs"] for ln in (reference, ceiling))
        if len(r) == len(c) and len(r) < len(more_seeds) and _needs(r, c):
            out.append(s)
    return out


def _needs(reference_costs, ceiling_costs) -> bool:
    from target_gym.scoring_runs import needs_more

    return needs_more(reference_costs, ceiling_costs)


def line_from_pieces(pieces, weights: dict, seeds: dict) -> dict:
    """A J3 line from ``reactor_ceiling.py`` pieces: every scenario on its seeds,
    once, or no line (``SystemExit``); converged only if every seed's solves were."""
    import reactor_ceiling

    stored = reactor_ceiling.assemble(pieces, weights, seeds)
    return charter.ceiling_line(stored, weights)


def part_reference(t: Task) -> dict:
    from target_gym.benchmark import shipped_policy

    if t.expert:
        res = (
            t.suite("pid")
            if t.protocol
            else t.L.run_suite(t.name, t.row, shipped_policy("pid"), seeds=t.seeds)
        )
        out = {"kept": "expert", "rungs": {"expert": res}, "left_out": {}}
        kept = res
    else:
        out = t.L.climb(t.name)  # the rungs on the ladder's seeds, as placed
        kept = t.on_seeds(out["rungs"][out["kept"]]) if out["kept"] else None
    rungs = {
        k: {"J": t.line(v)["J"], "trips": v["trips"]} for k, v in out["rungs"].items()
    }
    return {
        "kind": "expert" if t.expert else "ladder",
        "kept": out["kept"],
        "rungs": rungs,
        "left_out": out["left_out"],
        "line": t.line(kept) if kept else NM,
    }


def part_nulls(t: Task) -> dict:
    runs, broken = t.nulls(t.seeds)
    return {"lines": {k: t.line(v) for k, v in runs.items()}, "broken": broken}


def part_mpc(t: Task) -> dict:
    from target_gym.benchmark import shipped_policy

    why = charter.MPC_NOT_MEASURED.get(t.name)
    if why or t.spec.make_mpc is None:
        return {"line": NM, "why": why or "the task ships no oracle"}
    if t.protocol:
        if t.mpc_pieces:
            pieces = [json.loads(pathlib.Path(p).read_text()) for p in t.mpc_pieces]
            seeds = {s: t.S.seeds(t.name) for s in t.weights}
            return {"line": line_from_pieces(pieces, t.weights, seeds)}
        return {"line": t.line(t.suite("mpc"))}
    run = t.L.run_suite(t.name, t.row, shipped_policy("mpc"), seeds=t.seeds)
    return {"line": t.line(run)}


def part_ceiling(t: Task) -> dict:
    from target_gym.provenance import environment_fingerprint

    stored = charter.load_ceiling(t.name)
    part = charter.ceiling_part(stored, environment_fingerprint(t.spec), t.weights)
    # The shared rule (2026-11-27): the cheapest plan known per seed, the MPC
    # line's run against the foresight's; the foresight line stays on record.
    foresight = part["line"]
    part["line"] = charter.best_known(
        foresight, t.record.get("mpc", {}).get("line", NM), t.weights
    )
    if foresight != NM:
        part["foresight"] = {"J": foresight["J"], "converged": foresight["converged"]}
    source = charter.CEILINGS / f"{t.name}.json"
    part |= {"source": str(source.relative_to(charter.REPO)) if stored else None}
    if t.protocol:
        ref = t.record.get("reference", {}).get("line", NM)
        more = t.S.seeds(t.name, extended=True)
        part["extended"] = extended(ref, part["line"], t.weights, more)
    return part


def part_j2(t: Task) -> dict:
    commit, dirty = git_commit()
    seeds = charter.j2_seeds(commit, t.weights)
    if t.protocol:
        ref = t.suite(t.policy(), seeds)
    else:
        ref = t.L.run_suite(t.name, t.row, t.policy(), seeds=seeds)
    runs, broken = t.nulls(seeds)
    costs = {s: v["costs"] for s, v in ref["scenarios"].items()}
    null_costs = {
        k: {s: v["costs"] for s, v in r["scenarios"].items()} for k, r in runs.items()
    }
    wins = {k: charter.null_wins(costs, c) for k, c in null_costs.items()}
    # wins under the null margin are reported, not counted (2026-11-27)
    not_held = {k: charter.null_not_held(costs, c) for k, c in null_costs.items()}
    n = sum(len(v) for v in seeds.values())
    trips = sum(v["trips"] for v in ref["scenarios"].values())
    import family_sweep

    table = family_sweep.table(t.name)
    draws = None if table is None else family_sweep.run(t.name, table)
    sweep = charter.sweep_result(draws)
    return {
        "commit": commit,
        "dirty": dirty,
        "noise": t.L.NOISE,
        "seeds": seeds,
        "reference_trips": trips,
        "reference_trips_by_scenario": {
            s: v["trips"] for s, v in ref["scenarios"].items()
        },
        "null_wins": wins,
        "null_not_held": not_held,
        "null_share": {k: w / n for k, w in wins.items()},
        "null_trips": {k: r["trips"] for k, r in runs.items()},
        "broken": broken,
        "sweep": sweep,
        "sweep_not_held": charter.sweep_not_held(draws),
        "sweep_draws": draws,
        **charter.verdict(n, trips, wins, sweep),
    }


def part_anchor(t: Task) -> dict:
    seeds = list(charter.ANCHOR_SEEDS)
    if t.protocol:
        res = t.suite(t.policy(), seeds)
    else:
        res = t.L.run_suite(t.name, t.row, t.policy(), seeds=seeds)
    return {"seeds": seeds, "line": t.line(res)}


PARTS = {
    "reference": part_reference,
    "nulls": part_nulls,
    "mpc": part_mpc,
    "ceiling": part_ceiling,
    "j2": part_j2,
    "anchor": part_anchor,
}


def summary(part: str, block: dict) -> dict:
    if part == "j2":
        keys = ("reference_trips", "null_wins", "sweep", "accepted", "reasons")
        return {k: block[k] for k in keys}
    ln = block.get("line")
    if part == "nulls":
        return {k: v["J"] for k, v in block["lines"].items()}
    return {"J": ln["J"], "trips": ln["trips"]} if isinstance(ln, dict) else ln


def refresh_artefacts(name: str) -> list[str]:
    """The shared rule on fresh artefacts (2026-11-27): rebuild by its recipe each
    stored artefact the gates read that fails its own staleness check
    (``charter.stale_artefacts``), before they run. A current one is left alone."""
    from target_gym import records
    from target_gym.provenance import expert_fingerprint
    from target_gym.registry import REGISTRY

    spec = REGISTRY[name]
    stale = charter.stale_artefacts(
        records.load_row("null_policies", name), expert_fingerprint(spec)
    )
    if "null_policies" in stale:
        import record_credibility

        row = records.stamped(record_credibility.fitted_row(name))
        records.write("null_policies", name, row)
        print(f"{name}: null table refitted to the current expert", flush=True)
    return stale


def record(name: str, parts, force: bool, mpc_pieces=()) -> list[str]:
    """Run ``parts`` on ``name``, skipping current ones unless ``force``; returns
    the record's schema errors."""
    if {"nulls", "j2"} & set(parts):
        refresh_artefacts(name)
    now = charter.current(name)
    rec = charter.load(name) or {}
    stale = charter.stale_parts(rec, now)
    if rec.get("version") != now["version"]:
        rec = {}
    t = Task(name, rec)
    t.mpc_pieces = list(mpc_pieces)
    rec.update(task=name, version=now["version"], weights=t.weights)
    commit, dirty = git_commit()
    for part in [p for p in charter.PARTS if p in parts]:
        if part not in stale and not force:
            print(f"{name} {part}: current, skipped", flush=True)
            continue
        block = PARTS[part](t)
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        rec[part] = {
            **block,
            "fingerprint": now["fingerprints"][part],
            "generated": stamp,
        }
        charter.write(name, rec)
        charter.append_log(
            {"task": name, "part": part, "commit": commit, "dirty": dirty}
            | {"generated": stamp, "version": now["version"]}
            | {
                "fingerprint": now["fingerprints"][part],
                "summary": summary(part, block),
            }
        )
        print(f"{name} {part}: {json.dumps(summary(part, block))}", flush=True)
    return charter.validate(rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--parts", default=",".join(charter.PARTS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--mpc-pieces", nargs="*", default=[],
        help="planner-line pieces from reactor_ceiling.py --kind mpc, merged "
        "instead of running the line in one process",
    )  # fmt: skip
    args = ap.parse_args(argv)
    parts = [p for p in args.parts.split(",") if p]
    unknown = sorted(set(parts) - set(charter.PARTS))
    if unknown:
        ap.error(f"unknown parts {unknown}; choose from {charter.PARTS}")
    bad = 0
    for name in args.tasks:
        errors = record(name, parts, args.force, args.mpc_pieces)
        for e in errors:
            print(f"{name}: schema: {e}", file=sys.stderr)
        bad += bool(errors)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
