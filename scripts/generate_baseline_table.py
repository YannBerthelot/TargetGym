"""Write the baseline results page, ``docs/baseline-results.md``.

    uv run python scripts/generate_baseline_table.py            # write
    uv run python scripts/generate_baseline_table.py --check    # CI: exit 1 if stale

The page is the headline results of the whole library, so it is generated from
the recorded data rather than transcribed: the protocol scores from
``src/target_gym/data/protocol_results.json`` (``scripts/evaluate_baselines.py``)
and the episode returns from ``src/target_gym/data/baseline_returns.json``
(``scripts/record_baselines.py``). Hand-transcription is exactly the drift this
repository has spent a lot of effort eliminating everywhere else.

Returns are reported as a **cost per step**: minus the mean return divided
by the episode length. Every version-2 reward is minus a sum of non-negative
costs (docs/reward-shaping.md) -- tracking in floor-widths, avoidable
consumption, a failure charge -- so the per-step cost is comparable across
seeds of one plant and reads directly ("how many floor-widths off, on
average, over the episode"). It is *not* comparable across plants: the priced
plants are in dollars or euros per step, and an episode return mixes the
reach transient with the hold in a proportion set by the episode length. The
room to grow and the score (``target_gym.scoring``) are the numbers to compare
across plants.

Under version 1 the rewards were products of terms in [0, 1] and the table
reported the return as a share of the episode's ceiling; a positive mean return
now means a row was recorded against version 1 and is stale.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "baseline-results.md"
PROTOCOL = ROOT / "src" / "target_gym" / "data" / "protocol_results.json"

INTRO = """\
# Baseline results

<!-- Written by scripts/generate_baseline_table.py from
     src/target_gym/data/protocol_results.json and baseline_returns.json.
     Do not edit by hand. -->

Every environment ships two reference controllers, scored on its test
configuration (`spec.make_test_params()`):

- **The expert** (the `"pid"` slot): reads the observation only, as your agent
  does. It is the score to beat.
- **The oracle** (the `"mpc"` slot): reads the true state, including what the
  observation hides. It shows what information and planning are worth.

Costs on this page are per step, lower is better, in each environment's own
units. To score your own policy, see
[Evaluating against it](using-the-pids.md#evaluating-against-it).
How the controllers are built and measured is in
[How the baselines work](baselines.md).
"""


def _fmt(x) -> str:
    """Three significant figures, written the way a reader scans them: plain
    in the middle of the range, short scientific (``1.11e6``) at the ends."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x == 0:
        return "0"
    if 100 <= abs(x) < 1e5:
        return f"{x:,.0f}"
    if 1e-3 <= abs(x) < 100:
        return f"{x:.3g}"
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def _grouped(rows: dict, columns: int) -> list[str]:
    """Table rows in the sidebar's order, under a bold row per family."""
    from target_gym.registry import GROUPS, REGISTRY

    out = []
    for key, label in GROUPS.items():
        names = [n for n, s in REGISTRY.items() if s.group == key and n in rows]
        if names:
            out.append(f"| **{label}** |" + " |" * (columns - 1))
            out += [f"| [`{n}`](environments/{n}.md) | {rows[n]} |" for n in names]
    return out


def _seeds(counts) -> str:
    counts = sorted(set(counts))
    return (
        f"{counts[0]} seeds"
        if len(counts) == 1
        else f"{counts[0]} to {counts[-1]} seeds"
    )


def _room() -> list[str]:
    from target_gym.scoring import headroom, score

    data = json.loads(PROTOCOL.read_text())
    rows = {}
    for name, r in data.items():
        pid, mpc, c = r.get("pid"), r.get("mpc"), r["ceiling"]
        b = pid and pid["hold_tracking"]
        x = mpc and mpc["hold_tracking"]
        s = score(x, b, c) if pid and mpc else None
        rows[name] = (
            f"{_fmt(b)} | {_fmt(c)} | {headroom(b, c):.2f} | {_fmt(x)} | "
            f"{'—' if s is None or s != s else f'{s:.2f}'}"
        )
    return [
        "## Room to grow",
        "",
        "**Hold** is the tracking cost per step once a controller has settled on",
        "each target, after a settle fixed per environment. The **ceiling** is",
        "tracking every quantity within its instrument's resolution: as good as a",
        "real plant can verify. **Room** is the number of decades from the expert's",
        "hold down to the ceiling; 0 means the expert is already there. The",
        "**oracle score** places the oracle between the expert (0) and the ceiling",
        "(1). Score a policy the same way with `target_gym.scoring.score_policy`.",
        f"{_seeds(r['seeds'] for r in data.values())} each.",
        "",
        "| environment | expert hold | ceiling | room (decades) | oracle hold | oracle score |",
        "| --- | --- | --- | --- | --- | --- |",
        *_grouped(rows, 6),
    ]


def _episode() -> list[str]:
    data = json.loads(PROTOCOL.read_text())
    rows, failing = {}, []
    for name, r in data.items():
        pid, mpc = r.get("pid"), r.get("mpc")
        ratio = pid["gain"] / mpc["gain"] if pid and mpc and mpc["gain"] > 0 else None
        rows[name] = (
            f"{_fmt(pid and pid['gain'])} | {_fmt(mpc and mpc['gain'])} | "
            f"{_fmt(ratio)} | {_fmt(r.get('pid') and r['pid']['transient_cost'])} | "
            f"{_fmt(r.get('mpc') and r['mpc']['transient_cost'])}"
        )
        for who, m in (("expert", pid), ("oracle", mpc)):
            if m and m["failure_rate"] > 0:
                failing.append(f"the {who} on `{name}` ({m['failure_rate']:.3g})")
    lines = [
        "## Whole episode",
        "",
        "The **gain** is the mean cost per step after the plant's burn-in, moves",
        "between targets included: what the plant pays. **Oracle vs expert** is",
        "the expert's gain divided by the oracle's, so 10 means the oracle pays a",
        "tenth. The **transient** is the summed cost around each target change.",
        "",
        "| environment | expert gain | oracle gain | oracle vs expert | expert transient | oracle transient |",
        "| --- | --- | --- | --- | --- | --- |",
        *_grouped(rows, 6),
    ]
    if failing:
        lines += ["", "Trip rate per target change: " + ", ".join(failing) + "."]
    return lines


def _returns() -> list[str]:
    from target_gym.provenance import load_recorded_baselines

    recorded = load_recorded_baselines()
    rows, stale, losing, tripping, seeds = {}, [], [], [], []
    for name, row in recorded.items():
        if not row or "pid_returns" not in row:
            continue
        pid = np.asarray(row["pid_returns"], dtype=float)
        mpc = np.asarray(row["mpc_returns"], dtype=float)
        steps = int(row["steps"])
        # Version-2 rewards are costs: a positive mean return can only come
        # from a row recorded against the version-1 reward, and is stale.
        # Worth shouting about rather than publishing.
        if max(pid.mean(), mpc.mean()) > 1e-9:
            stale.append(
                f"{name}: positive mean return, recorded against the version-1 reward"
            )
        won = int((mpc > pid).sum())
        flag = " ⚠️" if won < len(pid) / 2 else ""
        if flag:
            losing.append(f"`{name}`")
        if row["mpc_trips"]:
            tripping.append(f"`{name}` ({row['mpc_trips']})")
        seeds.append(len(pid))
        rows[name] = (
            f"{steps} | {_fmt(-pid.mean() / steps)} | {_fmt(-mpc.mean() / steps)} | "
            f"{won}/{len(pid)}{flag}"
        )
    if stale:
        raise SystemExit(
            "refusing to publish: "
            + "; ".join(stale)
            + "\nRe-record those environments before regenerating the page."
        )
    lines = [
        "## Episode returns",
        "",
        "The cost per step over the whole test episode (minus the return divided by",
        "the episode length): reaching the targets and holding them, together.",
        f"{_seeds(seeds)}, paired: **oracle wins** counts the seeds where the oracle",
        "beat the expert.",
        "",
        "| environment | steps | expert cost/step | oracle cost/step | oracle wins |",
        "| --- | --- | --- | --- | --- |",
        *_grouped(rows, 5),
    ]
    if losing:
        lines += [
            "",
            "⚠️ The oracle loses more seeds than it wins here, so it is not an upper "
            "bound: " + ", ".join(losing) + ".",
        ]
    if tripping:
        lines += ["", "Oracle trips over the seeds: " + ", ".join(tripping) + "."]
    return lines


def _caveats() -> list[str]:
    from target_gym.registry import REGISTRY

    notes = [
        f"- [`{n}`](environments/{n}.md): {note}"
        for n, s in REGISTRY.items()
        for note in (s.baselines_note, getattr(s, "mpc_degraded", None))
        if note
    ]
    return ["## Caveats", "", *notes] if notes else []


def _page() -> str:
    sections = (_room(), _episode(), _returns(), _caveats())
    parts = [INTRO, *("\n".join(p) for p in sections if p)]
    return "\n\n".join(s.strip("\n") for s in parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    new = _page()
    old = DOC.read_text() if DOC.exists() else ""
    if new == old:
        print("baseline results page is current")
        return 0
    if args.check:
        print(f"  {DOC.relative_to(ROOT)} is out of date")
        print("run scripts/generate_baseline_table.py")
        return 1
    DOC.write_text(new)
    print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
