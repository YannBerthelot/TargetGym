"""Write the baseline results page, ``docs/baseline-results.md``, and its chart.

    uv run python scripts/generate_baseline_table.py            # write
    uv run python scripts/generate_baseline_table.py --check    # CI: exit 1 if stale

The page opens with a chart of each task's oracle and ceiling against its
expert: interactive on the docs site (``docs/javascripts/results-chart.js``
draws it from the rows embedded in the page), and as two static images for
the README, light and dark (``docs/images/results-vs-expert*.svg``).

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
IMAGES = {
    False: ROOT / "docs" / "images" / "results-vs-expert.svg",
    True: ROOT / "docs" / "images" / "results-vs-expert-dark.svg",
}

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


def chart_rows(data: dict) -> list[dict]:
    """One row per registered task in ``data``, in registry order: what the
    chart draws, as recorded."""
    from target_gym.registry import GROUPS, REGISTRY

    return [
        {
            "name": name,
            "group": GROUPS[spec.group],
            "expert": data[name]["pid"]["hold_tracking"],
            "oracle": data[name]["mpc"]["hold_tracking"],
            "ceiling": data[name]["ceiling"],
            "expert_gain": data[name]["pid"]["gain"],
            "oracle_gain": data[name]["mpc"]["gain"],
        }
        for name, spec in REGISTRY.items()
        if name in data
    ]


def _decades(x: float, ref: float) -> float:
    return float(np.log10(x / ref)) if x > 0 else -np.inf


#: The static image's colours, light and dark: the docs' chart palette on
#: GitHub's own page backgrounds.
_INK = {
    False: dict(
        bg="#ffffff", text="#0b0b0b", sub="#52514e", muted="#898781",
        grid="#e1e0d9", conn="#c3c2b7", tick="#52514e",
        expert="#2a78d6", oracle="#eb6834",
    ),
    True: dict(
        bg="#0d1117", text="#f0efec", sub="#c3c2b7", muted="#898781",
        grid="#2c2c2a", conn="#383835", tick="#c3c2b7",
        expert="#3987e5", oracle="#d95926",
    ),
}  # fmt: skip


def results_svg(rows: list[dict], dark: bool = False) -> str:
    """The chart anchored on the expert, as a standalone image: each task's
    oracle hold and ceiling in decades from its expert's hold (left is
    cheaper), grouped by family and sorted by the room the task leaves."""
    from target_gym.registry import GROUPS

    ink, lo, hi = _INK[dark], -5.0, 1.5
    width, left, right, row_h, head_h = 680, 170, 664, 19, 26

    def x(v):
        return left + (min(max(v, lo), hi) - lo) / (hi - lo) * (right - left)

    def text(px, py, s, size=11, fill="sub", anchor="start", extra=""):
        return (
            f'<text x="{px:.1f}" y="{py:.1f}" font-size="{size}" '
            f'fill="{ink[fill]}" text-anchor="{anchor}"{extra}>{s}</text>'
        )

    body, y = [], 58.0
    for label in GROUPS.values():
        group = [r for r in rows if r["group"] == label]
        group.sort(key=lambda r: -_decades(r["expert"], r["ceiling"]))
        if not group:
            continue
        body.append(text(0, y + 18, label, 12, "text", extra=' font-weight="500"'))
        body.append(
            f'<line x1="0" x2="{right}" y1="{y + 24}" y2="{y + 24}" '
            f'stroke="{ink["grid"]}"/>'
        )
        y += head_h
        for r in group:
            cy = y + row_h / 2
            oracle = _decades(r["oracle"], r["expert"])
            ceiling = _decades(r["ceiling"], r["expert"])
            mark = "off-axis" if not np.isfinite(oracle) else f"{oracle:.2f}"
            parts = [
                text(left - 24, cy + 4, r["name"], 11.5, anchor="end", extra=_MONO),
                f'<line x1="{x(0):.1f}" x2="{x(oracle):.1f}" y1="{cy}" y2="{cy}" '
                f'stroke="{ink["conn"]}" stroke-width="2" stroke-linecap="round"/>',
            ]
            for v in (oracle, ceiling):
                if v < lo:
                    parts.append(text(x(lo) - 11, cy + 4, "‹", 12, "muted", "middle"))
                if v > hi:
                    parts.append(text(x(hi) + 9, cy + 4, "›", 12, "muted", "middle"))
            parts += [
                f'<rect x="{x(ceiling) - 1:.1f}" y="{cy - 7}" width="2" '
                f'height="14" rx="1" fill="{ink["tick"]}"/>',
                f'<rect x="{x(oracle) - 5:.1f}" y="{cy - 5}" width="10" '
                f'height="10" rx="1.5" fill="{ink["oracle"]}" '
                f'stroke="{ink["bg"]}" stroke-width="2"/>',
                f'<circle cx="{x(0):.1f}" cy="{cy}" r="5.5" '
                f'fill="{ink["expert"]}" stroke="{ink["bg"]}" stroke-width="2"/>',
            ]
            body.append(
                f'<g data-task="{r["name"]}" data-oracle="{mark}" '
                f'data-ceiling="{ceiling:.2f}">' + "".join(parts) + "</g>"
            )
            y += row_h
    top, axis = 52.0, []
    ticks = [(-5, "10⁵×"), (-4, "10⁴×"), (-3, "1000×"), (-2, "100×"),
             (-1, "10×"), (0, "expert"), (1, "10× costlier")]  # fmt: skip
    for v, label in ticks:
        dash = ' stroke-dasharray="3 3"' if v == 0 else ""
        axis.append(
            f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{y}" '
            f'stroke="{ink["muted" if v == 0 else "grid"]}"{dash}/>'
        )
        axis.append(text(x(v), y + 16, label, 11, "muted", "middle"))
    axis.append(
        text((left + right) / 2, y + 36,
             "Hold cost relative to the expert (log scale; left is cheaper)",
             12, anchor="middle")
    )  # fmt: skip
    legend = [
        text(0, 16, "Oracle and ceiling against each task's expert", 13, "text",
             extra=' font-weight="500"'),
        f'<circle cx="6" cy="36" r="5" fill="{ink["expert"]}"/>',
        text(16, 40, "Expert", 12),
        f'<rect x="76" y="31" width="10" height="10" rx="1.5" fill="{ink["oracle"]}"/>',
        text(92, 40, "Oracle (full-state MPC)", 12),
        f'<rect x="250" y="29" width="2" height="14" rx="1" fill="{ink["tick"]}"/>',
        text(258, 40, "Ceiling (instrument resolution)", 12),
    ]  # fmt: skip
    height = y + 48
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" '
        f'width="{width}" height="{height:.0f}" font-family="{_SANS}" role="img" '
        "aria-label=\"Each task's oracle hold cost and ceiling relative to its "
        'expert, in decades">'
        f'<rect class="bg" width="100%" height="100%" fill="{ink["bg"]}"/>'
        + "".join(legend + axis + body)
        + "</svg>\n"
    )


_SANS = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
_MONO = ' font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"'


def _glance() -> list[str]:
    rows = chart_rows(json.loads(PROTOCOL.read_text()))
    return [
        "## At a glance",
        "",
        "How many times cheaper the oracle holds each target than the expert,",
        "and where the ceiling sits: tracking every quantity within its",
        "instrument's resolution. Switch the anchor to the ceiling to see how far",
        "each controller is from as good as a real plant can verify. Hover a row",
        "for the numbers, including the whole episode.",
        "",
        '<div class="tg-results-chart"></div>',
        '<script type="application/json" id="tg-results-data">'
        + json.dumps(rows, separators=(",", ":"))
        + "</script>",
    ]


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
    sections = (_glance(), _room(), _episode(), _returns(), _caveats())
    parts = [INTRO, *("\n".join(p) for p in sections if p)]
    return "\n\n".join(s.strip("\n") for s in parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rows = chart_rows(json.loads(PROTOCOL.read_text()))
    outputs = {DOC: _page(), **{p: results_svg(rows, d) for d, p in IMAGES.items()}}
    stale = [
        p for p, new in outputs.items() if (p.read_text() if p.exists() else "") != new
    ]
    if not stale:
        print("baseline results page and chart are current")
        return 0
    if args.check:
        for p in stale:
            print(f"  {p.relative_to(ROOT)} is out of date")
        print("run scripts/generate_baseline_table.py")
        return 1
    for p in stale:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(outputs[p])
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
