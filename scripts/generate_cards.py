"""Write each task's card (task protocol, item 8): ``docs/cards/<task>.md``.

    uv run python scripts/generate_cards.py            # write every card
    uv run python scripts/generate_cards.py --check    # fail if one is stale

A card is generated from two things. Its inputs, ``docs/cards/<task>.json``,
say what only the task's session knows, and the session writes them:

    {
      "job": "The job a real operator is paid to do, in one sentence.",
      "decision": "What the agent sets, and how often.",
      "unit": "the unit of the score's cost, as in 'EUR a step'",
      "prices": [
        {"field": "zone_price", "trades": "a second outside the pressure zone",
         "unit": "kJ/s", "source": "[C3], p. 4"}
      ],
      "not_modelled": ["what the task leaves out, one item each"],
      "breakdown": {
        "source": "what produced the figures: the script, its commit, the machine",
        "terms": {"Plan not taken": {"expert": 1.234, "mpc": 0.456}}
      }
    }

A price is a parameter that trades one goal against another; its ``source`` is
where it is published, or null where the price is the project's own. The
breakdown is optional: each cost term's mean a step, each scenario weighted as J
weights it, for the expert and one other recorded line. The card shows each
term's share of the saving and refuses a column that does not sum to its line's
recorded J, so the figures cannot drift from the record. The rest
comes from the tree: the scenarios, their weights and the scoring definition
(``target_gym.v1.manifest``), the baselines (the task's v1 record,
``scripts/record_v1.py``), each price's value in the scored configuration, and
the Gymnasium ID once the task is frozen.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import target_gym as tg  # noqa: E402
from target_gym import registry, scenarios, v1  # noqa: E402

CARDS = ROOT / "docs" / "cards"
NAMES = {
    "nothing": "Doing nothing (actuators held)",
    "expert": "Expert (the frozen reference)",
    "mpc": "MPC on the observation",
    "ppo": "PPO, 30 runs",
    "sac": "SAC, 30 runs",
}
ROOM = (
    "The room, the best line's saving over the expert, is an upper bound on what "
    "a deployed controller would gain: the MPC plans on the task's own model, "
    "estimating only the state, and no plant's controller has its model exact."
)


def _pct(x) -> str:
    return "n.m." if x is None or x != x else f"{100.0 * x:+.1f} %"


def _scoring_text(scoring: dict, unit: str) -> str:
    ev, run = scoring["evaluation"], scoring["run"]
    if run is not None:
        span = f"a {run['days']:g}-day run of {run['steps']} steps"
    elif ev["scored_steps"] is not None:
        start = ev["unscored_steps"]
        span = f"steps {start} to {start + ev['scored_steps']} of each episode"
    else:
        span = "each whole episode"
    # A run's protocol charges its end; an evaluation, its window's edges.
    edges = run if run is not None and run.get("edge_charge") else ev
    store, credit = edges.get("edge_charge"), edges.get("edge_credit", False)
    if not store:
        charge = ""
    elif credit:
        charge = (
            ", charged once for what the store lost and credited for what it "
            f"gained (`{store}`)"
        )
    else:
        charge = f", charged once for what the store lost (`{store}`)"
    seeds = scoring["seeds"]
    return (
        f"The cost is in {unit}, averaged over {span}{charge}. The score is the "
        f"saving over the expert, 1 - J / J_expert, where J weights each "
        f"scenario's mean cost by its share of operating time, on the scoring "
        f"seeds {seeds[0]} to {seeds[-1]} (`tg.score`)."
    )


def _baselines(rec: dict | None, unit: str) -> list[str]:
    if rec is None:
        return [
            "Not recorded yet: `scripts/record_v1.py` records them.",
        ]
    out = [
        f"Recorded on the scoring seeds at commit `{(rec.get('commit') or '?')[:10]}` "
        f"({rec['stamp']}). The interval is 95 %, over seeds, and over runs too for "
        "the learned agents.",
        "",
        f"| Baseline | J ({unit} a step) | Saving | Interval | Trips |",
        "|---|---|---|---|---|",
    ]
    for line in v1.LINES:
        got = rec["lines"].get(line)
        if got is None:
            out.append(f"| {NAMES[line]} | not recorded | | | |")
            continue
        lo, hi = got["saving_ci"]
        trips = sum(_total(row["policy_trips"]) for row in got["scenarios"])
        out.append(
            f"| {NAMES[line]} | {got['J']:.4g} | {_pct(got['saving'])} | "
            f"[{_pct(lo)}, {_pct(hi)}] | {trips} |"
        )
    trained = [rec["lines"][x]["training"] for x in ("ppo", "sac") if x in rec["lines"]]
    for run in trained:
        machine = run["machine"]
        out.append("")
        out.append(
            f"{run['algorithm']} trained {run['runs']} runs of {run['steps']:,} steps "
            f"in {run['training_seconds']:g} s on {machine['cores']} cores "
            f"({machine['backend']}, JAX {machine['jax']}, {run['library']} "
            f"{run['version']})."
        )
    return out


#: How far a breakdown's column may sum from its line's recorded J. The terms are
#: typed in, each rounded to a few digits, and a run on another machine sums in
#: another order; a column further off comes from another run than the record's,
#: and the card would contradict its own baselines table.
TOTAL_RTOL = 0.005


def _breakdown(task: str, breakdown: dict, rec: dict | None, unit: str) -> list[str]:
    """The saving by cost term: the expert's and one line's mean cost a step,
    each column checked against its line's recorded J."""
    terms = breakdown["terms"]
    lines = set(next(iter(terms.values())))
    if (
        "expert" not in lines
        or len(lines) != 2
        or any(set(v) != lines for v in terms.values())
    ):
        raise SystemExit(
            f"{task}: each term of the breakdown gives the expert and one other line"
        )
    (other,) = lines - {"expert"}
    recorded = (rec or {}).get("lines", {})
    totals = {x: sum(v[x] for v in terms.values()) for x in lines}
    for x in lines:
        if x not in recorded:
            raise SystemExit(f"{task}: the breakdown has {x}, which the record lacks")
        J = recorded[x]["J"]
        if abs(totals[x] - J) > TOTAL_RTOL * abs(J):
            raise SystemExit(
                f"{task}: the breakdown's {x} terms sum to {totals[x]:.4g}, the "
                f"record's J is {J:.4g}: take the breakdown on the recorded run"
            )
    saving = totals["expert"] - totals[other]
    out = [
        f"Where the saving comes from: each cost term's mean a step, in {unit}, each "
        f"scenario weighted as J weights it ({breakdown['source']}). Each column "
        f"sums to its line's recorded J, to within {100 * TOTAL_RTOL:g} %.",
        "",
        f"| Term | {NAMES['expert']} | {NAMES[other]} | Share of the saving |",
        "|---|---|---|---|",
    ]
    for name, v in [*terms.items(), ("Total", totals)]:
        share = (v["expert"] - v[other]) / saving if saving else None
        out.append(f"| {name} | {v['expert']:.4g} | {v[other]:.4g} | {_pct(share)} |")
    return out


def _expert(task: str) -> str:
    """Which controller the expert is (item 3, as read on 2026-11-28)."""
    from target_gym import reference

    frozen = reference.load(task)
    if frozen is None or frozen["kind"] == "expert":
        return "The expert is the task's shipped controller, frozen as its reference."
    return (
        f"The expert is the reference ladder's `{frozen['rung']}` rung, frozen as "
        "data: the ladder keeps the cheapest rung on its seeds, so its cost picked it."
    )


def _total(trips) -> int:
    return int(sum(trips)) if isinstance(trips, list) else int(trips)


def _prices(inputs: dict, task: str) -> list[str]:
    params = tg.make(task)[1]
    prices = inputs.get("prices", [])
    out = ["| Price | Value | Unit | Trades | Source |", "|---|---|---|---|---|"]
    for p in prices:
        if not hasattr(params, p["field"]):
            raise SystemExit(f"{task}: the card prices {p['field']!r}, not a parameter")
        source = p.get("source") or "ours, not sourced"
        value = getattr(params, p["field"])
        out.append(
            f"| `{p['field']}` | {value:g} | {p['unit']} | {p['trades']} | {source} |"
        )
    sourced = sum(1 for p in prices if p.get("source"))
    out += ["", f"{sourced} of {len(prices)} prices have a published source."]
    if sourced < len(prices):
        out.append(
            "A task waits until every one does (items 2 and 3, the owner's reading "
            "of 2026-11-28)."
        )
    return out


def card(task: str, inputs: dict, records=v1.RECORDS, manifests=v1.MANIFESTS) -> str:
    """Task ``task``'s card from its ``inputs``."""
    spec = registry.get(task)
    entry = v1.frozen(manifests).get(task)
    scoring = v1.manifest(task)["scoring"]
    suite = {s.name: s for s in scenarios.suite(spec)}
    title = registry.display_name(task)
    if entry is None:
        status = f"A v1 candidate, not frozen yet; its development stamp is {spec.versioned_name}."
    else:
        status = (
            f"Frozen as `{entry['id']}` on {entry['frozen']} at commit "
            f"`{(entry['commit'] or '?')[:10]}` ({entry['stamp']})."
        )
    lines = [f"# {title}: the task card", "", status, "", "## The job", ""]
    lines += [inputs["job"], "", f"**The decision.** {inputs['decision']}", ""]
    lines += ["## The score", "", _scoring_text(scoring, inputs["unit"]), ""]
    lines += ["| Scenario | Weight | What it is |", "|---|---|---|"]
    for name, part in scoring["scenarios"].items():
        lines.append(f"| `{name}` | {part['weight']:.3f} | {suite[name].description} |")
    lines += ["", "## Baselines", "", _expert(task), ""]
    rec = v1.record(task, records)
    lines += _baselines(rec, inputs["unit"])
    if "breakdown" in inputs:
        lines += ["", *_breakdown(task, inputs["breakdown"], rec, inputs["unit"])]
    lines += ["", ROOM, "", "## Prices", ""]
    lines += _prices(inputs, task)
    lines += ["", "## What it does not model", ""]
    lines += [f"- {item}" for item in inputs["not_modelled"]]
    if entry is not None:
        lines += [
            "",
            "## Use it",
            "",
            "```python",
            "import gymnasium",
            "import target_gym",
        ]
        lines += ["", f'env = gymnasium.make("{entry["id"]}")', "```"]
    return "\n".join(lines) + "\n"


def write(task: str) -> pathlib.Path | None:
    """Write ``task``'s card where it has inputs; returns its path."""
    source = CARDS / f"{task}.json"
    if not source.is_file():
        return None
    path = CARDS / f"{task}.md"
    path.write_text(card(task, json.loads(source.read_text())))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for source in sorted(CARDS.glob("*.json")):
        task, path = source.stem, source.with_suffix(".md")
        text = card(task, json.loads(source.read_text()))
        if args.check:
            if not path.is_file() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(text)
    if stale:
        print("stale cards, run scripts/generate_cards.py:", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
