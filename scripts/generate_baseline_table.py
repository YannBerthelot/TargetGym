"""Write the recorded baseline comparison into ``docs/baselines.md``.

    uv run python scripts/generate_baseline_table.py            # write
    uv run python scripts/generate_baseline_table.py --check    # CI: exit 1 if stale

The table used to be transcribed by hand from ``data/baseline_returns.json``,
including a "seeds won" column that nothing computed. That is the headline
results table of the whole library, and hand-transcription is exactly the drift
this repository has spent a lot of effort eliminating everywhere else.

Returns are reported as a **cost per step**: minus the mean return divided
by the episode length. Every version-2 reward is minus a sum of non-negative
costs (docs/reward-shaping.md) -- tracking in floor-widths, avoidable
consumption, a failure charge -- so the per-step cost is comparable across
seeds of one plant and reads directly ("how many floor-widths off, on
average, over the episode"). It is *not* comparable across plants: the priced
plants are in dollars or euros per step, and an episode return mixes the
reach transient with the hold in a proportion set by the episode length. The
protocol numbers in ``data/protocol_results.json`` (``scripts/evaluate_baselines.py``)
are the ones to compare across plants.

Under version 1 the rewards were products of terms in [0, 1] and the table
reported the return as a share of the episode's ceiling; a positive mean return
now means a row was recorded against version 1 and is stale.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "baselines.md"
BEGIN = "<!-- BEGIN GENERATED BASELINE TABLE -->"
END = "<!-- END GENERATED BASELINE TABLE -->"


def _ceiling(spec, row) -> int:
    """Env steps in an episode, the divisor of the per-step cost.

    ``max_steps_in_episode`` counts env steps on every plant, the reactor
    included since its clock was unified: ``control_period`` is how many
    physics sub-steps run *inside* one ``step_env`` call and changes nothing
    about how many rewards an episode pays.
    """
    return int(row["steps"])


def _table() -> str:
    from target_gym.provenance import load_recorded_baselines
    from target_gym.registry import REGISTRY

    recorded = load_recorded_baselines()
    rows: list = []
    stale: list[str] = []
    for name, spec in REGISTRY.items():
        row = recorded.get(name)
        if not row or "pid_returns" not in row:
            continue
        pid = np.asarray(row["pid_returns"], dtype=float)
        mpc = np.asarray(row["mpc_returns"], dtype=float)
        steps = _ceiling(spec, row)
        won = int((mpc > pid).sum())
        flag = " ⚠️" if won < len(pid) / 2 else ""
        # Version-2 rewards are costs: a positive mean return can only come
        # from a row recorded against the version-1 reward, and is stale.
        # Worth shouting about rather than publishing.
        if max(pid.mean(), mpc.mean()) > 1e-9:
            stale.append(
                f"{name}: positive mean return, recorded against the version-1 reward"
            )
        pid_cost, mpc_cost = -pid.mean() / steps, -mpc.mean() / steps
        rows.append(
            (
                (pid_cost - mpc_cost) / max(abs(pid_cost), 1e-12),
                f"| `{name}` | {steps} | {pid_cost:.4g} | {mpc_cost:.4g} | "
                f"{(pid_cost - mpc_cost) / max(abs(pid_cost), 1e-12):.3f} | "
                f"{won}/{len(pid)}{flag} | {row['mpc_terminated_early']} |",
            )
        )

    if stale:
        raise SystemExit(
            "refusing to publish: "
            + "; ".join(stale)
            + "\nRe-record those environments before regenerating the table."
        )

    rows.sort(key=lambda r: -r[0])
    return "\n".join(
        [
            BEGIN,
            "",
            "<!-- Written by scripts/generate_baseline_table.py from",
            "     data/baseline_returns.json. Do not edit by hand. -->",
            "",
            "| environment | steps | PID cost/step | MPC cost/step | MPC saves | MPC wins | term |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *[r for _, r in rows],
            "",
            "`cost/step` is minus the mean return over the episode length: tracking in",
            "floor-widths plus avoidable consumption (dimensionless plants) or dollars /",
            "euros per step (reactor, battery, wind turbine, HVAC). `MPC saves` is the",
            "fraction of the PID's cost the MPC removes. Episode returns mix the reach",
            "transient with the hold; the protocol table below separates them.",
            "",
            "`MPC wins` counts seeds where the MPC out-scored the PID, paired.",
            "A ⚠️ marks an environment where it loses more often than it wins, which",
            "means it is not the upper bound this table presents it as; those carry an",
            "`EnvSpec.mpc_degraded` note saying why.",
            "",
            "`term` counts seeds where the MPC ended the episode early. A permanent",
            "zero can mean the controller is safe or that the environment cannot",
            "terminate at all; `first_order` is the latter.",
            "",
            END,
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    block = _table()
    text = DOC.read_text()
    if BEGIN in text and END in text:
        i, j = text.index(BEGIN), text.index(END) + len(END)
        new = text[:i] + block + text[j:]
    else:
        new = text.rstrip() + "\n\n" + block + "\n"

    if new == text:
        print("baseline table is current")
        return 0
    if args.check:
        print(f"  {DOC.relative_to(ROOT)} is out of date")
        print("1 stale block; run scripts/generate_baseline_table.py")
        return 1
    DOC.write_text(new)
    print(f"wrote the baseline table into {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
