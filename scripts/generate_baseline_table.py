"""Write the recorded baseline comparison into ``docs/baselines.md``.

    uv run python scripts/generate_baseline_table.py            # write
    uv run python scripts/generate_baseline_table.py --check    # CI: exit 1 if stale

The table used to be transcribed by hand from ``data/baseline_returns.json``,
including a "seeds won" column that nothing computed. That is the headline
results table of the whole library, and hand-transcription is exactly the drift
this repository has spent a lot of effort eliminating everywhere else.

Returns are also reported as a **fraction of the ceiling**. Every reward in the
suite is a product of ``log_scaled_reward`` terms, each in [0, 1], times factors
of the form ``(1 - w*x)`` with ``x`` in [0, 1], so a step can score at most 1
and an episode's ceiling is exactly its number of env steps. Raw returns span
30 to 900 across the suite and say nothing on their own; the fraction says what
share of the available reward a controller actually captured, and is comparable
across every row.
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
    """Env steps in an episode, which is the most reward it can pay.

    Every shipped reward is a product of terms in [0, 1], so one env step pays
    at most 1 and the ceiling is just the step count.

    ``max_steps_in_episode`` counts env steps on every plant, the reactor
    included since its clock was unified: ``control_period`` is how many
    physics sub-steps run *inside* one ``step_env`` call and changes nothing
    about how many rewards an episode pays. (When the reactor's limit was still
    written in physics steps, this once divided by ``control_period`` and put
    the reactor's share at 1.251, which is what the guard below is for.)
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
        ceiling = _ceiling(spec, row)
        won = int((mpc > pid).sum())
        flag = " ⚠️" if won < len(pid) / 2 else ""
        # A share above 1 is arithmetically impossible under the current
        # rewards, every one of which is a product of terms in [0, 1]. It means
        # the row was recorded against a different reward or a different notion
        # of a step, and is stale. Worth shouting about rather than publishing:
        # this is precisely the inconsistency raw returns hide.
        over = max(pid.mean(), mpc.mean()) / ceiling
        if over > 1.0 + 1e-9:
            stale.append(f"{name}: share {over:.3f} exceeds 1, so the row is stale")
        rows.append(
            (
                mpc.mean() / ceiling - pid.mean() / ceiling,
                f"| `{name}` | {ceiling} | {pid.mean():.1f} | {mpc.mean():.1f} | "
                f"{pid.mean() / ceiling:.3f} | {mpc.mean() / ceiling:.3f} | "
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
            "| environment | steps | PID | MPC | PID share | MPC share | MPC wins | term |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            *[r for _, r in rows],
            "",
            "`share` is the mean return over the episode's ceiling, so 1.000 would be",
            "perfect tracking on every step. It is comparable across rows; the raw",
            "returns are not, because they are sums over episodes of different lengths.",
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
