"""Record a task's v1 baselines (task protocol, item 4): the command a task
session runs when it freezes its task.

    uv run --group learning python scripts/record_v1.py ph_neutralization
    uv run python scripts/record_v1.py hvac --lines nothing,expert,mpc

Each line is scored by ``tg.score`` against the frozen reference, on the
scoring seeds 200 to 209 of every scenario, and read with its statistic
(``scoring.headline``): the saving with its 95 % interval, per scenario too.

- ``nothing``: the actuators held where the reset left them;
- ``expert``: the frozen reference, ``tg.make_reference``;
- ``mpc``: the task's observation-only MPC, ``observation_only_mpc(env,
  params)`` in its package's ``oracle`` module, built on each scenario's
  configuration (``target_gym.estimation.ObservationOnly`` makes one);
- ``ppo``, ``sac``: 30 runs each at 204,800 steps (``target_gym.learned``),
  which need the ``learning`` group; their training wall-clock is recorded with
  the machine.

The record is ``src/target_gym/data/v1_records/<task>.json``, written after each
line, so a stopped run resumes with the lines it lacks; ``--again`` records
the named lines again. It is stamped with the task's parts
(``target_gym.v1.manifest``); a record of other parts is started afresh. At the
end it says what keeps the task from freezing (``target_gym.v1.ready``) and
writes the task's card (``scripts/generate_cards.py``). Run it on bob or
flanders: the MPC and the learned lines take hours.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import target_gym as tg  # noqa: E402
from target_gym import charter, registry, scoring_runs, v1  # noqa: E402
from target_gym.api import FullState  # noqa: E402

#: What a line's summary keeps of ``tg.score``'s result.
KEPT = ("saving", "saving_ci", "probability_of_improvement", "ties", "worst")
KEPT += ("run_savings", "beats_reference", "scenarios", "read_state", "days")


def scenario_params(task: str, scenario: str):
    """The configuration ``tg.score`` runs ``scenario`` of ``task`` on: the
    run's on a task scored on runs, the reference's otherwise."""
    if task in scoring_runs.PROTOCOLS:
        return scoring_runs.run_params(task, scenario)
    return tg.make_reference(task).params(scenario)


class Hold:
    """Doing nothing: the actuators held where the reset left them. It reads
    the state once, at an episode's first step, for where they are, as an
    operator reads the valve positions; a run's stream wraps the plant."""

    def __init__(self, env, params):
        self.env, self.params = env, params
        self.reset()

    def reset(self):
        self.held = None

    def __call__(self, obs, state):
        if self.held is None:
            plant = getattr(state, "env", state)
            self.held = np.asarray(self.env.initial_action(plant, self.params))
        return self.held


def nothing(task: str):
    env = registry.get(task).make_env()
    return {
        s: FullState(Hold(env, scenario_params(task, s)))
        for s in v1.scenario_names(task)
    }


def mpc(task: str):
    """The task's observation-only MPC, one per scenario's configuration."""
    module = type(registry.get(task).make_env()).__module__.rsplit(".", 1)[0]
    oracle = importlib.import_module(f"{module}.oracle")
    build = getattr(oracle, "observation_only_mpc", None)
    if build is None:
        raise LookupError(
            f"{task} has no observation-only MPC yet: add "
            f"observation_only_mpc(env, params) to {module}.oracle"
        )
    env = registry.get(task).make_env()
    return {s: build(env, scenario_params(task, s)) for s in v1.scenario_names(task)}


def summary(result: dict, seconds: float) -> dict:
    """What a line keeps of ``tg.score``'s result, with both ``J``: the cost a
    step, each scenario's mean weighted by its share of operating time."""
    out = {k: result[k] for k in KEPT if k in result}
    out["saving_ci"] = list(out["saving_ci"])
    rows = result["scenarios"]
    out["J"] = sum(r["weight"] * r["J_policy"] for r in rows)
    out["J_reference"] = sum(r["weight"] * r["J_reference"] for r in rows)
    out["scoring_seconds"] = round(seconds, 1)
    return out


def _plain(value):
    """A NumPy scalar as the Python value JSON writes."""
    return value.item() if hasattr(value, "item") else str(value)


def record_line(task: str, line: str, seeds=None, runs=None, steps=None) -> dict:
    """One line of ``task``'s record: its policy built, scored and summed up."""
    training = None
    if line in ("ppo", "sac"):
        from target_gym import learned

        policy, training = learned.train(
            task, line.upper(), runs or learned.RUNS, steps or learned.BUDGET
        )
    else:
        policy = {"nothing": nothing, "mpc": mpc}.get(line, tg.make_reference)(task)
    start = time.perf_counter()
    result = tg.score(task, policy, seeds=seeds)
    out = summary(result, time.perf_counter() - start)
    if training is not None:
        out["training"] = training
    return out


def _git(*args: str) -> str:
    run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return run.stdout.strip()


def record(
    task: str, lines, again=False, seeds=None, runs=None, steps=None, root=v1.RECORDS
):
    """Record ``lines`` of ``task`` into ``root``; returns the record."""
    entry = v1.manifest(task)
    rec = v1.record(task, root)
    if rec is None or rec["parts"] != entry["parts"]:
        rec = {
            "task": task,
            "parts": entry["parts"],
            "stamp": entry["stamp"],
            "lines": {},
        }
    rec["seeds"] = list(charter.ANCHOR_SEEDS if seeds is None else seeds)
    rec["weights"] = v1.scenario_names(task)
    path = pathlib.Path(root) / f"{task}.json"
    for line in lines:
        if line in rec["lines"] and not again:
            print(f"{line}: recorded, kept")
            continue
        try:
            rec["lines"][line] = record_line(task, line, seeds, runs, steps)
        except LookupError as missing:  # an MPC the task does not ship yet
            print(f"{line}: {missing}")
            continue
        # Each line's own commit: one resumed later may run other controller code.
        rec["commit"] = rec["lines"][line]["commit"] = _git("rev-parse", "HEAD") or None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(rec, indent=1, sort_keys=True, default=_plain) + "\n"
        )
        got = rec["lines"][line]
        lo, hi = got["saving_ci"]
        print(
            f"{line}: J {got['J']:.6g}, saving {got['saving']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        )
    return rec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task", choices=sorted(charter.FLAGSHIP))
    parser.add_argument("--lines", default=",".join(v1.LINES))
    parser.add_argument("--again", action="store_true", help="record the lines again")
    args = parser.parse_args(argv)
    lines = [x for x in args.lines.split(",") if x]
    unknown = sorted(set(lines) - set(v1.LINES))
    if unknown:
        parser.error(f"no such lines {unknown}; the lines are {list(v1.LINES)}")
    rec = record(args.task, lines, args.again)
    missing = v1.ready(rec, v1.manifest(args.task)["parts"])
    print("ready to freeze" if not missing else "not ready: " + "; ".join(missing))
    import generate_cards

    card = generate_cards.write(args.task)
    print(f"card: {card}" if card else f"no card inputs: docs/cards/{args.task}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
