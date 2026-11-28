"""Tasks frozen at v1: ``TargetGym/<task>-v1`` (task protocol, item 9).

A task that has the protocol's nine items is frozen once, by
``scripts/freeze_v1.py <task>``. Its manifest, ``data/v1/<task>.json``, holds a
fingerprint of each part its score depends on:

- ``plant``: the environment's own modules, the shared physics and the scored
  parameters' values (``provenance.environment_fingerprint``);
- ``scoring``: each scenario's weight and parameters, the scoring seeds, every
  field of the task's evaluation (its scored window, edge charge and credit)
  and of a run's protocol (its length, folds and end charge), but the charter
  lines' seeds, and the tie tolerance;
- ``reference``: the frozen reference's law and the code that builds it, or the
  shipped expert's gains and code for a task with no frozen reference.

``tests/test_v1_freeze.py`` fails when any part moves. A frozen task never
changes: a change becomes ``<task>-v2`` beside it. The development stamps
(``reactor-v10``) keep counting every change to the code, and a frozen task's
manifest records the one it froze at.

``import target_gym`` registers every frozen task with Gymnasium, so
``gymnasium.make("TargetGym/<task>-v1")`` finds it, on the episode its score
runs: the heaviest scenario by default, any other by ``scenario=``.

Kept out of every fingerprint: it records what a task is, and is no part of it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import pathlib

__all__ = ["FrozenTaskChanged", "NotReady", "changes", "freeze", "frozen", "gym_id"]
__all__ += ["make"]
__all__ += ["manifest", "ready", "record", "register", "scenario_names", "LINES"]

NAMESPACE = "TargetGym"
MANIFESTS = pathlib.Path(__file__).resolve().parent / "data" / "v1"
#: Each task's baselines (``scripts/record_v1.py``), the record a freeze reads.
RECORDS = MANIFESTS.parent / "v1_records"
#: The baselines of item 4: doing nothing, the expert, the observation-only
#: MPC, and PPO and SAC at one cheap budget.
LINES = ("nothing", "expert", "mpc", "ppo", "sac")
#: The lines item 5's room may come from: the MPC or a learned agent.
ROOM_LINES = ("mpc", "ppo", "sac")
#: The frozen reference's keys that decide its actions; the rest describe the
#: run that froze it.
_REFERENCE_KEYS = ("task", "kind", "rung", "law", "params", "stamp", "expert")


class FrozenTaskChanged(RuntimeError):
    """A frozen task whose tree no longer matches its manifest."""


def gym_id(task: str) -> str:
    """``task``'s Gymnasium ID once frozen: ``TargetGym/<task>-v1``."""
    return f"{NAMESPACE}/{task}-v1"


def _digest(value) -> str:
    text = json.dumps(value, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def scenario_names(task: str) -> dict[str, float]:
    """``task``'s scored scenarios with their weights, heaviest first: a task
    scored on runs folds each dropped twin into the scenario named for it."""
    from target_gym import registry, scenarios, scoring_runs

    if task in scoring_runs.PROTOCOLS:
        weights = scoring_runs.weights(task)
    else:
        weights = scenarios.weights(registry.get(task))
    return dict(sorted(weights.items(), key=lambda kv: -kv[1]))


def _scoring(task: str) -> dict:
    """The scoring definition of ``task``, as data."""
    from target_gym import charter, evaluation, registry, scenarios, scoring_runs
    from target_gym.provenance import _values

    spec = registry.get(task)
    weights = scenario_names(task)
    suite = {s.name: s for s in scenarios.suite(spec)}
    ev = evaluation.of(spec)
    protocol = scoring_runs.PROTOCOLS.get(task)
    out = {
        "scenarios": {
            name: {
                "weight": weights[name],
                "params": _values(scenarios.params_for(spec, suite[name])),
                "stream": {
                    "dwell_h": suite[name].dwell_h,
                    "events_per_h": suite[name].events_per_h,
                    "pinned_events_per_h": suite[name].pinned_events_per_h,
                    "overrides": dict(suite[name].stream_overrides),
                },
            }
            for name in weights
        },
        "seeds": list(charter.ANCHOR_SEEDS),
        "evaluation": _fields(ev, _LINE_FIELDS),
        "run": None,
        "tie": charter.tie_tolerance(task),
    }
    if protocol is not None:
        run = _fields(protocol, (*_LINE_FIELDS, "source"))
        out["run"] = {**run, "steps": scoring_runs.scored_steps(task)}
    return out


#: The fields of an evaluation or a run's protocol that set the charter's line
#: seeds, which ``tg.score`` does not read: it runs the scoring seeds.
_LINE_FIELDS = ("seeds", "extra_seeds", "more", "first")


def _fields(declared, skipped) -> dict:
    """Every field of dataclass ``declared`` but ``skipped``, so a field it
    gains later is part of the scoring without a change here."""
    return {
        f.name: getattr(declared, f.name)
        for f in dataclasses.fields(declared)
        if f.name not in skipped
    }


def _reference(task: str) -> dict:
    """What the comparator is: the frozen reference's law and the code that
    builds it, or the shipped expert's fingerprint."""
    from target_gym import provenance, reference, registry

    spec = registry.get(task)
    frozen = reference.load(task)
    if frozen is None or frozen["kind"] == "expert":
        return {"kind": "expert", "expert": provenance.expert_fingerprint(spec)}
    package = pathlib.Path(reference.__file__).parent
    code = [package / f for f in ("__init__.py", "row.py", "frozen.py")]
    code.append(package / "rows" / f"{task}.py")
    experts = package.parent / "experts"
    code += [experts / "pid.py", experts / "ladder.py"]
    return {
        "kind": frozen["kind"],
        "law": _digest({k: frozen.get(k) for k in _REFERENCE_KEYS}),
        "code": provenance._digest_paths(code),
    }


def manifest(task: str) -> dict:
    """``task``'s parts as the tree has them now, each a fingerprint, with the
    scoring definition in full beside them."""
    from target_gym import registry
    from target_gym.provenance import environment_fingerprint

    spec = registry.get(task)
    scoring = _scoring(task)
    return {
        "id": gym_id(task),
        "task": task,
        "stamp": spec.versioned_name,
        "parts": {
            "plant": environment_fingerprint(spec),
            "scoring": _digest(scoring),
            "reference": _digest(_reference(task)),
        },
        "scoring": scoring,
    }


def record(task: str, root: pathlib.Path = RECORDS) -> dict | None:
    """``task``'s v1 record, None if it has none."""
    path = pathlib.Path(root) / f"{task}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def ready(rec: dict | None, parts: dict) -> list[str]:
    """What keeps a task with v1 record ``rec`` from freezing on a tree whose
    parts are ``parts`` (:func:`manifest`'s), as far as its record can say:
    item 4, every baseline recorded on these parts, and item 5, a room line
    whose interval is above zero. Empty when nothing does."""
    if rec is None:
        return ["no v1 record: run scripts/record_v1.py"]
    out = [
        f"{p} changed since the record"
        for p in parts
        if rec["parts"].get(p) != parts[p]
    ]
    lines = rec.get("lines", {})
    out += [f"{line} not recorded" for line in LINES if line not in lines]
    beats = [
        line
        for line in ROOM_LINES
        if line in lines and lines[line]["saving_ci"][0] > 0.0
    ]
    if not beats:
        out.append("no room: neither the MPC nor a learned agent beats the expert")
    return out


def frozen(root: pathlib.Path = MANIFESTS) -> dict[str, dict]:
    """Every frozen task's manifest, by task."""
    root = pathlib.Path(root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else []
    return {p.stem: json.loads(p.read_text()) for p in paths}


def changes(task: str, root: pathlib.Path = MANIFESTS) -> list[str]:
    """The parts of frozen ``task`` that the tree no longer matches."""
    stored = frozen(root)[task]["parts"]
    now = manifest(task)["parts"]
    return [part for part in stored if now.get(part) != stored[part]]


class NotReady(RuntimeError):
    """A scored task whose record does not show items 4 and 5 (:func:`ready`)."""


def freeze(
    task: str,
    root: pathlib.Path = MANIFESTS,
    commit: str | None = None,
    records: pathlib.Path = RECORDS,
):
    """Write ``task``'s manifest, once: a frozen task is never frozen again. A
    task whose frozen reference no longer describes its plant or its expert is
    refused (``reference.StaleReference``): freeze the reference first. So is a
    scored task its record does not show ready (:func:`ready`, ``NotReady``)."""
    from target_gym import charter, reference

    root = pathlib.Path(root)
    path = root / f"{task}.json"
    if path.exists():
        raise FileExistsError(
            f"{task} is frozen as {gym_id(task)}; a change to it becomes "
            f"{task}-v2 beside it (docs/task-protocol.md, item 9)"
        )
    if reference.load(task) is not None:
        reference.build(task)
    entry = manifest(task)
    if charter.status(task) == charter.SCORED:
        missing = ready(record(task, records), entry["parts"])
        if missing:
            raise NotReady(f"{task} is not ready to freeze: {'; '.join(missing)}")
    entry["frozen"] = _dt.date.today().isoformat()
    entry["commit"] = commit
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    return entry


def _check(task: str, root) -> None:
    if task not in frozen(root):
        raise KeyError(f"{task} is not frozen; these are: {sorted(frozen(root))}")
    moved = changes(task, root)
    if moved:
        raise FrozenTaskChanged(
            f"{gym_id(task)} no longer matches its manifest: {', '.join(moved)} "
            f"changed. A frozen task never changes; register the change as "
            f"{task}-v2 beside it."
        )


def make(task: str, scenario: str | None = None, render_mode=None, root=MANIFESTS):
    """Frozen ``task`` as a Gymnasium environment, on the episode its score
    runs in ``scenario`` (the heaviest by default): the reference's
    configuration of it on a scored task, a task scored on runs on its run."""
    from target_gym import api, charter, reference, registry, scenarios, scoring_runs
    from target_gym.wrapper import GymnasiumEnv

    _check(task, root)
    names = scenario_names(task)
    scenario = next(iter(names)) if scenario is None else scenario
    if scenario not in names:
        raise KeyError(f"{task} is scored on {list(names)}, not {scenario!r}")
    if task in scoring_runs.PROTOCOLS:
        env, params = scoring_runs.stream(task, scenario)
        return GymnasiumEnv(env, params, render_mode=render_mode)
    spec = registry.get(task)
    frozen_reference = reference.load(task)
    if charter.status(task) == charter.SCORED and frozen_reference is not None:
        # Reference.params: the scenario with the changes its line runs on.
        changed = reference.changes(frozen_reference)
        params = api.make(task, scenario=scenario, **changed)[1]
    else:
        suite = {s.name: s for s in scenarios.suite(spec)}
        params = scenarios.params_for(spec, suite[scenario])
    return GymnasiumEnv(spec.make_env(), params, render_mode=render_mode)


def register(root: pathlib.Path = MANIFESTS) -> list[str]:
    """Register every frozen task with Gymnasium; returns the IDs. An ID
    already registered is left as it is."""
    tasks = frozen(root)
    if not tasks:
        return []
    import gymnasium

    for task in tasks:
        if gym_id(task) not in gymnasium.registry:
            gymnasium.register(
                id=gym_id(task),
                entry_point=make,
                kwargs={"task": task, "root": str(root)},
            )
    return [gym_id(t) for t in tasks]
