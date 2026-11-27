"""Recorded results for learned policies, and the shape they are recorded in.

The library ships a tuned PID and an MPC upper bound for every environment so a
learned policy has something real to beat. This is where the learned side of
that comparison is kept.

It is recorded rather than reproduced, for the same reason the MPC baselines
are: training costs GPU-hours, and the answer only moves when the environment
or the agent moves. The mechanism is the one already used for
``src/target_gym/data/baseline_returns/`` -- each record carries a fingerprint of what
determines it, and a record whose fingerprint no longer matches the tree is
refused rather than believed.

The fingerprint here is :func:`~target_gym.provenance.environment_fingerprint`,
which is narrower than the baselines' one: a tabula-rasa agent never runs a PID,
so re-tuning a controller must not throw away its training run, while a change
to the reward or the integrator must.

An expert-based agent does run the PID, in training or at deployment, so what it
learned depends on the gains. ``record_result(..., expert_based=True)`` stamps
:func:`~target_gym.provenance.expert_fingerprint` beside the environment's, and
a retune then marks that record stale. Records without the field, including any
written before it existed, are judged on the environment alone.

Writing results
---------------
Training happens outside this package -- the dependency runs one way, from the
RL library to here, so that installing TargetGym never drags in an RL framework.
A runner calls :func:`record_result` with what it measured and this module
stamps the provenance:

    from target_gym.rl_results import record_result

    record_result(
        env="plane",
        algorithm="SAC",
        library="ajax",
        library_version=ajax.__version__,
        returns=[...],          # one episode return per seed
        episode_steps=280,      # plane's scored episode
        total_timesteps=1_000_000,
        config={"n_envs": 64, "lr": 3e-4, ...},
    )

A result on the charter's protocol (docs/rl-protocol.md, section 4) also passes
``costs``, each scenario's cost per step on each of ``seeds`` (the scoring seeds),
one row per training run. It is stamped with the charter's fingerprints of the
reference it is read against (``charter.current``: the env version and the anchor
part's fingerprint), so a re-record of the reference marks it stale too, and
``scoring.headline`` reads it against the record's anchor line.

``config`` is free-form and is stored verbatim. It is not fingerprinted: two
runs of the same agent at different learning rates are different *results*, not
stale ones, and both are worth keeping. What the fingerprint protects is the
claim that a number describes *this* environment.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Iterable

from target_gym.provenance import (
    _ROOT,
    environment_fingerprint,
    expert_fingerprint,
)

RL_RESULTS_PATH = _ROOT / "data" / "rl_results.json"

#: Fields every record must carry. Kept explicit so a malformed record fails
#: when it is written rather than when someone reads it into a table.
REQUIRED_FIELDS = (
    "env",
    "env_version",
    "algorithm",
    "library",
    "env_fingerprint",
    "returns",
    "episode_steps",
    "total_timesteps",
    "recorded",
)


def result_key(env: str, algorithm: str, tag: str | None = None) -> str:
    """One record per environment, algorithm and optional variant tag."""
    return f"{env}/{algorithm}" + (f"/{tag}" if tag else "")


def load_results() -> dict:
    """Every recorded result, or an empty mapping if none exist yet."""
    if not RL_RESULTS_PATH.exists():
        return {}
    return json.loads(RL_RESULTS_PATH.read_text())


def record_result(
    env: str,
    algorithm: str,
    library: str,
    returns: Iterable[float],
    episode_steps: int,
    total_timesteps: int,
    config: dict[str, Any] | None = None,
    library_version: str | None = None,
    tag: str | None = None,
    notes: str | None = None,
    expert_based: bool = False,
    costs: dict[str, list[list[float]]] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Stamp a measured result with its provenance and store it.

    ``returns`` is one undiscounted episode return per seed, measured the same
    way the shipped baselines are, so the numbers sit in the same column as the
    PID's and the MPC's. Anything else -- a training curve, a normalised score --
    is not comparable to them and does not belong here.

    ``expert_based`` says the agent ran the expert, in training or at
    deployment, and stamps ``expert_fingerprint`` so a retune marks the record
    stale. A driver must set it: left at False, the retune goes unnoticed.

    ``costs`` and ``seeds``, the charter's protocol: every scenario of the task,
    each run's cost on each seed. ``ValueError`` for another scenario set or a
    run without a cost per seed.
    """
    from target_gym.registry import REGISTRY

    if env not in REGISTRY:
        raise KeyError(f"unknown environment {env!r}")
    values = [float(v) for v in returns]
    if not values:
        raise ValueError(f"{env}/{algorithm}: no returns given")

    record = {
        "env": env,
        # The citable identity, alongside the registry key. A published result
        # belongs to a *version* of an environment: the key says which entry it
        # came from, this says what that entry meant when the number was
        # measured, and the fingerprint below is what proves the two agree.
        "env_version": REGISTRY[env].versioned_name,
        "algorithm": algorithm,
        "library": library,
        "library_version": library_version,
        "env_fingerprint": environment_fingerprint(REGISTRY[env]),
        "expert_fingerprint": (
            expert_fingerprint(REGISTRY[env]) if expert_based else None
        ),
        "n_seeds": len(values),
        "returns": [round(v, 6) for v in values],
        "episode_steps": int(episode_steps),
        "total_timesteps": int(total_timesteps),
        "config": config or {},
        "notes": notes,
        "recorded": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }

    if costs is not None:
        record |= _charter_costs(env, costs, seeds)

    results = load_results()
    results[result_key(env, algorithm, tag)] = record
    results["_meta"] = {
        "note": (
            "Learned-policy results, recorded outside this package and checked "
            "here. Regenerate a stale entry by re-running its training; see "
            "docs/rl-baselines.md."
        ),
        "updated": record["recorded"],
    }
    RL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RL_RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return record


def _charter_stamp(env: str) -> dict:
    from target_gym import charter

    now = charter.current(env)
    return {"version": now["version"], "anchor": now["fingerprints"]["anchor"]}


def _charter_costs(env: str, costs: dict, seeds) -> dict:
    """The protocol's fields of a record, checked against the task's scenarios."""
    from target_gym import scenarios, scoring_runs
    from target_gym.registry import REGISTRY

    if env in scoring_runs.PROTOCOLS:
        names = set(scoring_runs.weights(env))
    else:
        names = set(scenarios.weights(REGISTRY[env]))
    seeds = [int(s) for s in seeds or ()]
    if set(costs) != names:
        raise ValueError(f"{env}: costs cover {sorted(costs)}, not {sorted(names)}")
    rows = {s: [[float(c) for c in run] for run in v] for s, v in costs.items()}
    if not seeds or any(len(r) != len(seeds) for v in rows.values() for r in v):
        raise ValueError(f"{env}: every run needs one cost per seed of {seeds}")
    return {"costs": rows, "seeds": seeds, "charter": _charter_stamp(env)}


def is_current(record: dict) -> bool:
    """Whether a record still describes the environment as it stands now, if its
    agent ran the expert, that expert, and, on the charter's protocol, the
    reference it is read against."""
    from target_gym.registry import REGISTRY

    spec = REGISTRY.get(record["env"])
    if spec is None or record["env_fingerprint"] != environment_fingerprint(spec):
        return False
    expert = record.get("expert_fingerprint")
    if expert is not None and expert != expert_fingerprint(spec):
        return False
    stamp = record.get("charter")
    return stamp is None or stamp == _charter_stamp(record["env"])
