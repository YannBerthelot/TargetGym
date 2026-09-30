"""Recorded results for learned policies, and the shape they are recorded in.

The library ships a tuned PID and an MPC upper bound for every environment so a
learned policy has something real to beat. This is where the learned side of
that comparison is kept.

It is recorded rather than reproduced, for the same reason the MPC baselines
are: training costs GPU-hours, and the answer only moves when the environment
or the agent moves. The mechanism is the one already used for
``src/target_gym/data/baseline_returns.json`` -- each record carries a fingerprint of what
determines it, and a record whose fingerprint no longer matches the tree is
refused rather than believed.

The fingerprint here is :func:`~target_gym.provenance.environment_fingerprint`,
which is narrower than the baselines' one: an agent never runs a PID, so
re-tuning a controller must not throw away a training run, while a change to the
reward or the integrator must.

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
        episode_steps=600,
        total_timesteps=1_000_000,
        config={"n_envs": 64, "lr": 3e-4, ...},
    )

``config`` is free-form and is stored verbatim. It is not fingerprinted: two
runs of the same agent at different learning rates are different *results*, not
stale ones, and both are worth keeping. What the fingerprint protects is the
claim that a number describes *this* environment.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Iterable

from target_gym.provenance import _ROOT, environment_fingerprint

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
) -> dict:
    """Stamp a measured result with its provenance and store it.

    ``returns`` is one undiscounted episode return per seed, measured the same
    way the shipped baselines are, so the numbers sit in the same column as the
    PID's and the MPC's. Anything else -- a training curve, a normalised score --
    is not comparable to them and does not belong here.
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
        "n_seeds": len(values),
        "returns": [round(v, 6) for v in values],
        "episode_steps": int(episode_steps),
        "total_timesteps": int(total_timesteps),
        "config": config or {},
        "notes": notes,
        "recorded": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }

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


def is_current(record: dict) -> bool:
    """Whether a record still describes the environment as it stands now."""
    from target_gym.registry import REGISTRY

    spec = REGISTRY.get(record["env"])
    if spec is None:
        return False
    return record["env_fingerprint"] == environment_fingerprint(spec)
