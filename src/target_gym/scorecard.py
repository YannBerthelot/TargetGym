"""The charter's comparison as one call (charter J4; outside review C1 and L2).

:func:`rollouts` runs a policy in the experts' form on every seed of a task at
once, compiled once: the episodes ``benchmark.run_policy`` runs one step at a
time, much faster. :func:`episode_costs` reads each as the task's evaluation
does, over its scored window and with its edge charge.

Kept out of every provenance fingerprint, like ``scoring_runs``: it decides how a
task is scored, not what the task is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import evaluation
from target_gym.benchmark import _per_seed, episode_keys

__all__ = ["BASELOAD", "episode_costs", "rollouts", "score"]


def rollouts(spec, params, policy, seeds, n_steps: int | None = None) -> dict:
    """``policy``'s episodes on ``seeds`` of ``spec`` at ``params``, every seed at
    once in one compiled program. ``policy`` has the experts' form:
    ``memory = policy.reset(n)``, then ``action, memory = policy.step(memory,
    obs)`` for one episode, pure, so it is vmapped over the seeds.

    Episode ``i`` is the one ``benchmark.run_policy`` runs on ``seeds[i]``: its
    reset and step keys, its parameters (``benchmark.seed_params``), and its plant
    frozen once it terminates. Returns per-episode, per-step ``reward``,
    ``tripped`` and, for a plant that stores what it buys, ``stored_value``, each
    ``(n, T)``; ``actions``, what the actuators did as fractions of their span,
    ``(n, T, n_act)``; and each episode's ``length``."""
    env = spec.make_env()
    n = len(seeds)
    T = int(params.max_steps_in_episode) if n_steps is None else int(n_steps)
    per_seed = hasattr(env, "params_for_seed")
    p = _per_seed(env, params, seeds, False) if per_seed else params
    pa = 0 if per_seed else None
    pairs = [episode_keys(jax.random.PRNGKey(int(s)), 1, T) for s in seeds]
    reset_keys = jnp.concatenate([r for r, _ in pairs])
    step_keys = jnp.concatenate([k for _, k in pairs], axis=1)
    stored = getattr(env, "stored_value", None)
    n_act = env.action_space(params).shape[0]

    def run(reset_keys, step_keys, p, memory):
        obs, state = jax.vmap(env.reset_env, in_axes=(0, pa))(reset_keys, p)
        step = jax.vmap(env.step_env, in_axes=(0, 0, 0, pa))
        act = jax.vmap(policy.step)

        def body(carry, keys):
            obs, state, memory, done = carry
            action, memory = act(memory, obs)
            action = jnp.reshape(jnp.asarray(action, jnp.float32), (n, n_act))
            obs_n, state_n, r, term, info = step(keys, state, action, p)
            live = ~done

            def freeze(new, old):
                shape = live.shape + (1,) * (new.ndim - 1)
                return jnp.where(jnp.reshape(live, shape), new, old)

            obs_n = freeze(obs_n, obs)
            state_n = jax.tree.map(freeze, state_n, state)
            tripped = info.get("tripped") if isinstance(info, dict) else None
            tripped = jnp.zeros(n, bool) if tripped is None else tripped.astype(bool)
            applied = info.get("applied_action") if isinstance(info, dict) else None
            applied = action if applied is None else jnp.reshape(applied, (n, n_act))
            out = {
                "reward": jnp.where(done, 0.0, r),
                "tripped": tripped & live,
                "terminated": done | term,
                "applied": applied,
            }
            if stored is not None:
                out["stored_value"] = jax.vmap(stored, in_axes=(0, pa))(state_n, p)
            return (obs_n, state_n, memory, done | term), out

        start = (obs, state, memory, jnp.zeros(n, bool))
        _, out = jax.lax.scan(body, start, step_keys)
        return out

    out = jax.jit(run)(reset_keys, step_keys, p, policy.reset(n))
    out = {k: np.swapaxes(np.asarray(v), 0, 1) for k, v in out.items()}
    space = env.action_space(params)
    low = np.broadcast_to(np.asarray(space.low, float), (n_act,))
    span = np.broadcast_to(np.asarray(space.high, float), (n_act,)) - low
    out["actions"] = np.clip((out.pop("applied") - low) / span, 0.0, 1.0)
    ended = out.pop("terminated")
    first = np.where(ended.any(axis=1), ended.argmax(axis=1) + 1, T)
    out["reward"] = out["reward"].astype(float)
    out["length"] = np.minimum(first, T).astype(int)
    return out


def episode_costs(spec, out: dict) -> np.ndarray:
    """Each episode of :func:`rollouts` as ``evaluation.episode_cost`` reads it:
    its mean cost per step over the scored window, with the edge charge."""
    from target_gym.eval import Episode

    costs = []
    for i, L in enumerate(np.asarray(out["length"], int)):
        terms = (
            {"stored_value": out["stored_value"][i, :L]}
            if "stored_value" in out
            else {}
        )
        episode = Episode(
            cost=-out["reward"][i, :L],
            in_band=np.zeros(L, bool),
            target_change=np.zeros(L, bool),
            failed=out["tripped"][i, :L],
            terms=terms,
        )
        costs.append(evaluation.episode_cost(spec, episode))
    return np.asarray(costs, float)


# --- the charter's comparison -------------------------------------------------

#: The scenario a load-following plant spends holding full power, which J4 reads
#: apart from its load-following days (owner, 2026-11-27).
BASELOAD = "baseload"


class _Pure:
    """A pure ``policy(obs) -> action`` in the experts' form, with no memory."""

    def __init__(self, f):
        self.f = f

    def reset(self, n_envs=None):
        return jnp.zeros(() if n_envs is None else (n_envs,), jnp.float32)

    def step(self, memory, obs):
        return self.f(obs), memory


def _form(policy, jit: bool):
    """How ``policy`` runs: compiled (the experts' ``reset``/``step`` form, or a
    pure callable), or on the host, one episode at a time (a ``FullState``, a
    callable that keeps its own memory, or ``jit=False``)."""
    from target_gym.api import FullState

    if isinstance(policy, FullState):
        return "host", policy
    if callable(getattr(policy, "step", None)) and callable(
        getattr(policy, "reset", None)
    ):
        return "compiled", policy
    if not callable(policy):
        raise TypeError(f"{policy!r} is not a policy")
    if not jit or callable(getattr(policy, "reset", None)):
        return "host", policy
    return "compiled", _Pure(policy)


def _scenario_run(name, spec, scenario, params, form, policy, seeds, protocol):
    """``{"costs", "trips", "travel", "reversals"}`` per seed of ``policy`` in one
    scenario, each cost as the charter reads it."""
    from target_gym import eval as E
    from target_gym import scoring_runs

    if protocol:
        if form == "compiled":
            act = policy.step
            out = scoring_runs.run_functional(
                name, scenario, act, policy.reset(), seeds
            )
            return {"costs": out["costs"], "trips": out["trips"]}
        from target_gym.benchmark import CallablePolicy

        costs, trips = [], []
        for seed in seeds:  # one at a time: the policy keeps its own memory
            from target_gym.api import as_policy
            from target_gym.runners.runners import reads_state

            own = as_policy(policy)
            if callable(getattr(own, "reset", None)):
                own.reset()
            call = own if reads_state(own) else (lambda o, s, own=own: own(o))
            adapter = CallablePolicy(
                lambda _t, call=call: call, name="policy", oracle=reads_state(own)
            )
            out = scoring_runs.run(name, scenario, adapter, [seed])
            costs += out["costs"]
            trips += out["trips"]
        return {"costs": costs, "trips": trips}
    if form == "compiled":
        out = rollouts(spec, params, policy, seeds)
        actions = [out["actions"][i, :L] for i, L in enumerate(out["length"])]
        return {
            "costs": [float(c) for c in episode_costs(spec, out)],
            "trips": [int(np.any(t)) for t in out["tripped"]],
            "travel": [E.travel(a) for a in actions],
            "reversals": [E.reversals(a) for a in actions],
        }
    from target_gym.api import as_policy

    episodes = []
    for seed in seeds:
        own = as_policy(policy)
        if callable(getattr(own, "reset", None)):
            own.reset()
        episodes.append(E.run_episode(spec, params, own, seed=int(seed)))
    return {
        "costs": [evaluation.episode_cost(spec, e) for e in episodes],
        "trips": [int(np.any(e.failed)) for e in episodes],
        "travel": [E.travel(e.actions) for e in episodes],
        "reversals": [E.reversals(e.actions) for e in episodes],
    }


def _lines(name: str, spec, weights: dict) -> dict:
    """J3's lines from the task's charter record, each ``J``, its room, and
    whether the record was measured on this plant."""
    from target_gym import charter
    from target_gym.provenance import environment_fingerprint

    record = charter.load(name)
    if record is None:
        return {"current": False, "J": {}, "room": charter.NOT_MEASURED}
    lines = charter.j3(record)
    J = {k: v["J"] if isinstance(v, dict) else v for k, v in lines.items()}
    best = lines["best_null"]
    # A record of another suite (a plant since changed) has no room on this one.
    same = all(
        isinstance(lines[k], dict) and set(lines[k]["scenarios"]) == set(weights)
        for k in ("reference", "ceiling")
    )
    room = (
        charter.room(lines["reference"], lines["ceiling"], weights)
        if same
        else charter.NOT_MEASURED
    )
    stamp = environment_fingerprint(spec)
    return {
        "current": record.get("version", {}).get("stamp") == stamp,
        "J": J,
        "best_null": best["name"] if isinstance(best, dict) else best,
        "room": room["J"] if isinstance(room, dict) else room,
        "room_by_scenario": (
            {s: v["room"] for s, v in room["scenarios"].items()}
            if isinstance(room, dict)
            else {}
        ),
    }


def _part(costs: dict, names, weights: dict, tie: float) -> dict:
    """The headline on ``names`` alone, their weights renormalized."""
    from target_gym.scoring import headline

    head = headline({s: costs[s] for s in names}, {s: weights[s] for s in names}, tie)
    return {k: head[k] for k in ("saving", "saving_ci", "probability_of_improvement")}


def _by_scenario(policy, weights) -> dict[str, list]:
    """Each scenario's runs: one policy or a list for every scenario, or a
    mapping naming each scenario with its own."""
    if not isinstance(policy, dict):
        policy = dict.fromkeys(weights, policy)
    if set(policy) != set(weights):
        raise ValueError(
            f"policies for {sorted(policy)}; the task has {sorted(weights)}"
        )
    out = {
        s: list(p) if isinstance(p, (list, tuple)) else [p] for s, p in policy.items()
    }
    if len({len(r) for r in out.values()}) != 1:
        raise ValueError("every scenario needs the same number of runs")
    return out


def score(name: str, policy, seeds=None, jit: bool = True) -> dict:
    """The charter's comparison of ``policy`` with scored task ``name``'s
    reference (charter J4): ``scoring.headline`` over every scenario, weighted by
    its share of operating time, on the scoring seeds, each episode read over the
    task's scored window with its edge charge; the reactor on its pinned 7-day
    streams. The reference runs beside the policy on the same seeds and worlds.

    ``policy`` is ``policy(obs) -> action``, compiled with JAX, or an object in
    the experts' form (``reset``, ``step(memory, obs)``), compiled too. A policy
    that keeps its own Python memory (it has ``reset`` but no ``step``), one that
    cannot be compiled (``jit=False``) or a ``FullState`` runs on the host, one
    episode at a time. A list of policies is one per training run; the interval
    then covers the runs too. A mapping gives each scenario its own policy, or
    its own list, for a controller built on each scenario's model such as an
    MPC; it names every scenario, with the same number of runs in each.

    ``seeds`` defaults to the scoring seeds (``charter.ANCHOR_SEEDS``), read
    once for a final result; choose checkpoints and settings on
    ``charter.VALIDATION_SEEDS`` instead.

    Returns the headline's keys, then ``scenarios``, one row per scenario
    (its weight, both ``J``, the saving, the trips and the room), ``lines``
    (the charter record's reference, best null, MPC and ceiling ``J``, the room,
    and whether the record was measured on this plant), ``reference`` (its rung,
    and its exits out of sample, on J2's hash seeds), ``read_state`` and, on the
    reactor, ``days``: the saving on baseload days and on load-following days.
    """
    from target_gym import charter, reference, scenarios, scoring_runs
    from target_gym.api import FullState, make_reference
    from target_gym.registry import get
    from target_gym.scoring import headline

    spec = get(name)
    if charter.status(name) != charter.SCORED:
        raise ValueError(
            f"{name} is {charter.status(name)}, not scored: its charter lines are "
            f"not held current. The scored tasks: {list(charter.FLAGSHIP)}"
        )
    ref = make_reference(name)
    seeds = tuple(int(s) for s in (charter.ANCHOR_SEEDS if seeds is None else seeds))
    protocol = name in scoring_runs.PROTOCOLS
    weights = scoring_runs.weights(name) if protocol else scenarios.weights(spec)
    by_scenario = _by_scenario(policy, weights)
    costs, table = {}, []
    for s in weights:
        runs = by_scenario[s]
        params = None if protocol else ref.params(s)
        theirs = _scenario_run(name, spec, s, params, "compiled", ref, seeds, protocol)
        ours = [
            _scenario_run(name, spec, s, params, *_form(p, jit), seeds, protocol)
            for p in runs
        ]
        one = len(runs) == 1
        row = {
            "policy": ours[0]["costs"] if one else [o["costs"] for o in ours],
            "reference": theirs["costs"],
            "policy_trips": ours[0]["trips"] if one else [o["trips"] for o in ours],
            "reference_trips": theirs["trips"],
        }
        for key in ("travel", "reversals"):
            if key in theirs and all(key in o for o in ours):
                row[f"reference_{key}"] = theirs[key]
                row[f"policy_{key}"] = ours[0][key] if one else [o[key] for o in ours]
        costs[s] = row
    tie = charter.tie_tolerance(name)
    head = headline(costs, weights, tie)
    lines = _lines(name, spec, weights)
    for s, w in weights.items():
        table.append(
            {
                "scenario": s,
                "weight": float(w),
                "J_policy": float(np.mean(costs[s]["policy"])),
                "J_reference": float(np.mean(costs[s]["reference"])),
                "saving": head["per_scenario"][s],
                "policy_trips": int(np.sum(costs[s]["policy_trips"])),
                "reference_trips": int(np.sum(costs[s]["reference_trips"])),
                "room": lines.get("room_by_scenario", {}).get(s, charter.NOT_MEASURED),
            }
        )
    frozen = reference.load(name)
    out = {
        "task": name,
        "seeds": list(seeds),
        **head,
        "scenarios": table,
        "lines": lines,
        "reference": {"rung": ref.rung, "exits": frozen.get("exits")},
        "read_state": any(
            isinstance(p, FullState) for runs in by_scenario.values() for p in runs
        ),
    }
    if protocol and BASELOAD in weights:
        follow = [s for s in weights if s != BASELOAD]
        out["days"] = {
            "baseload": _part(costs, [BASELOAD], weights, tie),
            "load_follow": _part(costs, follow, weights, tie),
        }
    return out
