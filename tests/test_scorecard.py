"""The scoring runner: every seed at once, compiled once (outside review L2).

Contract: for a policy in the experts' form (``memory = policy.reset(n)``, then
``action, memory = policy.step(memory, obs)``, pure), ``scorecard.rollouts`` runs
the episodes ``benchmark.run_policy`` runs on the same seeds: the same reset and
step keys, the same per-seed parameters (the compressor places its machine by
the seed), and an episode frozen once it terminates. It returns what each step
gave: the reward, whether the plant tripped, and the store's value on a plant
that stores what it buys (hvac), which an evaluation's edge charge reads.
``scorecard.episode_costs`` reads them as ``evaluation.episode_cost`` reads an
episode. The oracle is ``run_policy`` stepping the same controller one jitted
step at a time. Compiled as one program, XLA orders some of the plant's float32
arithmetic differently (flanders, 2026-11-27): a step's reward differed by up to
3e-5 of its size on compressor_surge, whose surge dynamics are the stiffest, and
by far less elsewhere; the scored costs of the four frozen references, over full
episodes on the scoring seeds, by at most 3.5e-6. Hence 1e-4 on a step's reward
and 1e-5 on an episode's cost.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym import evaluation, scorecard
from target_gym.benchmark import episodes_of, run_policy, task_info
from target_gym.registry import REGISTRY


class Stepped:
    """``expert`` stepped by ``run_policy``, one jitted step at a time."""

    name, oracle = "expert", False

    def __init__(self, expert):
        self.expert = expert
        self._step = jax.jit(jax.vmap(expert.step))

    def reset(self, key, obs, task):
        return self.expert.reset(task.n_episodes)

    def act(self, carry, obs, t):
        action, carry = self._step(carry, jnp.asarray(obs))
        return np.asarray(action, float), carry


def _both(name, steps, seeds):
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.make_test_params(max_steps_in_episode=steps)
    expert = tg.make_expert(name)
    got = scorecard.rollouts(spec, params, expert, seeds)
    task = task_info(name, env, params, len(seeds))
    want = run_policy(Stepped(expert), env, params, task, seeds=list(seeds))
    return spec, got, want


@pytest.mark.parametrize("name", ["compressor_surge", "cstr"])
def test_the_compiled_episodes_are_the_ones_run_policy_runs(name):
    _, got, want = _both(name, 40, (200, 201, 202))
    np.testing.assert_allclose(got["reward"], want["reward"], rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(
        got["reward"].mean(axis=1), want["reward"].mean(axis=1), rtol=1e-5
    )
    np.testing.assert_array_equal(got["tripped"], want["tripped"])
    np.testing.assert_array_equal(got["length"], want["length"])


def test_a_plants_store_is_read_as_the_evaluation_reads_it():
    # hvac's evaluation scores days 4 to 10 and charges the store's change across
    # them once: 1152 steps of 15 min, the first 288 unscored.
    spec, got, want = _both("hvac", 1152, (200, 201))
    stored = want["records"]["stored_value"]
    np.testing.assert_allclose(got["stored_value"], stored, rtol=1e-5)
    ours = scorecard.episode_costs(spec, got)
    theirs = [evaluation.episode_cost(spec, e) for e in episodes_of(want)]
    np.testing.assert_allclose(ours, theirs, rtol=1e-5)


# ``tg.score`` (charter J4; outside review C1, owner's pickers 2026-11-27). Contract:
# the saving is ``1 - J_policy / J_reference`` with ``J = sum_s w_s mean(cost |
# s)`` over the scenarios' shares of operating time, the policy and the reference
# run on the same seeds and worlds, each cost read over the scored window; one row
# per scenario beside the headline; the record's lines as context; whether the
# policy read the state. A task outside the scored suite is refused.

SEEDS = (200, 201)


@pytest.fixture(scope="module")
def expert_on_hvac():
    return tg.score("hvac", tg.make_expert("hvac"), seeds=SEEDS)


@pytest.mark.medium
def test_the_reference_against_itself_saves_nothing_and_ties_every_seed():
    ref = tg.make_reference("hvac")
    out = tg.score("hvac", ref, seeds=SEEDS)
    assert out["saving"] == 0.0
    assert out["ties"] == pytest.approx(1.0)
    assert out["probability_of_improvement"] == 0.0
    assert all(row["J_policy"] == row["J_reference"] for row in out["scenarios"])


@pytest.mark.medium
def test_the_scenario_table_adds_up_to_the_headline(expert_on_hvac):
    out = expert_on_hvac
    rows = out["scenarios"]
    assert [r["scenario"] for r in rows] == list(tg.list_scenarios("hvac"))
    assert sum(r["weight"] for r in rows) == pytest.approx(1.0)
    j_policy = sum(r["weight"] * r["J_policy"] for r in rows)
    j_reference = sum(r["weight"] * r["J_reference"] for r in rows)
    assert out["saving"] == pytest.approx(1.0 - j_policy / j_reference, rel=1e-9)
    for r in rows:
        assert r["saving"] == pytest.approx(1.0 - r["J_policy"] / r["J_reference"])
    assert out["seeds"] == list(SEEDS)
    assert out["read_state"] is False
    assert out["reference"]["rung"] == tg.make_reference("hvac").rung
    assert set(out["lines"]) >= {"current", "J", "room"}


@pytest.mark.medium
def test_a_pure_policy_costs_the_same_compiled_and_on_the_host():
    def hold(obs):  # both of pH's valves at mid-stroke
        return jnp.zeros(2, jnp.float32)

    compiled = tg.score("ph_neutralization", hold, seeds=(200,))
    host = tg.score("ph_neutralization", hold, seeds=(200,), jit=False)
    for a, b in zip(compiled["scenarios"], host["scenarios"]):
        assert a["J_policy"] == pytest.approx(b["J_policy"], rel=1e-5), a["scenario"]


@pytest.mark.medium
def test_a_full_state_policy_is_scored_and_said_to_read_the_state():
    probe = tg.FullState(lambda obs, state: np.zeros(2))
    out = tg.score("ph_neutralization", probe, seeds=(200,))
    assert out["read_state"] is True


def test_a_task_outside_the_scored_suite_is_refused():
    with pytest.raises(ValueError, match="not scored"):
        tg.score("plane", lambda obs: obs[:1])
