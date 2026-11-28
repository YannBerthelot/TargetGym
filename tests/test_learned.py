"""The learned baselines' runner (task protocol, item 4).

Contract, from docs/rl-protocol.md section 2 and the brief of 2026-11-28: a
learner trains on the measured reward, the one a plant computes from its
sensors, while the score reads the true one; PPO and SAC train 30 runs at once
at one budget, and each run's policy is scored by ``tg.score`` as one of a list,
whose interval then covers the runs.
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym import learned


def test_a_learner_trains_on_the_measured_reward():
    env, params = tg.make("ph_neutralization")
    wrapped = learned.MeasuredReward(env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = env.initial_action(state, params)
    *_, info = env.step_env(key, state, action, params)
    _, _, reward, _, _ = wrapped.step_env(key, state, action, params)
    assert float(reward) == float(info["measured_reward"])
    assert wrapped.action_space(params).shape == env.action_space(params).shape


@pytest.mark.learning
def test_trained_runs_are_scored_as_a_list(monkeypatch, tmp_path):
    """Two PPO runs of two iterations each, scored against pH's shipped expert
    frozen on today's plant: one saving per run, and the training's wall-clock."""
    pytest.importorskip("ajax")
    from target_gym import provenance, reference
    from target_gym.registry import REGISTRY

    spec = REGISTRY["ph_neutralization"]
    stand_in = {"task": "ph_neutralization", "kind": "expert", "rung": "expert"}
    stand_in |= {"params": {}, "format": 1}
    stand_in["stamp"] = provenance.environment_fingerprint(spec)
    stand_in["expert"] = provenance.expert_fingerprint(spec)
    reference.write("ph_neutralization", stand_in, tmp_path)
    build, load = reference.build, reference.load
    monkeypatch.setattr(reference, "build", lambda t, root=tmp_path: build(t, root))
    monkeypatch.setattr(reference, "load", lambda t, root=tmp_path: load(t, root))

    policies, run = learned.train("ph_neutralization", "PPO", runs=2, steps=2048)
    result = tg.score("ph_neutralization", policies, seeds=(200, 201))
    assert len(result["run_savings"]) == 2 and np.isfinite(result["saving"])
    assert run["runs"] == 2 and run["training_seconds"] > 0
