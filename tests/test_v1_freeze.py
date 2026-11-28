"""Frozen tasks never change (task protocol, item 9).

Contract, from the protocol's item 9 and W5 and W6 of the v1 task pages:

- a frozen task's manifest records its plant, its scoring definition and its
  reference, and the tree matches it until one of them moves; the guard then
  names the part that moved;
- a frozen task is never frozen again in place: a change becomes
  ``<task>-v2`` beside it;
- ``gymnasium.make("TargetGym/<task>-v1")`` finds every frozen task once
  ``target_gym`` is imported, on the episode its score runs, and refuses one
  whose tree moved.

``first_order`` stands in for a frozen task here, frozen into a temporary
folder: it is the cheapest task, and the real manifests are never touched.
"""

import json

import gymnasium
import numpy as np
import pytest

import target_gym as tg
from target_gym import charter, provenance, reference, v1
from target_gym.registry import REGISTRY

FROZEN = v1.frozen()


@pytest.mark.parametrize("task", sorted(FROZEN))
def test_a_frozen_task_still_matches_its_manifest(task):
    moved = v1.changes(task)
    assert not moved, (
        f"{v1.gym_id(task)} is frozen, and its {', '.join(moved)} changed. A "
        f"frozen task never changes: put the change in {task}-v2 beside it "
        "(docs/task-protocol.md, item 9)."
    )


@pytest.mark.parametrize("task", sorted(FROZEN))
def test_a_frozen_task_is_registered_on_import(task):
    assert gymnasium.spec(v1.gym_id(task)).kwargs["task"] == task


@pytest.fixture
def frozen_first_order(tmp_path):
    v1.freeze("first_order", root=tmp_path)
    return tmp_path


def test_a_frozen_task_matches_its_manifest(frozen_first_order):
    assert v1.changes("first_order", frozen_first_order) == []
    stored = json.loads((frozen_first_order / "first_order.json").read_text())
    assert stored["id"] == "TargetGym/first_order-v1"
    assert stored["stamp"] == REGISTRY["first_order"].versioned_name


def test_a_parameter_that_moves_moves_the_plant(frozen_first_order, monkeypatch):
    """The scored lag goes from 0.5 s to 0.6 s."""
    spec_type = type(REGISTRY["first_order"])
    scored = spec_type.make_test_params

    def slower(self, **overrides):
        return scored(self, **{"tau": 0.6, **overrides})

    monkeypatch.setattr(spec_type, "make_test_params", slower)
    assert "plant" in v1.changes("first_order", frozen_first_order)


def test_new_scoring_seeds_move_the_scoring_alone(frozen_first_order, monkeypatch):
    monkeypatch.setattr(charter, "ANCHOR_SEEDS", tuple(range(200, 211)))
    assert v1.changes("first_order", frozen_first_order) == ["scoring"]


def test_retuned_gains_move_the_reference_alone(
    frozen_first_order, tmp_path_factory, monkeypatch
):
    gains = json.loads(provenance.GAINS_PATH.read_text())
    gains["first_order"]["Kp"] *= 1.1
    path = tmp_path_factory.mktemp("gains") / "pid_gains.json"
    path.write_text(json.dumps(gains))
    monkeypatch.setattr(provenance, "GAINS_PATH", path)
    assert v1.changes("first_order", frozen_first_order) == ["reference"]


def test_a_frozen_task_is_never_frozen_again(frozen_first_order):
    with pytest.raises(FileExistsError, match="first_order-v2"):
        v1.freeze("first_order", root=frozen_first_order)


@pytest.fixture
def registered(frozen_first_order):
    ids = v1.register(frozen_first_order)
    yield frozen_first_order
    for i in ids:
        gymnasium.registry.pop(i, None)


def test_gymnasium_makes_a_frozen_task_on_its_scored_episode(registered):
    env = gymnasium.make("TargetGym/first_order-v1")
    _, scored = tg.make("first_order")
    assert vars(env.unwrapped.env_params) == vars(scored)
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and np.isfinite(reward)


def test_gymnasium_refuses_a_frozen_task_whose_tree_moved(registered, monkeypatch):
    monkeypatch.setattr(charter, "ANCHOR_SEEDS", tuple(range(200, 211)))
    with pytest.raises(v1.FrozenTaskChanged, match="scoring"):
        gymnasium.make("TargetGym/first_order-v1")


def test_a_task_whose_reference_is_stale_is_not_frozen(tmp_path, monkeypatch):
    """hvac's reference is frozen on its plant's stamp; with the comfort band's
    floor moved from 21 C to 20.5 C, it no longer describes the plant."""
    spec_type = type(REGISTRY["hvac"])
    scored = spec_type.make_test_params

    def moved(self, **overrides):
        if self.name == "hvac":
            overrides = {"band_low": 20.5, **overrides}
        return scored(self, **overrides)

    monkeypatch.setattr(spec_type, "make_test_params", moved)
    with pytest.raises(reference.StaleReference):
        v1.freeze("hvac", root=tmp_path)


@pytest.mark.medium
@pytest.mark.parametrize("task", charter.FLAGSHIP)
def test_a_scored_task_is_made_on_the_episode_its_score_runs(task, tmp_path):
    """With a manifest as ``freeze`` writes it, each scored task is made on its
    reference's configuration of its heaviest scenario (``Reference.params``),
    or on its scored run, and passes Gymnasium's own checker."""
    from gymnasium.utils.env_checker import check_env

    from target_gym import scoring_runs

    (tmp_path / f"{task}.json").write_text(json.dumps(v1.manifest(task)))
    env = v1.make(task, root=tmp_path)
    heaviest = next(iter(v1.scenario_names(task)))
    if task in scoring_runs.PROTOCOLS:
        steps = scoring_runs.scored_steps(task)
        assert env.env_params.max_steps_in_episode == steps
    else:
        changed = reference.changes(reference.load(task))
        expected = tg.make(task, scenario=heaviest, **changed)[1]
        assert vars(env.env_params) == vars(expected)
    check_env(env, skip_render_check=True)
