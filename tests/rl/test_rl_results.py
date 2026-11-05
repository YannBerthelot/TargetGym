"""The recorded learned-policy results, and the contract they are stored under.

Nothing here asserts that RL beats a PID. Whether it does is the question the
library exists to ask, and a test that presumed the answer would be worthless.
What is asserted is that a published number is well formed and still describes
the environment it claims to.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from target_gym import provenance, rl_results
from target_gym.provenance import environment_fingerprint
from target_gym.registry import REGISTRY

RECORDS = {k: v for k, v in rl_results.load_results().items() if k != "_meta"}


def test_result_keys_name_a_registered_environment():
    unknown = {
        k: r.get("env") for k, r in RECORDS.items() if r.get("env") not in REGISTRY
    }
    assert (
        not unknown
    ), f"results recorded for environments that do not exist: {unknown}"


@pytest.mark.skipif(not RECORDS, reason="no learned-policy results recorded yet")
@pytest.mark.parametrize("key", sorted(RECORDS))
def test_record_is_well_formed(key):
    record = RECORDS[key]
    missing = [f for f in rl_results.REQUIRED_FIELDS if f not in record]
    assert not missing, f"{key}: missing {missing}"

    returns = np.asarray(record["returns"], dtype=float)
    assert returns.size == record["n_seeds"], (
        f"{key}: n_seeds says {record['n_seeds']} but {returns.size} returns are "
        "stored"
    )
    assert np.isfinite(returns).all(), f"{key}: non-finite returns {record['returns']}"
    assert record["episode_steps"] > 0 and record["total_timesteps"] > 0


@pytest.mark.skipif(not RECORDS, reason="no learned-policy results recorded yet")
@pytest.mark.parametrize("key", sorted(RECORDS))
def test_record_still_describes_this_environment(key):
    """A number published against an environment that has since changed is a
    claim about code that no longer exists.

    This library has already invalidated every return-based figure it had once,
    when the reward contract was unified across all eighteen environments. That
    is exactly the event this catches: a training run that cost GPU-hours is
    still worth keeping, but not worth quoting, until it is re-run.
    """
    record = RECORDS[key]
    spec = REGISTRY[record["env"]]
    current = environment_fingerprint(spec)
    assert record["env_fingerprint"] == current, (
        f"{key}: recorded against {record['env_fingerprint']}, environment is now "
        f"{current}. Its dynamics, reward or parameters have changed since this "
        f"was trained, so the number no longer describes this environment. "
        f"Re-train and re-record, or drop the entry."
    )


@pytest.mark.skipif(not RECORDS, reason="no learned-policy results recorded yet")
@pytest.mark.parametrize("key", sorted(RECORDS))
def test_result_is_comparable_to_the_shipped_baselines(key):
    """Returns must be measured over the same episode the baselines use.

    A learned policy scored over a different horizon is not on the same axis as
    the PID and MPC numbers next to it, and putting them in one table would be
    the comparison equivalent of comparing a mean against a median.
    """
    record = RECORDS[key]
    spec = REGISTRY[record["env"]]
    episode = int(spec.make_test_params().max_steps_in_episode)
    assert record["episode_steps"] <= episode, (
        f"{key}: scored over {record['episode_steps']} steps, but the "
        f"environment's own episode is {episode}"
    )


@pytest.mark.skipif(not RECORDS, reason="no learned-policy results recorded yet")
@pytest.mark.parametrize("key", sorted(RECORDS))
def test_record_of_an_expert_based_agent_still_describes_its_expert(key):
    """An agent that ran the expert learned on that expert: a retune leaves
    the number describing a controller that no longer ships."""
    assert rl_results.is_current(RECORDS[key]), (
        f"{key}: its environment or the expert it ran has changed since this was "
        "trained. Re-train and re-record, or drop the entry."
    )


# The staleness contract (src/target_gym/rl_results.py): a record is current
# while the environment it names is, and, if the agent ran the expert, while
# that expert is. Records written before ``expert_fingerprint`` existed carry
# no such field and stay current.
TASK = "first_order"


def _recorded(tmp_path, monkeypatch, **kwargs) -> dict:
    """A result written through ``record_result`` and read back from its file."""
    monkeypatch.setattr(rl_results, "RL_RESULTS_PATH", tmp_path / "rl_results.json")
    rl_results.record_result(
        env=TASK,
        algorithm="SAC",
        library="ajax",
        returns=[-1.0, -2.0],
        episode_steps=150,
        total_timesteps=1_000,
        **kwargs,
    )
    return rl_results.load_results()[rl_results.result_key(TASK, "SAC")]


def _retune(tmp_path, monkeypatch) -> None:
    """Change the task's gains the way ``scripts/tune_pid.py`` would: in the
    gains file, leaving the environment's own fingerprint where it was."""
    gains = json.loads(provenance.GAINS_PATH.read_text())
    gains[TASK] = {"retuned": True}
    path = tmp_path / "pid_gains.json"
    path.write_text(json.dumps(gains))
    monkeypatch.setattr(provenance, "GAINS_PATH", path)


def test_a_retune_marks_an_expert_based_run_stale(tmp_path, monkeypatch):
    record = _recorded(tmp_path, monkeypatch, expert_based=True)
    assert rl_results.is_current(record)
    _retune(tmp_path, monkeypatch)
    assert not rl_results.is_current(record)


def test_a_retune_leaves_a_tabula_rasa_run_current(tmp_path, monkeypatch):
    """It never ran the expert, and training it cost GPU-hours."""
    record = _recorded(tmp_path, monkeypatch)
    _retune(tmp_path, monkeypatch)
    assert rl_results.is_current(record)


def test_a_record_older_than_the_expert_fingerprint_stays_current(
    tmp_path, monkeypatch
):
    record = _recorded(tmp_path, monkeypatch, expert_based=True)
    del record["expert_fingerprint"]
    _retune(tmp_path, monkeypatch)
    assert rl_results.is_current(record)


@pytest.mark.parametrize("expert_based", [False, True])
def test_an_environment_change_marks_any_run_stale(tmp_path, monkeypatch, expert_based):
    record = _recorded(tmp_path, monkeypatch, expert_based=expert_based)
    assert not rl_results.is_current({**record, "env_fingerprint": "older"})
