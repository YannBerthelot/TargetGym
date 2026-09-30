"""The recorded learned-policy results, and the contract they are stored under.

Nothing here asserts that RL beats a PID. Whether it does is the question the
library exists to ask, and a test that presumed the answer would be worthless.
What is asserted is that a published number is well formed and still describes
the environment it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest

from target_gym import rl_results
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
