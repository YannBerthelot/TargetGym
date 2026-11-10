"""The cement kiln as a continuing plant (long-strata draft 4, S1).

Contract:
- a window draws its free-lime target once; a stream changes it as events: a
  change started now (``start_event``) draws a new target within
  ``target_lime_range``, observed from the next step and held until the next;
- the changes keep the target's long-run law: each is a fresh draw over the
  whole range, not one value held or a walk.

Oracles: the declared target range, and a uniform draw's mean and spread.
"""

from __future__ import annotations

import jax
import numpy as np

from target_gym.stream import make_stream

STREAM = make_stream("cement_kiln")
ENV = STREAM.env
PARAMS = STREAM.default_params


def test_a_target_change_draws_a_new_target_in_range_and_holds_it():
    _, s = ENV.reset_env(jax.random.PRNGKey(0), PARAMS)
    before = float(s.target_lime)
    s = ENV.start_event(s, jax.random.PRNGKey(1), PARAMS)
    lo, hi = PARAMS.target_lime_range
    new = float(s.target_lime)
    assert new != before and lo <= new <= hi
    step = jax.jit(ENV.step_env)
    action = ENV.action_space(PARAMS).low * 0.0
    for k in range(3):
        obs, s, *_ = step(jax.random.PRNGKey(k), s, action, PARAMS)
        np.testing.assert_allclose(float(obs[7]), 100.0 * new, rtol=1e-6)


def test_target_changes_are_fresh_draws_over_the_whole_range():
    _, s = ENV.reset_env(jax.random.PRNGKey(0), PARAMS)
    keys = jax.random.split(jax.random.PRNGKey(2), 400)
    targets = np.asarray(
        jax.vmap(lambda k: ENV.start_event(s, k, PARAMS).target_lime)(keys)
    )
    lo, hi = PARAMS.target_lime_range
    assert targets.min() >= lo and targets.max() <= hi
    # Uniform over the range: mean at its middle within 4 standard errors, and
    # both ends reached within a twentieth of the range.
    se = (hi - lo) / np.sqrt(12.0 * len(targets))
    assert abs(targets.mean() - 0.5 * (lo + hi)) < 4.0 * se
    assert targets.min() < lo + 0.05 * (hi - lo)
    assert targets.max() > hi - 0.05 * (hi - lo)
