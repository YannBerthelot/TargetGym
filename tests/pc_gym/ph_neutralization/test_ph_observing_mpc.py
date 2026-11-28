"""pH's observation-only MPC (v1 item 5, as the owner read it on 2026-11-28).

Contract:

- The planner is the registry's MPC fed the shared estimator's belief
  (``estimation.ObservationOnly``), which reads the two pH probes, late and
  lagged as they are, by secants, the solve being a bisection.
- What the controller knows exactly it is told: the influent flow is metered
  (obs 6), so the belief's influent is the meter's, and no schedule the reset
  drew moves in its model (``known``).
- The influent's acid strength, which no reading shows and the reset schedules
  unannounced, is estimated as a random walk whose variance over an episode is
  what the scenario's upsets add: two, each uniform within its size, so
  ``2 size^2 / 3`` spread over the episode's steps; none on a day with no upsets.
- The four invariants walk by an allowance for the filter's own error, the same
  on each (the shared estimator's pH test sets its size).

Expected values: the meter's reading, and the walk by hand.
"""

import jax
import numpy as np
import pytest

from target_gym.pc_gym.ph_neutralization.env import PHParams, acid_scale, influent_flow
from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization

P = PHParams()


@pytest.mark.parametrize("step", [0, 700, 1439])
def test_the_belief_reads_the_metered_influent_and_holds_the_acid_still(step):
    from target_gym.pc_gym.ph_neutralization.oracle import observed

    bad = P.replace(acid_upset_size=0.2, flow_swing=0.2)
    env = PHNeutralization()
    obs, s = env.reset_env(jax.random.PRNGKey(4), bad)
    obs = obs.at[6].set(19.3)  # a metered influent, mL/s
    s = s.replace(time=step, upset_levels=s.upset_levels.at[0].set(1.17))
    known = observed(bad)["known"](s, obs)
    assert float(influent_flow(known, bad)) == pytest.approx(19.3, rel=1e-6)
    assert float(acid_scale(known, bad)) == pytest.approx(1.17, rel=1e-6)


@pytest.mark.parametrize(
    "size, want", [(0.0, 0.0), (0.2, np.sqrt(2 * 0.04 / 3 / 1440))]
)
def test_the_acid_walks_by_the_variance_the_scenarios_upsets_add(size, want):
    from target_gym.pc_gym.ph_neutralization.oracle import observed

    walk = observed(P.replace(acid_upset_size=size))["random_walk"]
    assert walk["upset_levels[0]"] == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_the_invariants_carry_one_allowance_for_the_filters_own_error():
    from target_gym.pc_gym.ph_neutralization.oracle import (
        INVARIANT_ALLOWANCE,
        observed,
    )

    walk = observed(P)["random_walk"]
    assert {walk[f] for f in ("Wa", "Wb", "Wa_s1", "Wb_s1")} == {INVARIANT_ALLOWANCE}
