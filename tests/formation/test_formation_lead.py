"""The formation's lead rolls at the parade rate of roll.

Contract (owner's decision of 2026-11-12, formation/PHYSICS.md): a formation
lead rolls into and out of its turns at a smooth, constant rate its wingmen can
match. VT-6's Formation Supplement (CNATRA TW-5, June 2022) gives the
benchmark: "A smooth, continuous roll rate of 3-5 seconds to reach 30 degrees
AOB (or roll level out of turn) is a good benchmark"; the owner chose its fast
end, 30 deg in 3 s, 10 deg/s. So, flying a route of full parade-turn
reversals, the expert's lead never moves its bank demand faster than 10 deg/s,
and it does roll at that rate through a reversal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane.dynamics import BANK_DEMAND_MAX
from target_gym.registry import REGISTRY

PARADE_ROLL = np.deg2rad(30.0) / 3.0  # rad/s [VT-6]
#: The route's fastest legs: the parade turn's 30 deg, both ways
#: (``lead_turn_rate_range``).
REVERSALS = jnp.array([0.0303, -0.0303] * 4)


def _lead_bank_demands(steps=200):
    spec = REGISTRY["formation"]
    env, params = spec.make_env(), spec.make_test_params()
    ctrl = spec.make_pid_controller()

    @jax.jit
    def run():
        _, state = env.reset_env(jax.random.PRNGKey(0), params)
        state = state.replace(lead_turn_schedule=REVERSALS, lead_turn_rate=REVERSALS[0])
        memory = jax.tree.map(jnp.asarray, ctrl.init_state)

        def body(carry, t):
            s, m = carry
            action, m = ctrl.act(m, env.get_obs(s, params))
            key = jax.random.fold_in(jax.random.PRNGKey(1), t)
            _, s, _, _, _ = env.step_env(key, s, action, params)
            return (s, m), action[2] * BANK_DEMAND_MAX

        return jax.lax.scan(body, (state, memory), jnp.arange(steps))[1]

    return np.asarray(run())


@pytest.mark.medium
def test_the_lead_rolls_at_the_parade_rate_through_route_reversals():
    # Whether the wingmen keep their slots through these reversals is the
    # wingman hold's contract, not this one's (formation/PHYSICS.md, findings).
    bank = _lead_bank_demands()
    rates = np.abs(np.diff(bank))  # per 1 s step
    # Never faster than the parade rate (float32 rounding aside) ...
    assert rates.max() <= PARADE_ROLL * (1.0 + 1e-5)
    # ... and at it while reversing: the 60 deg roll takes several steps at
    # the limit, from the first leg change (step 60) on.
    assert np.sum(rates[55:75] >= PARADE_ROLL * (1.0 - 1e-5)) >= 3
