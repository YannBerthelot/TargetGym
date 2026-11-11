"""The pH probe of ``ph_neutralization`` (docs/proposals/industrial-fixes.md, PR 4).

Contract, from the plan and PHYSICS.md's Readings section, not from the code: the
electrode has a 10 s first-order lag and reports what it saw 10 s ago, and the
5 s step puts that dead time at exactly 2 steps. So after a step in the tank's pH
at step 1 the probe shows nothing until step 3, then 1 - exp(-k / 2) of the step
on the k-th step after that. The plant and the score read the tank.
"""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.base import true_obs
from target_gym.pc_gym.ph_neutralization.env import solve_pH, steady_state_invariants
from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization


def test_a_step_in_the_tanks_ph_reaches_the_probe_late_and_lagged():
    env = PHNeutralization()
    params = env.default_params.replace(
        q2_noise_std=0.0, sensor_noise_scale=0.0, acid_upset_size=0.0, flow_swing=0.0
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    before = float(state.pH)

    # Both tanks jump to the steady state of a new base flow, and that flow
    # holds them (the trim valve shut).
    q3 = float(state.q3) + 1.0
    Wa, Wb = steady_state_invariants(q3, state.q2, params)
    after = float(solve_pH(Wa, Wb, params))
    assert abs(after - before) > 0.2
    state = state.replace(Wa=Wa, Wb=Wb, Wa_s1=Wa, Wb_s1=Wb)
    raw = 2.0 * (q3 - params.q3_min) / (params.q3_max - params.q3_min) - 1.0

    read = []
    for _ in range(5):
        obs, state, *_ = env.step_env(key, state, jnp.array([raw, -1.0]), params)
        read.append((float(obs[0]) - before) / (after - before))

    # 1e-3 of the step: the bisection and float32 resolve pH to about 1e-5.
    np.testing.assert_allclose(read, [0, 0, 0.39347, 0.63212, 0.77687], atol=1e-3)
    assert float(true_obs(env, state, params)[0]) == float(state.pH)
    assert abs(float(state.pH) - after) < 1e-3
