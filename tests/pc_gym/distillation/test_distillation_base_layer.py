"""The distillation expert is the base layer practice runs under its MPC.

Contract (PHYSICS.md section 8; Skogestad, "Distillation control: dos and
don'ts", sections 6.3 to 6.5): the LV-configuration with a temperature loop,
boilup holding the stage-15 temperature, reflux holding the distillate and
the temperature setpoint holding the bottoms. A colder stage 15 gets more
boilup; a distillate below spec gets more reflux; a bottoms above spec gets a
hotter stage 15. The switch to automatic is bumpless: on target, the first
move keeps the flows where they are. Stage 15 is the bottom-section stage the
maximum gain rule picks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.distillation.env import (
    T_BOIL_HEAVY,
    T_BOIL_LIGHT,
    DistillationParams,
)
from target_gym.pc_gym.distillation.env_jax import DistillationColumn
from target_gym.pc_gym.distillation.experts import make_distillation_pid_controller

P = DistillationParams()
#: [yD, xB, L_pct, V_pct, target_yD, target_xB, T_15]: on target, mid-range.
ON = np.array([0.99, 0.01, 0.5, 0.5, 0.99, 0.01, 351.0], np.float32)


def _first_then(change):
    """Flows (raw) a fresh expert asks for on ``ON``, then on ``ON`` changed."""
    pid = make_distillation_pid_controller().host()
    first = np.asarray(pid(ON))
    obs = ON.copy()
    for i, v in change.items():
        obs[i] = v
    return first, np.asarray(pid(obs))


def test_on_target_the_first_move_keeps_the_flows():
    first, _ = _first_then({})
    np.testing.assert_allclose(first, [0.0, 0.0], atol=1e-6)


def test_a_colder_stage_15_gets_more_boilup():
    first, then = _first_then({6: 350.0})
    assert then[1] > first[1]


def test_a_distillate_below_spec_gets_more_reflux():
    first, then = _first_then({0: 0.985})
    assert then[0] > first[0]


def test_a_bottoms_above_spec_heats_stage_15():
    """More light component in the bottoms raises the temperature setpoint,
    which the temperature loop answers with boilup."""
    first, then = _first_then({1: 0.015})
    assert then[1] > first[1]


@pytest.mark.large
def test_stage_15_has_the_largest_bottom_section_gain_to_boilup():
    """The maximum gain rule: of the bottom-section trays (stages 2 to 20),
    stage 15's temperature moves most per unit boilup at steady state."""
    env = DistillationColumn()
    p = P.replace(zF_noise_std=0.0)
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    state = state.replace(zF=jnp.float32(p.zF_nominal))

    def rest(V, s, steps=4000):
        raw = jnp.array(
            [
                2 * (2.706 - p.L_min) / (p.L_max - p.L_min) - 1,
                2 * (V - p.V_min) / (p.V_max - p.V_min) - 1,
            ],
            jnp.float32,
        )

        def body(c, _):
            return env.step_env(jax.random.PRNGKey(0), c, raw, p)[1], None

        return jax.lax.scan(body, s, None, length=steps)[0]

    base = rest(3.206, state)
    moved = rest(3.206 + 0.002, base)
    temp = lambda x: T_BOIL_LIGHT * x + T_BOIL_HEAVY * (1 - x)  # noqa: E731
    gain = np.abs(np.asarray(temp(moved.x) - temp(base.x)))
    assert int(np.argmax(gain[1:20])) + 2 == 15
