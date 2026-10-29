"""The four-tank expert: crossed PI loops through a static decoupler.

Contract (PHYSICS.md section 6; the crossed pairing from quadruple-tank
practice, the decoupler ours): with gamma1 + gamma2 < 1 each lower tank is
fed mostly by the far pump, so the loop on h1 drives pump 2 and the loop on
h2 pump 1. The decoupler makes each
loop's output move its own level alone at steady state, and the bias is the
pair of voltages that holds the targets. The oracle for both is the plant
itself, run to rest.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.four_tank.env import FourTankParams
from target_gym.pc_gym.four_tank.env_jax import FourTank
from target_gym.pc_gym.four_tank.experts import (
    make_four_tank_pid_controller,
    steady_voltages,
)

P = FourTankParams()
EXPERT = make_four_tank_pid_controller()
TARGETS = (0.15, 0.21)


def _raw(v):
    return 2.0 * (np.asarray(v) - P.v_min) / (P.v_max - P.v_min) - 1.0


def _rest(v, steps=3000):
    """Lower levels after ``steps`` s at constant voltages ``v``, from the
    reset state."""
    env = FourTank()
    _, state = env.reset_env(jax.random.PRNGKey(0), P)
    action = jnp.asarray(_raw(v), jnp.float32)

    def body(s, _):
        return env.step_env(jax.random.PRNGKey(0), s, action, P)[1], None

    state, _ = jax.lax.scan(body, state, None, length=steps)
    return np.array([float(state.h1), float(state.h2)])


def test_the_bias_voltages_hold_the_targets():
    v = steady_voltages(*TARGETS, EXPERT.params, np)
    np.testing.assert_allclose(_rest(v), TARGETS, atol=1e-4)


@pytest.mark.parametrize("loop", [0, 1])
def test_each_loop_moves_its_own_level_alone_at_rest(loop):
    """0.3 V on one loop's output moves its level by about 1 cm and the
    other by under a tenth of that: the decoupler is built from step tests
    at the training targets, not at this one, so it is close, not exact."""
    prm = EXPERT.params
    v0 = np.array(steady_voltages(*TARGETS, prm, np))
    D = np.array([[prm.D11, prm.D12], [prm.D21, prm.D22]])
    moved = _rest(v0 + D @ (np.eye(2)[loop] * 0.3)) - _rest(v0)
    assert abs(moved[loop]) > 5e-3
    assert abs(moved[1 - loop]) < 0.1 * abs(moved[loop])


def test_the_pairing_is_crossed():
    """A low tank 1 is answered mostly by pump 2, a low tank 2 by pump 1."""
    pid = make_four_tank_pid_controller().host()
    on = np.asarray(pid(np.array([*TARGETS, 0.3, 0.2, *TARGETS], np.float32)))
    pid.reset()
    low1 = np.asarray(pid(np.array([0.14, 0.21, 0.3, 0.2, *TARGETS], np.float32)))
    pid.reset()
    low2 = np.asarray(pid(np.array([0.15, 0.20, 0.3, 0.2, *TARGETS], np.float32)))
    assert (low1 - on)[1] > abs((low1 - on)[0])
    assert (low2 - on)[0] > abs((low2 - on)[1])
