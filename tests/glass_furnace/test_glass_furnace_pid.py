"""The glass furnace expert: the crown-temperature PI, SIMC-tuned.

Contract, from the tuning recipe and furnace practice (not the code):

* A PI on the measured crown temperature against its target, no derivative.
* SIMC on the integrating approximation of the stored step-test model
  ``k' e^(-theta s) / s``: ``Kc = 1 / (k' (tau_c + theta))`` and
  ``tau_I = 4 (tau_c + theta)`` (Skogestad 2003, Table 1).
* Practice puts an MPC over this loop; this is the base layer under it.
"""

import json

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.provenance import GAINS_PATH
from target_gym.registry import REGISTRY


def test_the_gains_follow_simc_for_the_stored_step_test():
    m = json.loads(GAINS_PATH.read_text())["glass_furnace"]["model"]
    k, theta, tau_c = m["k"], m["theta"], m["tau_c"]
    Kc = 1.0 / (k * (tau_c + theta))
    params = tg.make_expert("glass_furnace").params
    assert params.Kp == pytest.approx(Kc, rel=1e-9)
    assert params.Ki == pytest.approx(Kc / (4.0 * (tau_c + theta)), rel=1e-9)
    assert params.Kd == 0.0


def _pi(**changes):
    expert = REGISTRY["glass_furnace"].make_pid_controller()
    expert = expert.replace(params=expert.params.replace(**changes)).host()
    expert.reset()
    return expert


def _obs(crown, target=1590.0):
    return np.array([crown, 1200.0, 80.0, 1.0, target, 5.79], np.float32)


def test_the_filter_starts_on_the_first_reading():
    """No integral: 1 K of error at Kp 0.3 is 0.3, not a filter's fraction."""
    pi = _pi(Kp=0.3, Ki=0.0, alpha=0.1)
    assert float(pi(_obs(1589.0))[0]) == pytest.approx(0.3, rel=1e-4)


def test_a_reading_passes_its_filter():
    """A step of 2 K on a filter that passes 0.4 moves the PV 0.8 K."""
    pi = _pi(Kp=0.3, Ki=0.0, alpha=0.4)
    pi(_obs(1590.0))
    assert float(pi(_obs(1588.0))[0]) == pytest.approx(0.3 * 0.8, rel=1e-3)


def test_the_integral_stops_while_the_fuel_is_at_its_limit():
    """Ten steps pinned at +1, then 1 K high: only P acts on the way back."""
    pi = _pi(Kp=0.3, Ki=1e-4, alpha=1.0)
    for _ in range(10):
        assert float(pi(_obs(1570.0))[0]) == 1.0
    out = float(pi(_obs(1591.0))[0])
    assert out == pytest.approx(-0.3 - 1e-4 * 30.0, rel=1e-3)


# Spec, stated before measuring the crown error: a furnace's crown PID holds
# its temperature within 2 K. On held-out seeds, over the scored hold, the
# true crown is within 2 K of its target on at least 95 % of steps, and
# nothing trips.
HELD_OUT = range(200, 210)


@pytest.mark.large
@pytest.mark.xfail(
    strict=True,
    reason="Finding (2026-10-28): 88.8-97.0 % of hold steps within 2 K, 6 of 10 "
    "seeds under 95 %, peaks 4.3-6.9 K. The grid-tuned gains it replaces score "
    "the same (88.8-96.9 %, peaks 4.4-7.1 K): the misses are the firing "
    "interruptions at each regenerator reversal, which a PI on the crown "
    "only meets after the fact.",
)
def test_the_crown_holds_within_two_kelvin_on_held_out_seeds():
    from target_gym.eval import scored_burn_in

    spec = REGISTRY["glass_furnace"]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    burn = scored_burn_in("glass_furnace", p)
    for seed in HELD_OUT:
        expert = tg.make_expert("glass_furnace").host()
        expert.reset()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        inside = []
        for t in range(int(p.max_steps_in_episode)):
            key, k = jax.random.split(key)
            obs, state, _, _, info = step(k, state, expert(np.asarray(obs)), p)
            assert not bool(info["tripped"])
            if t >= burn:
                inside.append(abs(float(state.target_T_crown - state.T_crown)) <= 2.0)
        assert np.mean(inside) >= 0.95, f"seed {seed}: {np.mean(inside):.3f}"
