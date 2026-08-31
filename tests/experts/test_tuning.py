"""The PID tuners are runtime, not developer tooling.

Anyone who changes an environment's parameters -- a different tank geometry, a
slower actuator -- has invalidated the shipped gains and has to retune. That
makes ``relay_autotune`` and ``pid_tuning`` part of what the library is for,
and worth the same treatment as the environments.

The tuning rules are closed-form and are checked against their published
constants. The relay experiment and the gradient tuner are checked for the
properties that make them useful: that they identify an oscillation at all,
and that optimising actually reduces the loss.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from target_gym import FirstOrderParams, FirstOrderSystem
from target_gym.experts.relay_autotune import (
    TUNING_RULES,
    amigo,
    relay_experiment,
    relay_sweep,
    tyreus_luyben,
    ziegler_nichols,
)

KU, TU = 4.0, 2.0  # arbitrary but fixed ultimate gain and period


# ---------------------------------------------------------------------------
# Closed-form tuning rules
# ---------------------------------------------------------------------------


def test_ziegler_nichols_matches_the_published_constants():
    """Classic Z-N PID: Kp = 0.6 Ku, Ti = Tu/2, Td = Tu/8."""
    Kp, Ki, Kd = ziegler_nichols(KU, TU)
    assert Kp == pytest.approx(0.6 * KU)
    assert Ki == pytest.approx(Kp / (TU / 2.0))
    assert Kd == pytest.approx(Kp * (TU / 8.0))


def test_tyreus_luyben_matches_the_published_constants():
    """Tyreus-Luyben: Kp = Ku/3.2, Ti = 2.2 Tu, Td = Tu/6.3."""
    Kp, Ki, Kd = tyreus_luyben(KU, TU)
    assert Kp == pytest.approx(KU / 3.2)
    assert Ki == pytest.approx(Kp / (2.2 * TU))
    assert Kd == pytest.approx(Kp * (TU / 6.3))


def test_rules_are_ordered_by_aggressiveness():
    """The docstrings claim an ordering; hold them to it.

    Tyreus-Luyben advertises itself as less aggressive than Ziegler-Nichols
    and AMIGO as the balanced default, which means their proportional gains
    must come out in that order for the same identified plant.
    """
    kp_zn = ziegler_nichols(KU, TU)[0]
    kp_tl = tyreus_luyben(KU, TU)[0]
    kp_am = amigo(KU, TU)[0]
    assert kp_tl < kp_am < kp_zn, (
        f"expected Tyreus-Luyben < AMIGO < Ziegler-Nichols, got "
        f"{kp_tl:.3f}, {kp_am:.3f}, {kp_zn:.3f}"
    )


@pytest.mark.parametrize("name", sorted(TUNING_RULES))
def test_gains_scale_with_the_identified_plant(name):
    """Every rule is linear in Ku, and its Ki/Kd track the period as 1/Tu, Tu.

    This is what makes a gain schedule meaningful: the same rule applied at a
    different operating point must move the gains in the direction the
    identified dynamics moved.
    """
    rule = TUNING_RULES[name]
    Kp, Ki, Kd = rule(KU, TU)
    Kp2, Ki2, Kd2 = rule(2 * KU, TU)
    assert (Kp2, Ki2, Kd2) == pytest.approx((2 * Kp, 2 * Ki, 2 * Kd))

    _, Ki3, Kd3 = rule(KU, 2 * TU)
    assert Ki3 == pytest.approx(Ki / 2.0), f"{name}: Ki should fall as 1/Tu"
    assert Kd3 == pytest.approx(Kd * 2.0), f"{name}: Kd should grow with Tu"


# ---------------------------------------------------------------------------
# Relay experiment and sweep
# ---------------------------------------------------------------------------


def _first_order():
    env = FirstOrderSystem(integration_method="rk4_1")
    params = FirstOrderParams()

    def reset_fn(key, p, t):
        _, state = env.reset_env(key, p)
        return state.replace(target_x=t, x=t)

    return env, params, reset_fn


def test_relay_experiment_identifies_an_oscillation():
    """A relay experiment must come back with a usable Ku and Tu.

    Returning NaN, a non-positive period, or a zero gain all mean the relay
    never drove a limit cycle -- from which no tuning rule can produce gains.
    """
    env, params, reset_fn = _first_order()
    mid = float(np.mean(params.target_x_range))

    result = relay_experiment(
        env,
        params,
        reset_fn,
        state_index=0,
        setpoint_index=1,
        operating_point=mid,
        relay_amplitude=0.5,
        max_steps=3000,
    )

    assert np.isfinite(result["Ku"]), f"Ku is {result['Ku']}"
    assert np.isfinite(result["Tu"]), f"Tu is {result['Tu']}"
    assert result["Ku"] > 0.0, "non-positive ultimate gain"
    assert result["Tu"] > 0.0, "non-positive ultimate period"


def test_relay_sweep_produces_a_gain_schedule():
    """A sweep returns one set of gains per operating point, all finite."""
    env, params, reset_fn = _first_order()

    result = relay_sweep(
        env,
        params,
        reset_fn,
        state_index=0,
        setpoint_index=1,
        target_range=tuple(params.target_x_range),
        n_points=3,
        sign=1,
        tuning_rule="amigo",
        relay_amplitude=0.5,
        max_steps=3000,
    )

    n = len(result["operating_points"])
    assert n == 3
    for gain in ("Kp", "Ki", "Kd"):
        assert len(result[gain]) == n, f"{gain} has {len(result[gain])} of {n} points"
        assert all(np.isfinite(v) for v in result[gain]), f"{gain} contains non-finite"


# ---------------------------------------------------------------------------
# Gradient tuning
# ---------------------------------------------------------------------------


def test_gradient_tuning_reduces_the_tracking_loss():
    """Optimising must actually improve the objective it is given.

    The loss is a vmapped ITAE over setpoints spanning the target range; a
    tuner that returns gains no better than it started with is worse than
    useless, because its output is cached and shipped.
    """
    from target_gym.experts.pid_tuning import make_siso_pid_loss_fn, tune_pid_gains

    env, params, reset_fn = _first_order()
    loss_fn = make_siso_pid_loss_fn(
        env,
        params,
        state_index=0,
        setpoint_index=1,
        reset_fn=reset_fn,
        target_range=tuple(params.target_x_range),
        n_targets=4,
        n_steps=60,
    )

    init = (1.0, 0.1, 0.01)
    before = float(jax.jit(loss_fn)(*init))
    tuned = tune_pid_gains(loss_fn, *init, n_grad_steps=25, lr=0.1)
    after = float(jax.jit(loss_fn)(*tuned))

    assert np.isfinite(before) and np.isfinite(after)
    assert after <= before, f"tuning made the loss worse: {before:.4f} -> {after:.4f}"
    # Sign is documented as preserved through the log-magnitude parametrisation.
    assert all(
        np.sign(t) == np.sign(i) for t, i in zip(tuned, init)
    ), f"gain signs flipped during tuning: {init} -> {tuned}"
