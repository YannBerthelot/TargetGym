"""The kiln expert: a fuzzy supervisor on free lime over a burning-zone loop.

Contract, from Swain and Subudhi (1996), "Fuzzy logic based cement kiln
control", IE(I) Journal-EL 77, pp. 58-61 (a rule base run at a 2500 t/day dry
kiln), and kiln practice (not from the code):

* The supervisor reads free lime (LOW / OK / HIGH) and the kiln-speed trend
  (INCR / OK / DECR) and moves the burning-zone temperature setpoint by
  large / medium / small decreases, zero, or small / medium / large increases.
  Their rule block, rows free lime, columns speed INCR / OK / DECR:

      LOW   SINCR  MDECR  LDECR
      OK    MINCR  ZERO   MDECR
      HIGH  LINCR  MINCR  SDECR

* A loop on the burning-zone pyrometer holds that setpoint with fuel; kiln
  speed follows the measured feed so the bed depth stays put.
* Ours (the paper gives no breakpoints): triangles on the free-lime error with
  half-width ``b`` and on the speed's change over 5 min with half-width ``s``,
  shoulders beyond; the output sets are singletons at 1, 2, 3 steps of ``u``;
  min for AND and the centroid of the singletons. ``b`` and ``u`` are tuned on
  training seeds; ``s`` is the speed change one feed standard deviation
  causes.
"""

import json

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.provenance import GAINS_PATH
from target_gym.registry import REGISTRY

P = REGISTRY["cement_kiln"].make_test_params()


def _obs(lime, target=1.2, speed=40.0, feed=None, T_bz=1765.0):
    feed = float(P.raw_meal_nominal) if feed is None else feed
    return np.array([lime, T_bz, 600.0, 900.0, feed, 50.0, speed, target], np.float32)


def _expert():
    expert = tg.make_expert("cement_kiln").host()
    expert.reset()
    return expert


def _setpoint_move(lime, trend):
    """The setpoint step, in units of ``u``, after five minutes of steady
    speed and then a speed change of ``trend`` half-widths ``s``."""
    expert = _expert()
    params = tg.make_expert("cement_kiln").params
    for _ in range(10):
        expert(_obs(1.2, speed=40.0))
    before = float(expert.state.sp_trim)
    expert(_obs(lime, speed=40.0 + trend * params.speed_halfwidth))
    return (float(expert.state.sp_trim) - before) / params.step


def _lime(level):
    """Free lime in pp at ``level`` half-widths from the 1.2 pp target."""
    return 1.2 + 100.0 * level * tg.make_expert("cement_kiln").params.lime_halfwidth


@pytest.mark.parametrize(
    "lime, trend, move",
    [
        (-2, 2, 1.0),  # LOW, INCR: SINCR
        (-2, 0, -2.0),  # LOW, OK: MDECR
        (-2, -2, -3.0),  # LOW, DECR: LDECR
        (0, 2, 2.0),  # OK, INCR: MINCR
        (0, 0, 0.0),  # OK, OK: ZERO
        (0, -2, -2.0),  # OK, DECR: MDECR
        (2, 2, 3.0),  # HIGH, INCR: LINCR
        (2, 0, 2.0),  # HIGH, OK: MINCR
        (2, -2, -1.0),  # HIGH, DECR: SDECR
    ],
)
def test_the_rule_block_is_swain_and_subudhis(lime, trend, move):
    assert _setpoint_move(_lime(lime), trend) == pytest.approx(move, abs=1e-4)


def test_between_sets_the_centroid_interpolates():
    """Half a half-width high, speed steady: OK and HIGH at 0.5 each, so the
    centroid of ZERO (0) and MINCR (2) is 1."""
    assert _setpoint_move(_lime(0.5), 0) == pytest.approx(1.0, abs=1e-4)


def test_rules_fire_at_the_minimum_of_their_two_memberships():
    """Mamdani AND. Free lime a quarter half-width high (OK 0.75, HIGH 0.25)
    and the speed half a half-width up (OK 0.5, INCR 0.5): the four rules fire
    at 0.5, 0.5, 0.25 and 0.25 with ZERO, MINCR, MINCR and LINCR, so the
    centroid is (0 + 1 + 0.5 + 0.75) / 1.5 = 1.5 (a product would give
    1.375)."""
    assert _setpoint_move(_lime(0.25), 0.5) == pytest.approx(1.5, abs=1e-4)


def test_the_burning_zone_loop_follows_simc_for_its_step_test():
    m = json.loads(GAINS_PATH.read_text())["cement_kiln"]["model_inner"]
    k, tau, theta, tau_c = m["k"], m["tau"], m["theta"], m["tau_c"]
    Kc = tau / (k * (tau_c + theta))
    Ti = min(tau, 4.0 * (tau_c + theta))
    params = tg.make_expert("cement_kiln").params
    assert params.Kp_inner == pytest.approx(Kc, rel=1e-9)
    assert params.Ki_inner == pytest.approx(Kc / Ti, rel=1e-9)


def test_kiln_speed_follows_the_feed():
    """Bed depth held: rpm = rpm_nominal * feed / feed_nominal."""
    feed = 1.1 * float(P.raw_meal_nominal)
    a = _expert()(_obs(1.2, feed=feed))
    rpm = float(P.rpm_min) + 0.5 * (float(a[1]) + 1.0) * (
        float(P.rpm_max) - float(P.rpm_min)
    )
    assert rpm == pytest.approx(1.1 * float(P.rpm_nominal), rel=1e-5)


# Spec, stated before measuring: rule-based kiln control holds free lime to a
# few tenths of a percent. On held-out seeds, over the scored hold, the mean
# free-lime error is under 0.25 pp on every seed, and nothing trips.
HELD_OUT = range(200, 210)


@pytest.mark.slow
def test_free_lime_holds_on_held_out_seeds():
    from target_gym.eval import scored_burn_in

    spec = REGISTRY["cement_kiln"]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    burn = scored_burn_in("cement_kiln", p)
    for seed in HELD_OUT:
        expert = tg.make_expert("cement_kiln").host()
        expert.reset()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        errors = []
        for t in range(int(p.max_steps_in_episode)):
            key, k = jax.random.split(key)
            obs, state, _, _, info = step(k, state, expert(np.asarray(obs)), p)
            assert not bool(info["tripped"])
            if t >= burn:
                o = np.asarray(obs)
                errors.append(abs(o[0] - o[7]))
        assert np.mean(errors) < 0.25, f"seed {seed}: {np.mean(errors):.3f} pp"
