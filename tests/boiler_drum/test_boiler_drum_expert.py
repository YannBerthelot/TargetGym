"""The boiler drum expert: three-element level control and a pressure loop,
both lambda-tuned from single-run bump tests.

Contract, from boiler-control practice and the tuning recipe (not the code):

* Three-element level control: the feedwater command is the measured steam
  flow, trimmed by a level PI. At the level target, with fresh memory, feed
  equals steam.
* Firing is the steam's enthalpy rise as a feedforward, ``q_steam (h_g -
  h_fw)``, plus a pressure PI. ``h_g`` is saturated steam at the drum's 85 bar
  (IAPWS-IF97: 2749 kJ/kg at 8.5 MPa) and ``h_fw`` the plant's feedwater
  enthalpy.
* Level integrates, so it is lambda-tuned from the stored model ``k' e^(-theta
  s) / s``: ``Kc = (2 lam + theta) / (k' (lam + theta)^2)``, ``Ti = 2 lam + theta``.
* Steam follows pressure, so pressure regulates itself, and is lambda-tuned from
  its stored first-order model: ``Kc = tau1 / (k (lam + theta))``,
  ``Ti = min(tau1, 4 (lam + theta))``, C7's capped integral time on a long lag,
  which D7's lambda rule applies (docs/proposals/expert-protocols-process.md), on
  the action (the firing PI's gain on the [-1, 1] action is twice its gain on
  the heat fraction).
* The level trim has no limit of its own. Its integral stops at the feedwater's
  limits, while the error would push past them (Astrom and Hagglund 3.5).
* Each reading goes through its stored filter, ``Tf_level`` and ``Tf_pressure``.
"""

import json

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.provenance import GAINS_PATH
from target_gym.registry import REGISTRY

P = REGISTRY["boiler_drum"].make_test_params()
H_G = 2.749e6  # J/kg, saturated steam at 8.5 MPa, IAPWS-IF97


def _obs(level=0.0, pressure=85.0, q_steam=80.0, target_level=0.0, target_p=85.0):
    return np.array(
        [level, pressure, q_steam, 50.0, 40.0, target_level, target_p], np.float32
    )


def _act(obs):
    expert = tg.make_expert("boiler_drum").host()
    expert.reset()
    return np.asarray(expert(obs), float)


def test_on_target_feedwater_follows_the_steam_flow():
    fuel, feed = _act(_obs(q_steam=80.0))
    assert 0.5 * (feed + 1.0) * float(P.q_feed_max) == pytest.approx(80.0, rel=1e-5)


def test_on_target_firing_is_the_steam_enthalpy_rise():
    fuel, _ = _act(_obs(q_steam=80.0))
    heat = 0.5 * (fuel + 1.0) * float(P.Q_max)
    assert heat == pytest.approx(80.0 * (H_G - float(P.h_feedwater)), rel=1e-3)


def test_a_low_level_draws_more_feed_than_steam():
    _, feed = _act(_obs(level=-0.01))
    assert 0.5 * (feed + 1.0) * float(P.q_feed_max) > 80.0


def _stored():
    return json.loads(GAINS_PATH.read_text())["boiler_drum"]


def test_level_follows_the_lambda_rule_for_an_integrating_loop():
    m = _stored()["model_level"]
    k, theta, lam = m["k"], m["theta"], m["lam"]
    Kc = (2 * lam + theta) / (k * (lam + theta) ** 2)
    Ti = 2 * lam + theta
    params = tg.make_expert("boiler_drum").params
    assert params.Kp_level == pytest.approx(Kc, rel=1e-9)
    assert params.Ki_level == pytest.approx(Kc / Ti, rel=1e-9)


def test_pressure_follows_the_lambda_rule_for_a_first_order_loop():
    m = _stored()["model_pressure"]
    k, tau1, theta, lam = m["k"], m["tau1"], m["theta"], m["lam"]
    Kc = tau1 / (k * (lam + theta))
    Ti = min(tau1, 4 * (lam + theta))  # C7's cap on a long lag (D7)
    params = tg.make_expert("boiler_drum").params
    assert 2.0 * params.Kp_pressure == pytest.approx(Kc, rel=1e-9)
    assert 2.0 * params.Ki_pressure == pytest.approx(Kc / Ti, rel=1e-9)


def _feed(expert, obs):
    return 0.5 * (float(np.asarray(expert(obs))[1]) + 1.0) * float(P.q_feed_max)


def _expert(**changes):
    expert = REGISTRY["boiler_drum"].make_pid_controller()
    expert = expert.replace(params=expert.params.replace(**changes)).host()
    expert.reset()
    return expert


def test_the_trim_has_no_limit_of_its_own():
    """0.1 m low at 1000 kg/s per m asks for 100 kg/s over the steam's 80. The
    old 25 kg/s clamp would stop at 105."""
    expert = _expert(Kp_level=1000.0, Ki_level=0.0)
    assert _feed(expert, _obs(level=-0.1)) == pytest.approx(180.0, rel=1e-4)


def test_the_level_integral_stops_at_the_feed_limit():
    """Pinned at 200 kg/s for ten steps by a 0.5 m error, then 0.01 m high. The
    integral kept nothing from the pinned steps, so the feed is the steam less
    the P term and one step's integral: 80 - 1000 x 0.01 - 10 x 0.01 x 2."""
    expert = _expert(Kp_level=1000.0, Ki_level=10.0)
    pinned = [_feed(expert, _obs(level=-0.5)) for _ in range(10)]
    assert pinned == pytest.approx([200.0] * 10, rel=1e-5)
    assert _feed(expert, _obs(level=0.01)) == pytest.approx(69.8, rel=1e-3)


def test_the_level_integral_runs_off_a_limit_the_error_is_leaving():
    """Six steps of 1 m low at Ki 10 and 2 s build an integral of 12, 200 kg/s at
    80 of steam, exactly the limit. A steam swing to 100 then holds the feed past
    its limit, and a 0.01 m high level takes 0.02 off the integral: the update
    leaves the limit, so conditional integration lets it. Back at 80 of steam,
    the feed is 80 + 10 x 11.98, 199.8 and not 200."""
    expert = _expert(Kp_level=0.0, Ki_level=10.0)
    for _ in range(6):
        _feed(expert, _obs(level=-1.0))
    assert _feed(expert, _obs(level=0.01, q_steam=100.0)) == pytest.approx(
        200.0, rel=1e-5
    )
    assert _feed(expert, _obs(level=0.0)) == pytest.approx(199.8, rel=1e-5)


def test_a_reading_passes_its_filter():
    """A filter of 2 s on a 2 s step passes 1 - e^-1 = 0.632 of a level step."""
    expert = _expert(Kp_level=100.0, Ki_level=0.0, alpha_level=0.632)
    assert _feed(expert, _obs(level=0.0)) == pytest.approx(80.0, rel=1e-5)
    assert _feed(expert, _obs(level=0.1)) == pytest.approx(80.0 - 6.32, rel=1e-4)


def test_the_pressure_reading_passes_its_filter():
    """A 1 bar drop through a filter that passes half moves the PV 0.5 bar, and
    Kp 0.1 of the heat on that is 0.05 of Q_max, 0.1 on the [-1, 1] action."""
    expert = _expert(Kp_pressure=0.1, Ki_pressure=0.0, alpha_pressure=0.5)
    steady = float(np.asarray(expert(_obs(pressure=85.0)))[0])
    dropped = float(np.asarray(expert(_obs(pressure=84.0)))[0])
    assert dropped - steady == pytest.approx(0.1, rel=1e-3)


def test_the_filter_starts_on_the_first_reading():
    expert = _expert(Kp_level=100.0, Ki_level=0.0, alpha_level=0.1)
    assert _feed(expert, _obs(level=0.1)) == pytest.approx(70.0, rel=1e-4)


# Spec, stated before measuring the level: three-element control holds drum
# level within about an inch of its target. On held-out seeds, the mean level
# error over the scored hold is under 25 mm on every seed, and nothing trips.
HELD_OUT = range(200, 210)


@pytest.mark.large
@pytest.mark.xfail(
    strict=True,
    reason="Finding (PR 9, 2026-11-07): mean level error 26-52 mm over the held-out "
    "seeds, every seed above 25 mm, on the recipe's gains and the firing lag and "
    "rate limit. The old gains left 75 mm on this plant.",
)
def test_the_drum_level_holds_within_an_inch_on_held_out_seeds():
    from target_gym.eval import scored_burn_in

    spec = REGISTRY["boiler_drum"]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    burn = scored_burn_in("boiler_drum", p)
    for seed in HELD_OUT:
        expert = tg.make_expert("boiler_drum").host()
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
                errors.append(abs(o[0] - o[5]))
        assert np.mean(errors) < 0.025, f"seed {seed}: {np.mean(errors) * 1e3:.1f} mm"
