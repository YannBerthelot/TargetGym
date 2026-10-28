"""The boiler drum expert: three-element level control and a pressure loop,
both lambda-tuned.

Contract, from boiler-control practice and the tuning recipe (not the code):

* Three-element level control: the feedwater command is the measured steam
  flow, trimmed by a level PI. At the level target, with fresh memory, feed
  equals steam.
* Firing is the steam's enthalpy rise as a feedforward, ``q_steam (h_g -
  h_fw)``, plus a pressure PI. ``h_g`` is saturated steam at the drum's 85 bar
  (IAPWS-IF97: 2749 kJ/kg at 8.5 MPa) and ``h_fw`` the plant's feedwater
  enthalpy.
* Both loops are lambda-tuned for an integrating process from the stored
  step-test model ``k' e^(-theta s) / s``: ``Kc = (2 lam + theta) / (k' (lam +
  theta)^2)`` and ``Ti = 2 lam + theta`` (on the action; the firing PI's gain
  on the [-1, 1] action is twice its gain on the heat fraction).
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


@pytest.mark.parametrize("loop, scale", [("level", 1.0), ("pressure", 2.0)])
def test_both_loops_follow_the_lambda_rule_for_their_step_test(loop, scale):
    m = _stored()[f"model_{loop}"]
    k, theta, lam = m["k"], m["theta"], m["lam"]
    Kc = (2 * lam + theta) / (k * (lam + theta) ** 2)
    Ti = 2 * lam + theta
    params = tg.make_expert("boiler_drum").params
    assert scale * getattr(params, f"Kp_{loop}") == pytest.approx(Kc, rel=1e-9)
    assert scale * getattr(params, f"Ki_{loop}") == pytest.approx(Kc / Ti, rel=1e-9)


# Spec, stated before measuring the level: three-element control holds drum
# level within about an inch of its target. On held-out seeds, the mean level
# error over the scored hold is under 25 mm on every seed, and nothing trips.
HELD_OUT = range(200, 210)


@pytest.mark.large
@pytest.mark.xfail(
    strict=True,
    reason="Finding (2026-10-28): mean level error 19-34 mm over the held-out "
    "seeds, 6 of 10 above 25 mm. The lambda-tuned trim (tight, lam = theta) "
    "still lets steam-demand swings move the level; the shipped hand tuning "
    "left 28-79 mm on the protocol seeds.",
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
