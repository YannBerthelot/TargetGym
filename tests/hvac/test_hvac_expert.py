"""The HVAC expert: ASHRAE Guideline 36's zone heating loop with optimal start.

Contract, from the published sequence (not from the code):

* The heating loop is a PI with gain 0.1 per K and integral time 900 s on a
  0-1 heating signal (LBL Modelica Buildings, ``G36/ThermalZones/
  ControlLoops.mo``: ``kHeaCon = 0.1``, ``TiHeaCon = 900``). The action maps
  that signal onto [-1, 1], so 0 (no heat) is -1 and 1 (full heat) is +1.
* Optimal start follows LBL's ``OBC/Utilities/OptimalStart``, the block that
  implements G36's warm-up:
  - ``tOptMax`` (3 h) before occupancy it samples the zone, and the warm-up
    lasts ``(occupied setpoint - zone) / slope``, capped at 3 h;
  - the slope is the moving mean of the last three learned slopes, floored at
    1 K/h, so it is 1 K/h before anything has been learned;
  - a slope is learned from each warm-up, as the temperature gained over the
    time taken to reach the setpoint, and is pushed at midnight;
  - during warm-up the heating setpoint is the occupied one (G36 5.3.2.3).
* The occupied setpoint is the one seen in the last occupied period, and the
  design value (21 C) before the first.

The schedule (occupied 07:00-22:00, 15 min steps, episodes starting at
midnight) is the plant's, a BMS's programmed constants.
"""

import math

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.registry import REGISTRY

DT_H = 0.25  # one 15 min step, in hours
SETBACK, OCCUPIED = 17.0, 21.0


def _obs(hour, T_air, target):
    a = 2.0 * math.pi * (hour % 24.0) / 24.0
    return np.array(
        [T_air, 5.0, 0.0, 0.0, math.sin(a), math.cos(a), target], np.float32
    )


def _target(hour, occupied=OCCUPIED):
    return occupied if 7.0 <= hour % 24.0 < 22.0 else SETBACK


def _heat(expert, hour, T_air, occupied=OCCUPIED):
    """The heating signal (0-1) the expert commands at ``hour``."""
    raw = float(np.asarray(expert(_obs(hour, T_air, _target(hour, occupied))))[0])
    return 0.5 * (raw + 1.0)


def _night(expert, T_of_hour, until, start=0.0, occupied=OCCUPIED):
    """Feed the expert every step from ``start`` to ``until`` (hours, which may
    run past midnight), with the zone at ``T_of_hour(h)``; return
    ``{hour: heat}``."""
    out, h = {}, start
    while h < until - 1e-9:
        out[round(h % 24.0, 2)] = _heat(expert, h, T_of_hour(h), occupied)
        h += DT_H
    return out


def _expert():
    expert = tg.make_expert("hvac").host()
    expert.reset()
    return expert


def test_the_heating_loop_is_off_at_the_setpoint():
    expert = _expert()
    assert _heat(expert, 12.0, OCCUPIED) == pytest.approx(0.0, abs=1e-6)


def test_the_heating_signal_ramps_at_the_gain_over_the_integral_time():
    """A constant 0.5 K error adds kp * e * dt / Ti = 0.1 * 0.5 * 900 / 900
    = 0.05 of heating signal per step, whatever the discretisation."""
    expert = _expert()
    heats = [_heat(expert, 12.0 + k * DT_H, OCCUPIED - 0.5) for k in range(4)]
    np.testing.assert_allclose(np.diff(heats), 0.05, atol=1e-5)


def test_a_saturated_loop_leaves_full_heat_as_soon_as_the_error_reverses():
    """Anti-windup: ten steps pinned at full heat by a 10 K error, then the
    zone 0.2 K above the setpoint, and the heat is off at once."""
    expert = _expert()
    pinned = [_heat(expert, 8.0 + k * DT_H, OCCUPIED - 10.0) for k in range(10)]
    assert pinned == pytest.approx([1.0] * 10)
    assert _heat(expert, 10.5, OCCUPIED + 0.2) < 0.5


def test_before_any_warm_up_is_learned_the_start_is_one_hour_per_kelvin():
    """Zone at 19.1 C all night: 1.9 K below the 21 C design setpoint, so the
    warm-up lasts 1.9 h (6840 s). The heat comes on at the first step with
    less than that to 07:00, which is 05:15 (6300 s left; 05:00 has 7200)."""
    heat = _night(_expert(), lambda h: 19.1, until=7.0)
    assert max(v for h, v in heat.items() if h < 5.25) == 0.0
    assert heat[5.25] > 0.0


def test_the_warm_up_is_capped_at_three_hours():
    """Zone at 17.5 C: 3.5 h at 1 K/h, capped at 3 h. Sampled at 04:00, so
    the heat is off then (exactly 3 h left) and on at 04:15."""
    heat = _night(_expert(), lambda h: 17.5, until=7.0)
    assert max(v for h, v in heat.items() if h <= 4.0) == 0.0
    assert heat[4.25] > 0.0


def test_the_next_warm_up_uses_the_slope_learned_this_morning():
    """Day 1 at 1 K/h: the 3 h cap starts the warm-up at 04:15 with the zone
    at 17.5 C, which then gains 0.5 K per step and reaches 21 C seven steps
    later, at 06:00: 3.5 K in 6300 s, 2 K/h. Pushed at midnight, it is the
    whole moving mean on day 2, when the zone at 17.6 C (3.4 K short) needs
    6120 s: off at 05:15 (6300 s left), on at 05:30 (5400 s)."""

    def day1(h):
        return min(17.5 + 0.5 * max(round((h - 4.25) / DT_H), 0), OCCUPIED)

    expert = _expert()
    _night(expert, day1, until=24.0)
    heat = _night(expert, lambda h: 17.6, start=24.0, until=31.0)
    assert heat[5.25] == 0.0
    assert heat[5.5] > 0.0


def test_the_warm_up_aims_at_the_occupied_setpoint_last_seen():
    """Day 1 is occupied at 22 C. The zone never warms (19.6 C), so that
    morning's slope (1.4 K over the 3 h cap) is under the 1 K/h floor and the
    floor stays. Day 2's warm-up is to 22 C: 2.4 K, so 2.4 h (8640 s): off at
    04:30 (9000 s left), on at 04:45 (8100 s)."""
    expert = _expert()
    _night(expert, lambda h: 19.6, until=24.0, occupied=22.0)
    heat = _night(expert, lambda h: 19.6, start=24.0, until=31.0, occupied=22.0)
    assert heat[4.5] == 0.0
    assert heat[4.75] > 0.0


# Spec, stated before measuring: optimal start exists so that the zone is
# comfortable when occupancy begins. From day 2 (once a slope is learned), the
# air at the first occupied step is within the comfort tolerance below the
# target on at least 90 % of mornings, on held-out seeds the tuning never saw.
HELD_OUT = range(200, 210)


@pytest.mark.large
@pytest.mark.xfail(
    strict=True,
    reason="Finding (2026-10-28): 59 of 70 held-out mornings (84 %). Every miss "
    "(0.55-1.0 K cold) is a morning where the 04:00 sample found the zone less "
    "than uHigh = 0.5 K short, so OptimalStart ran no warm-up and the zone kept "
    "cooling until 07:00. That is the published block's single sample, not a "
    "fault of this copy.",
)
def test_the_zone_is_comfortable_when_occupancy_begins_on_held_out_seeds():
    spec = REGISTRY["hvac"]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    steps_per_day = int(round(86400.0 / float(p.delta_t)))
    first = int(round(float(p.occupied_start_h) * 3600.0 / float(p.delta_t)))
    ok, mornings = 0, 0
    for seed in HELD_OUT:
        expert = tg.make_expert("hvac").host()
        expert.reset()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        for t in range(int(p.max_steps_in_episode)):
            key, k = jax.random.split(key)
            obs, state, _, done, info = step(k, state, expert(np.asarray(obs)), p)
            assert not bool(info["tripped"])
            day, slot = divmod(t + 1, steps_per_day)
            if day >= 1 and slot == first:
                o = np.asarray(obs)
                mornings += 1
                ok += bool(o[0] >= o[6] - float(p.comfort_tolerance))
    assert ok >= 0.9 * mornings, f"{ok} of {mornings} mornings comfortable"
