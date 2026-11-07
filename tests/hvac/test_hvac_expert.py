"""The HVAC expert: outdoor reset with indoor feedback, and optimal start.

Contract, from the published sequences (not from the code):

* The heating curve, tekmar House Control 352 (D 352, 06/00). Heat supplied must
  equal heat lost, so on this plant's linear emitter the curve is ``Q = H (T_s -
  T_out)``, ``H`` the building's 159.4 W/K, from a starting point ``T_s`` at the
  setpoint in force. The heat stops when the outdoor temperature reaches it,
  which is warm weather shutdown. As a 0-1 signal on ``Q_heat_max``.
* Indoor feedback shifts the starting point up when the zone is cold and down
  when it is warm. It is a PI on the zone reading, with the recipe's gains
  (docs/proposals/expert-protocols-process.md, section 10). The integral stops
  while the signal is saturated.
* The action maps the signal onto [-1, 1], so 0 (no heat) is -1 and 1 is +1.
* Optimal start follows LBL's ``OBC/Utilities/OptimalStart``, the block that
  implements G36's warm-up:
  - ``tOptMax`` (3 h) before occupancy it samples the zone, and the warm-up
    lasts ``(occupied setpoint - zone) / slope``, capped at 3 h;
  - the slope is the moving mean of the last three learned slopes, floored at
    1 K/h, so it is 1 K/h before anything has been learned;
  - a slope is learned from each warm-up, as the temperature gained over the
    time taken to reach the setpoint, and is pushed at midnight;
  - during warm-up the setpoint in force is the occupied one (G36 5.3.2.3).
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


#: Past every curve's starting point, so only the room's loop runs. A warm day
#: leaves the warm-up's timing, the same on any day, to read by itself.
WARM = 25.0
H_PER_Q = 159.4 / 7200.0  # the curve's slope on the 0-1 signal, per K


def _obs(hour, T_air, target, T_out=WARM):
    a = 2.0 * math.pi * (hour % 24.0) / 24.0
    return np.array(
        [T_air, T_out, 0.0, 0.0, math.sin(a), math.cos(a), target], np.float32
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


def _on_curve(expert, hour, T_air, T_out, occupied=OCCUPIED):
    raw = expert(_obs(hour, T_air, _target(hour, occupied), T_out))
    return 0.5 * (float(np.asarray(raw)[0]) + 1.0)


def test_at_the_setpoint_the_curve_supplies_the_heat_lost():
    """At 21 C with 5 C outside: 159.4 W/K x 16 K = 2550 W of 7200, 0.354."""
    assert _on_curve(_expert(), 12.0, OCCUPIED, 5.0) == pytest.approx(0.354, abs=1e-3)


def test_a_colder_day_moves_the_curve_by_the_buildings_slope():
    cold = _on_curve(_expert(), 12.0, OCCUPIED, -3.0)
    mild = _on_curve(_expert(), 12.0, OCCUPIED, 5.0)
    assert cold - mild == pytest.approx(8.0 * H_PER_Q, abs=1e-5)


def test_warm_weather_shuts_the_heat_off():
    assert _on_curve(_expert(), 12.0, OCCUPIED, OCCUPIED) == 0.0
    assert _on_curve(_expert(), 12.0, OCCUPIED, OCCUPIED + 4.0) == 0.0


def test_the_setpoint_in_force_is_the_curves_starting_point():
    """At 01:00 the setback's 17 C starts the curve, 4 K under the day's."""
    night = _on_curve(_expert(), 1.0, SETBACK, 5.0)
    assert night == pytest.approx(12.0 * H_PER_Q, abs=1e-5)


def test_a_cold_zone_shifts_the_curve_up_and_a_warm_one_down():
    params = _expert().controller.params
    base = _on_curve(_expert(), 12.0, OCCUPIED, 5.0)
    cold = _on_curve(_expert(), 12.0, OCCUPIED - 0.5, 5.0)
    warm = _on_curve(_expert(), 12.0, OCCUPIED + 0.5, 5.0)
    step = params.kp * (0.5 + 0.5 * params.dt / params.Ti)
    assert cold - base == pytest.approx(step, abs=1e-5)
    assert warm - base == pytest.approx(-step, abs=1e-5)


def test_the_trim_ramps_at_the_gain_over_the_integral_time():
    """A constant 0.5 K error adds kp e dt / Ti of signal per step."""
    expert = _expert()
    p = expert.controller.params
    heats = [_on_curve(expert, 12.0 + k * DT_H, OCCUPIED - 0.5, 5.0) for k in range(4)]
    np.testing.assert_allclose(np.diff(heats), p.kp * 0.5 * p.dt / p.Ti, atol=1e-5)


def test_a_saturated_loop_leaves_full_heat_as_soon_as_the_error_reverses():
    """Anti-windup: ten steps pinned at full heat by a 10 K error, then the zone
    0.5 K above the setpoint: the signal is the curve less P and one step's
    integral, as if nothing had been pinned."""
    expert = _expert()
    pinned = [
        _on_curve(expert, 8.0 + k * DT_H, OCCUPIED - 10.0, 5.0) for k in range(10)
    ]
    assert pinned == pytest.approx([1.0] * 10)
    p = expert.controller.params
    after = _on_curve(expert, 10.5, OCCUPIED + 0.5, 5.0)
    assert after == pytest.approx(
        16.0 * H_PER_Q - p.kp * 0.5 * (1.0 + p.dt / p.Ti), abs=1e-5
    )


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
    reason="Finding (PR 9, 2026-11-07): 46 of 70 mornings against the 63 the spec "
    "asks. The curve and the recipe's PI reach the setpoint fast, so optimal start "
    "learns 5 K/h where the air's quick rise hides the mass, starts 0.7 h before "
    "occupancy, and the median morning is 0.16 K short, the worst 2.9 K. It was 67 "
    "of 70 on G36's slow loop, whose warm-ups ran to the 3 h cap.",
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
