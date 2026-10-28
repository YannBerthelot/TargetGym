"""Total energy control: throttle closes the energy error, pitch its balance.

TECS (Lambregts) is the longitudinal autopilot PX4 and ArduPilot fly. These
tests pin its defining behaviour: on target it asks for trim; short of height
it adds energy and pitches up; short of speed at the right height it adds
energy but pitches down, trading height for speed instead of climbing.
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.eval import scored_burn_in
from target_gym.plane.experts_tecs import AIRLINER, TECSParams, TECSState, tecs_step

P = TECSParams(climb_max=10.0, sink_min=3.0, sink_max=10.0, thr_trim=0.4)


def _step(h=5000.0, h_dot=0.0, v=230.0, h_dem=5000.0, v_dem=230.0):
    state = TECSState(h_ref=h_dem, v_ref=v_dem, v_prev=v, started=1.0)
    thr, pitch, _ = tecs_step(P, state, h, h_dot, v, h_dem, v_dem, 0.0, np)
    return float(thr), float(pitch)


def test_on_target_it_asks_for_trim():
    thr, pitch = _step()
    assert thr == P.thr_trim
    assert pitch == P.pitch_trim


def test_short_of_height_it_adds_energy_and_pitches_up():
    thr, pitch = _step(h=4950.0)
    assert thr > P.thr_trim
    assert pitch > P.pitch_trim


def test_short_of_speed_it_adds_energy_and_pitches_down():
    thr, pitch = _step(v=220.0)
    assert thr > P.thr_trim
    assert pitch < P.pitch_trim


def test_throttle_and_pitch_stay_in_their_limits():
    thr, pitch = _step(h=0.0, v=100.0)
    assert P.thr_min <= thr <= P.thr_max
    assert -P.pitch_max <= pitch <= P.pitch_max


def test_speed_weight_zero_leaves_speed_to_the_throttle():
    """TECS_SPDWEIGHT 0 is ArduPilot's height-only pitch: short of speed at
    the right height, pitch stays at trim and only the throttle answers."""
    p = P.replace(spd_weight=0.0)
    state = TECSState(h_ref=5000.0, v_ref=230.0, v_prev=220.0, started=1.0)
    thr, pitch, _ = tecs_step(p, state, 5000.0, 0.0, 220.0, 5000.0, 230.0, 0.0, np)
    assert float(pitch) == p.pitch_trim
    assert float(thr) > p.thr_trim


# The contract for the 2D accuracy test, set before measuring: the
# proposal's sanity target, small autopilots holding altitude within 1-2 m in
# flight tests (docs/proposals/scoring-and-experts.md). plane_energy's target
# jumps between rungs, so it is held to that once each rung can have been
# captured: the step at the airliner's declared climb or sink limit
# (``AIRLINER.climb_max``, ``sink_max``), then five TECS time constants.
ALTITUDE_M = 2.0


def _captured(targets, burn_in):
    """Indices after the burn-in at which every rung has had its capture
    time."""
    held = np.zeros(len(targets), bool)
    held[burn_in:] = True
    for i in np.flatnonzero(np.abs(np.diff(targets)) > 50.0) + 1:
        step = targets[i] - targets[i - 1]
        rate = AIRLINER.climb_max if step > 0 else AIRLINER.sink_max
        held[i : i + int(np.ceil(abs(step) / rate + 5 * AIRLINER.time_const))] = False
    return held


def _finding(reason):
    """A held-out case the expert misses, reported to the user 2026-10-27
    and kept as an expected failure on their decision. Strict, so a fix
    shows up as XPASS and the mark has to come off."""
    return pytest.mark.xfail(strict=True, reason=f"finding: {reason}")


_MISSES = {
    ("plane_sine", 200): _finding("plane_sine altitude 2.19 m on seed 200"),
    ("plane_energy", 202): _finding("plane_energy altitude 2.03 m on seed 202"),
}


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,seed",
    [
        pytest.param(n, s, marks=_MISSES[(n, s)]) if (n, s) in _MISSES else (n, s)
        for n in ("plane", "plane_sine", "plane_energy")
        for s in (200, 201, 202)
    ],
)
def test_the_2d_expert_holds_its_altitude(name, seed):
    """The 2D aircraft's expert is TECS over the pitch loop, the 3D expert's
    longitudinal half. On held-out seeds it holds the commanded altitude
    within what autopilots achieve in flight."""
    env, params = tg.make(name)
    pid = tg.make_expert(name).host()
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    seen = []
    for _ in range(int(params.max_steps_in_episode)):
        key, sub = jax.random.split(key)
        obs, state, _, _, info = step(sub, state, pid(np.asarray(obs)), params)
        assert not bool(info["tripped"])
        seen.append(np.asarray(obs))
    seen = np.array(seen)
    held = _captured(seen[:, 6], scored_burn_in(name, params))
    assert np.abs(seen[held, 6] - seen[held, 1]).mean() <= ALTITUDE_M
