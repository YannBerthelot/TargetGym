"""Total energy control: throttle closes the energy error, pitch its balance.

TECS (Lambregts) is the longitudinal autopilot PX4 and ArduPilot fly. These
tests pin its defining behaviour: on target it asks for trim; short of height
it adds energy and pitches up; short of speed at the right height it adds
energy but pitches down, trading height for speed instead of climbing.
"""

import jax
import numpy as np
import pytest

from target_gym import registry
from target_gym.eval import scored_burn_in
from target_gym.plane.experts import make_plane_pid_controller
from target_gym.plane.experts_tecs import TECSParams, TECSState, tecs_step

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


def _held(name, targets, burn_in):
    """The steps to score: after the burn-in, or on plane_energy, whose
    target jumps 360-720 m every 150 s, the last 25 s of each cycle. Its climbs
    at the airliner's 8 m/s fill most of a cycle, and are a climb rate, not
    an accuracy."""
    if name != "plane_energy":
        return np.arange(burn_in, len(targets))
    jumps = np.flatnonzero(np.abs(np.diff(targets)) > 50.0) + 1
    starts, ends = np.append(0, jumps), np.append(jumps, len(targets))
    cycles = [(a, b) for a, b in zip(starts, ends) if b - a >= 30]
    return np.concatenate([np.arange(b - 25, b) for a, b in cycles])


@pytest.mark.slow
@pytest.mark.parametrize("name", ["plane", "plane_sine", "plane_energy"])
def test_the_2d_expert_holds_its_altitude(name):
    """The 2D aircraft's expert is TECS over the pitch loop, the 3D expert's
    longitudinal half. On held-out seeds it holds the commanded altitude
    within what autopilots achieve in flight."""
    spec = registry.get(name)
    env, params = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    burn_in = scored_burn_in(name, params)
    for seed in (200, 201, 202):
        pid = make_plane_pid_controller().host()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        seen = []
        for _ in range(int(params.max_steps_in_episode)):
            obs, state, _, _, info = step(key, state, pid(np.asarray(obs)), params)
            assert not bool(info["tripped"])
            seen.append(np.asarray(obs))
        seen = np.array(seen)
        errors = np.abs(seen[:, 6] - seen[:, 1])[_held(name, seen[:, 6], burn_in)]
        assert errors.mean() < 2.5, (seed, errors.mean())
