"""The 3D aircraft experts: an autopilot's loops under the field's guidance laws.

Each task's expert is what a fixed-wing autopilot flies: a damped bank loop
under a course hold (heading) or L1 guidance (circle, racetrack, figure-8).
These tests pin the sign conventions, one of which once went wrong (the
heading and circle cascades fed roll rate back with the wrong sign), and how
closely each expert holds on held-out seeds, which its tuning never saw.
"""

import jax
import numpy as np
import pytest

from target_gym import registry
from target_gym.eval import scored_burn_in
from target_gym.plane3d import experts as X

FACTORIES = {
    "plane3d_heading": X.make_plane3d_heading_pid_controller,
    "plane3d_circle": X.make_plane3d_circle_pid_controller,
    "plane3d_racetrack": X.make_plane3d_racetrack_pid_controller,
    "plane3d_figure8": X.make_plane3d_figure8_pid_controller,
}
HELD_OUT = (200, 201, 202)


def _on_target(name):
    """A first observation moved onto the task's target, along its track and
    at the bank the path needs (level, but for the circle's steady turn), so
    nothing saturates."""
    spec = registry.get(name)
    obs, _ = spec.make_env().reset_env(jax.random.PRNGKey(0), spec.make_test_params())
    o = np.array(obs)
    track = np.arctan2(o[1], o[0])
    if name == "plane3d_figure8":
        o[12:15], o[15] = 0.0, track
        return o
    o[10] = o[2]
    if name == "plane3d_heading":
        o[11] = o[9]
    elif name == "plane3d_circle":
        o[11:13] = o[13] * np.cos(track - np.pi / 2), o[13] * np.sin(track - np.pi / 2)
        o[6] = np.arctan((o[0] ** 2 + o[1] ** 2) / (9.81 * o[13]))
    else:
        o[18], o[19], o[20] = 0.0, track, 0.0
    return o


@pytest.mark.parametrize("name", FACTORIES)
def test_the_bank_loop_damps_the_roll_rate(name):
    """The aileron commands a roll rate, so a roll already under way must be
    met with opposite aileron, whatever the guidance asks for."""
    obs = _on_target(name)
    ailerons = []
    for rate in (-0.1, 0.1):
        o = obs.copy()
        o[7] = rate
        ailerons.append(float(FACTORIES[name]().host()(o)[2]))
    assert ailerons[1] < ailerons[0]


def test_l1_flies_the_curvature_on_the_path():
    """On the path and along it, L1 asks for exactly the turn the path makes;
    off it to the side a positive bank turns toward, it steers back."""
    p = X.Plane3DGuidanceParams()
    obs = np.zeros(21, np.float32)
    obs[0] = 200.0
    on_path = X.l1_lateral_accel(p, 0.0, 0.0, 1.0e-4, obs, np)
    assert on_path == pytest.approx(200.0**2 * 1.0e-4, rel=1e-4)
    assert X.l1_lateral_accel(p, 50.0, 0.0, 0.0, obs, np) < 0.0
    assert X.l1_lateral_accel(p, 0.0, 0.1, 0.0, obs, np) > 0.0


def _errors(name, obs):
    """Altitude, and heading or path, errors from the task's observation."""
    if name == "plane3d_figure8":
        return np.abs(obs[:, 14]), np.hypot(obs[:, 12], obs[:, 13])
    altitude = np.abs(obs[:, 10] - obs[:, 2])
    if name == "plane3d_heading":
        return altitude, np.abs(np.angle(np.exp(1j * (obs[:, 11] - obs[:, 9]))))
    if name == "plane3d_circle":
        return altitude, np.abs(np.hypot(obs[:, 11], obs[:, 12]) - obs[:, 13])
    return altitude, np.abs(obs[:, 18])


#: Mean altitude (m) and heading (rad) or path (m) limits after the burn-in.
#: The figure-8's are looser for a physical reason: stall prevention keeps the
#: airliner fast enough that its tightest turn at 45 deg is wider than the
#: smaller lobes' tips (radius a/3), so an honest autopilot cuts them.
LIMITS = {
    "plane3d_heading": (2.5, np.deg2rad(1.0)),
    "plane3d_circle": (2.5, 5.0),
    "plane3d_racetrack": (2.5, 5.0),
    "plane3d_figure8": (15.0, 60.0),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", FACTORIES)
def test_the_expert_holds_like_an_autopilot(name):
    """After the scored burn-in, on held-out seeds, each expert holds about
    what fixed-wing autopilots achieve in flight tests (altitude within 1-2 m,
    a path within 2-3 m with L1 guidance), here in moderate turbulence and at
    one control step a second."""
    spec = registry.get(name)
    env, params = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    burn_in = scored_burn_in(name, params)
    alt_limit, limit = LIMITS[name]
    for seed in HELD_OUT:
        pid = FACTORIES[name]().host()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        seen = []
        for _ in range(int(params.max_steps_in_episode)):
            obs, state, _, _, info = step(key, state, pid(np.asarray(obs)), params)
            assert not bool(info["tripped"])
            seen.append(np.asarray(obs))
        altitude, lateral = _errors(name, np.array(seen)[burn_in:])
        assert altitude.mean() < alt_limit, (seed, altitude.mean())
        assert lateral.mean() < limit, (seed, lateral.mean())
