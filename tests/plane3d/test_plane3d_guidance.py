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

import target_gym as tg
from target_gym import registry
from target_gym.eval import scored_burn_in
from target_gym.plane3d import experts as X

TASKS = ("plane3d_heading", "plane3d_circle", "plane3d_racetrack", "plane3d_figure8")
HELD_OUT = (200, 201, 202)


def _on_target(name):
    """A first observation moved onto the task's target, along its track and
    at the bank the path needs (level, but for the circle's steady turn and
    the figure-8's lobe tip), so nothing saturates."""
    spec = registry.get(name)
    obs, _ = spec.make_env().reset_env(jax.random.PRNGKey(0), spec.make_test_params())
    o = np.array(obs)
    track = np.arctan2(o[1], o[0])
    if name == "plane3d_figure8":
        # It starts at a lobe tip, where the lemniscate's curvature is 3 / a,
        # turning toward increasing heading: as much of it as the 45 deg
        # bank limit allows.
        o[12:15], o[15] = 0.0, track
        o[6] = min(np.arctan((o[0] ** 2 + o[1] ** 2) * 3.0 / (9.81 * o[11])), np.pi / 4)
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


@pytest.mark.parametrize("name", TASKS)
def test_the_bank_loop_damps_the_roll_rate(name):
    """The aileron commands a roll rate, so a roll already under way must be
    met with opposite aileron, whatever the guidance asks for."""
    obs = _on_target(name)
    ailerons = []
    for rate in (-0.1, 0.1):
        o = obs.copy()
        o[7] = rate
        ailerons.append(float(tg.make_expert(name).host()(o)[2]))
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


def _on_the_figure8(tau, direction, a=10_000.0, z_mean=5_000.0, speed=160.0):
    """An observation on the scored figure-8, at its parameter ``tau``, flying
    along it (``direction`` +1 with tau, -1 against), at the bank its
    curvature needs. The pattern is the env's (PHYSICS.md): Bernoulli's
    lemniscate of half-width ``a``, rotated, its altitude rising and falling
    ``figure8_altitude_amplitude`` with sin(tau).

    The curvature is the classical result for the lemniscate, 3 r / a^2 at a
    distance r from its centre, signed by the way the path turns, which is
    read off the curve's own derivatives."""
    dz = registry.get("plane3d_figure8").make_test_params().figure8_altitude_amplitude
    orientation = 0.2

    def xy(t):
        x, y = a * np.cos(t), a * np.sin(t) * np.cos(t)
        x, y = x / (1.0 + np.sin(t) ** 2), y / (1.0 + np.sin(t) ** 2)
        c, s = np.cos(orientation), np.sin(orientation)
        return np.array([c * x - s * y, s * x + c * y])

    h = 1e-4
    d1 = direction * (xy(tau + h) - xy(tau - h)) / (2 * h)
    d2 = (xy(tau + h) - 2 * xy(tau) + xy(tau - h)) / h**2
    turn = np.sign(d1[0] * d2[1] - d1[1] * d2[0])
    curvature = turn * 3.0 * np.hypot(*xy(tau)) / a**2
    track = np.arctan2(d1[1], d1[0])

    o = np.zeros(19)
    o[0:2] = speed * np.cos(track), speed * np.sin(track)
    o[2] = z_mean + dz * np.sin(tau)
    o[6] = np.arctan(speed**2 * curvature / 9.81)
    o[9], o[10], o[11], o[15] = track, z_mean, a, track
    return o


@pytest.mark.parametrize("direction", (1, -1))
@pytest.mark.parametrize("tau", (0.0, 0.8, 2.6, np.pi, 4.4))
def test_the_figure8_expert_banks_for_the_curvature_on_the_path(tau, direction):
    """On the figure-8 and along it, at the bank its curvature needs, the
    expert holds that bank: it reads the curvature, which the observation
    does not carry, from where it is on the pattern. One observation a little
    earlier along the path comes first, as any second of flight would. The
    look-ahead, which banks ahead of the curvature for the roll loop's lag,
    is turned off so the bank is the curvature's own."""
    params = X.make_plane3d_figure8_pid_controller().params.replace(curvature_lead=0.0)
    state = X.Plane3DGuidanceState()
    _, state = X.plane3d_figure8_step(
        params, state, _on_the_figure8(tau - 0.02 * direction, direction), np
    )
    action, _ = X.plane3d_figure8_step(
        params, state, _on_the_figure8(tau, direction), np
    )
    assert action[2] == pytest.approx(0.0, abs=1e-4)


# The contract for the accuracy tests, set before measuring:
# - the proposal's sanity targets (docs/proposals/scoring-and-experts.md):
#   small autopilots in flight tests hold altitude within 1-2 m and, with L1
#   guidance, a path within 2-3 m;
# - heading within the AHRS resolution the task declares
#   (``heading_precision_floor``, 0.5 deg);
# - the figure-8's path target applies where its lobes are flyable at the
#   expert's declared envelope (``_lobes_flyable``); elsewhere it must still
#   hold its height and never trip.
ALTITUDE_M = 2.0
PATH_M = 3.0


def _finding(reason):
    """A held-out case the expert misses, reported to the user 2026-10-27
    and kept as an expected failure on their decision. Strict, so a fix
    shows up as XPASS and the mark has to come off."""
    return pytest.mark.xfail(strict=True, reason=f"finding: {reason}")


def _fly(name, seed):
    """One held-out episode of the shipped expert on the scored
    configuration: ``(first observation, observations after the scored
    burn-in, tripped, params)``."""
    env, params = tg.make(name)
    pid = tg.make_expert(name).host()
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    first, seen, tripped = np.asarray(obs), [], False
    for _ in range(int(params.max_steps_in_episode)):
        key, sub = jax.random.split(key)
        obs, state, _, _, info = step(sub, state, pid(np.asarray(obs)), params)
        tripped = tripped or bool(info["tripped"])
        seen.append(np.asarray(obs))
    return first, np.array(seen)[scored_burn_in(name, params) :], tripped, params


def _lobes_flyable(params, first):
    """Whether the figure-8's tightest turn, radius a / 3 at the lobe tips,
    is inside the expert's declared envelope: 45 deg of bank at the slowest
    speed stall prevention allows for it, v_min / sqrt(cos 45), turns on a
    radius v^2 / (g tan 45) = v_min^2 / (g cos 45). v_min is 104 m/s
    equivalent airspeed as a true airspeed at the lobe's highest point, by
    the standard atmosphere's density ratio."""
    top = first[10] + params.figure8_altitude_amplitude
    v_min = 104.0 / np.sqrt((1.0 - 2.2558e-5 * top) ** 4.2559)
    return v_min**2 / (9.81 * np.cos(np.pi / 4)) <= first[11] / 3.0


def _wrapped(angle):
    return np.abs(np.angle(np.exp(1j * angle)))


@pytest.mark.slow
@pytest.mark.parametrize("seed", HELD_OUT)
def test_the_heading_expert_holds_altitude_and_heading(seed):
    _, seen, tripped, params = _fly("plane3d_heading", seed)
    assert not tripped
    assert np.abs(seen[:, 10] - seen[:, 2]).mean() <= ALTITUDE_M
    assert _wrapped(seen[:, 11] - seen[:, 9]).mean() <= params.heading_precision_floor


@pytest.mark.slow
@pytest.mark.parametrize("seed", HELD_OUT)
def test_the_circle_expert_holds_altitude_and_path(seed):
    _, seen, tripped, _ = _fly("plane3d_circle", seed)
    assert not tripped
    assert np.abs(seen[:, 10] - seen[:, 2]).mean() <= ALTITUDE_M
    radial = np.abs(np.hypot(seen[:, 11], seen[:, 12]) - seen[:, 13])
    assert radial.mean() <= PATH_M


@pytest.mark.slow
@pytest.mark.parametrize(
    "seed",
    [
        200,
        pytest.param(201, marks=_finding("racetrack altitude 2.007 m on seed 201")),
        202,
    ],
)
def test_the_racetrack_expert_holds_altitude_and_path(seed):
    _, seen, tripped, _ = _fly("plane3d_racetrack", seed)
    assert not tripped
    assert np.abs(seen[:, 10] - seen[:, 2]).mean() <= ALTITUDE_M
    assert np.abs(seen[:, 18]).mean() <= PATH_M


@pytest.mark.slow
@pytest.mark.parametrize("seed", HELD_OUT)
def test_the_figure8_expert_keeps_its_height_on_every_pattern(seed):
    """Stall prevention gives up path, not height, where the lobes are too
    tight for the airliner."""
    _, seen, tripped, _ = _fly("plane3d_figure8", seed)
    assert not tripped
    assert np.abs(seen[:, 14]).mean() <= ALTITUDE_M


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(200, 210))
def test_the_figure8_expert_holds_its_path_where_the_lobes_are_flyable(seed):
    first, seen, _, params = _fly("plane3d_figure8", seed)
    if not _lobes_flyable(params, first):
        pytest.skip("the lobe tips are tighter than the expert's envelope")
    assert np.hypot(seen[:, 12], seen[:, 13]).mean() <= PATH_M
