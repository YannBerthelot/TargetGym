"""Behavioural flight-scenario tests for the 2D plane.

These assert things a pilot would recognise -- adding power climbs, pulling
back raises the nose, a stalled wing sinks, an engine-out aircraft glides
rather than falling -- rather than numerical values. They complement
``test_plane_physics.py``, which validates coefficients and figures of merit
against reference data.

Two reasons this layer matters:

* It is **gain-independent**. Every scenario uses open-loop control inputs, so
  these tests stay valid across controller re-tunes, unlike the closed-loop
  expert tests.
* It catches *coupled* errors. A sign flip or a broken moment balance can leave
  every coefficient individually correct while making the aircraft unflyable.

Sign conventions (from ``Airplane2D.step_env``):
    action = [power_raw, stick_raw], both in [-1, 1]
    throttle = (power_raw + 1) / 2      in [0, 1]
    elevator = stick_raw * 15 degrees   positive = nose up
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane.env import PlaneParams, PlaneState
from target_gym.plane.env_jax import Airplane2D

KEY = jax.random.PRNGKey(0)

# Throttle that roughly holds altitude at 5 000 m / 230 m/s (found by sweep).
TRIM_POWER_RAW = -0.65
CRUISE_ALT = 5_000.0
CRUISE_SPEED = 230.0
TRIM_THETA_DEG = 1.85


@pytest.fixture(scope="module")
def env():
    return Airplane2D()


@pytest.fixture(scope="module")
def params():
    # Long horizon so scenarios end on physics, not on the step limit.
    return PlaneParams(max_steps_in_episode=100_000)


def make_state(
    params,
    alt=CRUISE_ALT,
    speed=CRUISE_SPEED,
    theta_deg=TRIM_THETA_DEG,
    z_dot=0.0,
    power=0.5,
):
    """Build an explicit initial state; velocity is horizontal unless z_dot given."""
    return PlaneState(
        x=0.0,
        x_dot=speed,
        z=alt,
        z_dot=z_dot,
        theta=np.deg2rad(theta_deg),
        theta_dot=0.0,
        alpha=np.deg2rad(theta_deg),
        gamma=0.0,
        m=params.initial_mass,
        power=power,
        stick=0.0,
        fuel=params.initial_fuel_quantity,
        time=0,
        target_altitude=alt,
    )


def fly(env, params, power_raw, stick_raw, steps, state=None):
    """Roll out a constant control input. Returns a dict of trajectories."""
    state = state if state is not None else make_state(params)
    action = jnp.array([power_raw, stick_raw])
    z, v, theta, alpha, z_dot = [], [], [], [], []
    _jstep = jax.jit(env.step_env)
    for _ in range(steps):
        _, state, _, terminated, _ = _jstep(KEY, state, action, params)
        z.append(float(state.z))
        v.append(float(state.x_dot))
        theta.append(float(np.rad2deg(state.theta)))
        alpha.append(float(np.rad2deg(state.alpha)))
        z_dot.append(float(state.z_dot))
        if bool(terminated):
            break
    return {
        "z": np.array(z),
        "v": np.array(v),
        "theta": np.array(theta),
        "alpha": np.array(alpha),
        "z_dot": np.array(z_dot),
        "terminated": bool(terminated),
    }


# ---------------------------------------------------------------------------
# Thrust -> climb
# ---------------------------------------------------------------------------


def test_more_power_produces_more_climb(env, params):
    """Climb performance must increase monotonically with throttle."""
    climbs = [
        fly(env, params, power_raw=p, stick_raw=0.0, steps=250)["z"][-1] - CRUISE_ALT
        for p in (-1.0, 0.2, 1.0)
    ]
    assert climbs[0] < climbs[1] < climbs[2], f"non-monotonic climb: {climbs}"


def test_idle_power_causes_descent(env, params):
    """With the engines at idle the aircraft must lose altitude."""
    traj = fly(env, params, power_raw=-1.0, stick_raw=0.0, steps=250)
    assert traj["z"][-1] < CRUISE_ALT - 100.0


def test_full_power_causes_climb(env, params):
    traj = fly(env, params, power_raw=1.0, stick_raw=0.0, steps=250)
    assert traj["z"][-1] > CRUISE_ALT + 100.0


def test_climb_costs_airspeed_at_fixed_power(env, params):
    """Trading potential for kinetic energy: a climb at fixed thrust bleeds speed."""
    traj = fly(env, params, power_raw=0.2, stick_raw=0.0, steps=250)
    assert traj["z"][-1] > CRUISE_ALT, "expected a climb"
    assert traj["v"][-1] < CRUISE_SPEED, "climbed without losing any airspeed"


# ---------------------------------------------------------------------------
# Elevator -> pitch
# ---------------------------------------------------------------------------


def test_positive_stick_pitches_nose_up(env, params):
    traj = fly(env, params, power_raw=TRIM_POWER_RAW, stick_raw=0.4, steps=15)
    assert traj["theta"][-1] > TRIM_THETA_DEG


def test_negative_stick_pitches_nose_down(env, params):
    traj = fly(env, params, power_raw=TRIM_POWER_RAW, stick_raw=-0.4, steps=15)
    assert traj["theta"][-1] < TRIM_THETA_DEG


def test_elevator_response_is_monotonic_in_deflection(env, params):
    """More up-elevator must give more nose-up pitch, at equal time."""
    pitches = [
        fly(env, params, power_raw=TRIM_POWER_RAW, stick_raw=s, steps=12)["theta"][-1]
        for s in (-0.6, -0.2, 0.2, 0.6)
    ]
    assert all(a < b for a, b in zip(pitches, pitches[1:])), pitches


def test_neutral_stick_does_not_produce_sustained_pitch_rate(env, params):
    """At trim attitude with neutral elevator the aircraft must not tumble."""
    traj = fly(env, params, power_raw=TRIM_POWER_RAW, stick_raw=0.0, steps=200)
    assert np.abs(traj["theta"]).max() < 45.0


# ---------------------------------------------------------------------------
# Stall
# ---------------------------------------------------------------------------


def _vertical_accel(params, theta_deg, speed=CRUISE_SPEED, alt=CRUISE_ALT):
    """Vertical acceleration with velocity horizontal, so alpha == theta."""
    from target_gym.plane.dynamics import compute_acceleration

    velocities = jnp.array([speed, 0.0, 0.0])
    positions = jnp.array([0.0, alt, np.deg2rad(theta_deg)])
    accel, _ = compute_acceleration(
        velocities, positions, action=(120_000.0, 0.0), params=params
    )
    return float(accel[1])


def test_lift_supports_aircraft_below_stall(params):
    """Just below the stall angle the wing must be able to hold the aircraft up."""
    assert _vertical_accel(params, params.aoa_stall - 2.0) > 0.0


def test_stalled_wing_cannot_hold_the_aircraft_up(params):
    """Past the stall angle lift collapses and the aircraft accelerates downward.

    This is the defining behaviour of a stall: more angle of attack, *less*
    lift. Before the D1/D2 fix the lift curve peaked at 0.70 and this margin
    was far weaker.
    """
    pre_stall = _vertical_accel(params, params.aoa_stall - 2.0)
    post_stall = _vertical_accel(params, params.aoa_stall + 8.0)
    assert post_stall < 0.0, "stalled wing still producing net upward force"
    assert post_stall < pre_stall


def test_stall_recovery_by_lowering_the_nose(params):
    """Reducing angle of attack after a stall must restore lift."""
    stalled = _vertical_accel(params, params.aoa_stall + 8.0)
    recovered = _vertical_accel(params, params.aoa_stall - 5.0)
    assert recovered > stalled
    assert recovered > 0.0, "lift did not recover after unloading the wing"


def test_deep_stall_sinks_in_a_rollout(env, params):
    """Started beyond the stall angle, the aircraft must lose altitude."""
    state = make_state(params, theta_deg=params.aoa_stall + 8.0, speed=170.0)
    traj = fly(env, params, power_raw=-0.5, stick_raw=0.0, steps=60, state=state)
    assert traj["z"][-1] < CRUISE_ALT


# ---------------------------------------------------------------------------
# Glide (engine-out)
# ---------------------------------------------------------------------------


def test_engine_out_produces_a_steady_glide(env, params):
    """Idle power must give a bounded, sustained descent -- not a freefall."""
    traj = fly(env, params, power_raw=-1.0, stick_raw=0.0, steps=400)
    sink_rate = -np.diff(traj["z"])
    assert traj["z"][-1] < CRUISE_ALT, "did not descend with engines idle"
    # A gliding airliner sinks at metres per second, not tens of metres per second.
    assert 0.0 < np.mean(sink_rate) < 30.0, f"mean sink {np.mean(sink_rate):.1f} m/s"
    assert np.all(np.isfinite(traj["z"]))


def test_glide_is_shallower_than_ballistic_fall(env, params):
    """The wing must generate lift: descent far slower than free fall."""
    traj = fly(env, params, power_raw=-1.0, stick_raw=0.0, steps=200)
    dropped = CRUISE_ALT - traj["z"][-1]
    free_fall = 0.5 * params.gravity * (len(traj["z"]) * params.delta_t) ** 2
    assert dropped < 0.1 * free_fall


# ---------------------------------------------------------------------------
# Robustness / envelope
# ---------------------------------------------------------------------------


def test_flying_into_the_ground_terminates_the_episode(env, params):
    """Descending to ground level must end the episode.

    Note ``clamp_altitude`` in ``plane.dynamics`` is dead code -- defined,
    imported by ``plane3d.dynamics``, never called. Altitude is bounded purely
    by the ``z <= min_alt`` termination, so the *terminal* state may undershoot
    by up to one timestep of descent. Every earlier state must be above ground.
    """
    state = make_state(params, alt=250.0, speed=200.0, theta_deg=-10.0)
    traj = fly(env, params, power_raw=-1.0, stick_raw=-0.5, steps=200, state=state)
    assert traj["terminated"], "flew into the ground without terminating"
    assert np.all(traj["z"][:-1] >= 0.0), "went underground before terminating"
    # The terminal undershoot is bounded by one step of vertical travel.
    one_step = abs(float(traj["z_dot"][-1])) * params.delta_t
    assert traj["z"][-1] > -(one_step + 1.0)


@pytest.mark.parametrize(
    "power_raw,stick_raw",
    [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 0.0)],
)
def test_extreme_inputs_stay_numerically_finite(env, params, power_raw, stick_raw):
    """Saturated controls may depart controlled flight, but must not produce NaN."""
    traj = fly(env, params, power_raw=power_raw, stick_raw=stick_raw, steps=120)
    for name in ("z", "v", "theta", "alpha", "z_dot"):
        assert np.all(np.isfinite(traj[name])), f"non-finite {name}"


def test_level_flight_oscillation_is_bounded(env, params):
    """Near trim the aircraft phugoids, but the oscillation must not diverge.

    The phugoid (a slow exchange of altitude and airspeed, period ~100 s here)
    means level flight is never perfectly static. What matters is that the
    swings stay bounded rather than growing.
    """
    traj = fly(env, params, power_raw=TRIM_POWER_RAW, stick_raw=0.0, steps=600)
    excursion = np.abs(traj["z"] - CRUISE_ALT)
    first_half = excursion[: len(excursion) // 2].max()
    second_half = excursion[len(excursion) // 2 :].max()
    assert excursion.max() < 2_000.0, "phugoid amplitude implausibly large"
    assert second_half < 3.0 * first_half + 100.0, "oscillation appears to diverge"
