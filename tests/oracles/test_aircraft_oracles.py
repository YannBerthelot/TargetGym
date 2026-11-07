"""The aircraft oracles on the shared solver.

Contract, from docs/proposals/oracle-smoothing-protocol.md (T1r, T3, K2, K5) and
the plant's own limits (plane/PHYSICS.md):

- T3: the planner's rows read what the env's trip reads: the load factor at each
  inner sub-step, EAS and Mach on the state reached, in the wind it is left in;
- T1r: a rate limit inside the step binds a hold only if the surface cannot cross
  its travel within it; at 30 deg/s and one 1 s step it can, so no row is needed;
- K2: a smoothed clip sits within w ln 2 of the exact one and equals it at width 0;
  the attitude integral, frozen while the surface saturates, follows the clip's
  slope;
- a solve from a reset converges and plans inside every backed-off limit;
- S4: over a turn the heading error is the shorter way round, exact away from the
  half-turn, and a track past arctan2's cut stays on the plan's branch;
- T2p: at its best parameter a path's planned distance is the env's, to the
  sagitta of the env's own chords; the curves are the env's;
- X: a patrol's lead, gust and wake do not depend on the follower, so one run ahead
  of the solve gives them, and the slot terms are the env's.

Exact replay, the tie-break, G12's count and the no-expert imports for these tasks
are in ``test_process_oracles.py``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import enable_x64, float64, softplus_clip
from target_gym.patrol import env as patrol_env
from target_gym.patrol.oracle import lead_future, slot_terms
from target_gym.plane import oracle_model as om
from target_gym.plane.dynamics import PITCH_DEMAND_MAX, STICK_TRAVEL, attitude_axis
from target_gym.plane3d import env as plane3d_env
from target_gym.plane3d import oracle as p3
from target_gym.registry import REGISTRY

N_SITES = len(om.SITE_NAMES)


def _setup(name="plane", **overrides):
    spec = REGISTRY[name]
    env = spec.make_env()
    params = plan_params(spec, spec.make_test_params(**overrides))
    return spec, env, params


def _model_outputs(env, p, state, u):
    """One planned step from ``state``, its widths 0: the state and the outputs."""
    from target_gym.plane.oracle import plane_model

    m = plane_model(env, p)
    d = {"task": m.data(state), "widths": jnp.zeros(N_SITES), "end": 10**9}
    x1, _, y, _, _ = m.step(d, m.state_x(state), u, 0)
    return m, x1, y


def test_the_load_rows_hold_the_load_the_trip_reads():
    """A full pull from level flight, the load limits opened so the step does not
    restart the plant. The env keeps the sub-step's load furthest from 1 g, so of
    the four rows the planner bounds, that one is the env's."""
    _, env, p = _setup()
    with enable_x64():
        p = float64(p.replace(load_factor_max=100.0, load_factor_min=-100.0))
        _, state = env.reset_env(jax.random.PRNGKey(100), p)
        state = float64(state)
        u = jnp.array([1.0, 1.0])
        s1 = env.step_env(jax.random.PRNGKey(0), state, u, p)[1]
        _, _, y = _model_outputs(env, p, state, u)
    loads = np.asarray(y[2:6])
    furthest = loads[np.argmax(np.abs(loads - 1.0))]
    assert float(s1.load_factor) == pytest.approx(furthest, rel=1e-5)
    assert np.ptp(loads) > 0.01, "the sub-steps must differ for this to test anything"


@pytest.mark.parametrize("excess, trips", [(-1.0, False), (1.0, True)])
def test_the_eas_row_is_the_vmo_trip(excess, trips):
    """Level flight at 5000 m: EAS = V sqrt(rho / rho_0) with the ISA density.
    Set 1 m/s either side of VMO, the env trips exactly when the row passes it."""
    spec, env, p = _setup()
    rho_0 = float(p.air_density_at_sea_level)
    with enable_x64():
        p = float64(p.replace(load_factor_max=100.0, load_factor_min=-100.0))
        _, state = env.reset_env(jax.random.PRNGKey(100), p)
        rho = 0.7364  # ISA at 5000 m
        v = (float(p.vmo_eas) + excess) * math.sqrt(rho_0 / rho)
        state = float64(state.replace(z=5000.0, x_dot=v, z_dot=0.0, theta=0.0))
        u = jnp.array([0.0, 0.0])
        s1 = env.step_env(jax.random.PRNGKey(0), state, u, p)
        _, _, y = _model_outputs(env, p, state, u)
    assert bool(s1[4]["tripped"]) == trips
    assert (float(y[6]) > float(p.vmo_eas)) == trips


def test_the_stick_crosses_its_travel_within_one_step():
    """T1r's reach over a 1 s hold: 30 deg/s for 1 s is the whole 30 deg of
    travel, so no demand lies past it. From one stop, a full demand reaches the
    other within the step, short by what the 60 ms lag leaves of the last 7.5 deg
    sub-step, under 0.2 deg. The load limits are opened, or the pull would trip
    and restart the plant."""
    _, env, p = _setup()
    with enable_x64():
        p = float64(p.replace(load_factor_max=100.0, load_factor_min=-100.0))
        _, state = env.reset_env(jax.random.PRNGKey(100), p)
        state = float64(
            state.replace(stick=-STICK_TRAVEL, theta=-PITCH_DEMAND_MAX, pitch_integ=0.0)
        )
        s1 = env.step_env(jax.random.PRNGKey(0), state, jnp.array([0.0, 1.0]), p)[1]
    assert np.rad2deg(float(s1.stick)) == pytest.approx(15.0, abs=0.2)
    reach = om.surface_reach(
        float(p.surface_rate_limit), float(p.stick_response_rate), 1.0
    )
    assert reach >= 2.0 * STICK_TRAVEL


@pytest.mark.parametrize("z", np.linspace(-3.0, 3.0, 13))
def test_the_soft_clip_is_within_w_ln_2_of_the_clip(z):
    """CM's Prop. 2.2 for a single bound: 0 <= exact - smooth <= w ln 2 near the
    upper bound, and mirrored at the lower, with the other bound's own term 2
    apart, so e^(-2 / w) too small to see at w = 0.02. Float64: at z = 1 the gap
    is the bound itself."""
    w = 0.02
    with enable_x64():
        smooth = float(softplus_clip(jnp.float64(z), -1.0, 1.0, jnp.float64(w)))
    assert abs(smooth - float(np.clip(z, -1.0, 1.0))) <= w * math.log(2.0) + 1e-12


@pytest.mark.parametrize(
    "error, integral, grows", [(0.05, 0.125, True), (0.5, 0.125, False)]
)
def test_the_attitude_loop_matches_the_plants_at_width_zero(error, integral, grows):
    """Inside the clip the integral grows by Ki s^2 e dt; saturated it stands
    still (AP_RollController's anti-windup), and at width 0 the planner's law is
    the plant's on both sides."""
    args = (6.3662, 0.2546, 0.1273, error, 0.0, integral, 1.0, 0.25)
    exact = attitude_axis(*args)
    planned = om.attitude(*args, 0.0, 0.0)
    assert float(planned[0]) == pytest.approx(float(exact[0]))
    assert float(planned[1]) == pytest.approx(float(exact[1]))
    assert (float(exact[1]) > integral) == grows


def test_the_attitude_integral_stops_growing_past_the_soft_clip():
    """At 3 widths past the clip the logistic has closed to e^-3, so at most 5 %
    of the integral's growth inside it passes."""
    w = 0.02
    error = (1.0 + 3.0 * w) / 6.3662  # output = 6.3662 e, at the scaler's 1
    growth = 0.2546 * error * 0.25  # Ki e dt
    _, integral = om.attitude(
        6.3662, 0.2546, 0.1273, error, 0.0, 0.0, 1.0, 0.25, w, 0.0
    )
    assert 0.0 < float(integral) <= 0.05 * growth


@pytest.mark.medium
def test_a_plan_from_a_reset_converges_inside_its_limits():
    """Seed 100's reset, a 600 m capture at most, at the task's T_c. The cold solve
    converges inside the cap, and every planned load, EAS and Mach stays inside
    its limit, which the capture pushes against, the replay tripping nothing."""
    spec, env, params = _setup()
    oracle = spec.make_mpc(env, params)
    _, state = env.reset_env(jax.random.PRNGKey(100), params)
    oracle.step(None, state)
    report = oracle.solver_report()
    assert report["solver_failures"] == 0
    assert report["solver_capped"] == 0
    assert report["solver_max_error"] <= 1e-8
    assert report["solver_replay_trips"] == 0
    assert report["solver_guard_flags"] == 0


@pytest.mark.parametrize("track", [-3.1, -1.0, 0.0, 2.0, 3.1])
def test_polar_velocity_round_trips(track):
    x = jnp.zeros(16).at[3].set(200.0 * np.cos(track)).at[4].set(200.0 * np.sin(track))
    with enable_x64():
        x = x.astype(jnp.float64)
        polar = om.to_polar(x)
        back = om.to_cartesian(polar)
    assert float(polar[3]) == pytest.approx(200.0)
    assert float(polar[4]) == pytest.approx(track)
    np.testing.assert_allclose(back, x, atol=1e-5)


def test_the_track_keeps_its_branch_across_arctan2s_cut():
    """A step's turn of 0.2 rad from 3.1 rad ends at 3.3 rad, not at -2.98."""
    with enable_x64():
        v = 200.0 * jnp.array([np.cos(3.3), np.sin(3.3)], dtype=jnp.float64)
        x1 = jnp.zeros(16, jnp.float64).at[3:5].set(v)
        assert float(om.polar_step(x1, jnp.float64(3.1))[4]) == pytest.approx(3.3)
    assert float(om.on_branch(-3.1, 3.1)) == pytest.approx(2.0 * np.pi - 3.1)
    assert float(om.on_branch(3.1, 3.2)) == pytest.approx(3.1)


@pytest.mark.parametrize("error", [0.0, 0.3, -0.3, 2.0, -2.0, 3.0, -3.0])
def test_the_turn_is_the_shorter_way_round_at_width_zero(error):
    turn = om.turn_distance(jnp.sin(error), jnp.cos(error), 0.0)
    assert float(turn) == pytest.approx(abs(error), abs=1e-5)


def test_the_turn_is_exact_near_the_target_and_rounds_only_at_the_half_turn():
    """K2's min: within w ln 2 of the exact value at the half-turn, where the two
    turns meet and the exact slope jumps, and its slope is zero there, by symmetry.
    At 0.3 rad the other turn is e^(-(2 pi - 0.6) / w), about 1e-4 of w, away."""
    w = 2.0 * np.pi / 10.0
    with enable_x64():

        def turn(e):
            return om.turn_distance(jnp.sin(e), jnp.cos(e), jnp.float64(w))

        at_half = float(turn(jnp.float64(np.pi)))
        slope = float(jax.grad(turn)(jnp.float64(np.pi)))
        near = float(turn(jnp.float64(0.3)))
    assert np.pi - w * math.log(2.0) - 1e-9 <= at_half <= np.pi
    assert abs(slope) <= 1e-9
    assert near == pytest.approx(0.3, abs=1e-3)


def test_the_stadium_is_the_racetracks_curve_by_arc_length():
    """Every point of it is at the env's distance 0, and equal parameter steps are
    equal lengths along it, one lap closing on itself."""
    spec, env, p = _setup("plane3d_racetrack")
    with enable_x64():
        p = float64(p)
        s = float64(env.reset_env(jax.random.PRNGKey(3), p)[1])
        g = (
            s.target_x,
            s.target_y,
            s.target_altitude,
            s.target_radius,
            s.target_heading,
        )
        L = float(p3.STADIUM.period(g))
        grid = jnp.linspace(0.0, L, 2001)
        pts = jax.vmap(p3.stadium_at, (None, 0))(g, grid)
        d = jax.vmap(
            lambda q: plane3d_env.distance_to_racetrack(s.replace(x=q[0], y=q[1]))
        )(pts)
    steps = np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1)
    assert float(jnp.max(d)) <= 1e-6
    assert np.ptp(steps) <= 1e-3 * L / 2000.0
    assert np.linalg.norm(np.asarray(pts[0] - pts[-1])) <= 1e-6


def _closest(curve, g, position, around, span):
    """The best parameter near ``around`` by a dense scan and two refinements."""
    for _ in range(3):
        grid = jnp.linspace(around - span, around + span, 2001)
        pts = jax.vmap(curve.at, (None, 0))(g, grid)
        around = grid[jnp.argmin(jnp.sum((pts - position) ** 2, axis=1))]
        span = span / 500.0
    return around


def test_at_its_best_parameter_the_figure8s_distance_is_the_envs():
    """T2p. A point 40 m off the curve: the joint minimum's distance equals the env's
    to the sagitta of its chords, under 1 m by the env's own comment."""
    spec, env, p = _setup("plane3d_figure8")
    with enable_x64():
        p = float64(p)
        s = float64(env.reset_env(jax.random.PRNGKey(5), p)[1])
        amplitude = jnp.asarray(p.figure8_altitude_amplitude, jnp.float64)
        g = (s.target_x, s.target_y, s.target_altitude, s.target_radius,
             s.target_heading, amplitude)  # fmt: skip
        tau = jnp.float64(0.7)
        position = p3.lemniscate_at(g, tau) + jnp.array([28.0, 20.0, 20.0])
        best = _closest(p3.LEMNISCATE, g, position, tau, 0.05)
        planned = float(jnp.linalg.norm(p3.lemniscate_at(g, best) - position))
        at = s.replace(x=position[0], y=position[1], z=position[2])
        exact = float(plane3d_env.nearest_point_on_twisted_lemniscate(at, p)[3])
    assert planned == pytest.approx(exact, abs=1.0)
    assert exact == pytest.approx(40.0, abs=15.0)


def test_a_path_reference_starts_at_the_aircraft_and_moves_with_it():
    """The racetrack's reset sits on a leg, heading along it at 200 m/s: by arc length
    the reference is the nearest point plus 200 m a step, in the direction flown."""
    spec, env, p = _setup("plane3d_racetrack")
    with enable_x64():
        p = float64(p)
        s = float64(env.reset_env(jax.random.PRNGKey(3), p)[1])
        g = (
            s.target_x,
            s.target_y,
            s.target_altitude,
            s.target_radius,
            s.target_heading,
        )
        ref = np.asarray(p3.path_reference(p3.STADIUM, g, s, 4))
        here = p3.stadium_at(g, jnp.float64(ref[0] - 200.0))
    assert np.allclose(np.diff(ref), 200.0, atol=1e-6)
    assert float(jnp.hypot(here[0] - s.x, here[1] - s.y)) <= 40.0


def test_a_patrols_future_is_the_noise_free_envs():
    """The lead, run ahead from the state, lands where the env's own steps put it,
    whatever the follower does, and the slot terms read the env's."""
    spec, env, p = _setup("patrol")
    key = jax.random.PRNGKey(0)
    with enable_x64():
        p = float64(p)
        s = float64(env.reset_env(jax.random.PRNGKey(100), p)[1])
        ahead = lead_future(env, p, 6)(s)
        for _ in range(6):
            s = env.step_env(key, s, jnp.array([0.2, 0.1, 0.3]), p)[1]
            s = jax.tree_util.tree_map(
                lambda a: a.astype(jnp.float64) if a.dtype == jnp.float32 else a, s
            )
        lead = ahead.lead[-1]
        np.testing.assert_allclose(
            lead, [s.lead.x, s.lead.y, s.lead.z, s.lead.psi], atol=1e-5
        )
        x1 = om.to_polar(om.flight_x(s.follower))
        slot = jnp.stack([s.slot_back, s.slot_right, s.slot_up])
        squared, separation = slot_terms(x1, lead, slot, p)
    assert float(jnp.sqrt(squared)) == pytest.approx(
        float(patrol_env.slot_error(s)), abs=1e-3
    )
    assert float(separation) == pytest.approx(float(patrol_env.separation(s)), abs=1e-3)
