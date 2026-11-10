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
- a solve from a reset converges and plans inside every backed-off limit.

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
from target_gym.plane import oracle_model as om
from target_gym.plane.dynamics import PITCH_DEMAND_MAX, STICK_TRAVEL, attitude_axis
from target_gym.registry import REGISTRY

N_SITES = len(om.SITE_NAMES)


def _setup(name="plane_sine", **overrides):
    # plane_sine flies the 2D plant ``plane`` flies; ``plane``'s own oracle
    # waits on its rebuild for the Amendment 2 score (plane/PHYSICS.md).
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
def test_the_eas_row_is_the_vmo_warning(excess, trips):
    """Level flight at 5000 m: EAS = V sqrt(rho / rho_0) with the ISA density.
    Set 1 m/s either side of VMO, the env warns exactly when the row passes it.
    VMO is a warning, not a trip (Amendment 2 rule 14; plane/REALITY.md)."""
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
    assert bool(s1[4]["overspeed"]) == trips
    assert (float(y[6]) > float(p.vmo_eas)) == trips


def test_the_stick_crosses_its_travel_within_one_step():
    """T1r's reach over a 1 s hold: 30 deg/s for 1 s is the whole 30 deg of
    travel, so no demand lies past it. From one stop, a full demand reaches the
    other within the step, short by what the 60 ms lag leaves of the last 7.5 deg
    sub-step, under 0.2 deg. The load limits are opened, or the pull would trip
    and restart the plant, and the attitude loop's gain is raised so its output
    stays at its stop through the step: the reach is the surface's, not the
    loop's."""
    _, env, p = _setup()
    with enable_x64():
        p = float64(
            p.replace(
                load_factor_max=100.0,
                load_factor_min=-100.0,
                ap_load_increment=100.0,
                inner_loop=p.inner_loop.replace(Kp_theta=100.0),
            )
        )
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
