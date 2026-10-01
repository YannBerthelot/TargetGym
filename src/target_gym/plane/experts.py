"""Reference controller (the MPC slot) for
the 2D aircraft tasks (plane, plane_sine, plane_energy).

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax.numpy as jnp

from target_gym.experts.mpc import (
    GradientMPC,
    _done_value,
    _is_v1,
    _pid_rollout_plan,
    _v2_objective,
)

_PLANE_STALL_MARGIN = 1.3  # multiples of stall speed at which the barrier starts
_PLANE_BARRIER_WEIGHT = 10.0


def _plane_objective(state, params):
    """The aircraft's own reward, less a barrier on flying into the stall.

    The altitude reward scores one thing and the aircraft has two actuators, so
    the planner is free to buy altitude with airspeed. Over a 90 s window that
    is the *best* thing it can do when the target is thousands of metres away:
    a zoom climb converts kinetic energy to potential energy far faster than
    the engines can supply it. Measured on ``plane_steps`` seed 0, that is
    exactly what the plan did -- airspeed fell monotonically from 201 to 30 m/s
    over 90 s while the aircraft climbed 3300 m, touched the commanded altitude
    with nothing left, departed at 91 deg angle of attack and hit the ground at
    t=372. It happened on ``plane`` too; there the phugoid that followed
    happened to damp out, which is luck rather than control.

    Angle of attack is the wrong thing to fence, even though it is what
    actually stalls: through that whole manoeuvre it sat at 4-8 deg and only
    crossed 15 deg at t=90, one step from the departure and at the very end of
    the planning window. Airspeed decays through the entire climb, so a floor
    under it is a constraint the planner can see coming and descend.

    The floor is the stall speed at this mass and altitude rather than a fixed
    number, since both move: ``sqrt(2 m g / (rho S CL_max))`` is where the wing
    can no longer carry the weight, and the barrier switches on at 1.3 times
    it, the usual approach margin. Bounded in [0, 1] by construction, which is
    what lets ``done_value`` stay below the worst step the planner can plan, so
    flying into the ground cannot look better than flying slowly.

    Measured on ``plane_steps`` seed 0 over 400 s: return 304 against 83 for
    the unprotected planner, settled error 0.1 m, and airspeed held above
    128 m/s with a worst angle of attack of 7.7 deg. On ``plane``, 226 against
    189. Like the turbine's barrier, the constants barely matter -- weight 30
    scores 299 and a 1.15 margin 299 -- which is the signature of a term
    shaping the approach rather than trading against the objective.

    Two alternatives, both measured and both rejected. Doubling the tail to
    ``n_tail=120`` prices more of the aftermath and does fix seed 0 (274), but
    it costs two thirds again in compute and only moves the horizon at which
    the same trade becomes profitable. Making the planner hold airspeed
    outright, by planning against ``speed_weight=1.0``, flies beautifully for
    400 s (166, angle of attack 2.1 deg) and then crashes anyway at t=1260 on
    the full episode and at t=386 on seed 2: it changes which trim the planner
    settles into without ever putting a floor under the trade.
    """
    from target_gym.plane.env import compute_reward

    reward = compute_reward(state, params)
    return reward - _PLANE_BARRIER_WEIGHT * _plane_stall_barrier(state, params)


def _plane_stall_barrier(state, params):
    """Squared shortfall of airspeed below 1.3x the stall speed, in [0, 1]."""
    speed = jnp.sqrt(state.x_dot**2 + state.z_dot**2)
    v_stall = jnp.sqrt(
        2.0
        * state.m
        * params.gravity
        / (state.rho * params.wings_surface * params.CL_max)
    )
    margin = speed / (_PLANE_STALL_MARGIN * v_stall)
    return jnp.maximum(1.0 - margin, 0.0) ** 2


def make_plane_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 50,
    lr: float = 0.05,
    n_tail: int = 60,
    objective_fn=_plane_objective,
):
    """Gradient MPC for Airplane2D — optimises both power and stick in [-1, 1].

    Uses gradient-based MPC because the Plane has 9 coupled nonlinear ODEs
    including aerodynamic coefficients that are not expressible in CasADi
    without a full symbolic re-implementation.  dt=1.0 s; horizon=30.

    Two things make it fly. The objective carries a stall barrier, without
    which the planner trades all its airspeed for altitude on every
    acquisition and departs; see ``_plane_objective``, which is where that is
    measured. And ``n_tail=60``: optimising 30 s of flight and being charged
    for nothing beyond it, the plan climbed hard and left the aircraft outside
    the altitude envelope just past the horizon, settling 654x worse than the
    PID and crashing in one episode of two. Simulating 60 further seconds on
    the held action -- which for this aircraft is close to trim -- prices that
    ending into the objective. Measured over 600-step episodes, settled
    tracking error went from 2949 m to 0.083 m, with no terminations. Sixty is
    the knee: 120 is no better on a fixed setpoint and costs twice.

    The tail alone was not enough, which is why the barrier is here. It fixes
    the ending the plan can *see*; the zoom climb is an ending the plan likes,
    and no affordable horizon changes that.

    ``done_value`` sits below the worst step the objective can score, so a plan
    that reaches the ground is charged for the rest of the horizon rather than
    scoring the 0.0 that a barrier-free positive reward could rely on.
    """
    initial_plan = None
    if not _is_v1(params) and objective_fn is _plane_objective:
        from target_gym.experts.pid import make_plane_cascaded_pid
        from target_gym.plane.env import compute_reward

        objective_fn = _v2_objective(compute_reward, _plane_stall_barrier)
        done_value = _done_value(params)
        # Twenty seconds of horizon, not thirty. In turbulence the 30-step
        # plan never converged within the budget and the planner chattered
        # between its plan and the guide (stick moving 0.17 per step): it
        # held 4.6 m off with a 4.4 m bias where the PID held 2.5 m. Shorter
        # is myopic the other way -- airspeed answers the throttle slowly, so
        # at 15 steps the plan buys altitude with speed it will not see
        # itself pay for (23 m/s off cruise). Measured on seed 0 over the
        # hold, cost per step: 30 steps 4.6 (before the fix below, 10 at 25),
        # 15 steps 4.6, 20 steps 2.7 against the PID's 9; stick 0.009/step.
        if horizon == 30:
            horizon = 20
        # From a constant plan the planner parks at the edge of the altitude
        # tolerance 55 m/s below cruise: raising thrust alone pitches the
        # aircraft out of the band before the linear speed saving pays, and
        # the coordinated thrust-and-elevator move is out of reach of the
        # normalised steps. The cascaded PID's plan is a stabilising start.
        initial_plan = _pid_rollout_plan(env, make_plane_cascaded_pid, horizon, 2)
        guide_plan = initial_plan
        # No open-loop tail under version 2. Holding the last action for 60 s
        # lets the phugoid carry the aircraft far from the band, and under an
        # unbounded quadratic cost that tail dominated the objective (1e5
        # per solve against 26 per step realised): the planner optimised
        # what the held action did later, parked at the edge of the
        # tolerance and flew 55 m/s slow. Without it, and guided by the
        # PID's plan, it holds altitude to the metre at cruise speed.
        n_tail = 0
    else:
        done_value = -(_PLANE_BARRIER_WEIGHT + 1.0)
        guide_plan = None
    return GradientMPC(
        env,
        params,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
        n_tail=n_tail,
        objective_fn=objective_fn,
        initial_plan_fn=initial_plan,
        guide_plan_fn=guide_plan,
        done_value=done_value,
    )
