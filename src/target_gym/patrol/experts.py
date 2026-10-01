"""Reference controller (the MPC slot) for the patrol task.

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

# The follower's stall barrier, with the 2D aircraft planner's values
# (target_gym.plane.experts). Kept here so the patrol oracle depends only on
# its own package and the shared controller code.
_PATROL_STALL_MARGIN = 1.3  # multiples of stall speed at which the barrier starts
_PATROL_BARRIER_WEIGHT = 10.0

#: Fraction of ``slot_tolerance`` at which the patrol surrogate puts its
#: curvature. See :func:`_patrol_objective`.
_PATROL_ERROR_SCALE = 0.25


def _patrol_objective(state, params):
    """Slot tracking and heading alignment, with a floor under the follower's speed.

    Two departures from the environment's own reward, for the usual two
    reasons.

    The tracking term is ``1 / (1 + (e / scale)**2)`` rather than the shipped
    log-scaled reward. Both are bounded in [0, 1] and both are maximised at
    zero slot error, but a log-scaled reward's gradient decays like ``1/e``.

    ``scale`` is a quarter of ``slot_tolerance``, not the tolerance itself.
    The tolerance is a pass/fail bound; a controller that is actually good
    operates well inside it -- the shipped PID settles at 8-13 m against a 60 m
    tolerance -- so curvature at 60 m leaves the surrogate nearly flat across
    the whole range where the decisions are made. Chosen by measurement rather
    than argument, over two seeds at 300 iterations: a quarter of the tolerance
    scores 175.7, the raw log reward 144.6, and the full tolerance 138.2. The
    precision floor of 3 m is far worse again (19.4 at 50 iterations), so this
    is an interior optimum and not a monotone preference for tighter scaling.

    The barrier is the aircraft objective's, for the same reason it exists
    there. The follower is the same airframe with the same power and stick, and
    the slot can be several hundred metres away at reset, so a planner is free
    to buy position with airspeed and arrive at the slot with nothing left.
    Patrol terminates on the altitude envelope rather than on stall, so a
    departure costs the planner only the steps after it falls out of the sky --
    which a finite horizon may not reach.

    Multiplicative in the alignment factor, as the environment's reward is: a
    wingman flies the slot *parallel* to the lead, not merely at the point.
    """
    from target_gym.patrol.env import heading_alignment, slot_error

    err = slot_error(state) / (_PATROL_ERROR_SCALE * params.slot_tolerance)
    track = 1.0 / (1.0 + err**2)
    align = heading_alignment(state, params)

    f = state.follower
    speed = jnp.sqrt(f.x_dot**2 + f.y_dot**2 + f.z_dot**2)
    v_stall = jnp.sqrt(
        2.0 * f.m * params.gravity / (f.rho * params.wings_surface * params.CL_max)
    )
    margin = speed / (_PATROL_STALL_MARGIN * v_stall)
    penalty = _PATROL_BARRIER_WEIGHT * jnp.maximum(1.0 - margin, 0.0) ** 2
    return track * align - penalty


def _patrol_stall_barrier(state, params):
    """Squared shortfall of the follower's airspeed below 1.3x its stall
    speed, in [0, 1]: the 2D aircraft's barrier on the follower's state."""
    f = state.follower
    speed = jnp.sqrt(f.x_dot**2 + f.y_dot**2 + f.z_dot**2)
    v_stall = jnp.sqrt(
        2.0 * f.m * params.gravity / (f.rho * params.wings_surface * params.CL_max)
    )
    margin = speed / (_PATROL_STALL_MARGIN * v_stall)
    return jnp.maximum(1.0 - margin, 0.0) ** 2


def make_patrol_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 300,
    lr: float = 0.05,
    n_tail: int = 60,
):
    """Gradient MPC for the formation-keeping follower.

    A ``GradientMPC`` rather than a CasADi one, and that is what makes this
    tractable at all. The obstacle recorded against a patrol MPC was that the
    reference is a *manoeuvring lead*, so a symbolic model would need the
    lead's whole future trajectory wired in as a time-varying parameter. That
    is true of the CasADi route and irrelevant here: the lead is scripted and
    deterministic -- ``step_lead`` advances it with a heading autopilot at a
    fixed ``lead_turn_rate`` -- so a planner that differentiates the true
    ``step_env`` propagates the lead for free, exactly as it propagates the
    follower.

    The horizon is 30 s at the environment's 1 s step, which covers the slot
    capture from a 40 m spawn offset with room for the lead's turn to develop,
    and the settings that go with it are the 2D aircraft's for the reasons that
    file already records. ``n_tail=60`` charges the plan for twice the flight
    it optimises, so it cannot park the follower somewhere that leaves the
    altitude envelope just past the horizon. ``done_value`` sits below the
    worst step this objective can score: with the barrier subtracted the
    objective is no longer non-negative, so a ``done_value`` of 0 would make
    flying out of the envelope score *better* than any penalised step, and the
    planner takes that trade.

    ``n_iter`` is 300 where every other gradient planner here uses 50, and that
    single number is what decides whether this baseline is an upper bound at
    all. Measured over two seeds against a PID scoring ~105: 106.1 at 100
    iterations, 130.3 at 150, 153.8 at 300, 171.5 at 600 with no tail. The
    planner was not stuck, it was stopping early -- 90 decision variables under
    projected gradient descent -- and every objective and horizon variant tried
    before this was being compared at a non-converged optimum, which is why
    none of them looked decisive.

    The other tasks hide this. ``plane3d`` uses the same 50 iterations and wins
    enormously, but its PIDs score 0.16 of ceiling, so an under-converged plan
    clears them anyway. The patrol PID scores 0.50, and a bar that high is what
    made the under-convergence visible.

    The tail earns its keep here rather than costing: at 300 iterations
    ``n_tail=60`` scores 175.7 against 153.8 without it, which is better than
    doubling the iterations to 600 (171.5) and half the cost.
    """
    initial_plan = guide_plan = None
    if _is_v1(params):
        objective_fn = _patrol_objective
        done_value = -(_PATROL_BARRIER_WEIGHT + 1.0)
    else:
        # Under version 2 the planner descends the follower's own cost with
        # the stall barrier kept, and without the open-loop tail (as the 2D
        # aircraft: under an unbounded cost the tail dominates the solve).
        # The surrogate above has the version-1 minimiser, and once the
        # descent was made monotone -- a better solve of the surrogate --
        # the follower lost to its PID on two seeds by 18x: optimising the
        # wrong objective harder.
        from target_gym.experts.pid import make_patrol_stateful_pid
        from target_gym.patrol.env import compute_reward_patrol

        objective_fn = _v2_objective(compute_reward_patrol, _patrol_stall_barrier)
        done_value = _done_value(params)
        n_tail = 0
        # Twenty seconds of horizon, as on the 2D aircraft: in turbulence the
        # 30-step plan does not converge within the budget, and 15 is too
        # short to hold the slot on every seed (seeds 0 / 1 returns: 15 steps
        # -9449 / -536, 20 steps -732 / -702, 25 steps -1382 / -859, the PID
        # -3868 / -2033).
        if horizon == 30:
            horizon = 20
        # Started from, and at every step compared against, the shipped
        # PID's rollout under the planner's own objective, as the 2D
        # aircraft is: on its own the descent lost the slot on two seeds
        # (returns -5e5 against the PID's -6e4).
        initial_plan = _pid_rollout_plan(env, make_patrol_stateful_pid, horizon, 3)
        guide_plan = initial_plan
    return GradientMPC(
        env,
        params,
        action_dim=3,
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
