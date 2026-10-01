"""Reference controller (the MPC slot) for the wind turbine task.

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

_WT_BARRIER_ONSET = 0.85  # fraction of the trip speed at which the barrier starts
_WT_BARRIER_WEIGHT = 10.0


def _wind_turbine_objective(state, params):
    """Smooth stand-in for the turbine's reward, with the same minimiser.

    The environment used to score power tracking as ``clip(1-err/band, 0, 1)**2``
    minus a pitch-activity penalty: a fine thing to be scored on and a useless
    thing to descend, because one step off the operating point puts the error at
    nearly four times the band, where the term is clipped flat and the only
    surviving gradient belongs to the *penalty*. The optimiser was then correctly
    guided to stop moving the pitch, and returned ~0 for the rest of the episode.

    The reward is log-scaled now and has no flat region, so that premise is gone
    -- but the surrogate is still needed, for a subtler reason. Log-scaling makes
    the reward scale-free in *value* (each halving of the error is worth the same
    increment); it does not make the *gradient* scale-free. Differentiating
    ``1 - log1p(e/f)/log1p(E/f)`` gives ``-1/((f + e) * log1p(E/f))``, which
    decays like ``1/e``: the pull toward the setpoint is weakest exactly where
    the controller is furthest from it. A quadratic in the normalised error has
    the same minimiser and a gradient that instead *grows* with the error.

    Measured over six seeds against the log-scaled reward, that difference is
    still worth almost everything: 341.9 for this surrogate against 172.1 for
    the reward itself (with the barrier below in both). So the surrogate is not a
    workaround for a broken reward any more -- it is a planner-side
    reformulation, which is a normal thing for an MPC to carry.
    """
    from target_gym.energy.wind_turbine.env import electrical_power

    power = electrical_power(state.omega, state.torque, params)
    err = (state.target_power - power) / params.power_band
    activity = jnp.abs(state.pitch_cmd - state.pitch) / params.pitch_max

    # Soft barrier on the rotor-speed trip. The environment terminates outside
    # [underspeed, overspeed] x rated, and a *hard* stop is invisible to a
    # gradient planner: ``done`` is a boolean, so masking the reward after it
    # tells the optimiser what a trip costs while giving it no derivative
    # pointing away from one. Measured, that is exactly what happened -- the
    # planned pitch command stayed at 0.000 while the predicted rotor speed ran
    # past the trip, on every horizon from 60 to 200.
    #
    # A differentiable penalty that switches on before the boundary does give a
    # gradient, and it is what makes this controller stable: over twelve seeds
    # the worst episode goes from 22 to 307 and the spread from sd 152 to 24.
    # Re-checked against the log-scaled reward, where it is still the difference
    # between controlling and tripping: 172.1 with the barrier, 52.4 without,
    # and without it 6 of 6 episodes ended early on the overspeed trip.
    # The onset matters (0.85 beats 0.80); the weight barely does (10, 30 and
    # 100 land within 0.4 of each other), which is the signature of a term that
    # is shaping the approach rather than trading against the objective.
    barrier = _wind_turbine_barrier(state, params)

    # Offset so a healthy step scores ~1 and a terminated one scores
    # ``done_value`` = 0, making an early trip cost the rest of the horizon.
    return (
        1.0
        - err**2
        - params.pitch_activity_weight * activity
        - _WT_BARRIER_WEIGHT * barrier
    )


_WT_SPEED_BOX = 0.05  # rotor speed kept within this fraction of rated
_WT_SPEED_BOX_COST = 100.0  # floor units at a 10% excursion: ten floor-widths


def _wind_turbine_barrier(state, params):
    """Squared excursion of the rotor speed past 85% of the way to either trip."""
    from target_gym.energy.wind_turbine.env import omega_rated

    ratio = state.omega / omega_rated(params)
    over = ratio / params.overspeed_factor
    under = params.underspeed_factor / jnp.maximum(ratio, 1e-6)
    return (
        jnp.maximum(over - _WT_BARRIER_ONSET, 0.0) ** 2
        + jnp.maximum(under - _WT_BARRIER_ONSET, 0.0) ** 2
    )


def _wind_turbine_speed_box(state, params):
    """Soft box keeping the rotor within 5% of rated, in floor units.

    What a turbine's own supervisory logic does above rated wind (regulate
    rotor speed with the pitch, power with the torque), and it is there for
    the planner's horizon, not for the reward. Recovering a rotor that has
    slowed costs torque now and pays off over the rotor's ~100 s of inertia,
    beyond a 15 s plan; without the box the planner sat at 0.85x rated with a
    250 kW error rather than spend the torque (seed 1 of the test episode),
    and drifted the same way over a long hold. Mild by design: a 10%
    excursion costs ten floor-widths of tracking error (25 kW), so the plan
    is still the plant's own cost -- weighted like the trip, or like a 500 kW
    error, the box dominated the tracking term and the planner braked the
    rotor with the torque instead (a 350-700 kW error for a whole horizon,
    measured on seeds 1 and 2), where the PID rides a 16% overspeed with a
    50 kW error and the plant trips only at 25%.
    """
    from target_gym.energy.wind_turbine.env import omega_rated

    ratio = state.omega / omega_rated(params)
    excess = jnp.maximum(jnp.abs(ratio - 1.0) - _WT_SPEED_BOX, 0.0) / _WT_SPEED_BOX
    return _WT_SPEED_BOX_COST * excess**2


def make_wind_turbine_mpc(
    env,
    params,
    horizon: int = 60,
    n_iter: int | None = None,
    lr: float | None = None,
    n_tail: int = 0,
    objective_fn=_wind_turbine_objective,
):
    """Gradient MPC for the NREL 5 MW turbine.

    Gradient-based for the same reason as the aircraft and the column: the
    plant is already differentiable JAX, and the Cp surface is an empirical
    fit that would gain nothing from symbolic re-expression. Optimising pitch
    and torque jointly is the point -- the rate-limited pitch actuator means
    the useful move is often to start pitching *before* the rotor has
    accelerated, which a reactive loop cannot do.

    Two changes make it actually control. The objective is the smooth surrogate
    above rather than the environment's clipped reward, without which the
    controller scored a return of -0.02 against the PID's 393 at every horizon
    tried -- identical to two decimals at 20, 40 and 60, which is the signature
    of an optimiser that is not moving. The horizon is then 60 rather than 20,
    which only matters once the gradient is informative. Measured over 400-step
    episodes the return goes from -0.02 to 387, with no terminations.

    Scored over twelve seeds this reaches 385 against the PID's 392 -- 98%, and
    ahead on 7 of the 12 -- so it is on par with the PID rather than an upper
    bound over it. The remaining gap is one seed that still drops to ~307.

    Two things that look like explanations and are not, both measured: the wind
    forecast (the MPC plans with a fixed key while the episode uses its own, and
    the seed where they disagree scored *higher*), and the inner optimiser
    (Adam, and a decaying step size, are both worse here than the plain one).
    """
    # Version 1 keeps the planner it was recorded with (100 iterations at
    # lr 0.02, a zero warm start), so its recorded baseline reproduces; the
    # version-2 planner needs the larger budget for the move-suppressed,
    # squared-tracking objective (measured: 500 / 0.005 is the first setting
    # that beats the PID on every seed).
    v1 = _is_v1(params)
    n_iter = (100 if v1 else 500) if n_iter is None else n_iter
    lr = (0.02 if v1 else 0.005) if lr is None else lr
    done_value = 0.0
    if not v1 and objective_fn is _wind_turbine_objective:
        from target_gym.energy.wind_turbine.env import (
            compute_reward,
            compute_reward_terms,
        )

        objective_fn = _v2_objective(
            compute_reward,
            _wind_turbine_barrier,
            terms_fn=compute_reward_terms,
            shaping_fn=_wind_turbine_speed_box,
        )
        done_value = _done_value(params, squared=True)

    from target_gym.experts.pid import make_wind_turbine_stateful_pid

    initial_plan = (
        None
        if v1
        else _pid_rollout_plan(env, make_wind_turbine_stateful_pid, horizon, 2)
    )
    guide_plan = initial_plan

    def move_penalty(u0, u_prev, p):
        # Pitch command change between solves as a fraction of pitch_max
        # (raw range 2 <-> pitch_max), in floor units of the fatigue term.
        frac = jnp.abs(u0[0] - u_prev[0]) * 0.5
        return float(p.fatigue_weight) * (frac / p.c_hold) ** 2

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
        done_value=done_value,
        objective_fn=objective_fn,
        initial_plan_fn=initial_plan,
        guide_plan_fn=guide_plan,
        move_penalty_fn=None if v1 else move_penalty,
    )
