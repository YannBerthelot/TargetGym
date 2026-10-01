"""Reference controller (the MPC slot) for the battery task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

from target_gym.experts.mpc import (
    GradientMPC,
    _done_value,
    _is_v1,
    _v2_objective,
)


def _battery_objective(state, params):
    """Smooth stand-in for the battery's reward, with the same minimiser.

    This surrogate was written against a clipped ``clip(1-err/band, 0, 1)**2``
    tracking term that was exactly flat outside the band, leaving the optimiser
    nothing to descend but the degradation and state-of-charge terms, which both
    pull toward doing nothing. That reward is gone -- tracking is log-scaled now,
    and never flat. The surrogate stays anyway, for the reason given in
    :func:`_wind_turbine_objective`: log-scaling fixes the *value*, not the
    gradient, which still decays like ``1/err``. Re-measured over ten seeds
    against the log-scaled reward it is worth a median of +9 (155.8 against
    146.8).

    Read the mean with care in either case. It is carried by one seed where
    lookahead pays enormously (350 against the PID's 164); on the other nine the
    MPC is behind by 4 to 13, for a median of -4 against the PID and 1 win in
    10. So this is a large improvement over descending the reward directly and
    *not* an upper bound -- horizon, iterations and step size were all swept
    without closing the remainder. It is inside the 10% contract tolerance.
    """
    from target_gym.energy.battery.env import degradation_rate

    err = (state.target_power - state.power) / params.power_band
    fade = degradation_rate(state.current, state.T_cell, params) * params.delta_t
    headroom = (state.soc - 0.5) ** 2
    # Offset so a healthy step scores ~1, matching ``done_value`` = 0.
    return (
        1.0
        - err**2
        - params.degradation_weight * fade
        - params.soc_comfort_weight * headroom
    )


def make_battery_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 40,
    lr: float = 0.08,
    objective_fn=_battery_objective,
):
    """Gradient MPC for the grid battery.

    Horizon matters more here than in most environments: the battery has a
    *finite energy budget*, so the value of discharging now depends on what the
    dispatch is likely to ask for later. 30 steps is 2.5 min at dt = 5 s --
    long enough to see the state-of-charge limits coming, which is exactly what
    a reactive controller cannot do.
    """
    done_value = 0.0
    if not _is_v1(params) and objective_fn is _battery_objective:
        from target_gym.energy.battery.env import compute_reward, compute_reward_terms

        objective_fn = _v2_objective(compute_reward, terms_fn=compute_reward_terms)
        done_value = _done_value(params, squared=True)
    return GradientMPC(
        env,
        params,
        action_dim=1,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
        done_value=done_value,
        objective_fn=objective_fn,
    )
