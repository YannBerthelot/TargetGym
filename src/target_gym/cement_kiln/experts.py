"""Reference controller (the MPC slot) for the cement kiln task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

from target_gym.experts.mpc import (
    SamplingMPC,
    _is_v1,
    _v2_objective,
)


def _cement_kiln_objective(state, params):
    """Quadratic in the free-lime error, sharing the reward's minimiser.

    The environment's reward clips flat once free lime is more than
    ``lime_band`` from target -- exactly the situation the controller is called
    on to fix -- so a quadratic that stays informative far from target is what
    the optimiser needs.
    """
    err = (state.lime[-1] - state.target_lime) / params.lime_band
    fuel = (state.fuel - params.fuel_min) / (params.fuel_max - params.fuel_min)
    return -(err**2 + 0.02 * fuel)


def make_cement_kiln_mpc(
    env, params, horizon: int = 40, n_samples: int = 96, n_iter: int = 4, **kwargs
):
    """Sampling (CEM) MPC for the rotary kiln.

    Gradient-free by necessity, not preference -- see ``SamplingMPC`` for the
    measured reason.

    A 40-step horizon is 20 minutes at dt = 30 s, most of the ~25 minute
    transport delay. That is the point: a controller whose horizon is shorter
    than the delay is choosing fuel whose consequences it cannot see.
    """
    objective = _cement_kiln_objective
    if not _is_v1(params):
        from target_gym.cement_kiln.env import compute_reward

        objective = _v2_objective(compute_reward)
    return SamplingMPC(
        env,
        params,
        objective=objective,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_samples=n_samples,
        n_iter=n_iter,
        **kwargs,
    )
