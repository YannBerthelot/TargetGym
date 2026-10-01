"""Reference controller (the MPC slot) for the distillation task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

from target_gym.experts.mpc import (
    GradientMPC,
    _done_value,
)


def make_distillation_mpc(
    env, params, horizon: int = 15, n_iter: int = 40, lr: float = 0.08
):
    """Gradient MPC for the distillation column.

    Gradient-based rather than CasADi: the plant is 41 coupled stage balances
    that are already differentiable JAX, and re-expressing them symbolically
    would duplicate the whole model for no gain -- the same rationale as the
    aircraft. Optimises [L_raw, V_raw] jointly, which is the point on an
    ill-conditioned plant: the useful move is a *coordinated* change in reflux
    and boilup, exactly what independent diagonal loops cannot make.

    The objective is the environment's own reward; see ``make_plane3d_mpc``
    for why ``done_value`` follows the reward version.
    """
    return GradientMPC(
        env,
        params,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        done_value=_done_value(params),
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
    )
