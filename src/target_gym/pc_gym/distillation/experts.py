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
    env,
    params,
    horizon: int = 15,
    n_iter: int = 80,
    lr: float = 0.08,
    lr_end: float = 0.004,
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

    The step decays from ``lr`` to ``lr_end`` over each solve. A fixed
    normalised step cannot place the reflux and boilup closer than about
    ``lr`` to the optimum, and on this plant that was the whole hold error.
    Measured in the oracle audit (2026-10-01) on the protocol seeds, planning
    on the mean feed composition: the fixed step scored 0.34, the decaying
    step 0.0126 at 40 iterations and 0.0125 at 80, against a floor of about
    0.0122 set by seed 0's running cost. More iterations still shrink the
    tracking term, which is by then far below the analyser's resolution.
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
        lr_end=lr_end,
    )
