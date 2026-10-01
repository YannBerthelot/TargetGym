"""Reference controller (the MPC slot) for
the 3D aircraft tasks (plane3d_heading, plane3d_circle, plane3d_racetrack, plane3d_figure8).

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

from target_gym.experts.mpc import (
    GradientMPC,
    _done_value,
)


def make_plane3d_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter: int = 50,
    lr: float = 0.05,
):
    """Gradient MPC for the 3D plane tasks — optimises [power, stick, aileron] in [-1, 1].

    Same rationale as the 2D Plane MPC: the 3D dynamics extend the 2D
    aerodynamic model with roll, so it remains differentiable JAX but not
    expressible in CasADi. Works for all three task variants (Heading,
    Circle, FigureEight) since they share step_env.

    The objective is the environment's own reward. Under the version-2
    reward, a cost, a healthy step is negative, so a plan that leaves the
    envelope must be charged the failure cost for the rest of the horizon or
    crashing would read as an improvement over flying on; ``done_value`` is
    set to it (version 1 is non-negative and keeps 0).
    """
    return GradientMPC(
        env,
        params,
        action_dim=3,
        action_lb=-1.0,
        action_ub=1.0,
        done_value=_done_value(params),
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
    )
