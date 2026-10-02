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

# Stand-in for "the task's own default" in the factory's keyword arguments,
# so an explicit ``lr_end=None`` (a fixed step) stays expressible.
_TASK_DEFAULT = object()

#: Solver settings per task, keyed by the environment class. They live here,
#: not in the registry, because this file is in the four tasks' baseline
#: fingerprint and the registry is not: a setting passed from the registry
#: would change the oracle without staling its recorded scores.
#:
#: The step decays geometrically from ``lr`` (0.05) to ``lr_end`` over each
#: solve. A normalised descent cannot place an action closer than about its
#: step to the optimum. At a fixed 0.05 for 50 iterations the planner held
#: the circle 4.1 to 4.5 m off its altitude and 14.3 to 15.5 m off its path
#: (the hold rows recorded before the oracle audit, 2026-10). On the
#: circle's protocol seeds 0 and 1, under the -v2 floors, the audit measured
#: the mean errors after the burn-in: the decay at 50 iterations took them
#: to 1.6 to 1.7 m and 4.6 to 7.6 m, and at 200 iterations to 0.86 to 1.0 m
#: and 0.42 to 2.4 m. The figure-8 needs the 200: at 50 its protocol seed 0
#: cost 3.3x the fixed step's (40 m excursions at the lobe tips, where the
#: fixed step held a steady 15 m), at 200 it held 3.3 m. So does the
#: racetrack: under the -v3 floors its protocol cost on seeds 0 to 2 is 4.58
#: at 50 iterations (2.23, 7.54, 3.97) and 1.07 at 200 (0.960, 1.29, 0.962),
#: zero trips either way, for 3.5x the compute per seed (11 to 14 min against
#: 3.5 in the paired runs). The holds under the -v3 floors, the instrument
#: resolutions, are in PHYSICS.md.
_TASK_SETTINGS = {
    "Plane3DHeading": dict(n_iter=200, lr_end=0.002),
    "Plane3DCircle": dict(n_iter=200, lr_end=0.002),
    "Plane3DRacetrack": dict(n_iter=200, lr_end=0.002),
    "Plane3DFigureEight": dict(n_iter=200, lr_end=0.002),
}


def task_settings(env) -> dict:
    """The solver settings ``make_plane3d_mpc`` uses for ``env``'s task.

    An environment class not in ``_TASK_SETTINGS`` gets the settings every
    task used before the audit: 50 iterations at a fixed step.
    """
    return dict(_TASK_SETTINGS.get(type(env).__name__, dict(n_iter=50, lr_end=None)))


def make_plane3d_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter=_TASK_DEFAULT,
    lr: float = 0.05,
    lr_end=_TASK_DEFAULT,
):
    """Gradient MPC for the 3D plane tasks, optimising [power, stick, aileron] in [-1, 1].

    Same rationale as the 2D Plane MPC: the 3D dynamics extend the 2D
    aerodynamic model with roll, so it remains differentiable JAX but not
    expressible in CasADi. Works for all four task variants (Heading,
    Circle, Racetrack, FigureEight) since they share step_env.

    The objective is the environment's own reward. Under the version-2
    reward a trip is charged inside the rollout itself (the step that leaves
    the envelope pays the trip cost, ``base.failure_kernel``) and
    ``step_env`` never raises ``terminated``, so ``done_value``, the failure
    charge in floor units, is never applied (version 1 keeps 0). The plan
    therefore does not read ``rho_floor_tracking``; the floors and
    ``failure_cost`` weight it through the reward.

    ``n_iter`` and ``lr_end`` default to the task's entry in
    ``_TASK_SETTINGS``; pass them to override (``lr_end=None`` is the fixed
    step the planner used before the audit).
    """
    settings = task_settings(env)
    if n_iter is _TASK_DEFAULT:
        n_iter = settings["n_iter"]
    if lr_end is _TASK_DEFAULT:
        lr_end = settings["lr_end"]
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
        lr_end=lr_end,
    )
