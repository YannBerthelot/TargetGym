"""PD experts for Brax/Mujoco reaching tasks.

Each expert plugs into the same `FunctionalExpertPolicy` interface as
the CPG/PID experts (`target_gym.experts.cpg`, `target_gym.experts.pid`),
so Ajax's `expert_policy=` / `eval_expert_policy=` slots and the
gain-policy pipeline (SAC actor outputs per-parameter log-multipliers
on anchor gains) work out of the box.
"""

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import (
    FunctionalExpertPolicy,
    register_learnable_gains,
)


@struct.dataclass
class ReacherPDParams:
    """Brax `reacher` 2-link arm PD-on-Jacobian-transpose.

    Reacher's obs layout (Brax convention, 11 dims):
        [0:2]   cos(theta_1, theta_2)
        [2:4]   sin(theta_1, theta_2)
        [4:6]   target_xy
        [6:8]   tip_vel_xy
        [8:11]  tip_to_target_xyz   (tip - target)

    The expert recovers (theta_1, theta_2) from the cos/sin pair, builds
    the closed-form 2x2 Jacobian of the planar 2-link FK, and applies
    Jacobian-transpose control with EE-velocity damping:
        tau = kp * J^T (target - tip) - kd * J^T tip_vel
    Clipped to the [-1, 1] action box.

    Link lengths follow the standard MuJoCo Reacher MJCF (L1=0.1, L2=0.11).
    """

    kp: float
    kd: float
    L1: float
    L2: float


@struct.dataclass
class _PDState:
    # Stateless expert (memoryless); the dataclass exists only to match the
    # FunctionalExpertPolicy state-carrying interface.
    _: float


def _pd_reset(params: ReacherPDParams) -> _PDState:  # noqa: ARG001
    return _PDState(_=jnp.float32(0.0))


def reacher_pd_step(
    params: ReacherPDParams, state: _PDState, obs: jnp.ndarray
) -> tuple[jnp.ndarray, _PDState]:
    """Jacobian-transpose PD on the 2-link planar arm.

    Returns ``(action, new_state)``. The action drives the EE toward the
    target via ``J^T (target - tip)`` torques, with EE-velocity damping
    folded through the same Jacobian.
    """
    cos_t = obs[..., 0:2]
    sin_t = obs[..., 2:4]
    cos_1 = cos_t[..., 0]
    cos_2 = cos_t[..., 1]
    sin_1 = sin_t[..., 0]
    sin_2 = sin_t[..., 1]
    cos_12 = cos_1 * cos_2 - sin_1 * sin_2
    sin_12 = sin_1 * cos_2 + cos_1 * sin_2

    L1 = params.L1
    L2 = params.L2
    j00 = -L1 * sin_1 - L2 * sin_12
    j01 = -L2 * sin_12
    j10 = L1 * cos_1 + L2 * cos_12
    j11 = L2 * cos_12

    # tip_to_target = tip - target -> target - tip = -tip_to_target.
    # We only act in xy (Brax reacher tip is planar; z component is small).
    err_xy = -obs[..., 8:10]
    e_x = err_xy[..., 0]
    e_y = err_xy[..., 1]

    tip_vel = obs[..., 6:8]
    v_x = tip_vel[..., 0]
    v_y = tip_vel[..., 1]

    tau_1 = params.kp * (j00 * e_x + j10 * e_y) - params.kd * (j00 * v_x + j10 * v_y)
    tau_2 = params.kp * (j01 * e_x + j11 * e_y) - params.kd * (j01 * v_x + j11 * v_y)

    action = jnp.stack([tau_1, tau_2], axis=-1)
    action = jnp.clip(action, -1.0, 1.0)
    return action, state


_REACHER_PD_LEARNABLE = ("kp", "kd")
register_learnable_gains(reacher_pd_step, _REACHER_PD_LEARNABLE)


def make_reacher_pd(
    kp: float = 35.0,
    kd: float = 1.5,
    L1: float = 0.1,
    L2: float = 0.11,
) -> tuple[ReacherPDParams, _PDState]:
    """Build the Reacher PD expert params + initial state.

    Default ``kp=35, kd=1.5`` chosen by hand-tuning on the Brax Reacher
    physics: high enough kp to drive EE to the target inside the 50-step
    Brax episode, kd damps oscillation around the target.
    """
    params = ReacherPDParams(kp=float(kp), kd=float(kd), L1=float(L1), L2=float(L2))
    return params, _pd_reset(params)


def make_reacher_pd_expert() -> FunctionalExpertPolicy:
    """Return a FunctionalExpertPolicy-wrapped Reacher PD controller.

    Plugs into Ajax's ``expert_policy=`` / ``eval_expert_policy=`` slots.
    """
    params, zero_state = make_reacher_pd()
    return FunctionalExpertPolicy(params, zero_state, reacher_pd_step)
