"""The expert (the PID slot) for
the 2D aircraft tasks (plane, plane_sine, plane_energy).

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
from flax import struct

from target_gym.experts import pid as _pid_mod
from target_gym.plane.experts_tecs import (
    AIRLINER,
    TECSParams,
    TECSState,
    pitch_hold,
    tecs_step,
)

# ---------------------------------------------------------------------------
# The expert: TECS over the pitch loop (host NumPy and jax.jit)
# ---------------------------------------------------------------------------
#
# What a fixed-wing autopilot flies for height and speed (PX4, ArduPilot), and
# the 3D expert's longitudinal half with the same tuning (plane3d.experts).
# The cascade above stays as the oracle's speed loop and hold.


@struct.dataclass
class PlaneTECSParams:
    """TECS (``experts_tecs``) and the pitch loop its demand goes to, with
    the angle of attack kept inside ``alpha_max``."""

    Kp_theta: float = 8.0
    Ki_theta: float = 2.0
    Kd_theta: float = 0.3
    alpha_max: float = 0.19
    dt: float = 1.0
    tecs: TECSParams = AIRLINER


@struct.dataclass
class PlaneTECSState:
    """TECS's state and the pitch loop's integral."""

    tecs: TECSState = struct.field(default_factory=TECSState)
    theta_integ: float = 0.0


def plane_tecs_step(params, state, obs, xp=jnp):
    """One step of the expert: ``([power, stick], new_state)``."""
    x_dot, z, z_dot = obs[..., 0], obs[..., 1], obs[..., 2]
    theta, theta_dot, gamma = obs[..., 3], obs[..., 4], obs[..., 5]
    v = xp.sqrt(x_dot**2 + z_dot**2)
    power, pitch, tecs = tecs_step(
        params.tecs, state.tecs, z, z_dot, v, obs[..., 6], obs[..., 9], 0.0, xp
    )
    theta_cmd = xp.clip(pitch, gamma - params.alpha_max, gamma + params.alpha_max)
    stick, theta_integ = pitch_hold(
        params, state.theta_integ, theta_cmd, theta, theta_dot, xp
    )
    new_state = PlaneTECSState(tecs=tecs, theta_integ=theta_integ)
    return xp.stack([power, stick], axis=-1), new_state


def make_plane_pid_controller() -> _pid_mod.Controller:
    """The 2D aircraft's expert, TECS over the pitch loop, as a
    ``Controller``."""
    return _pid_mod.Controller(PlaneTECSParams(), PlaneTECSState(), plane_tecs_step)
