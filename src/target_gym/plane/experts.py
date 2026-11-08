"""The expert (the PID slot) for
the 2D aircraft tasks (plane, plane_sine, plane_energy).

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
from flax import struct

from target_gym.experts import pid as _pid_mod
from target_gym.plane.dynamics import PITCH_DEMAND_MAX
from target_gym.plane.experts_tecs import (
    AIRLINER,
    TECSParams,
    TECSState,
    tecs_step,
)

# ---------------------------------------------------------------------------
# The expert: TECS, whose pitch demand the plant's inner loop holds (host NumPy
# and jax.jit)
# ---------------------------------------------------------------------------
#
# What a fixed-wing autopilot flies for height and speed (PX4, ArduPilot), and
# the 3D expert's longitudinal half with the same tuning (plane3d.experts).
# The cascade above stays as the oracle's speed loop and hold.


@struct.dataclass
class PlaneTECSParams:
    """TECS (``experts_tecs``), with the angle of attack its pitch demand
    allows kept inside ``alpha_max``."""

    alpha_max: float = 0.19
    dt: float = 1.0
    tecs: TECSParams = AIRLINER


@struct.dataclass
class PlaneTECSState:
    """TECS's state."""

    tecs: TECSState = struct.field(default_factory=TECSState)


def plane_tecs_step(params, state, obs, xp=jnp):
    """One step of the expert: ``([throttle, pitch demand], new_state)``."""
    z, z_dot, gamma = obs[..., 1], obs[..., 2], obs[..., 5]
    # TECS holds airspeed, from air data [TECS], not ground speed.
    v = obs[..., 10]
    power, pitch, tecs = tecs_step(
        params.tecs, state.tecs, z, z_dot, v, obs[..., 6], obs[..., 9], 0.0, xp
    )
    theta_cmd = xp.clip(pitch, gamma - params.alpha_max, gamma + params.alpha_max)
    action = xp.stack([power, theta_cmd / PITCH_DEMAND_MAX], axis=-1)
    return action, PlaneTECSState(tecs=tecs)


def make_plane_pid_controller() -> _pid_mod.Controller:
    """The 2D aircraft's expert, TECS, as a ``Controller``."""
    return _pid_mod.Controller(PlaneTECSParams(), PlaneTECSState(), plane_tecs_step)
