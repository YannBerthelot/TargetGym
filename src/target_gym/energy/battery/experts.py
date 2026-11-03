"""The expert (the PID slot) for the battery task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import Controller, gain

# ---------------------------------------------------------------------------
# The shipped PID as one pure step (host and jit)
# ---------------------------------------------------------------------------


@struct.dataclass
class BatteryPIDParams:
    """Gains and limits of the dispatch-following battery controller.

    These are the values the baseline was scored with. Keep them Python
    floats so the host arithmetic on constants runs in double precision, as
    it did when the baseline was recorded.
    """

    Kp: float = 0.5
    Ki: float = 0.02
    guard_margin: float = 0.12
    soc_min: float = 0.05
    soc_max: float = 0.95
    power_max_MW: float = 1.0
    dt: float = 5.0


@struct.dataclass
class BatteryPIDState:
    """Integral of the power tracking error."""

    integral: float


def battery_pid_step(params, state, obs, xp=jnp):
    """Power feedforward with a PI trim, faded out near either SoC limit.

    Power is close to a direct feedthrough (commanding it delivers it, minus
    conversion losses), so the loop is mostly feedforward with a small
    proportional trim for the losses. Hitting either state-of-charge limit
    trips the pack, so the guard fades the demand out as the pack nears a
    limit in the direction that would breach it. Over the scored 30 min
    episode it is insurance: across 2000 seeds it acts on 4, and a copy
    without it never trips either (2026-10-01; see the battery's PHYSICS.md,
    section 5).

    obs: [soc, V_cell, T_cell, P_MW, target_P_MW]. Keep the arithmetic order:
    under ``numpy`` it reproduces the scored baseline bit for bit.
    """
    p = params
    soc = obs[..., 0]
    delivered_MW = obs[..., 3]
    target_MW = obs[..., 4]

    err = target_MW - delivered_MW
    integral = state.integral + err * p.dt
    demand = target_MW + p.Kp * err + p.Ki * integral

    # Fade the demand out only in the direction that would breach a limit.
    discharge_room = xp.clip((soc - p.soc_min) / p.guard_margin, 0.0, 1.0)
    charge_room = xp.clip((p.soc_max - soc) / p.guard_margin, 0.0, 1.0)
    demand = xp.where(demand > 0.0, demand * discharge_room, demand * charge_room)

    raw = xp.clip(demand / p.power_max_MW, -1.0, 1.0)
    # Anti-windup: stop integrating once the demand is clipped or guarded.
    integral = xp.where(xp.abs(raw) >= 1.0, integral - err * p.dt, integral)
    return raw, BatteryPIDState(integral=integral)


def make_battery_pid_controller() -> Controller:
    """The shipped battery PID as a :class:`Controller`, gains as tuned."""
    params = BatteryPIDParams(
        Kp=gain("battery", "Kp", 0.5),
        Ki=gain("battery", "Ki", 0.02),
        guard_margin=gain("battery", "guard_margin", 0.12),
    )
    return Controller(params, BatteryPIDState(integral=0.0), battery_pid_step)
