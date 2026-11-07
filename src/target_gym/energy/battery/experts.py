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
    """Gains and limits of the dispatch-following battery controller. The
    recipe in ``scripts/tune_industrial_experts.py`` stores the first three.
    Keep them Python floats, so the host arithmetic stays in double precision.
    """

    Kp: float
    Ki: float
    guard_margin: float
    # 5 % inside the pack's trips: MESA-1's operators advise that margin, and a
    # guard that fades to nothing on the trip itself rests on it.
    soc_min: float = 0.10
    soc_max: float = 0.90
    power_max_MW: float = 1.0
    dt: float = 5.0


@struct.dataclass
class BatteryPIDState:
    """Integral of the power tracking error."""

    integral: float


def battery_pid_step(params, state, obs, xp=jnp):
    """Power feedforward with a PI trim, faded out near either SoC limit.

    The inverter delivers its command, and the auxiliary load sits behind it, so
    the loop is a feedforward of the signal with a PI trim on the meter. On the
    block's last step the feedforward is the next block, published ahead.
    Leaving the charge window trips the pack, so the guard fades the demand out
    as the pack nears a limit in the direction that would breach it: the hard
    state-of-charge limits of a commercial controller (Xu et al., 2017), set
    5 % inside the trips.

    obs: [soc, V_cell, T_cell, P_MW, target_P_MW, next_block_MW, minutes_to_it].
    Returns the action ``(..., 1)``.
    """
    p = params
    soc = obs[..., 0]
    delivered_MW = obs[..., 3]
    target_MW = obs[..., 4]
    feedforward = xp.where(obs[..., 6] * 60.0 <= p.dt, obs[..., 5], target_MW)

    err = target_MW - delivered_MW
    demand = feedforward + p.Kp * err + p.Ki * state.integral

    # Fade the demand out only in the direction that would breach a limit.
    discharge_room = xp.clip((soc - p.soc_min) / p.guard_margin, 0.0, 1.0)
    charge_room = xp.clip((p.soc_max - soc) / p.guard_margin, 0.0, 1.0)
    room = xp.where(demand > 0.0, discharge_room, charge_room)

    # The integral stops under the guard and acts only outside it: a guard is not
    # a clip, so this step's share would reach the pack.
    shedding = room < 1.0
    grown = state.integral + err * p.dt
    demand = xp.where(shedding, demand, demand + p.Ki * err * p.dt) * room

    raw = xp.clip(demand / p.power_max_MW, -1.0, 1.0)
    # Anti-windup: stop integrating once the demand is clipped or guarded.
    held = xp.logical_or(xp.abs(raw) >= 1.0, shedding)
    integral = xp.where(held, state.integral, grown)
    return xp.stack([raw], axis=-1), BatteryPIDState(integral=integral)


def make_battery_pid_controller() -> Controller:
    """The shipped battery PID as a :class:`Controller`, gains as tuned."""
    params = BatteryPIDParams(
        Kp=gain("battery", "Kp"),
        Ki=gain("battery", "Ki"),
        guard_margin=gain("battery", "guard_margin"),
    )
    return Controller(params, BatteryPIDState(integral=0.0), battery_pid_step)
