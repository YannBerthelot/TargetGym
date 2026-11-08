"""The expert (the PID slot) for the battery task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and the
oracle each have a fingerprint of their own (``provenance.expert_fingerprint``,
``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import Controller


@struct.dataclass
class BatteryPIDParams:
    """The unit's numbers the baseline planner needs (BatteryParams defaults).
    Python floats, so the host arithmetic stays in double precision."""

    capacity_MWh: float = 3854.0e-3
    baseline_limit_MW: float = 0.4 * 1.927 / 1.4
    period_s: float = 1800.0
    target_soc: float = 0.5


@struct.dataclass
class BatteryPIDState:
    """Nothing to remember: the plan is read off the observation."""

    integral: float


def battery_pid_step(params, state, obs, xp=jnp):
    """Declare the open period's baseline to bring the reported energy back to
    the middle of the window over that period.

    NESO's guidance is to "return the stored energy to an appropriate level by
    following the submitted operational baselines" (NESO-GUIDE p. 37, item 4).
    The middle is where the export and import requirements are equally far.
    The energy the locked baselines will move before the open period starts
    is taken off first: the current period's rest, then the next two.

    obs: see ``env.get_obs``. Returns the action ``(..., 1)``.
    """
    p = params
    reported_MWh = obs[..., 0] * p.capacity_MWh
    to_next_h = obs[..., 11] / 3600.0
    locked_MWh = obs[..., 6] * to_next_h + (obs[..., 7] + obs[..., 8]) * (
        p.period_s / 3600.0
    )
    at_open_MWh = reported_MWh - locked_MWh
    baseline_MW = (at_open_MWh - p.target_soc * p.capacity_MWh) / (p.period_s / 3600.0)
    raw = xp.clip(baseline_MW / p.baseline_limit_MW, -1.0, 1.0)
    return xp.stack([raw], axis=-1), state


def make_battery_pid_controller() -> Controller:
    """The shipped battery baseline planner as a :class:`Controller`."""
    return Controller(
        BatteryPIDParams(), BatteryPIDState(integral=0.0), battery_pid_step
    )
