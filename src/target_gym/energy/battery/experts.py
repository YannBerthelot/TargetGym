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
    #: The plant's base day-ahead price, GBP/MWh, and round-trip efficiency.
    price_base: float = 74.0
    efficiency: float = 0.920
    #: Rule 22's back-off from the window's edge, as a fraction of the capacity:
    #: 2 sigma, sigma 0.0501 the reported charge's spread at period starts that
    #: the reference ladder's bump test measured (scripts/ladder_rows/battery.py,
    #: "charge on baseline", last placement round).
    back_off: float = 2 * 0.0501


@struct.dataclass
class BatteryPIDState:
    """Nothing to remember: the plan is read off the observation."""

    integral: float


def battery_pid_step(params, state, obs, xp=jnp):
    """Declare the open period's baseline to bring the reported energy to the
    period's target over that period, after the locked baselines.

    The recipe (owner, 2026-11-15), fixed before any run:

    - NESO's guidance is to "return the stored energy to an appropriate level by
      following the submitted operational baselines" (NESO-GUIDE p. 37, item 4).
      The energy the locked baselines will move before the open period starts is
      taken off first: the current period's rest, then the next two.
    - The appropriate level is priced. Stored energy is valued at the base price
      and buying it back loses the round trip (Sioshansi et al. 2009, sec. 2:
      a device charges 1/eta per unit it discharges). So holding more than the
      export requirement asks only loses, and storing pays when the open period's
      day-ahead price is below the base price times the efficiency.
    - The target is therefore the window's low edge plus rule 22's back-off,
      or, when storing pays, its high edge less the back-off. A price above the
      base keeps the low target: the energy above the requirement is sold down.

    obs: see ``env.get_obs``. Returns the action ``(..., 1)``.
    """
    p = params
    reported_MWh = obs[..., 0] * p.capacity_MWh
    to_next_h = obs[..., 11] / 3600.0
    locked_MWh = obs[..., 6] * to_next_h + (obs[..., 7] + obs[..., 8]) * (
        p.period_s / 3600.0
    )
    at_open_MWh = reported_MWh - locked_MWh
    low = obs[..., 9] / p.capacity_MWh + p.back_off
    high = 1.0 - obs[..., 10] / p.capacity_MWh - p.back_off
    store = obs[..., 16] < p.price_base * p.efficiency
    target = xp.where(store, high, low)
    baseline_MW = (at_open_MWh - target * p.capacity_MWh) / (p.period_s / 3600.0)
    raw = xp.clip(baseline_MW / p.baseline_limit_MW, -1.0, 1.0)
    return xp.stack([raw], axis=-1), state


def make_battery_pid_controller() -> Controller:
    """The shipped battery baseline planner as a :class:`Controller`."""
    return Controller(
        BatteryPIDParams(), BatteryPIDState(integral=0.0), battery_pid_step
    )
