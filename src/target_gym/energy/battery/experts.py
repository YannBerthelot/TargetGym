"""Reference controller (the MPC slot) for the battery task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.pid import Controller, _g


class ScheduleFeedforward:
    """Command the dispatch level the next step is scored against.

    The battery's delivered power equals its command within the step, and the
    target is the schedule's block level plus white noise drawn after the
    action. So the best a causal controller can do is command the level of
    the block the step is scored against, ``dispatch_block(t + 1)``: what is
    left is the noise, whose expected absolute value is ``e_floor`` exactly.
    No optimiser is needed: this is the tracking optimum.

    Measured on protocol seeds 0-2: gain
    7.82e-4 against 1.115e-3 for the GradientMPC it replaces (-30%), all of it
    tracking (mean error 1577 W against 3947 W, with e_floor 1596 W). A
    controller that knew the noise would remove only the noise. Shading the
    command toward zero to save degradation does slightly better, by about
    0.03% on the protocol seeds with a 100 W shade (measured in review), which
    is not worth a tuned constant.

    The state of charge is not guarded. Over 2000 seeds of the scored
    30-minute episode it stays within 0.17-0.85 and never trips (limits
    0.05-0.95); over the full 60-minute schedule, within 0.10-0.92, still with
    no trip. A guard would only ever move the command away from the target.

    It has no planning horizon, so scripts/audit_mpc_horizons.py reports n/a.
    """

    def __init__(self, env, params):
        self.env = env
        self.params = params

    def step(self, _obs, state):
        """Return the next action. ``_obs`` is ignored (kept for API symmetry)."""
        from target_gym.energy.battery.env import dispatch_block

        p = self.params
        level = float(state.dispatch_schedule[dispatch_block(state.time + 1, p)])
        return float(np.clip(level / p.power_max, -1.0, 1.0))

    def reset(self):
        pass

    def solver_report(self) -> dict:
        """No solver, so nothing to report."""
        return {}


def make_battery_mpc(env, params):
    """The battery's oracle: :class:`ScheduleFeedforward`."""
    return ScheduleFeedforward(env, params)


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
        Kp=_g("battery", "Kp", 0.5),
        Ki=_g("battery", "Ki", 0.02),
        guard_margin=_g("battery", "guard_margin", 0.12),
    )
    return Controller(params, BatteryPIDState(integral=0.0), battery_pid_step)
