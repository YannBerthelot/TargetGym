"""Reference controller (the MPC slot) for the battery task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import numpy as np


class ScheduleFeedforward:
    """Command the dispatch level the next step is scored against.

    The battery's delivered power equals its command within the step, and the
    target is the schedule's block level plus white noise drawn after the
    action. So the best a causal controller can do is command the level of
    the block the step is scored against, ``dispatch_block(t + 1)``: what is
    left is the noise, whose expected absolute value is ``e_floor`` exactly.
    No optimiser is needed: this is the tracking optimum.

    Measured in the oracle audit (2026-10-01, protocol seeds 0-2): gain
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
