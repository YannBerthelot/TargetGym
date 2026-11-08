"""The oracle (the MPC slot) for the battery task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). It is a stand-in, not run for room:
the expert's baseline plan on the true state of charge instead of the BMS's
report (PHYSICS.md, Reference ladder).
"""

import numpy as np


class TrueStateBaseline:
    """Declare the open period's baseline to bring the true stored energy to
    the middle of the window, after the locked baselines have run."""

    def __init__(self, env, params):
        self.env = env
        self.params = params

    def step(self, obs, state):
        from target_gym.energy.battery.env import baseline_limit

        p = self.params
        obs = np.asarray(obs, float)
        capacity_MWh = float(p.energy_capacity) / 3.6e9
        period_h = float(p.period_seconds) / 3600.0
        locked = obs[6] * obs[11] / 3600.0 + (obs[7] + obs[8]) * period_h
        at_open = float(state.soc) * capacity_MWh - locked
        baseline = (at_open - 0.5 * capacity_MWh) / period_h
        return float(np.clip(baseline / (float(baseline_limit(p)) / 1e6), -1.0, 1.0))

    def reset(self):
        pass

    def solver_report(self) -> dict:
        """No solver, so nothing to report."""
        return {}


def make_battery_mpc(env, params):
    """The battery's oracle slot: :class:`TrueStateBaseline`."""
    return TrueStateBaseline(env, params)
