"""The oracle (the MPC slot) for the battery task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The slot holds a stand-in for 0.8.0,
labelled non-compliant by the owner on 2026-11-07: it wins 9 of 10 seeds, not every
seed (PHYSICS.md). The solver's planner is in ``oracles/battery.py``, out of the slot.
"""

import numpy as np


def depth_bound(params) -> float:
    """The half-cycle depth past which the next MWh wears more than its imbalance:
    ``(pi eta + theta / eta) / R`` on the stress function's slope (Xu et al.,
    2017, eq. 11), with one price each way and one-way efficiency ``eta``."""
    from target_gym.energy.battery.env import round_trip_efficiency

    p = params
    eta = float(round_trip_efficiency(p.power_max, 0.5, p))
    pack_price = float(p.fade_price) * float(p.life_fade) * 1e3  # $/MWh of capacity
    pay = float(p.imbalance_price) * (eta + 1.0 / eta) / pack_price
    slope = float(p.stress_alpha) * float(p.stress_beta)
    return (pay / slope) ** (1.0 / (float(p.stress_beta) - 1.0))


class ScheduleFeedforward:
    """Command the signal's next value, inside the published cycle-depth bound.

    The target is the scheduled level plus an AR(1) movement drawn after the
    action, so the best a causal controller can do is the level of the block the
    step is scored against, plus the movement's mean. What is left is the
    innovation, whose expected absolute value is ``e_floor``.

    The energy decides how far to follow. Xu, Shi, Kirschen and Zhang (2017,
    Algorithm 1) keep the running lowest and highest state of charge, and stop
    following once their distance reaches ``depth_bound``, or at the charge
    window. They prove the gap to an offline optimum does not grow with time.

    It holds the slot for 0.8.0, labelled non-compliant: it wins 9 of 10 seeds, not
    every seed, so it does not meet the oracle rule (PHYSICS.md). It reads the true
    state, so it has no planning horizon, and scripts/audit_mpc_horizons.py reports
    n/a.
    """

    #: How far inside the charge window it stops: the guard against the trip.
    MARGIN = 0.01
    V_MARGIN = 0.02  # V per cell, inside the cell's voltage window

    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.bound = depth_bound(params)
        self.low = self.high = None

    def step(self, _obs, state):
        """Return the next action. ``_obs`` is ignored (kept for API symmetry)."""
        from target_gym.energy.battery.env import dispatch_block, open_circuit_voltage

        p = self.params
        soc = float(state.soc)
        self.low = soc if self.low is None else min(self.low, soc)
        self.high = soc if self.high is None else max(self.high, soc)
        level = float(state.dispatch_schedule[dispatch_block(state.time + 1, p)])
        mean = float(np.exp(-p.delta_t / p.regulation_tau)) * float(state.regulation)
        command = level + mean + float(p.aux_load)

        floor = max(float(p.soc_min) + self.MARGIN, self.high - self.bound)
        roof = min(float(p.soc_max) - self.MARGIN, self.low + self.bound)
        per_soc = float(p.pack_kWh) * 3.6e6 / float(p.delta_t)  # W per unit of charge
        command = np.clip(command, (soc - roof) * per_soc, (soc - floor) * per_soc)

        # The cell window binds before the charge window at high current.
        drive = float(open_circuit_voltage(soc, p)) - float(state.v_rc)
        v_low = (float(p.V_cell_min) + self.V_MARGIN) * float(p.n_series)
        v_high = (float(p.V_cell_max) - self.V_MARGIN) * float(p.n_series)
        most = v_low * max(drive - v_low, 0.0) / float(p.R0)
        least = -v_high * max(v_high - drive, 0.0) / float(p.R0)
        command = np.clip(command, least, most)
        return float(np.clip(command / p.power_max, -1.0, 1.0))

    def reset(self):
        self.low = self.high = None

    def solver_report(self) -> dict:
        """No solver, so nothing to report."""
        return {}


def make_battery_mpc(env, params):
    """The battery's oracle: :class:`ScheduleFeedforward`."""
    return ScheduleFeedforward(env, params)
