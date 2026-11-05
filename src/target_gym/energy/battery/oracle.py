"""The oracle (the MPC slot) for the battery task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator, with the raw
  move's clip removed and the move bounded instead (T1); the fade state left
  out, since it feeds nothing but the reward; the schedule's block levels
  along the plan, the white noise on them zeroed;
- objective: the imbalance through epigraph rows on every visited state,
  linear in |error| since e_tol = 0 and p = 1 (T2), and the fade above its
  hold rate as one epigraph term per sign of the current, which is max(c +
  k |I| - c_hold, 0) while the calendar rate c stays under c_hold: both are
  then never positive together (T2, the guard below);
- limits: the state of charge, cell voltage and temperature, on every visited
  state, elastic at the trip cost per unit (T3); the command within 0.999 of
  the deliverable power at each step's end (T1s);
- dropped guards (T4): the charge, discriminant and power clips of
  ``current_for_power`` and the charge clip of ``open_circuit_voltage``, which
  those limits keep off their edges; and the calendar rate under c_hold.

The reward is in dollars, so the tie-break's budget is what half a resolution
costs in one step in dollars (G10). The cold start holds the delivered power.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.energy.battery.env import (
    compute_velocity,
    current_for_power,
    degradation_rate,
    dispatch_block,
    max_deliverable_power,
    terminal_voltage,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost


def battery_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 1.0
    per_W = float(p.imbalance_price / 1.0e6 * p.delta_t / 3600.0)
    floor = per_W * float(p.e_floor)
    worth = float(p.fade_price * p.pack_kWh) / floor

    def velocity(x, action, params):
        return compute_velocity(x, action, params)[0][:3], None

    def step(d, x, u, t):
        schedule, start = d["task"]
        command = u[0] * p.power_max  # T1
        x1, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(velocity, action=command, params=p),
        )  # fmt: skip
        soc, v_rc, T = x1
        current = current_for_power(command, soc, v_rc, p)
        V = terminal_voltage(current, soc, v_rc, p)
        power = V * current
        level = schedule[dispatch_block(start + t + 1, p)]
        calendar = degradation_rate(0.0, T, p)  # a literal current: no abs
        cycling = (degradation_rate(1.0, T, p) - calendar) * current
        room = 0.999 * max_deliverable_power(soc, v_rc, p)
        y = jnp.stack([
            (level - power) / p.e_floor,
            worth * ((calendar + cycling) * p.delta_t - p.c_hold),
            worth * ((calendar - cycling) * p.delta_t - p.c_hold),
            soc, V / p.n_series, T,
            (room - command) / p.power_max, (room + command) / p.power_max,
        ])  # fmt: skip
        guard = p.c_hold - calendar * p.delta_t
        return x1, jnp.zeros((), x.dtype), y, power[None], (guard,)

    price = float(trip_cost(p))
    return TaskModel(
        nx=3, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.soc, s.v_rc, s.T_cell]),
        data=lambda s: (s.dispatch_schedule, s.time),
        hold=lambda s: raw_of(s.power, -p.power_max, p.power_max)[None],
        tracked=lambda s: s.power[None],
        resolution=(float(p.precision_floor),),
        hinges=(
            Hinge(weight=floor, absolute=True),
            Hinge(weight=floor), Hinge(weight=floor),
        ),
        limits=(
            Limit(p.soc_min, p.soc_max, price),
            Limit(p.V_cell_min, p.V_cell_max, price),
            Limit(hi=p.T_max, price=price),
            Limit(lo=0.0, price=price), Limit(lo=0.0, price=price),
        ),
        tie_budget=per_W * 0.5 * float(p.precision_floor),
    )  # fmt: skip


def make_battery_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``battery_model``."""
    return Oracle(env, params, "battery", battery_model)
