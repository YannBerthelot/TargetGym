"""The oracle (the MPC slot) for the unstable CSTR, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator behind the valve's
  lag, with the raw move's clip removed and the move bounded instead (T1),
  within the valve's slew from where it stands, so the limit never binds (T1r);
  the feed drift on its noise-free mean path, a^t times its measured value;
- objective: the tracking term against the schedule's ramped target as
  (e / e_floor)^2, exact since e_tol = 0 and p = 2 (T2);
- limits: the trip, T_valid_min < T < T_trip and C_a >= 0, on every visited
  state, elastic at the trip cost per unit (T3);
- terminal line-out (horizon protocol, D4): the plan's last state is an
  equilibrium under its last move, each balance's change over one step held
  at zero with the coolant at the move's own, elastic at the trip cost per
  unit;
- no smoothed site (T5) and no dropped guard (T4).

The cold start holds the valve where it stands.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.unstable_cstr.env import compute_velocity, scheduled_level
from target_gym.reward import trip_cost


def unstable_cstr_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0
    assert float(p.actuator_lag) > 0.0 and float(p.actuator_rate) > 0.0
    a = jnp.exp(-p.delta_t / p.Ti_tau)  # the drift's mean decay per step
    closes = 1.0 - jnp.exp(-p.delta_t / p.actuator_lag)  # of the valve's gap
    slew = 2.0 * p.actuator_rate * p.delta_t  # raw moves span 2

    def coolant(raw):
        return p.T_c_min + 0.5 * (raw + 1.0) * (p.T_c_max - p.T_c_min)  # T1

    def step(d, x, u, t):
        levels, clock, drift = d["task"]
        valve = x[3] + closes * (u[0] - x[3])
        dev = drift * a**t
        x1, _ = integrate_dynamics(
            positions=x[:3], delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                compute_velocity, action=coolant(valve), Ti_dev=dev, params=p
            ),
        )  # fmt: skip
        live = scheduled_level(levels, clock + t + 1, p)
        cost = ((live - x1[0]) / p.e_floor) ** 2
        rate = compute_velocity(x1, coolant(u[0]), dev * a, p)[0] * p.delta_t
        lineout = jnp.where(t == d["end"] - 1, rate, 0.0)  # D4, the last step
        y = jnp.concatenate([x1[1:2], x1[:1], lineout])
        return jnp.append(x1, valve), cost, y, x1[:1], ()

    def within_slew(x, u, steps):
        """T1r, rows >= 0: the valve's first move, its largest over a hold,
        stays inside its slew limit, so ``step`` needs no clip. A command past
        that moves the valve as one at the limit does."""
        moved = closes * (u[0] - x[3])
        return jnp.stack([slew - moved, slew + moved])

    price = float(trip_cost(p))
    return TaskModel(
        nx=4, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.C_a, s.T, s.T_j, s.actuator_held[0]]),
        data=lambda s: (s.target_levels, s.block_clock, s.Ti_dev),
        hold=lambda s: s.actuator_held,
        move_rows=within_slew,
        move_scale=lambda steps: (float(slew / closes),),
        tracked=lambda s: jnp.stack([s.C_a]),
        resolution=(float(p.precision_floor),),
        limits=(Limit(p.T_valid_min, p.T_trip, price), Limit(0.0, price=price))
        + (Limit(0.0, 0.0, price),) * 3,
    )  # fmt: skip


def make_unstable_cstr_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``unstable_cstr_model``."""
    return Oracle(env, params, "unstable_cstr", unstable_cstr_model)
