"""The oracle (the MPC slot) for the unstable CSTR, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator, with the raw
  move's clip removed and the move bounded instead (T1); the feed drift on
  its noise-free mean path, a^t times its measured value;
- objective: the tracking term against the schedule's live level as
  (e / e_floor)^2, exact since e_tol = 0 and p = 2 (T2);
- limits: the trip, T_valid_min < T < T_trip and C_a >= 0, on every visited
  state, elastic at the trip cost per unit (T3);
- terminal line-out (horizon protocol, D4): the plan's last state is an
  equilibrium under its last move, each state's change over one step held
  at zero, elastic at the trip cost per unit;
- no smoothed site (T5) and no dropped guard (T4).

The cold start holds the jacket where the reset put it.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.unstable_cstr.env import N_BLOCKS, compute_velocity
from target_gym.reward import trip_cost


def unstable_cstr_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0
    a = jnp.exp(-p.delta_t / p.Ti_tau)  # the drift's mean decay per step

    def step(d, x, u, t):
        levels, clock, drift = d["task"]
        T_c = p.T_c_min + 0.5 * (u[0] + 1.0) * (p.T_c_max - p.T_c_min)  # T1
        dev = drift * a**t
        x1, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(compute_velocity, action=T_c, Ti_dev=dev, params=p),
        )  # fmt: skip
        live = levels[jnp.minimum((clock + t + 1) // p.block_steps, N_BLOCKS - 1)]
        cost = ((live - x1[0]) / p.e_floor) ** 2
        rate = compute_velocity(x1, T_c, dev * a, p)[0] * p.delta_t
        lineout = jnp.where(t == d["end"] - 1, rate, 0.0)  # D4, the last step
        y = jnp.concatenate([x1[1:2], x1[:1], lineout])
        return x1, cost, y, x1[:1], ()

    price = float(trip_cost(p))
    return TaskModel(
        nx=3, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.C_a, s.T, s.T_j]),
        data=lambda s: (s.target_levels, s.block_clock, s.Ti_dev),
        hold=lambda s: raw_of(s.T_j, p.T_c_min, p.T_c_max)[None],
        tracked=lambda s: jnp.stack([s.C_a]),
        resolution=(float(p.precision_floor),),
        limits=(Limit(p.T_valid_min, p.T_trip, price), Limit(0.0, price=price))
        + (Limit(0.0, 0.0, price),) * 3,
    )  # fmt: skip


def make_unstable_cstr_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``unstable_cstr_model``."""
    return Oracle(env, params, "unstable_cstr", unstable_cstr_model)
