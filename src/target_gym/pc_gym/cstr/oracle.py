"""The oracle (the MPC slot) for the CSTR task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator, with the raw
  move's clip removed and the move bounded instead (T1);
- objective: the reward's tracking term as (e / e_floor)^2, exact since
  e_tol = 0 and p = 2 (T2);
- no limit: the task's trip test is documented unreachable and constant
  (T3, nothing binds); no smoothed site (T5) and no dropped guard (T4).

The reset state carries no coolant temperature (its ``T_c`` is 0 until the
first step), so the cold start holds the raw move that value maps to,
clipped to the box: full cooling.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.pc_gym.cstr.env import compute_velocity


def cstr_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0

    def step(d, x, u, t):
        T_c = p.T_c_min + 0.5 * (u[0] + 1.0) * (p.T_c_max - p.T_c_min)  # T1
        (C_a, T), _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(compute_velocity, action=T_c, params=p),
        )  # fmt: skip
        cost = ((d["task"] - C_a) / p.e_floor) ** 2
        return jnp.stack([C_a, T]), cost, jnp.zeros(0), jnp.stack([C_a]), ()

    return TaskModel(
        nx=2, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.C_a, s.T]),
        data=lambda s: s.target_CA,
        hold=lambda s: raw_of(s.T_c, p.T_c_min, p.T_c_max)[None],
        tracked=lambda s: jnp.stack([s.C_a]),
        resolution=(float(p.precision_floor),),
    )  # fmt: skip


def make_cstr_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``cstr_model``."""
    return Oracle(env, params, "cstr", cstr_model)
