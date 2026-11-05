"""The oracle (the MPC slot) for the first-order task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator, with the raw
  move's clip removed and the move bounded instead (T1);
- objective: the reward's tracking term as (e / e_floor)^2, exact since
  e_tol = 0 and p = 2 (T2);
- limit: the trip band x_min < x < x_max on every visited state, elastic at
  the trip cost per unit (T3; it never binds);
- no smoothed site (T5) and no dropped guard (T4).
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.first_order.env import compute_velocity
from target_gym.reward import trip_cost


def first_order_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0

    def step(d, x, u, t):
        action = p.u_min + 0.5 * (u[0] + 1.0) * (p.u_max - p.u_min)  # no clip: T1
        (x1,), _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(compute_velocity, action=action, params=p),
        )  # fmt: skip
        cost = ((d["task"] - x1) / p.e_floor) ** 2
        return jnp.stack([x1]), cost, jnp.stack([x1]), jnp.stack([x1]), ()

    return TaskModel(
        nx=1, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.x]),
        data=lambda s: s.target_x,
        hold=lambda s: raw_of(s.u, p.u_min, p.u_max)[None],
        tracked=lambda s: jnp.stack([s.x]),
        resolution=(float(p.precision_floor),),
        limits=(Limit(p.x_min, p.x_max, float(trip_cost(p))),),
    )  # fmt: skip


def make_first_order_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``first_order_model``."""
    return Oracle(env, params, "first_order", first_order_model)
