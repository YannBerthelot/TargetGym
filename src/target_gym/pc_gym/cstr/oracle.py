"""The oracle (the MPC slot) for the CSTR task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem (version 5,
charter Amendment 3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator on the reactor and
  jacket, the cooling water's clip removed and the move bounded (T1), the
  feed and the scenario's event as the state carries them, held over the plan
  (certainty equivalence: a later event is not seen);
- objective: the temperature's tracking term as (e / e_floor)^2, exact since
  e_tol = 0 and p = 2 (T2);
- limit: the interlock's temperature on every visited state, elastic at the
  trip cost per kelvin (T3).

Not re-run since the rebuild.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.cstr.env import compute_velocity, events
from target_gym.reward import trip_cost


def cstr_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0

    def step(d, x, u, t):
        target, T_feed, Caf, k0 = d["task"]
        F_w = p.Fw_min + 0.5 * (u[0] + 1.0) * (p.Fw_max - p.Fw_min)  # T1
        y, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                compute_velocity, action=F_w, params=p, extra=(T_feed, Caf, k0)
            ),
        )  # fmt: skip
        cost = ((target - y[1]) / p.e_floor) ** 2
        return y, cost, y[1:2], y[1:2], ()

    def data(s):
        target, Caf, k0, _ = events(s, p, s.time * p.delta_t)
        return (s.target_T, s.T_feed, Caf, k0)

    return TaskModel(
        nx=3, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.C_a, s.T, s.T_j]),
        data=data,
        hold=lambda s: raw_of(s.F_w, p.Fw_min, p.Fw_max)[None],
        tracked=lambda s: jnp.stack([s.T]),
        resolution=(float(p.precision_floor),),
        limits=(Limit(hi=float(p.T_trip), price=float(trip_cost(p))),),
    )  # fmt: skip


def make_cstr_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``cstr_model``."""
    return Oracle(env, params, "cstr", cstr_model)
