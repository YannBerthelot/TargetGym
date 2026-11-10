"""The oracle (the MPC slot) for the four-tank task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator, with the raw
  moves' clips removed and the moves bounded instead (T1);
- the extra outlet as the state carries it at the solve, held open over the
  plan (certainty equivalence, as the CSTR holds its event in force): an
  outlet not yet open, its closing, the cup and the setpoint moves are not
  in the plan;
- objective: both lower tanks' tracking terms as (e / e_floor)^2, exact
  since e_tol = 0 and p = 2 (T2);
- limits: every level inside the tank, 0 <= h <= h_max, on every visited
  state, elastic at the worst step's tracking cost per mm (T3; ours): the rig
  has no trip, so its trip cost (0) cannot price a limit, and the solver
  divides by the price. An empty or overflowing tank is physical, so any
  price no plan breaches for will do; the worst step,
  2 (h_max / e_floor)^2, is one no tracking gain outweighs;
- dropped guard: the outflow's sqrt(max(h, 0)) (T4), kept off its edge by
  the band; every plan's levels are checked against 0.

The reset state holds both pumps at the operating point.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.four_tank.env import compute_velocity


def four_tank_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0

    def step(d, x, u, t):
        targets, leak = d["task"]
        v = p.v_min + 0.5 * (u + 1.0) * (p.v_max - p.v_min)  # T1
        h, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(compute_velocity, action=v, params=p, leak=leak),
        )  # fmt: skip
        cost = jnp.sum(((targets - h[:2]) / p.e_floor) ** 2)
        return h, cost, h, h[:2], h  # the guard's edge is h = 0

    def data(s):
        # The outlet open at the solve, as compute_next_state opens it.
        t = s.time
        open_ = (t >= s.leak_start) & (t < s.leak_start + p.leak_duration)
        area = jnp.where(open_, p.leak_fraction, 0.0) * jnp.stack(
            [(s.leak_tank == 0) * p.a3, (s.leak_tank == 1) * p.a4]
        )
        return jnp.stack([s.target_h1, s.target_h2]), area

    worst_step = 2.0 * (float(p.h_max) / float(p.e_floor)) ** 2
    band = Limit(0.0, p.h_max, worst_step * 1e3)  # per mm, in metres
    return TaskModel(
        nx=4, nu=2, step=step,
        state_x=lambda s: jnp.stack([s.h1, s.h2, s.h3, s.h4]),
        data=data,
        hold=lambda s: raw_of(jnp.stack([s.v1, s.v2]), p.v_min, p.v_max),
        tracked=lambda s: jnp.stack([s.h1, s.h2]),
        resolution=(float(p.precision_floor),) * 2,
        limits=(band,) * 4,
    )  # fmt: skip


def make_four_tank_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``four_tank_model``."""
    return Oracle(env, params, "four_tank", four_tank_model)
