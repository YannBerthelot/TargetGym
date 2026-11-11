"""The oracle (the MPC slot) for the LDPE autoclave, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem (version 5,
charter Amendment 3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own step (``env.advance``: the initiator exact per
  substep, RK4 on the monomer and temperature), the initiator's clip removed
  and the move bounded (T1), the scenario's event as the state carries it,
  held over the plan (a later event is not seen);
- objective: the temperature's tracking term as (e / e_floor)^2, exact since
  e_tol = 0 and p = 2 (T2);
- limit: the trip temperature on every visited state, elastic at the trip cost
  per kelvin (T3).

Not run since the rebuild.
"""

import jax.numpy as jnp

from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.unstable_cstr.env import advance, events
from target_gym.reward import trip_cost


def autoclave_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0

    def step(d, x, u, t):
        target, T_f, f, scale = d["task"]
        I_f = p.I_f_min + 0.5 * (u[0] + 1.0) * (p.I_f_max - p.I_f_min)  # T1
        y = advance(x, I_f, p, (T_f, f, scale), env.integration_method)
        cost = ((target - y[2]) / p.e_floor) ** 2
        return y, cost, y[2:3], y[2:3], ()

    def data(s):
        _, T_f, f, scale = events(s, p, s.time * p.delta_t)
        return (s.target_T, T_f, f, scale)

    return TaskModel(
        nx=3, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.I, s.M, s.T]),
        data=data,
        hold=lambda s: raw_of(s.I_f, p.I_f_min, p.I_f_max)[None],
        tracked=lambda s: jnp.stack([s.T]),
        resolution=(float(p.precision_floor),),
        limits=(Limit(hi=float(p.T_trip), price=float(trip_cost(p))),),
    )  # fmt: skip


def make_unstable_cstr_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``autoclave_model``."""
    return Oracle(env, params, "unstable_cstr", autoclave_model)


#: The reward terms the ceiling's solver keeps (``pc_gym.foresight``):
#: the tracking term; the trip is the margin.
CEILING_TERMS = ("tracking",)

#: The ceiling's stretches (``foresight.solve``'s ``window``): 160 steps
#: planned, twice the horizon P = 80 of ``data/oracle_horizons``, the first 20
#: kept. The autoclave is open-loop unstable, so float32 rounding grows along a
#: plan; over the whole 1800 s episode it swamps the plan's gradient.
CEILING_WINDOW = (160, 20)


def _x(s):
    return jnp.stack([s.I, s.M, s.T])


def _put(s, x):
    return s.replace(I=x[0], M=x[1], T=x[2])


#: The ceiling's replay (``foresight.TrackingPolicy``): its plan held on its
#: path by the reactor temperature, entry 2 of (I, M, T). The plant is open-
#: loop unstable, so the benchmark's batched float32 rounding, a bit off the
#: planner's, would otherwise grow along the plan.
CEILING_FEEDBACK = (_x, _put, 2)


def ceiling_margins(state, p):
    """The ceiling's margins (``pc_gym.foresight``): the trip temperature,
    ``T_trip - T`` in kelvin (``e_floor``), backed off one; the price is the
    trip's cost."""
    return jnp.stack([(p.T_trip - state.T) / p.e_floor - 1.0]), trip_cost(p)
