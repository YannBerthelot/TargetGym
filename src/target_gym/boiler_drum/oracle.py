"""The oracle (the MPC slot) for the boiler drum task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator behind the fuel's
  and the feedwater's rate limits and lags: each move is an actuator's rate, a
  fraction of its limit, sent as the command that makes it (T1rr,
  ``oracles.rates``), with that command inside the box (T1); the turbine valve
  on the episode's load plan, which the oracle knows ahead as the operator does,
  upset included (a ceiling cheats on purpose); the upset's firing and feed caps;
- objective: the score's steam off the plan past the meter's resolution, at
  the imbalance price in GBP per step (T2), the absolute value and the dead
  band each smoothed by a square root of a square;
- limits: the level between the two trips on every visited state, elastic at
  the trip cost per unit (T3). The trip's 10 s delay is not used: the plan
  never leaves the band, which is stricter than the plant;
- rest (G12): feedwater is a spare move, the level being limited but not
  priced, so it rests at normal water level;
- dropped guards (T4): the void, circulation, pressure and volume clips,
  which the level limits keep off their edges.

Not yet run on this plant (rule 9's room is "not measured").

The cold start holds both actuators where they stand.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.boiler_drum.env import (
    actuator_caps,
    compute_velocity,
    drum_level,
    imbalance_cost,
    load_plan,
    steam_draw,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel
from target_gym.oracles.rates import RateLimit
from target_gym.oracles.shooting import Limit
from target_gym.reward import trip_cost


def boiler_model(env, p) -> TaskModel:
    rate = RateLimit.of(env, p)

    def step(d, x, u, t):
        plan, level_ref, now = d["task"]
        held = rate.reached(x[4:], u)
        Q, q_feed = (
            0.5 * (held[0] + 1.0) * p.Q_max, 0.5 * (held[1] + 1.0) * p.q_feed_max
        )  # fmt: skip
        Q_cap, q_cap = actuator_caps(plan, (now + t) * p.delta_t, p)
        Q, q_feed = jnp.minimum(Q, Q_cap), jnp.minimum(q_feed, q_cap)
        q_demand = load_plan(plan, (now + t + 1) * p.delta_t, p)
        x1, _ = integrate_dynamics(
            positions=x[:4], delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                compute_velocity, action=(Q, q_feed), q_demand=q_demand, params=p
            ),
        )  # fmt: skip
        level = drum_level(x1[1], x1[2], x1[3], level_ref, x1[0], p)
        off = steam_draw(q_demand, x1[0], p) - q_demand
        past = jnp.sqrt(off * off + 1e-6) - p.steam_resolution
        kg = 0.5 * (past + jnp.sqrt(past * past + 1e-6)) * p.delta_t
        cost = imbalance_cost(kg, p)
        tracked = jnp.stack([level, x1[0]])
        return jnp.concatenate([x1, held]), cost, jnp.stack([level]), tracked, ()

    price = float(trip_cost(p))
    return TaskModel(
        nx=6, nu=2, step=step,
        state_x=lambda s: jnp.concatenate([
            jnp.stack([s.pressure, s.V_wt, s.alpha_r, s.V_sd]), s.actuator_held,
        ]),
        data=lambda s: (s, s.level_ref, s.time),
        hold=lambda s: jnp.zeros(2, s.actuator_held.dtype),
        action=lambda s, u: rate.command(s.actuator_held, u),
        move_rows=lambda x, u, steps: rate.rows(x[4:], u, steps),
        tracked=lambda s: jnp.stack([s.level, s.pressure]),
        resolution=(
            float(p.level_precision_floor), float(p.pressure_precision_floor)
        ),
        limits=(Limit(p.level_trip_low, p.level_trip_high, price),),
        # G12: feedwater is a spare move, since the level is limited and not
        # priced; it rests at normal water level, 1 at the high trip.
        rest=lambda x: drum_level(
            x[1], x[2], x[3], p.A_d * p.normal_level_height, x[0], p
        ) / p.level_trip_high,
    )  # fmt: skip


def make_boiler_drum_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``boiler_model``."""
    return Oracle(env, params, "boiler_drum", boiler_model)
