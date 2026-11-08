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
  the imbalance price in GBP per step: an absolute epigraph row with the
  meter's dead band as its tolerance (T2), so the planned cost of a plan is
  the env's (``tests/oracles/test_thermal_energy_oracles.py``);
- limits: the level between the two trips on every visited state, elastic at
  the trip cost per unit (T3). The trip's 10 s delay is not used: the plan
  never leaves the band, which is stricter than the plant;
- rest (G12): feedwater is a spare move, the level being limited but not
  priced, so it rests at normal water level;
- dropped guards (T4): the void, circulation, pressure and volume clips,
  which the level limits keep off their edges. Not the steam under the water
  surface's floor at 0 (``compute_next_state``) on the ceiling's plans: a plan
  over the whole episode drove it below 0 in its last steps, where the env
  holds it at 0 and the level rose 0.19 m above the model's into a trip, so
  the ceiling holds it at or above 0 as a limit (``ceiling=True``).

The plant draws nothing after its reset, so the reset state is the whole
future: ``ceiling_data`` is the solve's data for a plan over the episode, the
ceiling of charter J3 (``data/oracle_ceilings/boiler_drum.json``). The
registry's receding slot keeps the resting target; the ceiling plans the score
alone.

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
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost


def boiler_model(env, p, rest: bool = True, ceiling: bool = False) -> TaskModel:
    rate = RateLimit.of(env, p)
    # GBP per (kg/s) off the plan for one step (``imbalance_cost``).
    per_kg_s = float(imbalance_cost(p.delta_t, p))

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
        tracked = jnp.stack([level, x1[0]])
        cost = jnp.zeros((), x.dtype)
        y = jnp.stack([off, level] + ([x1[3]] if ceiling else []))
        return jnp.concatenate([x1, held]), cost, y, tracked, ()

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
        hinges=(
            Hinge(weight=per_kg_s, absolute=True, tol=float(p.steam_resolution)),
        ),
        limits=(Limit(p.level_trip_low, p.level_trip_high, price),)
        + ((Limit(lo=0.0, price=price),) if ceiling else ()),
        # G12: feedwater is a spare move, since the level is limited and not
        # priced; it rests at normal water level, 1 at the high trip.
        rest=(lambda x: drum_level(
            x[1], x[2], x[3], p.A_d * p.normal_level_height, x[0], p
        ) / p.level_trip_high) if rest and not ceiling else None,
    )  # fmt: skip


def ceiling_backoffs(env, p) -> tuple:
    """How far inside each limit the ceiling plans: one sigma of the level's
    reading (0.5 mm). The plan with foresight rides the trip levels for
    minutes (a runback's shrink is cheapest spent there), and the float32
    plant the score runs ends 1e-6 m off a float64 plan, which trips it.
    None on the floor of the steam under the water, which the env clips."""
    (sigma,) = [r.sigma for r in env.readings if r.field == "level"]
    return (float(sigma), 0.0)


def ceiling_data(env, p, state, step_keys):
    """The data of a plan over the whole episode from the reset ``state``:
    the load plan and its upset are drawn at the reset, so the step keys add
    nothing."""
    del env, p, step_keys
    return state, state.level_ref, state.time


def make_boiler_drum_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``boiler_model``."""
    return Oracle(env, params, "boiler_drum", boiler_model)
