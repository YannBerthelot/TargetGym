"""The oracle (the MPC slot) for the patrol tasks, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.2), run by ``target_gym.oracles.controller.Oracle``:

- the follower's model: ``plane.oracle_model``, as the 3D tasks';
- the lead, its gust and its wake: exogenous (X), since nothing in them reads the
  follower. Each solve runs them forward on the noise-free env (``lead_future``);
  the wake's effect on the follower depends on where it flies, so that part stays
  in the model, on the env's own ``wake_air``, which is smooth but for its cut
  ahead of the lead, a dropped guard: the follower flies behind;
- objective: the slot error and the heading against the lead's as
  (e / e_floor)^2, exact since e_tol = 0 and p = 2, the heading over the shorter
  turn (T5, S4);
- limits: the 3D aircraft's on the follower, and the separation and the slot loss
  on every visited state, elastic at the trip cost per unit (T3; separation is
  nonconvex, so a local optimum), backed off by the planner's error on them (K11).

No resting target: the slot and the heading price all three moves. The same
controller fills ``patrol_range_bearing``'s slot, labelled a full-state bound
(:func:`make_patrol_range_bearing_mpc`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from target_gym import records
from target_gym.oracles.controller import Oracle, TaskModel
from target_gym.oracles.shooting import Limit
from target_gym.patrol.env import (
    advance_air,
    air_speed,
    compute_next_state_patrol,
    slot_delay,
    slot_error,
    wake_air,
    wrap_angle,
)
from target_gym.plane import oracle_model as om
from target_gym.reward import trip_cost


class Future(NamedTuple):
    """What a plan flies against, one row per step ahead."""

    gust: jnp.ndarray  # the follower's air before the wake
    wake: jnp.ndarray  # the wake's history at the step's start
    delay: jnp.ndarray  # the wake's delay to the slot
    lead: jnp.ndarray  # the lead's x, y, z, track at the step's end


def lead_future(env, p, steps):
    """``state -> Future``: the noise-free env run forward. The follower is the
    lead's copy, which no field read here depends on."""
    key = jax.random.PRNGKey(0)
    method = env.integration_method

    def one(s, _):
        s = s.replace(follower=s.lead)
        speed = air_speed(s.lead, s.gust_history[0], p)
        back = jnp.atleast_1d(s.slot_back)
        _, _, gusts, _ = advance_air(
            s.gust_history, s.gust_own, back, jnp.atleast_1d(s.slot_right),
            jnp.atleast_1d(s.slot_up), speed, p, key, s.time,
        )  # fmt: skip
        wake, delay = s.wake_history, slot_delay(back, speed, p)
        s, _ = compute_next_state_patrol(jnp.zeros(3), s, p, method, key)
        lead = jnp.stack([s.lead.x, s.lead.y, s.lead.z, s.lead.psi])
        return s, Future(gusts[1], wake, delay, lead)

    return jax.jit(lambda s: jax.lax.scan(one, s, None, length=steps)[1])


def wake_on(future, x, t, p):
    """The step's air at the follower's position: ``(gust, roll gust)``."""
    gust, roll = wake_air(future.wake[t], x[:1], x[1:2], x[2:3], future.delay[t], p)
    return future.gust[t] + gust[0], roll[0]


def slot_terms(x1, lead, slot, p):
    """The slot error, squared with the env's guard, and the separation, from the
    state reached and the lead's: ``slot_error_vector`` in the lead's frame."""
    sin_l, cos_l = jnp.sin(lead[3]), jnp.cos(lead[3])
    dx, dy, up = x1[0] - lead[0], x1[1] - lead[1], x1[2] - lead[2]
    back = -(dx * cos_l + dy * sin_l)
    lateral = dx * sin_l - dy * cos_l
    errors = jnp.stack([back - slot[0], lateral - slot[1], up - slot[2]])
    return errors @ errors + 1e-8, jnp.sqrt(dx**2 + dy**2 + up**2 + 1e-8)


def patrol_model(env, p) -> TaskModel:
    assert float(p.e_tol_slot) == 0.0 and float(p.tracking_exponent) == 2.0
    tc = int(records.load_row("oracle_horizons", env.registry_name)["T_c_steps"])
    future = lead_future(env, p, int(p.max_steps_in_episode) + tc)
    price = float(trip_cost(p))

    def tracking(d, x1, u, t, widths):
        run, slot = d["task"]
        lead = run.lead[t]
        squared, separation = slot_terms(x1, lead, slot, p)
        e = x1[4] - lead[3]
        turn = om.turn_distance(jnp.sin(e), jnp.cos(e), widths[len(om.SITE_NAMES)])
        cost = squared / p.e_floor_slot**2 + (turn / p.e_floor_heading) ** 2
        error = jnp.sqrt(squared)
        return cost, jnp.stack([error, turn]), jnp.stack([separation, error])

    def tracked(s):
        heading = wrap_angle(s.follower.psi - s.lead.psi)
        return jnp.stack([slot_error(s), jnp.abs(heading)])

    return om.flight_model(
        env, p, tracking,
        data=lambda s: (future(s), jnp.stack([s.slot_back, s.slot_right, s.slot_up])),
        air=lambda d, x, t: wake_on(d["task"][0], x, t, p),
        tracked=tracked,
        resolution=(float(p.slot_precision_floor), float(p.heading_precision_floor)),
        limits=(
            Limit(lo=float(p.min_separation), price=price),
            Limit(hi=float(p.max_slot_error), price=price),
        ),
        rest=False,
        state_x=lambda s: om.to_polar(om.flight_x(s.follower)),
        hold=lambda s: om.hold_flight(s.follower, p),
    )  # fmt: skip


def make_patrol_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``patrol_model``."""
    return Oracle(env, params, env.registry_name, patrol_model)


def make_patrol_range_bearing_mpc(env, params) -> Oracle:
    """``patrol_range_bearing``'s slot: the patrol oracle, reading the TRUE state.
    A full-state bound, not a range-and-bearing controller: on a task defined by
    what its observation withholds, its number bounds information and control
    together. It never reads ``obs``, so its costs equal patrol's seed for seed."""
    return make_patrol_mpc(env, params)
