"""The oracle (the MPC slot) for the 3D aircraft, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.2), run by ``target_gym.oracles.controller.Oracle``:

- model: ``compute_next_state_3d`` restated in ``plane.oracle_model``
  (``flight_step``), the plant's fixed inner loop at 4 Hz included, its sites
  smoothed at the K5 widths (T5) and its guards dropped (T4), the raw moves' clips
  removed and the moves bounded (T1), the gust on its noise-free decay;
- objective: each tracking error as (e / e_floor)^2, exact since e_tol = 0 and
  p = 2 (T2); the heading error over the shorter turn, smoothed where the two
  turns meet (T5, S4), with the track a state of the plan on its own branch, so
  arctan2's cut never enters (T3); the speed's resting target at the schedule
  (G12), since no reward here prices it; a path's distance as a path parameter
  per visited state, chosen by the planner (T2p), on the circle in closed form,
  its leg-arc joins left C1 on the racetrack (S9);
- limits: the altitude band, the load factor at each inner sub-step, the bank, VMO
  and MMO on every visited state, elastic at the trip cost per unit (T3), backed
  off by the planner's error on them (K11, in the widths file).

The cold start holds the throttle and the attitude where they are.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from target_gym import records
from target_gym.oracles.controller import Oracle, TaskModel
from target_gym.plane import oracle_model as om
from target_gym.plane3d.env import (
    _RACETRACK_LEG as RACETRACK_LEG,
)
from target_gym.plane3d.env import (
    distance_to_circle,
    distance_to_racetrack,
    nearest_point_on_twisted_lemniscate,
    wrap_angle,
)

#: The heading task's sites: the airframe's, then the turn's.
N_SITES = len(om.SITE_NAMES)


#: How far a plan's path parameter may sit from its reference, as a share of the
#: curve: a bound that must not bind, wide of any drift of the plan's speed.
WINDOW = 0.25
#: Each path parameter's typical size in raw moves, for the variable scaling.
PATH_SCALE = 1e-3
#: Points of the search for the path parameter at a solve's start.
SEARCH = 4096


def gust_of(s):
    return jnp.stack([s.gust_x, s.gust_y, s.gust_z])


def altitude_term(p, target, z):
    return ((target - z) / p.e_floor_altitude) ** 2


def heading_model(env, p) -> TaskModel:
    """The heading task, with the position dropped."""
    assert float(p.e_tol_altitude) == 0.0 and float(p.tracking_exponent) == 2.0

    def tracking(d, x1, u, t, widths):
        target_z, target_psi, _ = d["task"]
        e = x1[4] - target_psi
        turn = om.turn_distance(jnp.sin(e), jnp.cos(e), widths[N_SITES])
        cost = altitude_term(p, target_z, x1[2]) + (turn / p.e_floor_heading) ** 2
        return cost, jnp.stack([x1[2], turn]), ()

    return om.flight_model(
        env, p, tracking, drop=2,
        data=lambda s: (s.target_altitude, s.target_heading, gust_of(s)),
        tracked=lambda s: jnp.stack(
            [s.z, jnp.abs(wrap_angle(s.psi - s.target_heading))]
        ),
        resolution=(float(p.precision_floor), float(p.heading_precision_floor)),
    )  # fmt: skip


def circle_model(env, p) -> TaskModel:
    """The circle: the distance in closed form, its sqrt off its kink at the
    centre (T4)."""
    assert float(p.e_tol_altitude) == 0.0 and float(p.tracking_exponent) == 2.0

    def tracking(d, x1, u, t, widths):
        target_z, centre, radius, _ = d["task"]
        distance = jnp.hypot(x1[0] - centre[0], x1[1] - centre[1]) - radius
        cost = altitude_term(p, target_z, x1[2]) + (distance / p.e_floor_path) ** 2
        return cost, jnp.stack([x1[2], distance]), ()

    return om.flight_model(
        env, p, tracking,
        data=lambda s: (
            s.target_altitude, jnp.stack([s.target_x, s.target_y]), s.target_radius,
            gust_of(s),
        ),
        tracked=lambda s: jnp.stack([s.z, distance_to_circle(s)]),
        resolution=(float(p.precision_floor), float(p.position_precision_floor)),
    )  # fmt: skip


class Curve(NamedTuple):
    """A closed path: ``at(geometry, s)`` the point at parameter ``s``, periodic in
    ``period``, and ``rate(geometry, s)`` the path's length per unit of ``s``."""

    at: Callable
    rate: Callable
    period: Callable


def stadium_at(g, s):
    """The racetrack by arc length from the near leg's start: leg, turn, leg, turn.
    C1 at the four joins (S9)."""
    _, _, _, r, heading = g
    half = r * RACETRACK_LEG
    s = jnp.mod(s, 4.0 * half + 2.0 * jnp.pi * r)
    leg, turn = 2.0 * half, jnp.pi * r
    around = lambda c, a: (c + r * jnp.cos(a), r * jnp.sin(a))  # noqa: E731
    u, v = -half + s, jnp.full_like(s, r)
    over = s - leg
    u, v = (
        jnp.where(over >= 0.0, b, a)
        for a, b in zip((u, v), around(half, 0.5 * jnp.pi - over / r))
    )
    over = s - leg - turn
    u = jnp.where(over >= 0.0, half - over, u)
    v = jnp.where(over >= 0.0, -r, v)
    over = s - 2.0 * leg - turn
    u, v = (
        jnp.where(over >= 0.0, b, a)
        for a, b in zip((u, v), around(-half, -0.5 * jnp.pi - over / r))
    )
    c, s_ = jnp.cos(heading), jnp.sin(heading)
    return jnp.stack([g[0] + c * u - s_ * v, g[1] + s_ * u + c * v, g[2]])


STADIUM = Curve(
    stadium_at,
    lambda g, s: jnp.ones_like(s),
    lambda g: (4.0 * RACETRACK_LEG + 2.0 * jnp.pi) * g[3],
)


def lemniscate_at(g, tau):
    """The figure-8 on its analytic curve, of which the env samples 800 points."""
    x0, y0, z_mean, a, orientation, dz = g
    d = 1.0 + jnp.sin(tau) ** 2
    bx, by = a * jnp.cos(tau) / d, a * jnp.sin(tau) * jnp.cos(tau) / d
    c, s = jnp.cos(orientation), jnp.sin(orientation)
    return jnp.stack(
        [x0 + bx * c - by * s, y0 + bx * s + by * c, z_mean + dz * jnp.sin(tau)]
    )


LEMNISCATE = Curve(
    lemniscate_at,
    lambda g, tau: jnp.linalg.norm(jax.jacfwd(lemniscate_at, 1)(g, tau)),
    lambda g: 2.0 * jnp.pi,
)


def path_reference(curve, geometry, state, steps):
    """A plan's reference along ``curve``, one parameter per step ahead: from the
    nearest point at the solve's start, at the speed there, in the direction it
    flies. Exogenous data: the plan's own parameters are its moves (T2p)."""
    grid = jnp.linspace(0.0, curve.period(geometry), SEARCH, endpoint=False)
    points = jax.vmap(curve.at, (None, 0))(geometry, grid)
    position = jnp.stack([state.x, state.y, state.z])
    first = grid[jnp.argmin(jnp.sum((points - position) ** 2, axis=1))]
    tangent = jax.jacfwd(curve.at, 1)(geometry, first)
    velocity = jnp.stack([state.x_dot, state.y_dot, state.z_dot])
    sign = jnp.where(tangent @ velocity >= 0.0, 1.0, -1.0)
    speed = jnp.linalg.norm(velocity)

    def advance(s, _):
        s = s + sign * speed / curve.rate(geometry, s)
        return s, s

    return jax.lax.scan(advance, first, None, length=steps)[1]


def path_model(env, p, curve, geometry, floors, distance) -> TaskModel:
    """A path scored by its distance, a min over the curve (T2p): one parameter per
    visited state, a move of the plan held by nothing, its value the planner's
    choice, so the joint minimum over moves and parameters is the env's. ``floors``
    is the cost and tracked outputs of the altitude term, if the task has one."""
    assert float(p.tracking_exponent) == 2.0
    tc = int(records.load_row("oracle_horizons", env.registry_name)["T_c_steps"])
    steps = int(p.max_steps_in_episode) + tc

    def tracking(d, x1, u, t, widths):
        g, reference, window, *rest = d["task"]
        s = reference[t] + window * u[3 + t % tc]
        off = x1[:3] - curve.at(g, s)
        length = jnp.sqrt(off @ off + 1e-8)
        cost, outputs = floors(d, x1, off, length)
        return cost, outputs, ()

    def data(s):
        g = geometry(s)
        window = WINDOW * curve.period(g)
        return g, path_reference(curve, g, s, steps), window, gust_of(s)

    return om.flight_model(
        env, p, tracking, nu=3 + tc, data=data, tracked=distance,
        hold=lambda s: jnp.concatenate([om.hold_flight(s, p), jnp.zeros(tc)]),
        action=lambda s, u: u[:3],
        move_scale=lambda steps: (1.0, 1.0, 1.0, *(PATH_SCALE,) * tc),
        resolution=_resolution(env, p),
    )  # fmt: skip


def _resolution(env, p):
    if env.registry_name == "plane3d_racetrack":
        return (float(p.precision_floor), float(p.position_precision_floor))
    return (float(p.position_precision_floor),)


def racetrack_model(env, p) -> TaskModel:
    """The racetrack on its stadium, in the plane of the path: its height is the
    altitude term's."""
    assert float(p.e_tol_altitude) == 0.0

    def floors(d, x1, off, length):
        horizontal = jnp.sqrt(off[0] ** 2 + off[1] ** 2 + 1e-8)
        cost = altitude_term(p, d["task"][0][2], x1[2])
        cost = cost + (horizontal / p.e_floor_path) ** 2
        return cost, jnp.stack([x1[2], horizontal])

    return path_model(
        env, p, STADIUM,
        lambda s: (
            s.target_x, s.target_y, s.target_altitude, s.target_radius,
            s.target_heading,
        ),
        floors,
        lambda s: jnp.stack([s.z, distance_to_racetrack(s)]),
    )  # fmt: skip


def figure8_model(env, p) -> TaskModel:
    """The figure-8, whose distance is in 3D: the task scores no altitude term."""

    def floors(d, x1, off, length):
        return (length / p.e_floor_path) ** 2, length[None]

    return path_model(
        env, p, LEMNISCATE,
        lambda s: (
            s.target_x, s.target_y, s.target_altitude, s.target_radius,
            s.target_heading, jnp.asarray(p.figure8_altitude_amplitude, s.z.dtype),
        ),
        floors,
        lambda s: nearest_point_on_twisted_lemniscate(s, p)[3][None],
    )  # fmt: skip


MODELS = {
    "plane3d_heading": heading_model,
    "plane3d_circle": circle_model,
    "plane3d_racetrack": racetrack_model,
    "plane3d_figure8": figure8_model,
}


def make_plane3d_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on the task's model."""
    return Oracle(env, params, env.registry_name, MODELS[env.registry_name])
