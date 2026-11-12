"""Measure the smoothing widths of the oracles on the shared solver (K5).

docs/proposals/oracle-smoothing-protocol.md, K4 to K6 and K10. For each site
a task's planner model smooths (T5): the largest softplus width on the
ladder s/10, ..., s/1e8 (s the clipped interval's span, K4) whose worst
one-step effect on every tracked output stays under half its resolution,
at the states the expert visits on training seeds 100 to 109, every 10th
step, with its command (``target_gym.oracles.widths``). K6: the largest
change of the site's argument over one integrator substep at those probes,
and a flag when the width is below it (reported, the width kept). K11: each
hard limit's back-off at the same probes: the same gap bound on its limited
quantity, with the widths found, plus one noise innovation's quantile.

Writes ``data/oracle_widths/<task>.json`` for every task on the shared
solver: its sites (none but the compressor's) and its dropped guards (T4).
Run once, before the re-record; the file is in the oracle's fingerprint.

    uv run python scripts/measure_oracle_widths.py [--tasks cstr ...]
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import records
from target_gym.benchmark import episode_keys
from target_gym.oracles.controller import enable_x64, float64
from target_gym.oracles.rates import RateLimit
from target_gym.oracles.widths import kink_width, limit_backoff, noise_backoff
from target_gym.plane.dynamics import INNER_STEPS, PITCH_DEMAND_MAX
from target_gym.registry import REGISTRY
from target_gym.runners.runners import pid_policy

SEEDS, EVERY = range(100, 110), 10
PROTOCOL = "docs/proposals/oracle-smoothing-protocol.md, K4 to K6, K10 and K11"
#: K11's risk of any trip from unforeseen noise over the ten recorded seeds.
RISK, RECORDED_SEEDS = 0.01, 10
#: Tasks whose moves are rates (``oracles.rates``).
BY_RATE = ("boiler_drum",)
AIRCRAFT_2D = ("plane", "plane_energy", "plane_sine")
#: The tasks whose reward prices a heading, a min over turn directions (S4).
TURNS = ()
AIRCRAFT = AIRCRAFT_2D + TURNS
#: The dropped guards (T4) of the aircraft (section 5.2).
_AIRFRAME = {
    "mach": "plane/dynamics.py, the shock stall, the drag rise and the caps past "
    "M_crit, which MMO does not keep off: checked",
    "section_cap": "plane/dynamics.py, the section's lift cap of +-2",
    "angular_clip": "plane/env.py, the clip of the angular accelerations",
    "incidence_wrap": "plane/dynamics.py, the wraps at +-180 deg",
    "scaler": "plane/dynamics.py, airspeed_scaler's clip, from below",
    "fuel": "plane/env.py, the tanks' clip",
}
#: The dropped guards (T4) of each task, by the protocol's inventory.
GUARDS = {
    **{n: _AIRFRAME for n in AIRCRAFT},
    "cstr": {},
    "first_order": {},
    "four_tank": {
        "level_sqrt": "pc_gym/four_tank/env.py:204, sqrt(max(h, 0)); "
        "the level band keeps h >= h_min"
    },
    "unstable_cstr": {},
    "ph_neutralization": {
        "diversion": "pc_gym/ph_neutralization/env.py:432 to :437, the sewer "
        "floor's switch and the pump-back's cap; the permit keeps them off",
        "buffer_clip": "pc_gym/ph_neutralization/env.py:426, the buffer flow's "
        "clip, which its mean path never meets",
    },
    "hvac": {},
    "battery": {
        "current_guards": "energy/battery/env.py:299, :329, :330, the charge, "
        "power and discriminant clips; the limits keep them off",
        "fade_floor": "energy/battery/env.py:537, max(fade - c_hold, 0); the "
        "fade stays above c_hold",
    },
    "boiler_drum": {
        "plant_guards": "boiler_drum/env.py:299, :312, :348, :354, :380, :389, "
        ":398, :449 to :452; the level and pressure bands keep them off",
        "condensation": "boiler_drum/env.py:399, max(Q - q_dc subcool, 0)",
    },
    "wind_turbine": {
        "speed_guards": "energy/wind_turbine/env.py:293, :294, :382; the speed "
        "band keeps them off",
        "actuator_clips": "energy/wind_turbine/env.py:383, :384; a lag toward an "
        "in-range command stays in range",
        "betz": "energy/wind_turbine/env.py:288, Cp's ceiling, never reached",
    },
    "reactor": {
        "rod_range": "reactor/env.py:569, the rods' range clip; the demand is in "
        "range",
    },
    "cement_kiln": {
        "plant_guards": "cement_kiln/env.py:313, :363, :366, :392, :416, :444, "
        ":445, :456, :457, :546, :547, :548, :549; the limits and the speed box "
        "keep them off, calcination_step keeps alpha a fraction, and lime stays "
        "inside [0, 1] below the trip",
    },
    "compressor_surge": {
        "characteristic": "compressor_surge/env.py:314, the tangent past "
        "phi_join (K9)"
    },
}


def probes(name: str) -> list:
    """(state, command) every ``EVERY`` steps of the expert's episodes."""
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    T = int(params.max_steps_in_episode)
    step = jax.jit(env.step_env)
    out = []
    for seed in SEEDS:
        policy = pid_policy(spec)
        k_reset, k_steps = episode_keys(jax.random.PRNGKey(seed), 1, T)
        obs, state = env.reset_env(k_reset[0], params)
        for t in range(T):
            u = np.clip(
                np.atleast_1d(np.asarray(policy(np.asarray(obs)), float)), -1, 1
            )
            if t % EVERY == 0:
                out.append((state, u))
            obs, state, *_ = step(k_steps[t, 0], state, jnp.asarray(u), params)
    return out


def _plan_moves(name: str, points) -> list:
    """The probes in float64, each command as the plan's move: on a task planned
    by rate, the rate the env gave it."""
    pts = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in points]
    if name not in BY_RATE:
        return pts
    spec = REGISTRY[name]
    rate = RateLimit.of(spec.make_env(), float64(spec.make_test_params()))
    return [(s, rate.move(s.actuator_held, u)) for s, u in pts]


def compressor_sites(points) -> tuple[dict, list]:
    """The drive's and the valve's rate limits (env.py:511, :514), each
    evaluated once per substep, against the header pressure in kPa; and the
    back-off of the trip limit on the step's soft minimum of Phi."""
    from target_gym.compressor_surge.env import KPA, _substeps
    from target_gym.compressor_surge.oracle import compressor_model, compressor_step

    spec = REGISTRY["compressor_surge"]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model = compressor_model(env, p)
    n = _substeps(env.integration_method)
    h = p.delta_t / n
    sites = {
        "drive_rate": ("compressor_surge/env.py:511", 2.0 * p.drive_rate * h),
        "valve_rate": (
            "compressor_surge/env.py:514",
            (p.valve_open_rate + p.valve_close_rate) * h,
        ),
    }
    pts = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in points]
    out = {}
    for i, (name, (where, scale)) in enumerate(sorted(sites.items())):

        @jax.jit
        def site(probe, delta, i=i):
            state, u = probe
            deltas = jnp.zeros((2, n)).at[i].set(delta)
            x1 = compressor_step(
                env, p, model.state_x(state), u, 0, model.data(state), (0.0, 0.0),
                deltas,
            )[0]  # fmt: skip
            return (x1[1] / KPA)[None]

        width = kink_width(site, float(scale), pts, [p.precision_floor], n)
        moved = max(
            float(np.max(np.abs(np.diff(np.asarray(args)[:, i]))))
            for args in (
                compressor_step(
                    env, p, model.state_x(s), u, 0, model.data(s), (0.0, 0.0)
                )[3]
                for s, u in pts
            )
        )
        out[name] = {
            "site": where, "scale": float(scale), "width": width, "evaluations": n,
            "k6_substep_change": moved, "k6_flag": width is None or width < moved,
        }  # fmt: skip

    @jax.jit
    def phi_min(probe, deltas):
        state, u = probe
        return compressor_step(
            env, p, model.state_x(state), u, 0, model.data(state), (0.0, 0.0), deltas
        )[1][None]

    widths = [out[name]["width"] or 0.0 for name in sorted(sites)]
    return out, [float(b) for b in limit_backoff(phi_min, widths, pts, n)]


def compressor_zone(points, sites) -> dict:
    """K11 on the header zone's edge (the owner's call of 2026-11-11: the plan
    rode the band and a step slipped out): the planner's worst one-step error
    on the header pressure, kPa, at the probes: the drive's and valve's
    smoothing gap plus one innovation of the demand deviation."""
    from target_gym.compressor_surge.env import KPA, _substeps
    from target_gym.compressor_surge.oracle import compressor_model, compressor_step

    name = "compressor_surge"
    spec = REGISTRY[name]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model, n = compressor_model(env, p), _substeps(env.integration_method)
    pts = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in points]

    @jax.jit
    def header(probe, deltas):
        state, u = probe
        x1 = compressor_step(
            env, p, model.state_x(state), u, 0, model.data(state), (0.0, 0.0), deltas
        )[0]  # fmt: skip
        return (x1[1] / KPA)[None]

    @jax.jit
    def header_by_demand(probe, d):
        state, u = probe
        task = _with_deviation(name, model.data(state), d)
        x1 = compressor_step(env, p, model.state_x(state), u, 0, task, (0.0, 0.0))[0]
        return (x1[1] / KPA)[None]

    widths = [sites[k]["width"] or 0.0 for k in sorted(sites)]
    gap = float(limit_backoff(header, widths, pts, n)[0])
    a, sigma = NOISY[name][1](p)
    std = float(sigma) * float(np.sqrt(1.0 - a**2))
    steps = RECORDED_SEEDS * int(p.max_steps_in_episode)
    noise = float(noise_backoff(header_by_demand, std, pts, RISK, steps)[0])
    return {"backoff_kpa": gap + noise, "gap_kpa": gap, "noise_kpa": noise}


def _table_jumps(state, p):
    """The largest jump in the slope of ROSCO's table, in Cp per cell, across
    the nodes around a state, along each axis, pitch then tip-speed ratio: a
    softplus on a cell's clip misses by w log 2 times that (K5's site output)."""
    from target_gym.energy.wind_turbine.cp_table import (
        CP_TABLE,
        PITCH_STEP,
        TSR_FIRST,
        TSR_STEP,
    )

    lam = float(state.omega) * float(p.R) / float(state.v_wind)
    row = (lam - TSR_FIRST) / TSR_STEP
    col = float(state.pitch) / PITCH_STEP
    last_row, last_col = CP_TABLE.shape[0] - 1, CP_TABLE.shape[1] - 1
    rows, cols = [int(row), int(row) + 1], [int(col), int(col) + 1]
    near_rows, near_cols = np.clip(rows, 0, last_row), np.clip(cols, 0, last_col)
    # np.diff(.., 2) drops the end nodes, whose slope has no jump.
    along_pitch = np.abs(np.diff(CP_TABLE, 2, axis=1))
    along_tsr = np.abs(np.diff(CP_TABLE, 2, axis=0))
    return (
        float(along_pitch[near_rows][:, np.clip(cols, 1, last_col - 1) - 1].max()),
        float(along_tsr[np.clip(rows, 1, last_row - 1) - 1][:, near_cols].max()),
    )


def wind_sites(points) -> tuple[dict, list]:
    """Cp's floor, the table's two axes and the two rate limits, inside the
    right-hand side: each delta enters every evaluation of a step at once, the
    sensitivities of one step sharing their sign, against the electrical power;
    and the back-off of the speed limit."""
    from target_gym.energy.wind_turbine.env import electrical_power
    from target_gym.energy.wind_turbine.oracle import BETZ, wind_model, wind_step

    spec = REGISTRY["wind_turbine"]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model = wind_model(env, p)
    sites = {
        "cp_floor": ("energy/wind_turbine/env.py:288", BETZ),
        "cp_pitch": ("energy/wind_turbine/cp_table.py, a cell of the pitch axis", 1.0),
        "cp_tsr": ("energy/wind_turbine/cp_table.py, a cell of the tip-speed axis", 1.0),
        "pitch_rate": (
            "energy/wind_turbine/env.py:334", 2.0 * p.pitch_rate_max
        ),
        "torque_rate": (
            "energy/wind_turbine/env.py:337", 2.0 * p.torque_rate_max
        ),
    }  # fmt: skip
    pts = [
        (float64(s), jnp.asarray(u, jnp.float64), *_table_jumps(s, p))
        for s, u in points
    ]

    def run(probe, deltas):
        state, u, *_ = probe
        return wind_step(
            env, p, model.state_x(state), u, 0, model.data(state), (0.0,) * 5, deltas
        )

    def on_plant(probe, deltas):
        """A delta per site, name order, onto Cp, the pitch rate, the torque rate;
        a table axis's in units of its slope jump."""
        _, _, pitch_jump, tsr_jump = probe
        cp = deltas[0] + deltas[1] * pitch_jump + deltas[2] * tsr_jump
        return jnp.stack([cp, deltas[3], deltas[4]])

    out = {}
    for i, (name, (where, scale)) in enumerate(sorted(sites.items())):

        @jax.jit
        def site(probe, delta, i=i):
            x1 = run(probe, on_plant(probe, jnp.zeros(5).at[i].set(delta[0])))
            return electrical_power(x1[0], x1[2], p)[None]

        width = kink_width(site, float(scale), pts, [p.precision_floor])
        out[name] = {
            "site": where,
            "scale": float(scale),
            "width": width,
            "evaluations": 1,
        }

    moved = jax.jit(
        lambda probe: jnp.abs(run(probe, jnp.zeros(3)) - model.state_x(probe[0]))
    )
    for name, index, tau in (
        ("pitch_rate", 1, p.pitch_tau), ("torque_rate", 2, p.torque_tau),
    ):  # fmt: skip
        rate = max(float(moved(q)[index]) for q in pts) / 2.0 / float(tau)
        width = out[name]["width"]
        out[name]["k6_substep_change"] = rate
        out[name]["k6_flag"] = width is None or width < rate

    @jax.jit
    def speed(probe, deltas):
        return run(probe, on_plant(probe, deltas.sum(axis=-1)))[0][None]

    widths = [out[name]["width"] or 0.0 for name in sorted(sites)]
    return out, [float(b) for b in limit_backoff(speed, widths, pts, 1)]


def reactor_sites(points) -> tuple[dict, list]:
    """The rods' rate limit (env.py:568), once per sub-step, against Tavg; and the
    back-off of the limits on every sub-step's state."""
    from target_gym.reactor.oracle import reactor_model, reactor_step, rod_travel

    spec = REGISTRY["reactor"]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model = reactor_model(env, p)
    pts = _plan_moves("reactor", points)
    n, scale = env.control_period, float(2.0 * rod_travel(p))

    def run(probe, deltas):
        state, u = probe
        task = model.data(state)
        return reactor_step(env, p, model.state_x(state), u, 0, task, 0.0, deltas)

    @jax.jit
    def site(probe, delta):
        return run(probe, delta)[1][-1:, 4]

    width = kink_width(site, scale, pts, [p.precision_floor], n)
    out = {
        "rod_rate": {
            "site": "reactor/env.py:568", "scale": scale, "width": width,
            "evaluations": n,
        }
    }  # fmt: skip

    @jax.jit
    def limited(probe, deltas):
        return run(probe, deltas[0])[1][:, 1:].ravel()

    return out, [float(b) for b in limit_backoff(limited, [width or 0.0], pts, n)]


def _aircraft(name, points):
    """A flight task's params, probes and ``run(probe, deltas, gust, roll)``, one
    planned step at width 0 giving its tracked outputs and limited quantities."""
    from target_gym.plane import oracle_model as om

    spec = REGISTRY[name]
    env, p = spec.make_env(), float64(spec.make_test_params())
    pts = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in points]
    zero = jnp.zeros(len(om.SITE_NAMES))
    method = env.integration_method
    if name in AIRCRAFT_2D:
        from target_gym.plane.oracle import plane_step, state_x

        def run(probe, deltas, gust, roll):
            state, u = probe
            x1, (loads, eas, mach, _) = plane_step(
                p, state_x(state), u, 0, gust, method, zero, deltas
            )
            limited = jnp.concatenate([x1[:1], loads, eas[None], mach[None]])
            return x1[:1], limited

        return p, pts, run, ("gust_x", "gust_z")

    def run(probe, deltas, gust, roll):
        state, u = probe
        x1, flown = om.flight_step(
            p, om.flight_x(state), u, 0, gust, method, zero, deltas, roll
        )
        tracked = jnp.stack([x1[2], jnp.arctan2(x1[4], x1[3])])
        return tracked, om.flight_outputs(x1, flown)

    return p, pts, run, ("gust_x", "gust_y", "gust_z")


def _gust(state, fields):
    return jnp.stack([getattr(state, f) for f in fields])


def aircraft_sites(name, points) -> tuple[dict, list]:
    """An aircraft's sites, each delta entering every evaluation of a step at once,
    against its tracked outputs, and the back-off of each limit."""
    from target_gym.plane import oracle_model as om

    p, pts, run, fields = _aircraft(name, points)
    sites = {
        "buffet": ("plane/dynamics.py, the buffet line's cap", p.CL_max - p.CL_min),
        "cl_stall": ("plane/dynamics.py, the stall clip", p.CL_max - p.CL_min),
        "separation": ("plane/dynamics.py, abs(incidence)", 2.0 * p.aoa_stall),
        "surface_clip": ("plane/dynamics.py, attitude_axis", 2.0),
        "surface_rate": (
            "plane/dynamics.py, compute_next_surface",
            2.0 * p.surface_rate_limit * p.delta_t / INNER_STEPS,
        ),
        "tropopause": ("plane/dynamics.py, the sound speed's switch", 288.15 - om.T11),
        "protection": (
            "plane/dynamics.py, protected_pitch_demand's clip",
            2.0 * PITCH_DEMAND_MAX,
        ),
    }
    resolutions = [p.precision_floor]
    if name not in AIRCRAFT_2D:
        resolutions.append(p.heading_precision_floor)
    n = len(om.SITE_NAMES)
    zero = jnp.zeros(n)
    no_roll = jnp.zeros(())

    def tracked(probe, deltas):
        return run(probe, om.sites(deltas), _gust(probe[0], fields), no_roll)[0]

    out = {}
    for i, site_name in enumerate(om.SITE_NAMES):
        where, scale = sites[site_name]

        @jax.jit
        def site(probe, delta, i=i):
            return tracked(probe, zero.at[i].set(delta[0]))

        out[site_name] = {
            "site": where,
            "scale": float(scale),
            "width": kink_width(site, float(scale), pts, resolutions),
            "evaluations": 1,
        }

    @jax.jit
    def limited(probe, deltas):
        deltas = om.sites(deltas.sum(axis=-1))
        return run(probe, deltas, _gust(probe[0], fields), no_roll)[1]

    widths = [out[n_]["width"] or 0.0 for n_ in om.SITE_NAMES]
    return out, [float(b) for b in limit_backoff(limited, widths, pts, 1)]


def turn_site() -> dict:
    """The min over turn directions (S4): it prices the reward and moves no state,
    so K5 takes the first rung of the turn's 2 pi, as the kiln's soft max does."""
    scale = 2.0 * float(np.pi)
    return {
        "site": "plane3d/env.py, wrap_angle in the heading cost",
        "scale": scale,
        "width": scale / 10.0,
        "evaluations": 1,
    }


def aircraft_noise(name, points) -> list:
    """K11's noise term of each limit: one innovation of the gust, the axes
    independent, so their terms add in quadrature (derived), and on a 3D aircraft
    the rolling gust's held white draw at the slowest probe, its largest."""
    from target_gym.plane import oracle_model as om
    from target_gym.plane3d.dynamics import rolling_gust

    p, pts, run, fields = _aircraft(name, points)
    std = float(p.turbulence_sigma) * float(np.sqrt(p.delta_t))
    steps = RECORDED_SEEDS * int(p.max_steps_in_episode)
    no_roll = jnp.zeros(())
    terms = []
    for axis, _ in enumerate(fields):

        @jax.jit
        def by_gust(probe, d, axis=axis):
            gust = _gust(probe[0], fields).at[axis].add(d)
            return run(probe, om.NO_DELTA, gust, no_roll)[1]

        terms.append(noise_backoff(by_gust, std, pts, RISK, steps))
    if name not in AIRCRAFT_2D:

        @jax.jit
        def by_roll(probe, d):
            return run(probe, om.NO_DELTA, _gust(probe[0], fields), d)[1]

        slowest = min(float(jnp.hypot(s.x_dot, s.y_dot)) for s, _ in pts)
        one = jax.vmap(lambda k: rolling_gust(k, slowest, p), axis_size=20000)(
            jax.random.split(jax.random.PRNGKey(0), 20000)
        )
        terms.append(noise_backoff(by_roll, float(jnp.std(one)), pts, RISK, steps))
    return [float(b) for b in np.sqrt(np.sum(np.square(terms), axis=0))]


def battery_sites(_points) -> tuple[dict, list]:
    """The charge a new half-cycle jumps to at the hysteresis (env.py:366): it
    moves no tracked output, so K5 takes the ladder's first rung, and no limit."""
    p = REGISTRY["battery"].make_test_params()
    scale = float(p.cycle_hysteresis)
    return {
        "half_cycle": {
            "site": "energy/battery/env.py:366", "scale": scale,
            "width": scale / 10.0, "evaluations": 1,
        }
    }, []  # fmt: skip


def kiln_sites(_points) -> tuple[dict, list]:
    """The burning zone's soft max in the cold limit (env.py:627): it moves no
    tracked output, so K5 takes the ladder's first rung; K3's shift keeps it
    conservative, so it needs no back-off."""
    p = REGISTRY["cement_kiln"].make_test_params()
    scale = float(p.T_bz_max - p.T_bz_min)
    return {
        "burning_zone": {
            "site": "cement_kiln/env.py:576, :627", "scale": scale,
            "width": scale / 10.0, "evaluations": 1,
        }
    }, []  # fmt: skip


SITES = {
    **{n: functools.partial(aircraft_sites, n) for n in AIRCRAFT},
    "reactor": reactor_sites,
    "battery": battery_sites,
    "cement_kiln": kiln_sites,
    "compressor_surge": compressor_sites,
    "wind_turbine": wind_sites,
}
#: Each task's count of smoothed sites, which its planner's ``widths`` hold.
N_SITES = {
    "reactor": 1,
    "battery": 1,
    "cement_kiln": 1,
    "compressor_surge": 2,
    "wind_turbine": 5,
}
#: A noisy task's planner model, and its deviation's (decay per step, std).
#: boiler_drum and unstable_cstr draw their upsets at reset and step without
#: noise, so K11 has no innovation to back off from there.
NOISY = {
    "reactor": (
        "target_gym.reactor.oracle:reactor_model",
        lambda p: _wobble(p, 6),
    ),
    "compressor_surge": (
        "target_gym.compressor_surge.oracle:compressor_model",
        lambda p: (np.exp(-p.demand_theta * p.delta_t), p.demand_sigma),
    ),
    "wind_turbine": (
        "target_gym.energy.wind_turbine.oracle:wind_model",
        lambda p: _turbulence(p, max(p.v_mean_range)),
    ),
    "cement_kiln": (
        "target_gym.cement_kiln.oracle:kiln_model",
        lambda p: (
            1.0 - _theta("cement_kiln.env", "FEED_OU_THETA") * p.delta_t,
            p.feed_noise_std,
        ),
    ),
    "ph_neutralization": (
        "target_gym.pc_gym.ph_neutralization.oracle:ph_model",
        lambda p: (
            1.0 - _theta("pc_gym.ph_neutralization.env", "BUFFER_OU_THETA") * p.delta_t,
            p.q2_noise_std,
        ),
    ),
    "hvac": (
        "target_gym.hvac.oracle:hvac_model",
        lambda p: (
            1.0 - _theta("hvac.env", "WEATHER_OU_THETA") * p.delta_t,
            p.T_out_noise_std,
        ),
    ),
}


def _turbulence(p, v_mean):
    """The wind's turbulence over a step at the mean wind ``v_mean``: its
    decay, and its stationary std (env.py, Kaimal time scale, IEC sigma). The
    innovation grows with the mean wind, so the range's top bounds it."""
    from target_gym.energy.wind_turbine.env import (
        turbulence_sigma,
        turbulence_time_constant,
    )

    decay = np.exp(-p.delta_t / float(turbulence_time_constant(v_mean, p)))
    return decay, float(turbulence_sigma(v_mean, p))


def _theta(module, name):
    """A noise's reversion rate, imported when its task is measured, so a task
    whose constant has moved does not stop the others."""
    return getattr(importlib.import_module(f"target_gym.{module}"), name)


def _wobble(p, period):
    """The reactor's load wobble over a control step: its decay, and the
    stationary deviation its innovations imply."""
    a = 1.0 - p.demand_theta * p.delta_t
    return a**period, p.demand_sigma * np.sqrt(p.delta_t / (1.0 - a**2))


#: Where a task's deviation sits in its model's task data, if not last.
DEVIATION_AT = {"compressor_surge": 3, "hvac": 1, "reactor": 3, "wind_turbine": 2}


def _with_deviation(name, task, d):
    """The task data with ``d`` added to its deviation."""
    i = DEVIATION_AT.get(name, len(task) - 1)
    return (*task[:i], task[i] + d, *task[i + 1 :])


def noise_terms(name: str, points) -> list:
    """K11's noise term of each limit, in ``TaskModel.limits`` order: one
    innovation of the deviation, a field of the model's task data."""
    where, decay = NOISY[name]
    module, attr = where.split(":")
    build = getattr(importlib.import_module(module), attr)
    spec = REGISTRY[name]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model = build(env, p)
    if not model.limits:  # nothing to back off (hvac prices its band)
        return []
    a, sigma = decay(p)
    std = float(sigma) * float(np.sqrt(1.0 - a**2))
    n_sites = N_SITES.get(name, 0)
    hinges = len(model.hinges)

    @jax.jit
    def limited(probe, d):
        state, u = probe
        task = model.data(state)
        data = {
            "task": _with_deviation(name, task, d), "end": 10**9,
            "widths": jnp.zeros(n_sites),
        }  # fmt: skip
        return jnp.atleast_1d(model.step(data, model.state_x(state), u, 0)[2])[hinges:]

    pts = _plan_moves(name, points)
    steps = RECORDED_SEEDS * int(p.max_steps_in_episode)
    return [float(b) for b in noise_backoff(limited, std, pts, RISK, steps)]


def main(tasks) -> None:
    for name in tasks:
        points = probes(name) if name in SITES or name in NOISY else []
        with enable_x64():
            sites, gaps = SITES[name](points) if name in SITES else ({}, [])
            if name in TURNS:
                sites["turn"] = turn_site()
            if name in AIRCRAFT:
                noise = aircraft_noise(name, points)
            else:
                noise = noise_terms(name, points) if name in NOISY else []
            zone = (
                {"zone": compressor_zone(points, sites)}
                if name == "compressor_surge"
                else {}
            )
        n = max(len(gaps), len(noise))
        gaps, noise = gaps or [0.0] * n, noise or [0.0] * n
        backoffs = [g + b for g, b in zip(gaps, noise)]
        row = {
            "sites": sites, "limit_backoffs": backoffs, "limit_gaps": gaps,
            "limit_noise": noise, "guards": GUARDS[name], "protocol": PROTOCOL,
            **zone,
            "probes": f"the expert's states, training seeds {SEEDS.start} to "
            f"{SEEDS.stop - 1}, every {EVERY}th step" if points else None,
        }  # fmt: skip
        path = records.write("oracle_widths", name, records.stamped(row), compact=True)
        print(name, json.dumps(sites), backoffs, "->", path.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", default=list(GUARDS))
    main(ap.parse_args().tasks)
