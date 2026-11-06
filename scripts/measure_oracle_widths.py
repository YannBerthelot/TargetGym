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
import importlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import records
from target_gym.benchmark import episode_keys
from target_gym.boiler_drum.env import STEAM_OU_THETA
from target_gym.cement_kiln.env import FEED_OU_THETA
from target_gym.energy.wind_turbine.env import WIND_OU_THETA
from target_gym.glass_furnace.env import M_PULL_AR_RHO
from target_gym.hvac.env import WEATHER_OU_THETA
from target_gym.oracles.controller import enable_x64, float64
from target_gym.oracles.widths import kink_width, limit_backoff, noise_backoff
from target_gym.registry import REGISTRY
from target_gym.runners.runners import pid_policy

SEEDS, EVERY = range(100, 110), 10
PROTOCOL = "docs/proposals/oracle-smoothing-protocol.md, K4 to K6, K10 and K11"
#: K11's risk of any trip from unforeseen noise over the ten recorded seeds.
RISK, RECORDED_SEEDS = 0.01, 10
#: The dropped guards (T4) of each task, by the protocol's inventory.
GUARDS = {
    "cstr": {},
    "first_order": {},
    "four_tank": {
        "level_sqrt": "pc_gym/four_tank/env.py:204, sqrt(max(h, 0)); "
        "the level band keeps h >= h_min"
    },
    "unstable_cstr": {},
    "hvac": {},
    "battery": {
        "current_guards": "energy/battery/env.py:283, :313, :314, :378, the "
        "charge, power and discriminant clips; the limits keep them off",
        "calendar_rate": "energy/battery/env.py:336, the two fade hinges are "
        "the env's while the calendar rate stays under c_hold",
    },
    "boiler_drum": {
        "plant_guards": "boiler_drum/env.py:289, :301, :329, :334, :360, :369, "
        ":378, :429 to :432; the level and pressure bands keep them off",
        "condensation": "boiler_drum/env.py:379, max(Q - q_dc subcool, 0)",
    },
    "wind_turbine": {
        "speed_guards": "energy/wind_turbine/env.py:262, :264, :276, :277, "
        ":355; the speed band keeps them off",
        "actuator_clips": "energy/wind_turbine/env.py:356, :357; a lag toward "
        "an in-range command stays in range",
        "betz": "energy/wind_turbine/env.py:271, Cp's ceiling, never reached",
    },
    "glass_furnace": {
        "flame": "glass_furnace/env.py:522, the flame's clip",
        "batch_heat": "glass_furnace/env.py:593, heat to the batch max(., 0)",
        "stored_blanket": "glass_furnace/env.py:806, max(m_batch, 0)",
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


def wind_sites(points) -> tuple[dict, list]:
    """Cp's floor and the pitch rate (env.py:271, :309), inside the right-hand
    side: each delta enters every evaluation of a step at once, the sensitivities
    of one step sharing their sign, against the electrical power; and the
    back-off of the speed limit."""
    from target_gym.energy.wind_turbine.env import electrical_power
    from target_gym.energy.wind_turbine.oracle import BETZ, wind_model, wind_step

    spec = REGISTRY["wind_turbine"]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model = wind_model(env, p)
    sites = {
        "cp_floor": ("energy/wind_turbine/env.py:271", BETZ),
        "pitch_rate": ("energy/wind_turbine/env.py:309", 2.0 * p.pitch_rate_max),
    }
    pts = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in points]

    def run(probe, deltas):
        state, u = probe
        return wind_step(
            env, p, model.state_x(state), u, 0, model.data(state), (0.0, 0.0), deltas
        )[0]

    out = {}
    for i, (name, (where, scale)) in enumerate(sorted(sites.items())):

        @jax.jit
        def site(probe, delta, i=i):
            x1 = run(probe, jnp.zeros(2).at[i].set(delta[0]))
            return electrical_power(x1[0], x1[2], p)[None]

        width = kink_width(site, float(scale), pts, [p.precision_floor])
        out[name] = {
            "site": where,
            "scale": float(scale),
            "width": width,
            "evaluations": 1,
        }

    moved = jax.jit(
        lambda probe: jnp.abs(run(probe, jnp.zeros(2)) - model.state_x(probe[0]))
    )
    rate = max(float(moved(q)[1]) for q in pts) / 2.0 / float(p.pitch_tau)
    out["pitch_rate"]["k6_substep_change"] = rate
    out["pitch_rate"]["k6_flag"] = (
        out["pitch_rate"]["width"] is None or out["pitch_rate"]["width"] < rate
    )

    @jax.jit
    def speed(probe, deltas):
        return run(probe, deltas.sum(axis=-1))[0][None]

    widths = [out[name]["width"] or 0.0 for name in sorted(sites)]
    return out, [float(b) for b in limit_backoff(speed, widths, pts, 1)]


def glass_sites(points) -> tuple[dict, list]:
    """The blanket's guards on what the right-hand side and the flame read
    (env.py:543, :562, :758): the delta enters every evaluation of a step at
    once, against the crown; and the back-off of the three bands."""
    from target_gym.glass_furnace.oracle import glass_model, glass_step

    spec = REGISTRY["glass_furnace"]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model = glass_model(env, p)
    pts = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in points]

    def run(probe, delta):
        state, u = probe
        return glass_step(
            env, p, model.state_x(state), u, 0, model.data(state), 0.0, delta
        )[0]

    @jax.jit
    def site(probe, delta):
        return run(probe, delta[0])[:1]

    scale = float(p.m_batch_full)
    width = kink_width(site, scale, pts, [p.precision_floor])
    out = {
        "blanket": {
            "site": "glass_furnace/env.py:543, :562, :758", "scale": scale,
            "width": width, "evaluations": 1,
        }
    }  # fmt: skip

    @jax.jit
    def bands(probe, deltas):
        return run(probe, deltas.sum())[:3]

    return out, [float(b) for b in limit_backoff(bands, [width or 0.0], pts, 1)]


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
    "cement_kiln": kiln_sites,
    "compressor_surge": compressor_sites,
    "glass_furnace": glass_sites,
    "wind_turbine": wind_sites,
}
#: Each task's count of smoothed sites, which its planner's ``widths`` hold.
N_SITES = {
    "cement_kiln": 1,
    "compressor_surge": 2,
    "glass_furnace": 1,
    "wind_turbine": 2,
}
#: A noisy task's planner model, and its deviation's (decay per step, std).
NOISY = {
    "compressor_surge": (
        "target_gym.compressor_surge.oracle:compressor_model",
        lambda p: (np.exp(-p.demand_theta * p.delta_t), p.demand_sigma),
    ),
    "unstable_cstr": (
        "target_gym.pc_gym.unstable_cstr.oracle:unstable_cstr_model",
        lambda p: (np.exp(-p.delta_t / p.Ti_tau), p.Ti_sigma),
    ),
    "boiler_drum": (
        "target_gym.boiler_drum.oracle:boiler_model",
        lambda p: (1.0 - STEAM_OU_THETA * p.delta_t, p.q_steam_noise_std),
    ),
    "wind_turbine": (
        "target_gym.energy.wind_turbine.oracle:wind_model",
        lambda p: (1.0 - WIND_OU_THETA * p.delta_t, p.turbulence_std),
    ),
    "glass_furnace": (
        "target_gym.glass_furnace.oracle:glass_model",
        lambda p: (M_PULL_AR_RHO, p.m_pull_noise_std),
    ),
    "cement_kiln": (
        "target_gym.cement_kiln.oracle:kiln_model",
        lambda p: (1.0 - FEED_OU_THETA * p.delta_t, p.feed_noise_std),
    ),
    "hvac": (
        "target_gym.hvac.oracle:hvac_model",
        lambda p: (1.0 - WEATHER_OU_THETA * p.delta_t, p.T_out_noise_std),
    ),
}


def noise_terms(name: str, points) -> list:
    """K11's noise term of each limit, in ``TaskModel.limits`` order: one
    innovation of the deviation, the last field of the model's task data."""
    where, decay = NOISY[name]
    module, attr = where.split(":")
    build = getattr(importlib.import_module(module), attr)
    spec = REGISTRY[name]
    env, p = spec.make_env(), float64(spec.make_test_params())
    model = build(env, p)
    a, sigma = decay(p)
    std = float(sigma) * float(np.sqrt(1.0 - a**2))
    n_sites = N_SITES.get(name, 0)
    hinges = len(model.hinges)

    @jax.jit
    def limited(probe, d):
        state, u = probe
        task = model.data(state)
        data = {
            "task": (*task[:-1], task[-1] + d), "end": 10**9,
            "widths": jnp.zeros(n_sites),
        }  # fmt: skip
        return jnp.atleast_1d(model.step(data, model.state_x(state), u, 0)[2])[hinges:]

    pts = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in points]
    steps = RECORDED_SEEDS * int(p.max_steps_in_episode)
    return [float(b) for b in noise_backoff(limited, std, pts, RISK, steps)]


def main(tasks) -> None:
    for name in tasks:
        points = probes(name) if name in SITES or name in NOISY else []
        with enable_x64():
            sites, gaps = SITES[name](points) if name in SITES else ({}, [])
            noise = noise_terms(name, points) if name in NOISY else []
        n = max(len(gaps), len(noise))
        gaps, noise = gaps or [0.0] * n, noise or [0.0] * n
        backoffs = [g + b for g, b in zip(gaps, noise)]
        row = {
            "sites": sites, "limit_backoffs": backoffs, "limit_gaps": gaps,
            "limit_noise": noise, "guards": GUARDS[name], "protocol": PROTOCOL,
            "probes": f"the expert's states, training seeds {SEEDS.start} to "
            f"{SEEDS.stop - 1}, every {EVERY}th step" if points else None,
        }  # fmt: skip
        path = records.write("oracle_widths", name, records.stamped(row), compact=True)
        print(name, json.dumps(sites), backoffs, "->", path.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", default=list(GUARDS))
    main(ap.parse_args().tasks)
