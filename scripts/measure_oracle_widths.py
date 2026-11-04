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


SITES = {"compressor_surge": compressor_sites}
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
    n_sites = 2 if name == "compressor_surge" else 0
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
