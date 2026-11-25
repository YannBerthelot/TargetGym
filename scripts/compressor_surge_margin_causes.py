"""Prototype, unregistered: what makes a compressor plant keep its surge margin.

Run with ``uv run python scripts/compressor_surge_margin_causes.py OUT.json``
(about 15 min on one core; run it on a server). It registers nothing and writes
no record: each variant is a params override or a subclass with lagged readings,
built here. Findings and the decision they led to are in
``docs/proposals/compressor-surge-margin.md``.

The plant has since gained both causes (owner, 2026-11-23). The script therefore
starts from the plant without them, the map's own machine read by instant
transmitters (``BARE``), and adds one cause at a time, as it was first run.

One cause at a time, from ``compressor_surge/REALITY.md`` ("Why the margin is
kept"):

- transmitter dynamics: a first-order lag on the flow and pressure readings
  (flow dP 100 to 150 ms, at most 200 ms [C10]);
- the surge line's position: the plant's true line a few percent off the map the
  controller reads (2.6 to 7 % at low suction pressure [C14]), as the cubic's
  ``W`` scaled while the reference keeps the nominal map;
- inlet temperature, with the map taken at 288 K [C15].

The reference is the ladder's override rung at a given control margin, bump-tested
on the nominal plant with the variant's transmitters (commissioning), then run on
the variant, seeds 100 to 109, noise 1. Per variant: ``J``, the quasi-steady floor
of the recycle energy on the variant's own surge line, and per scenario the cost,
its zone share, the surges per seed, the trips and the true margin held.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import reference_ladder as L  # noqa: E402

from target_gym import scenarios  # noqa: E402
from target_gym.benchmark import (  # noqa: E402
    episode_keys,
    episodes_of,
    run_policy,
    task_info,
)
from target_gym.compressor_surge import env as E  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

TASK, SEEDS = "compressor_surge", tuple(range(100, 110))
LINE_RIGHT, LINE_LEFT = {"W": 0.2625}, {"W": 0.2375}  # the true line 5 % off the map
#: The plant before it had a hidden line: every variant starts from the map's machine.
BARE = {"line_shift_max": 0.0}
#: (name, params override, lag on the dp and flow readings in s, control margins).
VARIANTS = (
    ("plant as it is", {}, 0.0, (0.10, 0.08, 0.07, 0.06, 0.05)),
    ("readings lag 0.2 s", {}, 0.2, (0.10, 0.09, 0.08, 0.07, 0.05)),
    ("line 5 % right of the map", LINE_RIGHT, 0.0, (0.10, 0.12, 0.15)),
    ("line 5 % right, lag 0.2 s", LINE_RIGHT, 0.2, (0.10, 0.12, 0.15)),
    ("line 5 % left of the map", LINE_LEFT, 0.0, (0.10,)),
    ("line 5 % left, lag 0.2 s", LINE_LEFT, 0.2, (0.15,)),
    ("inlet 268 K", {"T01": 268.15}, 0.0, (0.10, 0.15)),
    ("inlet 308 K", {"T01": 308.15}, 0.0, (0.10,)),
)


def env_with(lag: float):
    """The task's env, its flow and pressure readings behind ``lag`` seconds."""
    base = type(REGISTRY[TASK].make_env())
    lagged = tuple(
        (
            dataclasses.replace(r, lag=lag)
            if getattr(r, "field", "") in ("dp", "m_c")
            else r
        )
        for r in base.readings
    )
    return type("Lagged", (base,), {"readings": lagged})()


def row_at(margin: float):
    """The ladder's row with its control line ``margin`` right of the surge line."""
    path = L.ROWS_DIR / f"{TASK}.py"
    module_spec = importlib.util.spec_from_file_location(f"row_{margin}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.CONTROL_MARGIN = margin
    return module.row()


def tracer(env, p):
    """Per step of a plan: the step's soft minimum of Phi and the opening."""

    def step(s, ka):
        k, a = ka
        s1 = env.step_env(k, s, a, p)[1]
        s1 = jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), s1, s)
        opening = E.demand_opening(
            s1.demand_levels, s1.block_clock * p.delta_t, s1.demand_dev, p,
            ramp=E.schedule_ramp(s1, p),
        )  # fmt: skip
        return s1, (s1.phi_min, opening)

    return jax.jit(lambda s0, keys, plan: jax.lax.scan(step, s0, (keys, plan))[1])


def floor(opening, p) -> float:
    """Quasi-steady recycle energy (kJ per step) for the demand trace: the
    compressor on its own surge line at the speed whose head there is the
    header pressure, the consumers at that pressure, the recycle the rest; the
    cheapest pressure in the zone."""
    rho, area, tip, w = (
        float(E.suction_density(p)),
        float(p.A_c),
        float(p.U_r),
        float(p.W),
    )
    sp, band = float(p.header_setpoint), float(p.header_band)
    psi = float(E.characteristic(2.0 * w, p, np))

    def power(dp):
        speed = np.sqrt(dp / (rho * tip**2 * psi))
        taken = np.asarray(E.consumer_flow(dp, opening, p, np))
        return (
            np.maximum(rho * area * tip * speed * 2.0 * w - taken, 0.0) * dp / rho / 1e3
        )

    grid = np.linspace(sp - band, sp + band, 25)
    return float(np.min(np.stack([power(d) for d in grid]), axis=0).mean() * p.delta_t)


def run(spec, env, rung, row, override) -> dict:
    """The rung on every scenario of the suite, on the variant's plant."""
    out = {}
    for s in scenarios.suite(spec):
        p = scenarios.params_for(
            spec, s, sensor_noise_scale=1.0, **row.score, **BARE, **override
        )
        steps = int(p.max_steps_in_episode)
        task = task_info(TASK, env, p, len(SEEDS))
        rollout = run_policy(
            L.Reference("override", rung), env, p, task, seeds=list(SEEDS)
        )
        trace, rows = tracer(env, p), []
        for i, (seed, ep) in enumerate(zip(SEEDS, episodes_of(rollout))):
            resets, keys = episode_keys(jax.random.PRNGKey(seed), 1, steps)
            s0 = env.reset_env(resets[0], p)[1]
            moves = jnp.asarray(rollout["action"][i], jnp.float32)
            phi, opening = (np.asarray(v) for v in trace(s0, keys[:, 0], moves))
            rows.append(
                {
                    "cost": float(ep.cost.mean()),
                    "zone": float(ep.terms["zone"].mean()),
                    "surges": round(
                        float(ep.terms["surge"].sum()) / float(p.surge_price)
                    ),
                    "trips": int(ep.failed.sum()),
                    "margin": float((phi / (2.0 * float(p.W)) - 1.0).mean()),
                    "floor": floor(opening, p),
                }
            )
        out[s.name] = {
            **{
                k: float(np.mean([r[k] for r in rows]))
                for k in ("cost", "zone", "margin", "floor")
            },
            "surges": [r["surges"] for r in rows],
            "trips": sum(r["trips"] for r in rows),
        }
    return out


def main(out_path: str) -> None:
    spec, weights, results, started = REGISTRY[TASK], None, {}, time.time()
    weights = scenarios.weights(spec)
    for name, override, lag, margins in VARIANTS:
        env = env_with(lag)
        spec_v = dataclasses.replace(spec, env_factory=lambda lag=lag: env_with(lag))
        for margin in margins:
            row = row_at(margin)
            row.test.update(BARE)  # commissioned on the map's machine too
            rung = L.build(TASK, row, spec=spec_v)["override"]
            per = run(spec, env, rung, row, override)
            entry = {
                "J": sum(weights[k] * v["cost"] for k, v in per.items()),
                "J_floor": sum(weights[k] * v["floor"] for k, v in per.items()),
                "scenarios": per,
            }
            results[f"{name}, margin {margin:.2f}"] = entry
            surged = {k: sum(1 for x in v["surges"] if x) for k, v in per.items()}
            print(f"{name}, margin {margin:.2f}: J {entry['J']:.3f}, floor "
                  f"{entry['J_floor']:.3f}, seeds with a surge {surged}", flush=True)  # fmt: skip
            pathlib.Path(out_path).write_text(json.dumps(results, indent=1) + "\n")
    print(f"done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1])
