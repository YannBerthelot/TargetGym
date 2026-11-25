"""Glass room study, probe 5: the same furnace worked harder. At each pull, where the
crown's cap and the fining pass mark leave less room between them, what the cascade
loses on the scored normal day, against holding the crown at its cap.

Run: PYTHONPATH=src python scripts/glass_room/probe5.py <out.json> [pull factors]
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

E, P = L.E, L.P
H = L.HOUR
SEEDS = tuple(range(100, 110))
factors = [float(x) for x in sys.argv[2:]] or [1.0, 1.10, 1.15, 1.20, 1.24]
row = L.ladder_row()
base = L.scenario_params("normal")
res = {}


def settled_start(params, controller, seed, plant_state):
    """The scored reset, its plant replaced by one settled at this pull."""
    st, mem, obs = L.start(controller, params, seed)
    s = plant_state
    st = st.replace(
        plant=s.plant,
        cooling=s.cooling,
        integral=s.integral,
        fuel=s.fuel,
        limit=s.limit,
    )
    return st, mem, L.ENV.get_obs(st, params)


for fac in factors:
    p = base.replace(pull=base.pull * fac)
    quiet = p.replace(walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
    # the cap's own steady state: ask the most, the override holds 1615 C
    top = L.cascade(setpoint=0.99, bias=row.balance)
    keys = jax.random.split(jax.random.PRNGKey(11), 40 * 24 * H)
    carry, tr = jax.jit(lambda: L.roll(top, quiet, L.start(top, quiet, 0), keys))()
    x_cap = float(np.asarray(tr["X"])[-1])
    cap_state = carry[0]
    cap_cost = float(np.asarray(tr["cost"])[-24 * H :].mean())
    line = dict(
        pull=fac,
        X_at_cap=x_cap,
        fuel_at_cap=float(np.asarray(tr["valve"])[-1]),
        cost_at_cap=cap_cost,
    )
    print(
        f"pull x{fac}: at the cap X {x_cap:.4f} valve {line['fuel_at_cap']:.4f} cost {cap_cost:.2f}",
        flush=True,
    )
    for label, sp in (
        ("placed 0.5523", 0.5523),
        ("2 sigma own 0.52", 0.52),
        ("edge 0.505", 0.505),
        ("cap", 0.99),
    ):
        c = L.cascade(setpoint=sp, bias=row.balance)
        # settle this controller at this pull first, quiet, from the cap's state
        k2 = jax.random.split(jax.random.PRNGKey(12), 20 * 24 * H)
        c0 = (cap_state, jax.tree.map(jnp.asarray, c.init_state), carry[2])
        (s_set, m_set, _), _ = jax.jit(lambda: L.roll(c, quiet, c0, k2))()

        def one(seed):
            st, mem, obs = settled_start(p, c, seed, s_set)
            return L.roll(c, p, (st, m_set, obs), L.episode_keys(seed, 5040)[1])[1]

        f = jax.jit(one)
        rows = [L.summary(f(seed)) for seed in SEEDS]
        agg = {
            k: float(np.mean([r[k] for r in rows]))
            for k in (
                "cost",
                "fuel",
                "lost_h",
                "X_mean",
                "X_min",
                "X_sd",
                "hot_max_C",
                "capped_h",
            )
        }
        agg["seeds_losing"] = int(sum(r["lost_h"] > 0 for r in rows))
        agg["costs"] = [r["cost"] for r in rows]
        line[label] = agg
        print(
            f"  {label}: cost {agg['cost']:.2f} fuel {agg['fuel']:.2f} lost {agg['lost_h']:.1f} h "
            f"({agg['seeds_losing']}/10 seeds) X {agg['X_mean']:.3f} min {agg['X_min']:.3f} "
            f"capped {agg['capped_h']:.0f} h",
            flush=True,
        )
    res[str(fac)] = line
json.dump(res, open(sys.argv[1], "w"), indent=1)
