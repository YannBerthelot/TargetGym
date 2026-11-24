"""Glass room study: the size of the oracle's gradient at its design start, with and
without the defects. SLSQP's first step is the scaled gradient itself, so this sets the
solver's objective scale (``oracle.FIRST_STEP``).

Run: PYTHONPATH=src python scripts/glass_room/oracle_scale.py [scenario] [seed]
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.glass_furnace import oracle as O  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "normal"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
p = L.scenario_params(name)
n = int(p.max_steps_in_episode)
knot = int(round(O.KNOT_HOURS * 3600.0 / float(p.delta_t)))
reset, keys = O.episode_keys(seed, n)
_, st = L.ENV.reset_env(reset, p)
pp = plan_params(L.SPEC, p)
z0 = jnp.full(-(-n // knot), float(st.fuel))


def running(z, params):
    """The episode's fuel and rejected plates, EUR per step, as the oracle plans them."""
    out = O.rollout(L.ENV, params, st, keys, jnp.repeat(z, knot)[:n])
    return jnp.sum(out[1]) / n


no_defects = pp.replace(plant=pp.plant.replace(block_release=0.0, reboil_plate=0.0))
grad = jax.jit(jax.jacfwd(running))
for tag, q in (("fuel and plates", pp), ("fuel alone", no_defects)):
    g = np.abs(np.asarray(grad(z0, q), float))
    print(
        f"{name} {seed}, {tag}: |gradient| per knot, EUR per step per kg/s: "
        f"largest {g.max():.2f}, median {np.median(g):.2f}; first step at scale "
        f"0.01 {0.01 * g.max():.4f} kg/s, at 0.001 {0.001 * g.max():.4f} kg/s",
        flush=True,
    )
