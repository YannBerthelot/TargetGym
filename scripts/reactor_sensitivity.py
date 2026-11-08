"""How much the reactor's room rests on its `ours` prices.

For each scenario, the reference and the saved ceiling plans are re-scored with one
price at a time scaled by 1/3 and by 3: rod travel, effluent, and the offset
clock's derate. The plans stay as solved at the nominal prices, so a re-scored
ceiling is an upper bound on that price's best cost, and the room it gives is a
lower bound. The reference's actions do not depend on prices, so it is simply
rerun.

    uv run python scripts/reactor_sensitivity.py --tune ladder.json --plans DIR
"""

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import reactor_foresight as foresight  # noqa: E402
import reactor_ladder as ladder  # noqa: E402

from target_gym import scenarios  # noqa: E402
from target_gym.reactor import Reactor, ReactorParams  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

PRICES = ("rod_step_price", "effluent_price", "derate_price_per_MWh")
SCALES = (1.0 / 3.0, 1.0, 3.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", required=True)
    parser.add_argument("--plans", required=True, help="plans_<scenario>.npy here")
    parser.add_argument("--ipopt", default=None, help="plan_<seed>_2880_1.npy here")
    args = parser.parse_args()
    saved = json.load(open(args.tune))
    env = Reactor()
    seeds = list(range(100, 110))
    out = {}
    for sc in scenarios.suite(REGISTRY["reactor"]):
        plans = np.load(pathlib.Path(args.plans) / f"plans_{sc.name}.npy")
        if args.ipopt and sc.name == "edf_daily":
            for k, seed in enumerate(seeds):
                f = pathlib.Path(args.ipopt) / f"plan_{seed}_2880_1.npy"
                if f.exists():
                    plans[k] = np.load(f)
        base = ReactorParams(**sc.overrides)
        rows = {}
        for price in PRICES:
            for scale in SCALES:
                if scale == 1.0 and price != PRICES[0]:
                    continue
                name = "nominal" if scale == 1.0 else f"{price} x {scale:.3g}"
                p = base.replace(**{price: getattr(base, price) * scale})
                ref = ladder.reference(saved, p)
                ref_cost = jax.jit(
                    jax.vmap(lambda s: foresight.reference_actions(env, p, ref, s)[1])
                )(jnp.asarray(seeds))
                ceiling = jax.jit(
                    jax.vmap(lambda s, a: foresight.plan_cost(env, p, s, a)[0])
                )(jnp.asarray(seeds), jnp.asarray(plans, jnp.float32))
                # The ceiling is the cheaper of the plan and the reference itself.
                ceiling = jnp.minimum(ceiling, ref_cost)
                room = np.asarray(ref_cost) / np.asarray(ceiling)
                rows[name] = {
                    "reference": [round(float(x), 1) for x in ref_cost],
                    "ceiling": [round(float(x), 1) for x in ceiling],
                    "room": [round(float(x), 2) for x in room],
                    "room_mean": round(float(room.mean()), 2),
                }
                print(sc.name, name, rows[name]["room_mean"], flush=True)
        out[sc.name] = {"weight": sc.weight, "rows": rows}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
