"""Glass room study: the kept cascade's fuel and lost glass on the ceiling's own seeds
(five scenarios, seeds 100 to 104, sensor noise 1, a week from the reset), to set beside
the foresight plans of ``scripts/glass_slow_oracle.py``.

Run: PYTHONPATH=src python scripts/glass_room/reference_split.py <out.json>
"""

import json
import pathlib
import sys

import jax
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym import scenarios  # noqa: E402

SEEDS = (100, 101, 102, 103, 104)
CASCADE = L.cascade()
res = {}
for name in scenarios.weights(L.SPEC):
    p = L.scenario_params(name)
    f = jax.jit(jax.vmap(lambda seed, p=p: L.run_controller(CASCADE, p, seed)))
    tr = {k: np.asarray(v) for k, v in f(np.asarray(SEEDS)).items()}
    mark = (tr["canal"] < float(p.canal_target)) | (tr["X"] < float(p.fining_pass))
    res[name] = dict(
        cost=tr["cost"].mean(1).tolist(),
        fuel=tr["fuel_cost"].mean(1).tolist(),
        glass=(tr["cost"] - tr["fuel_cost"]).mean(1).tolist(),
        steps_under_a_mark=mark.sum(1).tolist(),
        fining_mean=tr["X"].mean(1).tolist(),
        fining_min=tr["X"].min(1).tolist(),
        hot_sd=tr["hot"].std(1).tolist(),
        valve_min=tr["valve"].min(1).tolist(),
        valve_max=tr["valve"].max(1).tolist(),
        valve_move=np.abs(np.diff(tr["valve"], axis=1)).sum(1).tolist(),
    )
    print(name, {k: np.round(v, 3).tolist() for k, v in res[name].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
