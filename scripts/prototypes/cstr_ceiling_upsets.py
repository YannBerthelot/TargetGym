"""Does the foresight ceiling keep cstr-v8's temperature in its band through the
three upsets? A quick check, not a record: the shipped expert and the prototype's
ceiling (``cstr_production.ceiling``) on a few seeds of each upset scenario.

    python scripts/prototypes/cstr_ceiling_upsets.py OUT_DIR [N_SEEDS] [SOLVES]
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "prototypes"))

import cstr_production as P  # noqa: E402
import numpy as np  # noqa: E402

from target_gym.benchmark import shipped_policy  # noqa: E402

UPSETS = ("catalyst_loss", "feed_buildup", "cooling_loss")


def main(out, n=3, solves=2):
    out.mkdir(parents=True, exist_ok=True)
    env, base = P.SPEC.make_env(), P.CSTRParams(sensor_noise_scale=1.0)
    seeds, result = P.SEEDS[:n], {}
    for name, (p, _) in P.scenarios(base).items():
        if name not in UPSETS:
            continue
        ref_policy = shipped_policy("pid")
        ref_policy.bind(env, p, seeds)
        ref, moves, _ = P.run(env, p, ref_policy, seeds)
        top = P.ceiling(env, p, name, seeds, moves, ref["cost"], solves)
        result[name] = {"reference": ref, "ceiling": top}
        P.show(name, "reference", ref)
        P.show(name, "ceiling", top)
        (out / "ceiling_upsets.json").write_text(
            json.dumps(result, indent=1, default=float)
        )
    for name, r in result.items():
        held = int(np.sum(np.array(r["ceiling"]["off_spec"]) == 0))
        print(f"{name}: ceiling holds the band on {held} of {n} seeds", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(pathlib.Path(args[0]), *(int(a) for a in args[1:]))
