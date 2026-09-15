"""The building's NEA reference: the shipped MPC's lowest per-seed hold cost.

Run with ``uv run python scripts/hvac_floor.py [--seeds N]``. The building
has no certified floor -- its hold error is overheating the weather sets,
and no reduced model of it has been solved under the version-2 cost -- so
the reference is the best the shipped MPC held on any one seed of the test
episode, in EUR per step: comfort alone (``rho_floor_tracking``) and with the
gas charged in full (``rho_floor``). The weather moves the cost about 2x
between seeds, which is why the minimum and not the mean is the reference:
a per-seed minimum is what no run of the reference sits below.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.eval import evaluate, hold_settings, run_episode  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.runners.runners import baseline_policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    spec = REGISTRY["hvac"]
    p = spec.make_test_params()
    burn_in = min(hold_settings("hvac")["burn_in"], int(p.max_steps_in_episode) // 2)
    rows = []
    for seed in range(args.seeds):
        ep = run_episode(spec, p, baseline_policy(spec, "mpc", p), seed=seed)
        m = evaluate([ep], burn_in=burn_in)
        rows.append((m["hold_tracking"], m["hold"]))
        print(
            f"seed {seed}: comfort {m['hold_tracking']:.4f}  total {m['hold']:.4f} EUR/step"
        )
    print(f"rho_floor_tracking = {min(r[0] for r in rows):.4f}")
    print(f"rho_floor          = {min(r[1] for r in rows):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
