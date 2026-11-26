"""Glass room study, prototype: the 30 % pull rise again, with the second loop on the
bath's true mean temperature against the held crown and the floor couple.

Run: PYTHONPATH=src python scripts/glass_room/proto_bath_rise.py <proto_bath.json> <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import proto as M  # noqa: E402
import proto_bath as B  # noqa: E402

fits = json.load(open(sys.argv[1]))["step"]
p0 = M.BASE.replace(pull=M.PULL0 * 0.8)
p = p0.replace(pull_change=0.30, pull_change_window_h=(96.0, 96.0))
par = tuple(jnp.asarray(v) for v in (0.5523, 0.0, 1.0e9, 0.0))
settled = jax.jit(lambda: M.settle(8, par, p0))()
res = {}
loops = {"crown held, table on the pull": None}
for which, name in (
    ("bath", "loop on the true bath mean"),
    ("floor3", "loop on the floor couple under bay 3"),
):
    k, theta, tau = (
        fits[which]["k"],
        max(fits[which]["theta_h"] * 3600.0, M.DT),
        fits[which]["tau_h"] * 3600.0,
    )
    loops[name] = (
        "bath" if which == "bath" else "floor",
        tau / (k * 2.0 * theta),
        min(tau, 8.0 * theta),
        0.0,
    )
for label, loop in loops.items():
    f = jax.jit(lambda seed, loop=loop: B.run(p, seed, B.N, loop=loop, settled=settled))
    rows = [B.week(f(seed)) for seed in (100, 101, 102)]
    r = B.mean_of(rows)
    res[label] = r
    print(
        f"{label:40s} blocks rise {r['travel']:.1f} K (end zones {r['end']:.1f}), crown span {r['crown_span']:.0f} K, fuel {r['fuel']:.2f}, "
        f"marks lost {r['marks_h']:.1f} h, X min {r['X_min']:.3f}, glass lost at q10 1 % {100*r['lost']:.2f} %, total {r['total']:.2f}",
        flush=True,
    )
json.dump(res, open(sys.argv[2], "w"), indent=1)
