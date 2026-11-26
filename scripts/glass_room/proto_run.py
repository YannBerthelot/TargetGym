"""Glass room study, prototype runs: does a price on temperature rises make the
reference pay, and is there a trade between moving in time and moving gently?

    PYTHONPATH=src python scripts/glass_room/proto_run.py normal <out.json>
    PYTHONPATH=src python scripts/glass_room/proto_run.py rise <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import proto as M  # noqa: E402

H = M.H
N, BURN = 10 * 24 * H, 3 * 24 * H
INF = 1.0e9
Q10 = (0.0, 0.001, 0.01, 0.1)
P1 = 0.1


def run(ch, variants, p0, p, seeds):
    """``{label: [per-seed info]}`` for ``variants`` = ``{label: (sp, outer, rate, offset)}``."""
    labels = list(variants)
    par = [jnp.asarray([variants[k][i] for k in labels], jnp.float32) for i in range(4)]
    f = jax.jit(
        jax.vmap(
            jax.vmap(
                lambda a, b, c, d, seed: M.episode(ch, (a, b, c, d), p0, p, seed, N),
                in_axes=(None, None, None, None, 0),
            ),
            in_axes=(0, 0, 0, 0, None),
        )
    )
    out = f(*par, jnp.asarray(seeds))
    out = {k: np.asarray(v) for k, v in out.items()}
    return {
        lab: [{k: v[i, j] for k, v in out.items()} for j in range(len(seeds))]
        for i, lab in enumerate(labels)
    }


def report(label, infos):
    rows = []
    for info in infos:
        r = {f"q{q}": M.costs(info, q, 0.0, BURN) for q in Q10}
        r["reboil"] = M.costs(info, 0.0, P1, BURN)
        hot = info["hot"][BURN:]
        r["travel"] = float(info["rise"][BURN:].sum())
        r["travel_end"] = float(info["rise_end"][BURN:].sum())
        r["crown_span"] = float(hot.max() - hot.min())
        r["X_mean"], r["X_min"] = float(info["X"][BURN:].mean()), float(
            info["X"][BURN:].min()
        )
        rows.append(r)
    m = lambda f: float(np.mean([f(r) for r in rows]))  # noqa: E731
    line = dict(
        fuel=m(lambda r: r["q0.0"]["fuel"]),
        marks_h=m(lambda r: r["q0.0"]["cliff_h"]),
        travel=m(lambda r: r["travel"]),
        travel_end=m(lambda r: r["travel_end"]),
        crown_span=m(lambda r: r["crown_span"]),
        X_mean=m(lambda r: r["X_mean"]),
        X_min=m(lambda r: r["X_min"]),
        total={str(q): m(lambda r: r[f"q{q}"]["total"]) for q in Q10},
        contact_share={
            str(q): m(lambda r: r[f"q{q}"]["contact_share"]) for q in Q10 if q
        },
        reboil_share=m(lambda r: r["reboil"]["reboil_share"]),
    )
    print(
        f"{label:34s} fuel {line['fuel']:.2f} | marks lost {line['marks_h']:.1f} h | blocks rise {line['travel']:.1f} K "
        f"(end zones {line['travel_end']:.1f}) | crown span {line['crown_span']:.0f} K | X {line['X_mean']:.3f} min {line['X_min']:.3f} | "
        + "total at q10 "
        + ", ".join(f"{100*q:g}%: {line['total'][str(q)]:.2f}" for q in Q10)
        + f" | glass lost at 1%: {100*line['contact_share']['0.01']:.2f}% | reboil share {100*line['reboil_share']:.2f}%",
        flush=True,
    )
    return line


mode, out = sys.argv[1], sys.argv[2]
res = {}
if mode == "normal":
    p = M.BASE
    variants = {
        "cascade as tuned (chases)": (0.5523, 1.0, INF, 0.0),
        "outer loop at 0.3 of its gain": (0.5523, 0.3, INF, 0.0),
        "outer loop at 0.1 of its gain": (0.5523, 0.1, INF, 0.0),
        "outer loop at 0.03 of its gain": (0.5523, 0.03, INF, 0.0),
        "crown held, no quality loop": (0.5523, 0.0, INF, 0.0),
        "crown held 5 K hotter": (0.5523, 0.0, INF, 5.0),
        "crown held 10 K hotter": (0.5523, 0.0, INF, 10.0),
        "cascade, setpoint 0.58": (0.58, 1.0, INF, 0.0),
        "outer at 0.1, setpoint 0.58": (0.58, 0.1, INF, 0.0),
        "crown held 20 K hotter": (0.5523, 0.0, INF, 20.0),
    }
    infos = run(8, variants, p, p, (100, 101, 102, 103, 104))
    for lab in variants:
        res[lab] = report(lab, infos[lab])
else:
    p0 = M.BASE.replace(pull=M.PULL0 * 0.8)
    for notice in (8.0, 24.0):
        p = p0.replace(
            pull_change=0.30, pull_change_window_h=(96.0, 96.0), pull_notice_h=notice
        )
        if notice == 8.0:
            now = {
                f"on the pull charged, {('step' if r >= INF else f'{r:g} K/h')}": (
                    0.5523,
                    1.0,
                    r,
                    0.0,
                )
                for r in (INF, 20.0, 10.0, 5.0, 2.0, 1.0)
            }
            now["on the pull charged, 5 K/h, outer 0.1"] = (0.5523, 0.1, 5.0, 0.0)
            now["on the pull charged, step, outer 0.1"] = (0.5523, 0.1, INF, 0.0)
            infos = run(8, now, p0, p, (100, 101, 102))
            for lab in now:
                res[lab] = report(lab, infos[lab])
        early = {
            f"on the plan {notice:g} h ahead, {('step' if r >= INF else f'{r:g} K/h')}": (
                0.5523,
                1.0,
                r,
                0.0,
            )
            for r in ((INF, 10.0, 5.0, 2.0) if notice == 8.0 else (5.0, 2.0, 1.0))
        }
        early[f"on the plan {notice:g} h ahead, 2 K/h, outer 0.1"] = (
            0.5523,
            0.1,
            2.0,
            0.0,
        )
        infos = run(9, early, p0, p, (100, 101, 102))
        for lab in early:
            res[lab] = report(lab, infos[lab])
    # the least the blocks must rise: the settled state at the high pull over the low one
    par = tuple(jnp.asarray(v) for v in (0.5523, 1.0, INF, 0.0))
    lo, _ = jax.jit(lambda: M.settle(8, par, p0))()
    hi, _ = jax.jit(lambda: M.settle(8, par, p0.replace(pull=p0.pull * 1.30)))()
    d = np.maximum(np.asarray(hi.plant.T - lo.plant.T), 0.0)
    need = float((d * np.asarray(M.CELL_MASS)).sum() / np.asarray(M.CELL_MASS).sum())
    end = float(
        (d * np.asarray(M.CELL_MASS))[:, -4:].sum()
        / np.asarray(M.CELL_MASS)[:, -4:].sum()
    )
    print(
        f"settled change, low to high pull: blocks rise {need:.1f} K (end zones {end:.1f} K), "
        f"crown {float(M.P.sensors(hi.plant, p0.grid, p0.plant)['hot'] - M.P.sensors(lo.plant, p0.grid, p0.plant)['hot']):.0f} K"
    )
    res["settled change"] = dict(rise=need, rise_end=end)
json.dump(res, open(out, "w"), indent=1)
