"""Judging step 3's two checks on a finished plan.

    python glass_rebuilt_polish.py whole SEED PLAN.npz OUT.npy
    python glass_rebuilt_polish.py flat SEED OUT.npy

``whole`` polishes a plan with no window dropped, as charter draft 3 counts room.
``flat`` plans again from a flat valve, to show the start doesn't set the answer.
Run with JAX_ENABLE_X64=1 so that episodes draw alike.
"""

import sys
import time

import glass_rebuilt_ladder as L
import glass_rebuilt_oracle as O
import jax
import numpy as np

from target_gym.glass_furnace import rebuilt as B

LEAD = (0.314, 0.314 / 1513.0, 0.916, 600.0, 120.0)

if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2])
    p = B.RebuiltParams()
    if mode == "whole":
        O.SKIP = 0
        u, levels = np.load(sys.argv[3])["u"], ((1, 300),)
    else:
        mean = np.asarray(jax.jit(lambda: L.episode(seed, LEAD, p))()[3]).mean()
        u, levels = np.full(p.max_steps_in_episode, float(mean)), ((20, 300), (5, 300))
    print("start", O.true_cost(seed, u, p), flush=True)
    for every, iters in levels:
        t0 = time.time()
        u, res = O.plan(seed, u, p, every, iters)
        print(
            every * 30, O.true_cost(seed, u, p), res.nit, time.time() - t0, flush=True
        )
        np.save(sys.argv[-1], u)
