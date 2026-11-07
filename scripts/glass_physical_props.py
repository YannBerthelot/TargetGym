"""Attempt 2's properties, no controller. Every path is a difference from a baseline run
on the same clock, so the reversal cancels. Reading = couple through its sheath lag.

    python glass_physical_props.py ORBIT.npz
"""

import sys

import glass_physical_settle as Z
import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace.env import firing_fraction

g = P.Grid()
p = P.PhysicalParams()
a = P.layout(g)
d = np.load(sys.argv[1])
f0 = float(d["fuel"])
S0 = P.Fields(**{k: jnp.asarray(d[k]) for k in d if k != "fuel"})
PULL, CULLET, THICK = 5.79, 0.25, 0.2
HOURS = (0.25, 1, 6, 24, 96)
N = 2880 * 4 + 40
nm = a.nm


def run(s, fuel, charge, cullet, thick, sulfate):
    q = p.replace(sulfate_batch=sulfate)

    def f(s, t):
        n, b = P.plant_step(
            s, fuel * firing_fraction(t, q), charge, cullet, thick, t, g, q
        )
        hot, tc, bottom, canal = P.sensors(n, Z.DEPTH, Z.BAY, g)
        glass = jnp.sum(a.dz * n.T[:, nm - 1]) / a.depth
        into = jnp.sum(b["q"] * a.dx * a.width) + b["in_blanket"]
        return n, (
            hot,
            tc,
            bottom,
            canal,
            glass,
            into,
            b["foam"].mean(),
            jnp.sum(P.cover(n.m)),
        )

    return jax.lax.scan(f, s, jnp.arange(N))[1]


def lag(y, tau):
    out, v = np.zeros_like(y), y[0]
    alpha = 1 - np.exp(-p.delta_t / tau)
    for i, x in enumerate(y):
        v += alpha * (x - v)
        out[i] = v
    return out


def kick(s, skin=0.0, bath=0.0, refiner=0.0, batch=1.0):
    T = s.T.at[:3, 6:10].add(skin).at[:, :nm].add(bath).at[:, nm:].add(refiner)
    return s.replace(T=T, T_top=s.T_top + 0 * skin, m=s.m * batch)


R = (f0, PULL, CULLET, THICK, 0.01)
CASES = {
    "gas +1 %": (S0, (f0 * 1.01,) + R[1:]),
    "pull -2 %": (S0, (f0, PULL * 0.98) + R[2:]),
    "pull +2 %": (S0, (f0, PULL * 1.02) + R[2:]),
    "cullet +5 pts": (S0, (f0, PULL, 0.30, THICK, 0.01)),
    "cullet -5 pts": (S0, (f0, PULL, 0.20, THICK, 0.01)),
    "cullet +10 pts": (S0, (f0, PULL, 0.35, THICK, 0.01)),
    "blanket +20 %": (S0, (f0, PULL, CULLET, 0.24, 0.01)),
    "blanket -20 %": (S0, (f0, PULL, CULLET, 0.16, 0.01)),
    "sulfate +50 %": (S0, (f0, PULL, CULLET, THICK, 0.015)),
    "skin +5 K": (kick(S0, skin=5.0), R),
    "bath +5 K": (kick(S0, bath=5.0), R),
    "refiner +5 K": (kick(S0, refiner=5.0), R),
    "batch +10 %": (kick(S0, batch=1.1), R),
}
batch = jax.jit(jax.vmap(run))
base = [np.asarray(v) for v in jax.jit(run)(S0, *R)]
ss = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[c[0] for c in CASES.values()])
args = [jnp.array([c[1][i] for c in CASES.values()]) for i in range(5)]
out = [np.asarray(v) for v in batch(ss, *args)]
tc0 = lag(base[1], 120.0)
print(
    "reading, K, at", HOURS, "| peak | rate K/h | first over 0.36 K, h | half peak, h"
)
for i, name in enumerate(CASES):
    y = lag(out[1][i], 120.0) - tc0
    at = [y[int(h * 120) - 1] for h in HOURS]
    w = 30  # 15 min
    rate = np.abs(y[w:] - y[:-w]).max() * 4.0
    pk = np.abs(y[:2880]).max()
    over = np.argmax(np.abs(y) > 0.36) / 120.0 if (np.abs(y) > 0.36).any() else np.inf
    half = np.argmax(np.abs(y) > 0.5 * pk) / 120.0
    print(
        f"{name:15s}", np.round(at, 2), f"{pk:6.2f} {rate:6.2f} {over:6.2f} {half:6.2f}",
        "| glass@6h %.2f @24h %.2f" % tuple(out[4][i][[719, 2879]] - base[4][[719, 2879]]),
        "heat in %.2f%% foam %.4f->%.4f cover %.2f->%.2f bottom@24 %.2f canal@24 %.2f" % (
            100 * (out[5][i][2000:2880].mean() / base[5][2000:2880].mean() - 1),
            base[6][-200:].mean(), out[6][i][-200:].mean(), base[7][-200:].mean(),
            out[7][i][-200:].mean(), out[2][i][2879] - base[2][2879],
            out[3][i][2879] - base[3][2879],
        ),
    )  # fmt: skip

# Lookalikes: each path over its 24 h value.
names = ["gas +1 %", "pull -2 %", "cullet +5 pts"]
paths = {}
for n in names:
    i = list(CASES).index(n)
    y = lag(out[1][i], 120.0) - tc0
    paths[n] = np.array([y[int(h * 120) - 1] for h in (1, 6, 24, 96)]) / y[2879]
    print(n, "scaled", np.round(paths[n], 3))
print("worst gap", max(np.abs(paths[x] - paths[y]).max() for x in names for y in names))

# Fuel step: dead time and settling, on the reading.
i = 0
y = lag(out[1][i], 120.0) - tc0
final = y[-1]
print("fuel step +1 %: first 8 steps", np.round(y[:8], 3), "t10", np.argmax(y > 0.1 * final) / 120,
      "t63", np.argmax(y > 0.63 * final) / 120, "96 h", final)  # fmt: skip
hot = out[0][0] - base[0]
print(
    "hot face: t10",
    np.argmax(hot > 0.1 * hot[-1]) / 120,
    "t63",
    np.argmax(hot > 0.63 * hot[-1]) / 120,
)
tail = base[0][-200:]
print("swing hot face p-p", tail.max() - tail.min(), "reading p-p", np.ptp(tc0[-200:]))
tb = np.ptp(base[2][-200:]), np.ptp(base[3][-200:])
print("bottom, canal swing", tb)

# Wrong way: the reading against the heat entering the glass, and the glass against it.
for n in (
    "sulfate +50 %",
    "cullet +5 pts",
    "cullet +10 pts",
    "blanket -20 %",
    "gas +1 %",
):
    i = list(CASES).index(n)
    y = lag(out[1][i], 120.0) - tc0
    at = [int(h * 120) - 1 for h in (0.1, 0.25, 0.5, 1, 3, 6, 12)]
    heat = 100 * (out[5][i] / base[5] - 1)
    win = [np.mean(heat[max(j - 20, 0) : j + 1]) for j in at]
    print(n, "reading", np.round(y[at], 2), "heat %", np.round(win, 2),
          "glass", np.round((out[4][i] - base[4])[at], 2))  # fmt: skip
i = list(CASES).index("cullet +5 pts")
y = lag(out[1][i], 120.0) - tc0
c = int(np.argmax((y[240:] > 0))) + 240
print(
    "cullet +5: reading crosses zero at",
    c / 120,
    "h, glass then",
    (out[4][i] - base[4])[c],
)
for a_, b_ in (("cullet +10 pts", "blanket -20 %"), ("gas +1 %", "pull -2 %")):
    u_ = [lag(out[1][list(CASES).index(n)], 120.0) - tc0 for n in (a_, b_)]
    k = int(6 * 120) - 1
    sc = u_[0][k] / u_[1][k]
    hours = np.array([0.5, 1, 3, 6]) * 120
    print(a_, "vs", b_, "scaled", sc, np.round(u_[0][hours.astype(int) - 1], 2),
          np.round(sc * u_[1][hours.astype(int) - 1], 2))  # fmt: skip
