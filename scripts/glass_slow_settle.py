"""Settle attempt 3's plant: the fuel that holds the crown's face at 1590 C in the bay of
W2000b's peak, with the refiner cooling that holds the canal 1 K over its target.

    python glass_slow_settle.py OUT.npz [days]
    python glass_slow_settle.py tune SETTLED.npz

Both are found by secant steps on settled runs, the cooling inside the fuel. ``tune``
steps the cooling by 5 % with the fuel held and sets the refiner loop by SIMC, with the
closed-loop time the larger of the dead time and 30 min, the middle of the 20 to 40 min
a glass loop answers in (Andritz). Nothing here feeds back into a parameter of the plant.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import physical as P

CROWN = 1590.0 + 273.15  # K, the hot face, W2000b's peak at 14 m
EP = E.GlassFurnace().default_params
G, PP = EP.grid, EP.plant
CANAL = EP.canal_target + EP.cooling_above


def hold(s, fuel, cooling, steps):
    """Fields after ``steps`` held, and the last hour's mean crown face and canal."""

    def f(s, _):
        n, _ = P.plant_step(s, fuel, EP.pull, EP.cullet, EP.thickness, cooling, G, PP)
        r = P.sensors(n, G, PP)
        return n, jnp.stack([n.T_roof[0, P.COUPLE_BAY], r["canal"]])

    s, y = jax.lax.scan(f, s, None, length=steps)
    return s, y[-30:].mean(0)


def secant(f, x0, x1, target, tol, lo, hi, n=8):
    y0, y1 = f(x0), f(x1)
    for _ in range(n):
        if abs(y1 - target) < tol or y1 == y0:
            break
        x0, x1, y0 = (
            x1,
            float(np.clip(x1 + (target - y1) * (x1 - x0) / (y1 - y0), lo, hi)),
            y1,
        )
        y1 = f(x1)
    return x1, y1


def settle(days=30):
    steps = int(days * 86400 / PP.delta_t)
    run = jax.jit(lambda s, u, q: hold(s, u, q, steps))
    state = {"s": P.initial_fields(G, PP)}  # built outside any trace

    def canal_at(fuel):
        def f(q):
            s, y = run(state["s"], fuel, q)
            state["s"], state["y"] = s, y
            return float(y[1])

        q, _ = secant(f, 0.8e6, 1.2e6, CANAL, 0.2, 0.0, 2.0e7)
        print(f"  fuel {fuel:.4f}: cooling {q / 1e6:.3f} MW", flush=True)
        state["q"] = q
        return float(state["y"][0])

    fuel, crown = secant(canal_at, 0.47, 0.48, CROWN, 0.3, 0.3, 0.8)
    return fuel, state["q"], state["s"], crown, float(state["y"][1])


def tune(path):
    d = np.load(path)
    fuel, q = float(d["fuel"]), float(d["cooling"])
    s = P.Fields(**{k: jnp.asarray(d[k]) for k in d if k not in ("fuel", "cooling")})
    n = int(2 * 86400 / PP.delta_t)

    def canal(cool):
        def f(s, _):
            s, _ = P.plant_step(s, fuel, EP.pull, EP.cullet, EP.thickness, cool, G, PP)
            return s, P.sensors(s, G, PP)["canal"]

        return jax.lax.scan(f, s, None, length=n)[1]

    y = np.asarray(jax.jit(canal)(q * 1.05) - jax.jit(canal)(q))
    final = y[-30:].mean()
    t = lambda frac: np.argmax(np.abs(y) > frac * abs(final)) * PP.delta_t
    tau = 1.5 * (t(0.632) - t(0.283))
    theta = max(t(0.632) - tau, PP.delta_t)
    k = final / (0.05 * q)  # K per W, negative
    tc = max(theta, 1800.0)
    kc = tau / (-k * (tc + theta))
    ti = min(tau, 4.0 * (tc + theta))
    print(
        f"canal {final:.2f} K for +5 % cooling: gain {k * 1e6:.1f} K/MW, lag {tau / 60:.0f} min,"
        f" dead time {theta / 60:.0f} min -> cooling_gain {kc:.4g} W/K, cooling_ti {ti:.0f} s"
    )


if __name__ == "__main__" and sys.argv[1] == "tune":
    tune(sys.argv[2])
elif __name__ == "__main__":
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    fuel, q, s, crown, canal = settle(days)
    np.savez(
        sys.argv[1], fuel=fuel, cooling=q,
        **{k: np.asarray(getattr(s, k)) for k in s.__dataclass_fields__},
    )  # fmt: skip
    print(
        f"fuel {fuel:.4f} kg/s, cooling {q / 1e6:.3f} MW, crown face {crown - 273.15:.1f} C,"
        f" canal {canal - 273.15:.1f} C, SEC {fuel * P.mean_firing(PP) * PP.LHV / EP.pull / 1e6:.2f} GJ/t",
        flush=True,
    )
