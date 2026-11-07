"""Judging step 1: the control properties of the rebuilt furnace, no controller.

Fuel held, noise off, moves off unless a row makes one. Like phases of the reversal
cycle are compared by averaging over one 50 min cycle, as the proposal's table did.
"""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import rebuilt as B

DT = 30.0
CYC = 100
env = B.RebuiltFurnace()
base = B.RebuiltParams(
    sensor_noise_scale=0.0, m_pull_noise_std=0.0, lhv_noise_std=0.0,
    charge_duty=1.0, charge_mass_jitter=0.0, cullet_move=0.0, thickness_move=0.0,
)  # fmt: skip
KEY = jax.random.PRNGKey(0)
HOURS = lambda n: np.arange(n) * DT / 3600  # noqa: E731


def start(T, p, days=25):
    """The true orbit at a hot-face mean, by holding the table's firing, a cycle
    boundary, on params ``p``."""
    _, s = env.reset_env(KEY, p)
    raw, x = env.settled(jnp.asarray(T))
    fuel = p.fuel_min + 0.5 * (raw + 1) * (p.fuel_max - p.fuel_min)
    s = s.replace(
        T_crown=x[0], T_roof=x[: B.N_ROOF], T_bath=x[B.BS : B.WORK], T_work=x[B.WORK],
        T_rA=x[B.RA : B.RB], T_rB=x[B.RB :], m_batch=x[B.BATCH],
        fuel_pipeline=jnp.full((2,), fuel), lhv_deviation=jnp.zeros(()),
        move_sizes=jnp.zeros_like(s.move_sizes),
        T_crown_meas=B.brick_temperature(x[: B.N_ROOF], p.tc_depth) + p.tc_offset,
    )  # fmt: skip
    return raw, s


def run(state, raw, p, steps):
    def f(s, _):
        s = B.next_state(raw, s, p, KEY, "rk4_2")
        return s, jnp.stack(
            [s.T_crown, s.T_crown_meas, s.T_bath[2], s.T_work, s.m_batch]
        )

    return np.asarray(jax.jit(lambda: jax.lax.scan(f, state, None, length=steps))()[1])


def ave(y):
    return np.convolve(y, np.ones(CYC) / CYC, "valid")


def settle(T, p):
    raw, s = start(T, p)
    s = jax.jit(
        lambda: jax.lax.scan(
            lambda c, _: (B.next_state(raw, c, p, KEY, "rk4_2"), None),
            s, None, length=30 * 2880,
        )[0]
    )()  # fmt: skip
    return raw, s.replace(time=0)


def step_props(tag, T, p):
    raw, s = settle(T, p)
    N = 20 * 2880
    a = run(s, raw, p, N)
    b = run(s, raw + 0.1, p, N)
    H = HOURS(N - CYC + 1)
    dfuel = 0.05 * (p.fuel_max - p.fuel_min)
    for name, col in (("hot face", 0), ("reading", 1)):
        d = ave(b[:, col] - a[:, col])
        fin = d[-300:].mean()
        t10 = H[np.argmax(d >= 0.1 * fin)]
        t63 = H[np.argmax(d >= 0.632 * fin)]
        t95 = H[np.argmax(d >= 0.95 * fin)]
        raw_d = b[:600, col] - a[:600, col]
        print(
            f"{tag:18s} {name:9s} gain {fin / dfuel:6.0f} K per kg/s  t10 {t10:5.2f} h"
            f" t63 {t63:5.2f} h t95 {t95:5.1f} h  share at 30 min {raw_d[60:160].mean() / fin:.2f}"
            f" at 2 h {d[240] / fin:.2f}"
        )
    d = b[:, 1] - a[:, 1]
    print(f"   reading, first steps K: {np.round(d[:12], 3)}")


print("-- fuel step, by operating point")
for tag, T, pm in (
    ("crown 1570", 1570.0, 1.0),
    ("crown 1590", 1590.0, 1.0),
    ("crown 1605", 1605.0, 1.0),
    ("1590, pull -10%", 1590.0, 0.9),
    ("1590, pull +10%", 1590.0, 1.1),
):
    step_props(tag, T, base.replace(m_pull=base.m_pull * pm))

print("-- hidden kicks seen in the reading, fuel held (noise 0.36 K)")
raw, s = settle(1590.0, base)
N = 20 * 2880
a = run(s, raw, base, N)
kicks = {
    "hot-spot surface +5 K": s.replace(T_bath=s.T_bath.at[2].add(5.0)),
    "all bath +5 K": s.replace(T_bath=s.T_bath + 5.0),
    "working end +5 K": s.replace(T_work=s.T_work + 5.0),
    "batch +10 %": s.replace(m_batch=s.m_batch * 1.1),
}
for tag, k in kicks.items():
    d = ave(run(k, raw, base, N)[:, 1] - a[:, 1])
    print(
        f"{tag:24s} reading at 1,6,24,96 h:",
        [round(float(d[int(h * 120)]), 2) for h in (1, 6, 24, 96)],
    )

print("-- lookalike upsets: reading after each, fuel held")


def moved_state(which, size):
    mt = s.move_times.at[which, 0].set(-1e7)
    ms = s.move_sizes.at[which, 0].set(size)
    return s.replace(move_times=mt, move_sizes=ms)


rows = {
    "gas +1 %": (s, base.replace(LHV=base.LHV * 1.01)),
    "pull -2 %": (s, base.replace(m_pull=base.m_pull * 0.98)),
    "pull +2 %": (s, base.replace(m_pull=base.m_pull * 1.02)),
    "cullet +5 points": (moved_state(0, 0.05), base),
    "cullet -5 points": (moved_state(0, -0.05), base),
    "thickness +20 %": (moved_state(1, 0.20), base),
}
for tag, (st, p) in rows.items():
    r = run(st, raw, p, N)
    d = ave(r[:, 1] - a[:, 1])
    print(
        f"{tag:18s} at 1,6,24,96 h:",
        [round(float(d[int(h * 120)]), 2) for h in (1, 6, 24, 96)],
        " batch +%:",
        round(100 * float(r[:400, 4].max() / a[:400, 4].mean() - 1), 1),
    )

print("-- upset speed: fastest slope of the cycle-mean reading, K/h")
for tag, (st, p) in rows.items():
    r = run(
        st.replace(
            move_times=(
                st.move_times.at[:, 0].set(600.0 * 30)
                if "pull" not in tag and "gas" not in tag
                else st.move_times
            )
        ),
        raw,
        p,
        2880,
    )
    d = ave(r[:, 1] - a[:2880, 1])
    print(f"{tag:18s} {np.abs(np.gradient(d, DT / 3600)).max():.2f} K/h")

c = a[:300]
print(
    "reversal swing p-p: hot face",
    round(float(np.ptp(c[:, 0])), 2),
    "reading",
    round(float(np.ptp(c[:, 1])), 2),
)
