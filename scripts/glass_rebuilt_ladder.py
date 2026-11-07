"""Judging steps 2 and 3's reference: the ladder on the rebuilt furnace.

A written recipe, run once: a noise-free bump on the reading gives SIMC gains with
tc = theta, then pull feedforward from the plant's own balance, then a lead on the
reading. No cascade rung: no reading of the bath exists. Nothing here looks at a
result to set a number.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import rebuilt as B

DT = 30.0
env = B.RebuiltFurnace()
TF = 82.2  # s, the shipped expert's reading filter
SEEDS = jnp.arange(100, 112)
SKIP = 240  # the first 2 h, as the proposal's evidence dropped its first hour or so


def bump():
    """(k' K/s per raw, theta s), on the reading: tangent at its steepest slope,
    in the cycle-averaged sense, from a step placed clear of a firing gap."""
    p = B.RebuiltParams(
        sensor_noise_scale=0.0, m_pull_noise_std=0.0, lhv_noise_std=0.0,
        charge_duty=1.0, charge_mass_jitter=0.0, cullet_move=0.0, thickness_move=0.0,
    )  # fmt: skip
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    raw, x = env.settled(jnp.asarray(1590.0))
    fuel = p.fuel_min + 0.5 * (raw + 1) * (p.fuel_max - p.fuel_min)
    s = s.replace(
        T_roof=x[: B.N_ROOF], T_bath=x[B.BS : B.WORK], T_work=x[B.WORK],
        T_rA=x[B.RA : B.RB], T_rB=x[B.RB :], m_batch=x[B.BATCH],
        fuel_pipeline=jnp.full((2,), fuel), lhv_deviation=jnp.zeros(()),
        actuator_held=jnp.full((1,), raw),
        T_crown_meas=B.brick_temperature(x[: B.N_ROOF], p.tc_depth) + p.tc_offset,
    )  # fmt: skip

    def path(step):
        def f(c, t):
            u = raw + step * (t >= 2)
            _, c, *_ = env.step_env(jax.random.PRNGKey(0), c, jnp.full((1,), u), p)
            return c, c.T_crown_meas

        return np.asarray(jax.jit(lambda: jax.lax.scan(f, s, jnp.arange(48))[1])())

    d = path(0.2) - path(0.0)
    slope = np.gradient(d, DT)
    i = int(np.argmax(slope))
    return float(slope[i]) / 0.2, max(i * DT - d[i] / slope[i] - 2 * DT, DT)


def pull_gain():
    """Raw fuel per kg/s of pull that holds the crown: the balance, from the two
    orbits' firings and the plant's steady gain."""
    p = B.RebuiltParams(
        sensor_noise_scale=0.0, m_pull_noise_std=0.0, lhv_noise_std=0.0,
        charge_duty=1.0, charge_mass_jitter=0.0, cullet_move=0.0, thickness_move=0.0,
    )  # fmt: skip
    from glass_rebuilt_settle import orbit  # the same orbit routine, same folder

    a = float(orbit(0.05, p)[0])
    b = float(orbit(0.05, p.replace(m_pull=p.m_pull * 1.02))[0])
    c = float(orbit(0.15, p)[0])
    per_raw = (c - a) / 0.10
    return (a - b) / per_raw / (0.02 * p.m_pull)


def episode(seed, c, p):
    """A whole episode under gains ``c``: (Kp, Ki, ff, lead, Tf)."""
    key = jax.random.PRNGKey(seed)
    ob, st = env.reset_env(key, p)
    Kp, Ki, ff, lead, Tf = c
    alpha = 1.0 - jnp.exp(-DT / Tf)
    raw0 = env.initial_action(st, p)[0]

    def f(carry, t):
        st, ob, y, pf, I = carry
        y = y + alpha * (ob[0] - y)
        pf = pf + (1.0 - jnp.exp(-DT / 3600.0)) * (ob[5] - pf)
        est = y + lead * (ob[0] - y) / Tf
        e = ob[4] - est
        I2 = I + e * DT
        u = Kp * e + Ki * I2 + ff * (pf - p.m_pull)
        uc = jnp.clip(u, -1.0, 1.0)
        I = jnp.where(u == uc, I2, I)
        kt = jax.random.fold_in(key, t)
        ob, st, r, d, info = env.step_env(kt, st, uc[None], p)
        return (st, ob, y, pf, I), (
            -r, st.T_crown - st.target_T_crown, st.T_crown_meas - st.target_T_crown,
            uc, st.T_bath, st.T_work, st.fuel_flow, info["tripped"],
        )  # fmt: skip

    I0 = raw0 / jnp.maximum(Ki, 1e-12)
    carry = (st, ob, ob[0], ob[5], I0)
    return jax.lax.scan(f, carry, jnp.arange(p.max_steps_in_episode))[1]


def score(name, c, p, seeds=SEEDS):
    run = jax.jit(jax.vmap(lambda s: episode(s, c, p)))
    cost, e, m, u, bath, work, fuel, trip = [np.asarray(v) for v in run(seeds)]
    e2 = e[:, SKIP:]
    track = (e2**2).mean(1)
    row = dict(
        cost=cost[:, SKIP:].mean(1), hot_rms=np.sqrt(track), read_sd=m[:, SKIP:].std(1),
        trips=trip.any(1), u=u, bath=bath, fuel=fuel[:, SKIP:].mean(1),
    )  # fmt: skip
    print(
        f"{name:32s} cost {row['cost'].mean():7.2f}  hot face rms {row['hot_rms'].mean():6.3f}"
        f"  reading sd {row['read_sd'].mean():5.3f}  trips {int(row['trips'].sum())}"
        f"  fuel {row['fuel'].mean():.4f}",
        flush=True,
    )
    return row


if __name__ == "__main__":
    k, theta = bump()
    Kc = 1.0 / (k * 2 * theta)
    Ti = 4 * 2 * theta
    print(
        f"bump k' {k:.5f} K/s per raw, theta {theta:.0f} s -> Kc {Kc:.3f}, Ti {Ti:.0f}"
    )
    ffg = pull_gain()
    print(f"pull feedforward {ffg:.3f} raw per kg/s")
    p = B.RebuiltParams()
    pi = (Kc, Kc / Ti, 0.0, 0.0, TF)
    rows = {}
    rows["PI"] = score("PI, SIMC on the reading", pi, p)
    rows["FF"] = score("+ pull feedforward", (Kc, Kc / Ti, ffg, 0.0, TF), p)
    for depth_lag in (600.0,):
        rows["lead"] = score(
            "+ lead on the reading, 10 min", (Kc, Kc / Ti, ffg, depth_lag, 120.0), p
        )
    np.savez(
        sys.argv[1] if len(sys.argv) > 1 else "ladder.npz",
        **{f"{k}_u": v["u"] for k, v in rows.items()},
        **{f"{k}_cost": v["cost"] for k, v in rows.items()},
        Kc=Kc, Ti=Ti, ffg=ffg,
    )  # fmt: skip
