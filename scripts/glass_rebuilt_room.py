"""Judging steps 3 and 4: the ladder's best rung against the foresight plan.

    python glass_rebuilt_room.py CASE SEED LEVELS OUT.npz

CASE is a row of CASES, one named factor removed. The ladder's gains are the recipe's
on the full plant and don't move. Run with JAX_ENABLE_X64=1 so that episodes draw alike.
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import glass_rebuilt_ladder as L  # noqa: E402
import glass_rebuilt_oracle as O  # noqa: E402
from target_gym.glass_furnace import rebuilt as B  # noqa: E402

#: The recipe's printed result on the full plant.
KC, TI, FFG = 0.314, 1513.0, 0.916
LEVELS = {
    "full": ((20, 200), (5, 300), (2, 300), (1, 300)),
    "coarse": ((20, 200), (5, 300)),
}
CASES = {
    "full": {},
    "no firing gap": dict(reversal_dead_time=0.0),
    "no pull wander": dict(m_pull_noise_std=0.0),
    "no gas drift": dict(lhv_noise_std=0.0),
    "no cullet moves": dict(cullet_move=0.0),
    "no blanket moves": dict(thickness_move=0.0),
    "no fuel slew": dict(actuator_rate=1.0),
    "no charger pulses": dict(charge_duty=1.0, charge_mass_jitter=0.0),
    "no sensor noise": dict(sensor_noise_scale=0.0),
    "no sheath lag": dict(tau_thermocouple=1.0),
    "bath well mixed": dict(current_scale=1.0),
    "reading 5 min": dict(tc_depth=0.025),
    "reading 20 min": dict(tc_depth=0.050),
}


def with_offset(p):
    """A depth's own reading offset, by the same measure as ``tc_offset``."""
    if p.tc_depth == B.RebuiltParams().tc_depth:
        return p
    q = p.replace(
        sensor_noise_scale=0.0, m_pull_noise_std=0.0, lhv_noise_std=0.0,
        charge_duty=1.0, charge_mass_jitter=0.0, cullet_move=0.0, thickness_move=0.0,
    )  # fmt: skip
    _, s = L.env.reset_env(jax.random.PRNGKey(0), q)
    raw, x = L.env.settled(jnp.asarray(1590.0))
    s = s.replace(
        T_roof=x[: B.N_ROOF], T_bath=x[B.BS : B.WORK], T_work=x[B.WORK],
        T_rA=x[B.RA : B.RB], T_rB=x[B.RB :], m_batch=x[B.BATCH], lhv_deviation=jnp.zeros(()),
        fuel_pipeline=jnp.full((2,), q.fuel_min + 0.5 * (raw + 1) * (q.fuel_max - q.fuel_min)),
    )  # fmt: skip
    key = jax.random.PRNGKey(0)

    def f(c, _):
        c = B.next_state(raw, c, q, key, "rk4_2")
        return c, c.T_crown - B.brick_temperature(c.T_roof, q.tc_depth)

    return p.replace(tc_offset=float(jax.lax.scan(f, s, None, length=1000)[1].mean()))


if __name__ == "__main__":
    case, seed, levels, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    p = with_offset(B.RebuiltParams().replace(**CASES[case]))
    rungs = {
        "PI": (KC, KC / TI, 0.0, 0.0, L.TF),
        "FF": (KC, KC / TI, FFG, 0.0, L.TF),
        "lead": (KC, KC / TI, FFG, 600.0, 120.0),
    }
    res = {}
    for name, c in rungs.items():
        res[name] = L.score(f"{case}: {name}", c, p, jnp.array([seed]))
    best = min(rungs, key=lambda k: res[k]["cost"][0])
    u = np.asarray(res[best]["u"][0], dtype=np.float64)
    if u.ndim > 1:
        u = u[:, 0]
    print("ladder best", best, O.true_cost(seed, u, p), flush=True)
    for every, iters in LEVELS[levels]:
        t0 = time.time()
        u, r = O.plan(seed, u, p, every, iters)
        print(
            f"{case} seed {seed} knots {every * 30} s: {O.true_cost(seed, u, p)}"
            f" objective {r.fun:.4f} iters {r.nit} {time.time() - t0:.0f} s",
            flush=True,
        )
        np.savez(out, u=u, ladder=res[best]["cost"][0], best=best, every=every)
