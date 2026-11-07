"""Print what the rebuilt furnace's reset needs: two orbit rows, the reading's offset
and the cullet's gain on the melting heat. Rerun if the plant's balance changes."""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import rebuilt as B
from target_gym.glass_furnace.env import SETTLED_LOW as SHIPPED
from target_gym.utils import convert_raw_action_to_range

DAYS = 40
CYCLE = 100  # steps in two reversals, 50 min
FUEL_DOWN = 0.025  # 2.5 % less fuel per 10 points of cullet


def quiet(H_scale=1.0):
    return B.RebuiltParams(
        sensor_noise_scale=0.0, m_pull_noise_std=0.0, lhv_noise_std=0.0,
        charge_duty=1.0, charge_mass_jitter=0.0, H_melt=2.2e6 * H_scale,
    )  # fmt: skip


def start(raw, p):
    fuel = convert_raw_action_to_range(raw, min_action=p.fuel_min, max_action=p.fuel_max)
    x = np.zeros(B.N_POS)
    x[: B.N_ROOF] = [1590, 1570, 1520, 1430, 1270]
    x[B.BS : B.WORK] = 1550
    x[B.WORK], x[B.BATCH] = 1450, 10000.0
    x[B.RA :] = np.array(SHIPPED)[5:]
    x = jnp.array(x)
    return B.RebuiltState(
        time=0, T_gas=1620.0, T_roof=x[: B.N_ROOF], T_bath=x[B.BS : B.WORK],
        T_work=x[B.WORK], T_rA=x[B.RA : B.RB], T_rB=x[B.RB :], m_batch=x[B.BATCH],
        T_crown=x[0], target_T_crown=1590.0, target_schedule=jnp.full(5, 1590.0),
        m_pull_disturbance=0.0, lhv_deviation=0.0,
        move_times=jnp.zeros((2, B.N_MOVES)), move_sizes=jnp.zeros((2, B.N_MOVES)),
        T_crown_meas=1590.0, fuel_pipeline=jnp.full(2, fuel), fuel_flow=fuel,
        T_air_preheat=1200.0, T_stack=500.0,
    )  # fmt: skip


def orbit(raw, p):
    """``(hot face's cycle mean, reading's offset, end state)``, run for DAYS in
    whole reversal cycles, so the state is where an episode starts."""
    key = jax.random.PRNGKey(0)

    def step(s, _):
        s = B.next_state(raw, s, p, key, "rk4_2")
        return s, s.T_crown - B.brick_temperature(s.T_roof, p.tc_depth)

    @jax.jit
    def run(s):
        s, _ = jax.lax.scan(step, s, None, length=DAYS * 2880)
        s = s.replace(time=0)
        _, y = jax.lax.scan(lambda c, _: step(c, 0), s, None, length=CYCLE)
        mean = jax.lax.scan(lambda c, _: (lambda n: (n, n.T_crown))(
            B.next_state(raw, c, p, key, "rk4_2")), s, None, length=CYCLE)[1].mean()
        return mean, y.mean(), s

    return run(start(raw, p))


def row(raw):
    p = quiet()
    mean, offset, s = orbit(raw, p)
    x = B.positions_of(s)
    return float(mean), float(offset), [float(mean), raw] + [float(v) for v in x]


def crown_of(raw, H_scale):
    return float(orbit(raw, quiet(H_scale))[0])


if __name__ == "__main__":
    low, off_low, row_low = row(-0.05)
    high, off_high, row_high = row(0.20)
    print("SETTLED_LOW =", tuple(round(v, 2) for v in row_low))
    print("SETTLED_HIGH =", tuple(round(v, 2) for v in row_high))
    print("offset", off_low, off_high)
    # The melting heat that, at the same crown, burns 2.5 % less fuel.
    slope = (high - low) / 0.25
    span = quiet().fuel_max - quiet().fuel_min
    down_raw = FUEL_DOWN * 0.59 / (0.5 * span)
    raw0 = 0.15
    a, b = 0.9, 1.0
    want = crown_of(raw0, 1.0)
    for _ in range(14):
        m = 0.5 * (a + b)
        if crown_of(raw0 - down_raw, m) < want:
            b = m
        else:
            a = m
    print("slope K per raw", slope, "H scale", m, "cullet_gain", (1 - m) / 0.10)
