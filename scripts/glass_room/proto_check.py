"""Checks of the prototype's tracers against the plant itself.

1. The second plant step the wrapper runs for its flows gives the env's own plant.
2. With fining switched off, the plant carries its fining share by transport alone, and
   the wrapper's transport of that same field must land on the plant's.
3. Each tracer's books close: what the blocks made is in the tank or has left.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import proto as M  # noqa: E402

L, P, E, H = M.L, M.P, M.E, M.H
p = M.BASE
par = (jnp.asarray(0.5523), jnp.asarray(1.0), jnp.asarray(1.0e9), jnp.asarray(0.0))

# 1 and 3: two days of the scored normal day, with its walks
info = jax.jit(lambda: M.episode(8, par, p, p, 100, 2 * 24 * H))()
t = {k: np.asarray(v) for k, v in info.items()}
print(
    "1. largest gap between the env's plant and the wrapper's second step, K:",
    t["mismatch"].max(),
)
assert t["mismatch"].max() < 1e-2
made, left, held = t["made"].sum(0), t["left"].sum(0), t["held"][-1]
for i, name in enumerate(("contact", "reboil")):
    gap = abs(made[i] - left[i] - held[i]) / max(made[i], 1e-30)
    print(
        f"3. {name}: made {made[i]:.4g}, left {left[i]:.4g}, in the tank {held[i]:.4g}, books off by {gap:.2e}"
    )
    assert made[i] == 0 or gap < 2e-3

# 2: transport of the fining share with fining off
slow = p.replace(
    walk_sd=(0.0,) * 5,
    sensor_noise_scale=0.0,
    plant=p.plant.replace(fining_time=1.0e30),
)
_, st = L.ENV.reset_env(jax.random.PRNGKey(0), slow)


def f(s, k):
    _, s1, _, _, _ = L.ENV.step_env(k, s, E.to_action(jnp.asarray(0.48), slow), slow)
    pp = slow.plant.replace(
        delta_t=slow.delta_t, sulfate_batch=slow.sulfate * (1.0 + s1.walk[4])
    )
    _, books = P.plant_step(
        s.plant,
        s1.fuel,
        s1.pull,
        s1.cullet,
        slow.thickness * (1.0 + s1.walk[3]),
        s.cooling,
        slow.grid,
        pp,
        lhv=E.gas_factor(s1.walk, s1.time, s, slow),
    )
    mine, _ = M.transport(s.plant.fining, books, pp)
    return s1, jnp.max(jnp.abs(mine - s1.plant.fining))


_, gap = jax.jit(
    lambda: jax.lax.scan(f, st, jax.random.split(jax.random.PRNGKey(5), 24 * H))
)()
print(
    "2. largest gap between the wrapper's transport and the plant's, fining share:",
    float(np.asarray(gap).max()),
)
assert float(np.asarray(gap).max()) < 1e-5
print(
    "contact area per kg of glass, m2/kg:",
    M.CONTACT_PER_KG,
    "| plate density for q10 = 1 %:",
    M.d_contact(0.01),
)
print("upward travel of the blocks over the two days, K (mass mean):", t["rise"].sum())
