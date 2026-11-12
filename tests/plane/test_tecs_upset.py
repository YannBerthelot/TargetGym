"""The airliner's altitude hold recovers from an upset (long-strata E1).

Contract: an altitude hold returns to its level after an upset and stays
there; a settling time that never ends fails acceptance (long-strata R1/E1,
docs/proposals/re-record-pass.md step 2). The band is the TECS tuning guide's
10 m ("height changes greater than 10m" is an oscillation [G-TECS]), inside
RVSM's altitude-keeping tolerance of +-65 ft, 20 m [RVSM A.4.1.4,
plane/REALITY.md]. Ours: back inside it within 5 min, and still inside 40 min
after the upset.

The upset: in calm air, with the expert settled for 10 min, the aircraft is
displaced off its level, the expert's memory kept, as a gust would leave it.
Sizes: 20 m, the edge of the RVSM band; 50 m, about the largest error a
turbulence zone had left before a lasting cycle began in #190's E1 run (45.6 m,
plane, seed 2); 100 m, twice that (ours). The flight conditions are the two
where E1 found the cycles: plane at FL298 and 67 t, plane3d_heading at 3 600 m
and 77 t. Before the TECS fix of 2026-11-12 a 50 m upset on plane, and a 20 m
one on plane3d_heading, fell into a cycle of +-450 to +-600 m.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.registry import REGISTRY

BAND_M = 10.0
SETTLE_S = 300
WATCH_S = 2400

CASES = {"plane": (9091.0, 66956.0), "plane3d_heading": (3600.0, 76988.0)}


def _after_upset(name, dz):
    level, mass = CASES[name]
    spec = REGISTRY[name]
    env, ctrl = spec.make_env(), spec.make_pid_controller()
    params = spec.make_test_params().replace(
        target_altitude_range=(level, level),
        initial_altitude_range=(level, level),
        initial_mass=mass,
        turbulence_sigma=0.0,
        max_steps_in_episode=600 + WATCH_S,
    )
    if hasattr(params, "initial_altitude_offset_range"):
        params = params.replace(initial_altitude_offset_range=(0.0, 0.0))

    def fly(state, memory, steps):
        def body(carry, t):
            s, m = carry
            action, m = ctrl.act(m, env.get_obs(s, params))
            key = jax.random.fold_in(jax.random.PRNGKey(7), t)
            _, s, _, _, info = env.step_env(key, s, action, params)
            return (s, m), (s.z - s.target_altitude, info["tripped"])

        return jax.lax.scan(body, (state, memory), steps)

    @jax.jit
    def run():
        _, state = env.reset_env(jax.random.PRNGKey(3), params)
        memory = jax.tree.map(jnp.asarray, ctrl.init_state)
        (state, memory), _ = fly(state, memory, jnp.arange(600))
        state = state.replace(z=state.z + dz)
        _, (error, tripped) = fly(state, memory, 600 + jnp.arange(WATCH_S))
        return error, tripped

    error, tripped = (np.asarray(v) for v in run())
    return error, tripped


@pytest.mark.medium
@pytest.mark.parametrize("dz", (20.0, 50.0, -50.0, 100.0))
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_altitude_hold_settles_after_an_upset(name, dz):
    error, tripped = _after_upset(name, dz)
    assert not tripped.any()
    late = np.abs(error[SETTLE_S:])
    assert late.max() <= BAND_M, (
        f"{name}: {dz:+.0f} m upset, |error| {late.max():.0f} m after "
        f"{SETTLE_S} s (worst at {SETTLE_S + int(late.argmax())} s)"
    )
