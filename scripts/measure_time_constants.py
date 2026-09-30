"""Control-relevant time constant, and the discount the protocol derives from it.

Run with ``uv run python scripts/measure_time_constants.py``. Re-run it after
changing an actuator, a plant's dynamics or an episode length: the numbers in
docs/rl-protocol.md section 6 come from here, and a discount quoted against
dynamics that have moved is the same kind of stale claim the recorded baselines
guard against.

: the tracked output's response to the actuator.

Linearising the whole state matrix gives the slowest mode in the plant, which is
often not the one a controller acts through -- for the 2D aircraft it is fuel
burn at 29 694 s, and for the reactor it is xenon at 61 908 s. Neither says how
long the plant takes to answer the stick or the rod.

What a control engineer means by "the time constant" is the open-loop step
response from actuator to tracked output: hold the input steady, step it, and
measure the time to 63.2% of the total change. That is the quantity relay
tuning assumes and the one the discount should be set from.

Plants whose output integrates the input (a tank level, a drum level) have no
steady state to settle to, and are reported as integrating rather than fitted.
"""

import warnings

# Ahead of the imports below: importing target_gym pulls in do-mpc, which warns
# about optional features this package does not use.
warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.runners.runners import _as_tuple  # noqa: E402

out = {}
print(f"  {'environment':20s} {'dt':>8s} {'tau_63':>10s} {'steps':>7s}  shape")
for name, spec in REGISTRY.items():
    params = spec.make_test_params()
    env = spec.make_env()
    dt = float(getattr(params, "delta_t", 1.0))
    idx = list(_as_tuple(env.obs_value_index))[:1]
    step = jax.jit(env.step_env)
    adim = int(
        np.atleast_1d(env.action_space(params).sample(jax.random.PRNGKey(0))).shape[0]
    )
    n = int(params.max_steps_in_episode)

    def run(level):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset_env(key, params)
        ys = [float(np.asarray(obs)[idx][0])]
        a = jnp.full((adim,), level, dtype=jnp.float32)
        for _ in range(n):
            key, sub = jax.random.split(key)
            obs, state, _, term, _ = step(sub, state, a, params)
            ys.append(float(np.asarray(obs)[idx][0]))
            if bool(term):
                break
        return np.array(ys)

    try:
        lo, hi = run(-0.4), run(0.4)
    except Exception as exc:
        print(f"  {name:20s} step test failed: {type(exc).__name__}")
        continue

    m = min(len(lo), len(hi))
    d = hi[:m] - lo[:m]  # differential response kills drift
    if m < 5 or not np.isfinite(d).all():
        print(f"  {name:20s} unusable response")
        continue
    total = d[-1] - d[0]
    if abs(total) < 1e-12:
        print(f"  {name:20s} {dt:8.2f}  no response to the actuator")
        continue

    # Integrating if the response is still climbing at a near-constant rate at
    # the end: compare the last quarter's slope with the first quarter's.
    q = max(m // 4, 2)
    early = (d[q] - d[0]) / (q * dt)
    late = (d[-1] - d[-q]) / (q * dt)
    integrating = abs(late) > 0.5 * abs(early) and abs(late) > 1e-12

    target = d[0] + 0.632 * total
    cross = np.argmax((d - d[0]) / total >= 0.632) if total != 0 else 0
    tau = float(cross * dt) if cross > 0 else float("nan")
    shape = "integrating" if integrating else "first-order-ish"
    out[name] = {
        "dt": dt,
        "tau": tau,
        "steps": tau / dt if tau == tau else None,
        "shape": shape,
        "episode": n,
    }
    print(
        f"  {name:20s} {dt:8.2f} {tau:10.1f} {tau/dt if tau==tau else float('nan'):7.1f}  {shape}"
    )

print("\n  gamma = 1 - 1/N per the protocol; five time constants must fit in N:")
for n, r in out.items():
    if r["steps"] and 5 * r["steps"] > r["episode"]:
        print(f"    {n:20s} needs {5 * r['steps'] / r['episode']:.1f}x its episode")
