"""Score one family of controllers under four candidate reward shapes.

Produces the table in docs/reward-shaping.md. The controllers are the shipped
cascaded PID detuned by scaling its outer altitude gain, so they differ in
tracking precision and nothing else; the same trajectories are then scored
under every shape, each normalised to 1 at the target and 0 at the edge of the
envelope so the columns are comparable.

    python scripts/compare_reward_shapes.py
"""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "src")
import jax
import jax.numpy as jnp
import numpy as np

from target_gym.registry import REGISTRY

spec = REGISTRY["plane"]
env = spec.make_env()
p = spec.params_cls(max_steps_in_episode=600)
SPAN = p.max_alt - p.min_alt
BAND = 50.0  # band / pseudo-Huber transition scale
FLOOR = 1.0  # precision floor


# each normalised so r(0)=1 and r(SPAN)=0, so the columns are comparable
def r_original(e):
    return (1 - e / SPAN) ** 10


def r_huber(e):
    h = lambda x: np.sqrt(1 + (x / BAND) ** 2) - 1
    return 1 - h(e) / h(SPAN)


def r_band(e):
    return 1 / (1 + (e / BAND) ** 2)


def r_log(e):
    return 1 - np.log1p(e / FLOOR) / np.log1p(SPAN / FLOOR)


FORMS = {
    "original (1-e/span)^10": r_original,
    "pseudo-Huber (quad/lin)": r_huber,
    "band 1/(1+(e/b)^2)": r_band,
    "log-scaled": r_log,
}


def rollout(kp_scale, seeds=(0, 1, 2)):
    """Settled |error| for a PID detuned by scaling its outer altitude gain."""
    errs = []
    for seed in seeds:
        pid = spec.make_pid()
        pid.Kp_alt = pid.Kp_alt * kp_scale
        pid.reset()
        key = jax.random.PRNGKey(seed)
        obs, st = env.reset_env(key, p)
        step = jax.jit(env.step_env)
        traj = []
        for _ in range(p.max_steps_in_episode):
            obs, st, _, term, _ = step(
                key, st, jnp.atleast_1d(jnp.asarray(pid(obs))), p
            )
            o = np.asarray(obs)
            traj.append(abs(float(o[1]) - float(o[6])))
            if bool(term):
                break
        traj = np.array(traj)
        errs.append(traj[len(traj) // 2 :])  # settled half
    return np.concatenate(errs)


scales = [0.15, 0.3, 0.6, 1.0, 1.6, 2.4]
rows = []
for s in scales:
    e = rollout(s)
    rows.append((s, e.mean(), {k: f(e).mean() for k, f in FORMS.items()}))

print(f"  {'Kp_alt':>7} {'settled |err| m':>16}  " + "".join(f"{k:>26}" for k in FORMS))
for s, m, sc in rows:
    print(f"  x{s:<6.2f} {m:16.2f}  " + "".join(f"{sc[k]:26.4f}" for k in FORMS))

print("\n  Discrimination: reward gained by going from the sloppiest to the tightest,")
print("  and the share of that gain earned in the last (most precise) step.")
best, worst = rows[-1], rows[0]
by_err = sorted(rows, key=lambda r: r[1])
tight, loose = by_err[0], by_err[-1]
print(f"\n  tightest = {tight[1]:.2f} m   loosest = {loose[1]:.2f} m")
print(f"  {'form':28s} {'loose->tight':>13} {'2nd-best->best':>16} {'share':>8}")
second = by_err[1]
for k in FORMS:
    total = tight[2][k] - loose[2][k]
    last = tight[2][k] - second[2][k]
    print(f"  {k:28s} {total:13.4f} {last:16.4f} {(last/total if total else 0):7.1%}")
