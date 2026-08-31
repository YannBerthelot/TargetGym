"""Check every MPC's horizon against the time its plant needs to reach the target.

A receding-horizon controller can only optimise what it can see. If the horizon
is much shorter than the time the tracking error takes to close, every feasible
plan leaves the error essentially unchanged, the objective goes flat in the
variables that matter, and the MPC degenerates into greedy local behaviour. On
the 2D aircraft that meant flying into the altitude envelope, because nothing
within 30 s signalled that the boundary was coming.

The criterion:

    horizon * mpc_dt  >=  tau_close

``tau_close`` is the time a *viable* controller needs to bring the tracking
error to 1/e of its initial value and keep it there. One time constant rather
than full settling, because the final approach is asymptotic and is exactly
what re-planning handles well; what the horizon has to cover is the dominant
transient, where the shape of the plan matters.

"Viable" is doing real work in that sentence. An earlier version of this script
measured the *fastest* closure under any constant action and passed the 2D
aircraft -- an MPC already measured to fail at its shipped horizon and to
improve twofold when the horizon was extended. The fastest closure was a zoom
climb: full power and full stick trading airspeed for altitude, covering 1653 m
in 8 s and then departing controlled flight. A horizon sized on a manoeuvre
that ends in a crash is not a horizon that reaches the target. So a candidate
only counts here if its episode survives and the error *stays* closed rather
than grazing the threshold once.

The fix for a myopic MPC need not be more decision variables: the CasADi
controllers accept ``mpc_dt`` independently of the environment's ``delta_t``,
so a coarser prediction step buys more covered time at the same optimisation
cost. The criterion is in seconds, not steps.

    python scripts/audit_mpc_horizons.py
"""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "src")
import jax
import jax.numpy as jnp
import numpy as np

from target_gym.registry import REGISTRY

MAXSTEPS, SEEDS, HOLD = 400, (0, 1), 10


def as_tuple(i):
    return tuple(i) if isinstance(i, (tuple, list)) else (int(i),)


def trace(env, p, step, key, st, policy, vi, ti):
    errs, obs = [], None
    for _ in range(MAXSTEPS):
        a = policy(obs) if obs is not None else policy(env.get_obs(st, p))
        obs, st, _, term, _ = step(key, st, jnp.atleast_1d(jnp.asarray(a)), p)
        o = np.asarray(obs)
        errs.append(float(np.abs(o[list(vi)] - o[list(ti)]).mean()))
        if bool(term):
            return np.array(errs), True
    return np.array(errs), False


print(
    f"  {'environment':20s} {'e0':>10} {'t63 viable':>11} {'horizon':>9} {'ratio':>7}  verdict"
)
for name, spec in REGISTRY.items():
    if not spec.has_mpc:
        continue
    env = spec.make_env()
    p = spec.params_cls(max_steps_in_episode=MAXSTEPS)
    vi, ti = as_tuple(env.obs_value_index), as_tuple(env.obs_target_index)
    step = jax.jit(env.step_env)
    t63, e0s = np.inf, []
    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        obs0, st0 = env.reset_env(key, p)
        o = np.asarray(obs0)
        e0 = float(np.abs(o[list(vi)] - o[list(ti)]).mean())
        e0s.append(e0)
        if e0 <= 0 or not spec.has_pid:
            continue
        pid = spec.make_pid()
        pid.reset()
        errs, crashed = trace(env, p, step, key, st0, lambda ob: pid(ob), vi, ti)
        if crashed:
            continue
        below = errs <= 0.37 * e0
        # first index from which it stays below for HOLD steps -- closed, not grazed
        for i in range(len(below) - HOLD):
            if below[i : i + HOLD].all():
                t63 = min(t63, i + 1)
                break
    mpc = spec.make_mpc(env, p)
    H = getattr(mpc, "horizon", None)
    mdt = getattr(mpc, "mpc_dt", None) or float(p.delta_t)
    H_env = H * mdt / float(p.delta_t) if H else np.nan
    ratio = H_env / t63 if np.isfinite(t63) and t63 > 0 else np.nan
    v = (
        "ok"
        if (np.isfinite(ratio) and ratio >= 1.0)
        else ("MYOPIC" if np.isfinite(ratio) else "n/a")
    )
    print(
        f"  {name:20s} {np.mean(e0s):10.4g} {(f'{t63:.0f}' if np.isfinite(t63) else 'never'):>11} "
        f"{H_env:9.0f} {ratio:7.2f}  {v}",
        flush=True,
    )
