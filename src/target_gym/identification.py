"""Bump tests and step-response fits, shared by the tuning protocols.

The expert protocols (docs/proposals/expert-protocols-process.md, C1 and C2)
and the oracle horizon protocol (docs/proposals/oracle-horizon-protocol.md,
H7 to H11) identify a plant the same way:

- the plant runs under its shipped expert from each training seed's reset,
  then the loops go to manual at the expert's average output;
- each manipulated variable (MV) is stepped on its own, and the response is
  the stepped run minus an unstepped run on the same noise (C1, H8), up to the
  first trip of either run (H9);
- each response is classed (H10) and fitted (H11). No fit is dropped,
  whatever its residual.

Not hashed into any fingerprint (``provenance``): it measures, and what it
measures is written to data files.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

__all__ = ["TRAIN_SEEDS", "Response", "bump_test", "fit_fopdt"]
__all__ += ["fit_integrating_lag", "identify", "is_integrating"]

#: The training seeds every protocol identifies on; 200-209 are held out.
TRAIN_SEEDS = tuple(range(100, 110))


@dataclass(frozen=True)
class Response:
    """One bump: ``dy[i]`` is the change ``(i + 1) dt`` after a step ``du``."""

    dy: np.ndarray
    dt: float
    du: float

    @property
    def t(self) -> np.ndarray:
        return self.dt * np.arange(1, len(self.dy) + 1)


def away_from_nearer_limit(u: float, size: float) -> float:
    """C2's direction for a raw MV in [-1, 1]: down from the upper half."""
    return -size if u > 0 else size


def _hold(state, mem, u):
    return u, mem


def bump_test(
    spec,
    params,
    cvs,
    sizes,
    *,
    n_settle,
    n_window,
    window=20,
    seeds=TRAIN_SEEDS,
    policy=_hold,
    memory=lambda state: jnp.zeros(()),
    start=None,
    direction=away_from_nearer_limit,
):
    """``{(mv, cv): [Response per seed]}`` for each MV stepped alone.

    The expert runs ``n_settle`` steps from each seed's reset; the MVs then
    hold ``start(actions, memories)``, by default the mean of the expert's
    last ``window`` commands. ``policy(state, mem, u) -> (action, mem)``
    applies a held ``u`` (by default as the raw action), its memory starting
    at ``memory(state)``. MV ``j`` is stepped by ``direction(u_j, sizes[j])``;
    each CV ``cvs[name](obs, state, params)`` is read for ``n_window`` steps.
    """
    env, ctrl = spec.make_env(), spec.make_pid_controller()
    names, n_mv = list(cvs), env.action_space(params).shape[0]
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))

    def expert(key):
        def body(carry, t):
            obs, state, mem = carry
            a, mem = ctrl.step(ctrl.params, mem, obs, jnp)
            a = jnp.reshape(a, (n_mv,))
            obs, state, *_ = env.step_env(jax.random.fold_in(key, t), state, a, params)
            return (obs, state, mem), (a, mem)

        carry0 = (*env.reset_env(key, params), ctrl.init_state)
        return jax.lax.scan(body, carry0, jnp.arange(n_settle))

    def manual(key, obs, state, u):
        def body(carry, t):
            obs, state, mem = carry
            a, mem = policy(state, mem, u)
            key_t = jax.random.fold_in(key, n_settle + t)
            obs, state, _, _, info = env.step_env(key_t, state, a, params)
            y = jnp.stack([jnp.reshape(cvs[n](obs, state, params), ()) for n in names])
            return (obs, state, mem), (y, info["tripped"])

        carry0 = (obs, state, memory(state))
        return jax.lax.scan(body, carry0, jnp.arange(n_window))[1]

    (obs, state, _), (acts, mems) = jax.jit(jax.vmap(expert))(keys)
    take = start or (lambda a, m: a[:, -window:].mean(axis=1))
    u0 = np.asarray(take(np.asarray(acts), mems), float)
    run = jax.jit(jax.vmap(manual))
    y0, trip0 = (np.asarray(x) for x in run(keys, obs, state, jnp.asarray(u0)))
    dt, out = float(params.delta_t), {}
    for j in range(u0.shape[1]):
        du = np.array([direction(u[j], sizes[j]) for u in u0])
        u1 = u0 + np.outer(du, np.eye(u0.shape[1])[j])
        y1, trip1 = (np.asarray(x) for x in run(keys, obs, state, jnp.asarray(u1)))
        for s in range(len(seeds)):
            tripped = np.flatnonzero(trip0[s] | trip1[s])
            m = int(tripped[0]) if tripped.size else n_window
            for i, name in enumerate(names):
                dy = y1[s, :m, i].astype(float) - y0[s, :m, i]
                out.setdefault((j, name), []).append(Response(dy, dt, float(du[s])))
    return out


def is_integrating(d) -> bool:
    """H10: the mean slope over the window's last quarter is at least half
    that over its first; ``d[0]`` is the value at the step. As
    ``scripts/measure_time_constants.py``, compared in magnitude, and a flat
    end is never a ramp."""
    d = np.asarray(d, float)
    q = max((len(d) - 1) // 4, 1)
    early, late = d[q] - d[0], d[-1] - d[-1 - q]
    return bool(late != 0 and abs(late) >= 0.5 * abs(early))


def _fit(model, guesses, lo, hi, t, dy):
    best = None
    for g in guesses:
        r = least_squares(lambda q: model(t, q) - dy, g, bounds=(lo, hi), x_scale="jac")
        best = r if best is None or r.cost < best.cost else best
    rms = float(np.sqrt(np.mean(best.fun**2)))
    return best.x, rms


def _delay_guess(t, dy) -> float:
    """The last sample before the response leaves 2 % of its largest move."""
    moved = np.flatnonzero(np.abs(dy) > 0.02 * np.max(np.abs(dy)))
    return float(t[moved[0] - 1]) if moved.size and moved[0] > 0 else 0.0


def fit_fopdt(t, dy, du) -> dict:
    """Least-squares ``k e^(-theta s) / (tau s + 1)`` to a step of ``du``."""
    t, dy = np.asarray(t, float), np.asarray(dy, float)

    def model(t, q):
        k, tau, theta = q
        return k * du * (1 - np.exp(-np.maximum(t - theta, 0.0) / tau))

    k0, th0 = dy[-1] / du, _delay_guess(t, dy)
    reached = np.flatnonzero(np.abs(dy) >= 0.632 * abs(dy[-1]))
    tau0 = max(float(t[reached[0]]) - th0 if reached.size else t[-1], t[0])
    guesses = [(k0, tau0, th0), (k0, tau0, 0.0)]
    lo, hi = [-np.inf, 1e-9, 0.0], [np.inf, np.inf, t[-1]]
    (k, tau, theta), rms = _fit(model, guesses, lo, hi, t, dy)
    return {"model": "fopdt", "k": k, "tau": tau, "theta": theta, "rms": rms}


def fit_integrating_lag(t, dy, du) -> dict:
    """Least-squares ``k' e^(-theta s) / (s (tau2 s + 1))`` to a step of
    ``du``; ``k'`` is the final slope per unit step."""
    t, dy = np.asarray(t, float), np.asarray(dy, float)

    def model(t, q):
        kp, tau2, theta = q
        x = np.maximum(t - theta, 0.0)
        return kp * du * (x - tau2 * (1 - np.exp(-x / tau2)))

    h = len(t) // 2
    kp0 = (dy[-1] - dy[h]) / (t[-1] - t[h]) / du
    lag = min(max(t[-1] - dy[-1] / (kp0 * du), 0.0), t[-1]) if kp0 else 0.0
    guesses = [(kp0, max(lag / 2, t[0]), lag / 2), (kp0, t[0], _delay_guess(t, dy))]
    lo, hi = [-np.inf, 1e-9, 0.0], [np.inf, t[-1], t[-1]]
    (kp, tau2, theta), rms = _fit(model, guesses, lo, hi, t, dy)
    fit = {"k": kp, "tau2": tau2, "theta": theta, "rms": rms}
    return {"model": "integrating_lag", **fit}


def identify(r: Response) -> dict:
    """H10 and H11 on one response: its class and the fit the class names.
    A response with nothing to fit (no move, or a trip within four steps)
    is reported as such, never fitted."""
    if len(r.dy) < 4:
        return {"class": "too short", "samples": len(r.dy)}
    if not np.any(r.dy):
        return {"class": "no response", "samples": len(r.dy)}
    integrating = is_integrating(np.concatenate([[0.0], r.dy]))
    fit = (fit_integrating_lag if integrating else fit_fopdt)(r.t, r.dy, r.du)
    cls = "integrating" if integrating else "self-regulating"
    fit = {k: v if isinstance(v, str) else float(v) for k, v in fit.items()}
    return {"class": cls, "samples": len(r.dy), "du": r.du, **fit}
