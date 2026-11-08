"""The reference ladder's row for ``unstable_cstr`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from reference_ladder import (  # noqa: E402
    Row,
)

from target_gym.experts.pid import Controller
from target_gym.registry import REGISTRY


def _lee_cascade(p):
    """The inner PID by Lee, Lee and Park (2000), lambda = theta, on the unstable
    pole at the band's fastest end, under an outer SIMC PI. From plant numbers only.
    The form is derived, not copied: the paper itself was not read."""
    from target_gym.pc_gym.unstable_cstr.env import steady_coolant, steady_temperature

    def poles(L):
        T = steady_temperature(L, p, np)
        k = (p.q / p.V) * (p.Caf - L) / L
        kT = k * p.EA_over_R / T**2
        A, b, q = -p.deltaHr / (p.rho * p.C), p.UA / (p.rho * p.C * p.V), p.q / p.V
        J = np.array([[-q - k, -kT * L], [A * k, -q - b + A * kT * L]])
        return np.linalg.eigvals(J).real.max(), k, kT, b

    band = np.linspace(*p.target_CA_range, 21)
    Ld = band[int(np.argmax([poles(L)[0] for L in band]))]
    pole, k, kT, b = poles(Ld)
    well = 10.0 / 60.0
    t2, th = well + p.tau_j / 2, p.tau_j / 2 + p.actuator_lag + p.delta_t / 2
    lam = th
    tau, K = 1 / pole, b / pole
    beta = tau * ((lam / tau + 1) ** 2 * np.exp(th / tau) - 1)
    d = [2 * lam - beta + th, lam**2 - th**2 / 2 + beta * th]
    d.append(th**3 / 6 - beta * th**2 / 2)
    n = [-1 / K, (tau - beta) / K, tau * beta / K]
    f0 = n[0] / d[0]
    f1 = (n[1] - f0 * d[1]) / d[0]
    f2 = (n[2] - f0 * d[2] - f1 * d[1]) / d[0]
    Ki, Kp, Kd = f0, f1 + f0 * t2, f2 + f1 * t2
    tho = 2 * lam + th
    t1 = 1 / (p.q / p.V + k)
    ko = kT * Ld * t1
    Kco, Tio = t1 / (ko * 2 * tho), min(t1, 8 * tho)
    a = -np.expm1(-p.delta_t / beta)
    b_f = -np.expm1(-p.delta_t * 10 * Kp / Kd)
    span = float(p.T_c_max - p.T_c_min)
    low = float(p.T_c_min)
    dt = float(p.delta_t)

    def step(_, m, obs, xp=jnp):
        Ca, T, L = obs[0], obs[1], obs[3]
        e = L - Ca
        io, ii, sp, f, started = m
        first = started > 0
        sp = xp.where(first, sp, T)
        f_new = xp.where(first, f + b_f * (T - f), T)
        rate = (f_new - xp.where(first, f, T)) / dt
        raw = steady_temperature(L, p, xp) - Kco * (e + io / Tio)
        sp = sp + a * (raw - sp)
        ei = sp - T
        coolant = steady_coolant(L, 0.0, p, xp) + Kp * ei + ii - Kd * rate
        held = xp.clip(coolant, low, low + span)
        free = held == coolant
        io = xp.where(free, io + e * dt, io)
        ii = xp.where(free, ii + Ki * ei * dt, ii)
        action = xp.reshape(2 * (held - low) / span - 1, (1,))
        return action, (io, ii, sp, f_new, xp.ones_like(started))

    zero = (0.0, 0.0, 0.0, 0.0, 0.0)
    return Controller(None, tuple(np.float32(x) for x in zero), step)


def row() -> Row:
    band = {"target_CA_range": (0.50, 0.70)}
    p = REGISTRY["unstable_cstr"].make_test_params(**band)
    return Row(
        1200,
        None,
        score=band,
        published=lambda s: _lee_cascade(p),
        why={
            "pi": "open-loop unstable: no bump test without an inner loop",
            "feedforward": "as pi",
            "cascade": "the published rule is the cascade",
            "decoupler": "one MV",
            "override": "the published rule holds the trip on its nominal seeds, or fails",
        },
    )
