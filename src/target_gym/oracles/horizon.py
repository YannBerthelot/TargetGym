"""The planning horizon by Cooper's DMC recipe, as pure functions.

docs/proposals/oracle-horizon-protocol.md, section 2 with H2 to H5 and H12,
under the owner decisions of 2026-11-03: the sample time is Cooper's T_c,
P = N + M, and every CV counts in the maximum. Notation as there: T is the
controller's sample time T_c, dt the task's step, Int(x) the floor.

``scripts/measure_oracle_horizons.py`` applies it to the bump-test fits of
``target_gym.identification`` and writes ``data/oracle_horizons/``.
"""

from __future__ import annotations

import math

__all__ = ["cooper_horizons", "dead_time_samples", "effective_dead_time"]
__all__ += ["episode_samples", "horizon_at", "sample_steps", "sample_time"]
__all__ += ["ramp_samples", "task_horizon", "task_ramp"]


def _int(x: float) -> int:
    """Cooper's Int(x), the floor, with float round-off below 1e-9 forgiven
    (0.15 / 0.05 is 2.9999999999999996 in binary)."""
    return math.floor(x + 1e-9)


def sample_time(fit: dict) -> float:
    """H2 for one fit: T_c = max(0.1 tau, 0.5 theta). An integrating fit's
    tau is its lag tau2."""
    tau = fit["tau"] if fit["model"] == "fopdt" else fit["tau2"]
    return max(0.1 * tau, 0.5 * fit["theta"])


def sample_steps(sample_times, dt: float) -> int:
    """H2 for a task: the smallest T_c over its pairs and seeds, rounded down
    to whole task steps, at least one."""
    return max(1, _int(min(sample_times) / dt))


def dead_time_samples(theta: float, T: float) -> int:
    """H3: k = Int(theta / T) + 1, theta the fitted delay."""
    return _int(theta / T) + 1


def effective_dead_time(theta: float, T: float, tau2: float) -> float:
    """H4: theta_e = theta + T / 2 + tau2 / 2."""
    return theta + T / 2 + tau2 / 2


def cooper_horizons(fit: dict, T: float) -> tuple[int, int, int]:
    """``(N, M, P)`` for one fit at sample time ``T``: N = Int(5 tau_x / T) + k,
    M = Int(tau_x / T) + k, P = N + M, with tau_x the fit's tau when it is
    self-regulating and tau_CL = sqrt(10) theta_e when it integrates."""
    k = dead_time_samples(fit["theta"], T)
    if fit["model"] == "fopdt":
        tau_x = fit["tau"]
    else:
        tau_x = math.sqrt(10.0) * effective_dead_time(fit["theta"], T, fit["tau2"])
    N, M = _int(5 * tau_x / T) + k, _int(tau_x / T) + k
    return N, M, N + M


def task_horizon(fits: dict, dt: float):
    """H2 and H5 over a task's valid fits, ``{key: fit}`` with a key per pair
    and seed: ``(T_c in steps, {key: (N, M, P)}, the key with the largest P)``.
    With no valid fit there is no protocol horizon, ``(None, {}, None)``:
    the task keeps its current oracle (Amendment 2, N3, replacing H20)."""
    if not fits:
        return None, {}, None
    tc = sample_steps([sample_time(f) for f in fits.values()], dt)
    rows = {key: cooper_horizons(f, tc * dt) for key, f in fits.items()}
    return tc, rows, max(rows, key=lambda key: rows[key][2])


def ramp_samples(span: float, gain: float, rate: float, T: float) -> int:
    """H21 for one fit: the samples of ``T`` a move limited to ``rate`` takes
    to carry its CV across ``span``, span / (|gain| rate), rounded up."""
    return math.ceil(span / (abs(gain) * rate) / T - 1e-9)


def task_ramp(fits: dict, cv: str, span: float, rates: dict, T: float):
    """H21 for a task, as H5 takes the largest: over its fits ``{(mv, cv,
    seed): fit}`` of tracked ``cv`` and the rate-limited MVs in ``rates``, the
    longest ramp and the key that sets it, ``(0, None)`` with none. A bump
    test is too small to reach a rate limit, so a large move's ramp is added
    to the horizon Cooper's recipe gives."""
    rows = {
        key: ramp_samples(span, fit["k"], rates[key[0]], T)
        for key, fit in fits.items()
        if key[1] == cv and key[0] in rates
    }
    if not rows:
        return 0, None
    top = max(rows, key=rows.get)
    return rows[top], top


def episode_samples(n_steps: int, tc_steps: int) -> int:
    """N_ep in T_c samples: the solves an episode of ``n_steps`` holds, the
    last one cut short if ``tc_steps`` does not divide it."""
    return -(-n_steps // tc_steps)


def horizon_at(P: int, n_ep: int, k: int) -> int:
    """H12: H_k = min(P, N_ep - k) at sample ``k`` of an ``n_ep``-sample
    episode, so the plan never runs past the episode's end."""
    if not 0 <= k < n_ep:
        raise ValueError(f"sample {k} is outside an episode of {n_ep} samples")
    return min(P, n_ep - k)
