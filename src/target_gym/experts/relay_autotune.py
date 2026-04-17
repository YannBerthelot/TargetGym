"""
Model-free relay autotuning (Åström–Hägglund method).

Treats the environment as a black-box plant: injects a relay signal,
measures the sustained oscillation, and extracts the ultimate gain (Ku)
and period (Tu). PID gains are then computed from standard tuning rules
(AMIGO, Tyreus–Luyben, or Ziegler–Nichols).

This mirrors what industrial autotuners (Honeywell, ABB, Siemens) do on
real plants. Budget: 1 relay experiment ≈ 5 oscillation cycles per
operating point. With N=8 operating points, the total cost is ~40 episodes
worth of interaction — realistic for commissioning a real control loop.

Usage
-----
    from target_gym.experts.relay_autotune import relay_sweep, TUNING_RULES

    result = relay_sweep(
        env, params,
        reset_fn=lambda key, p, t: env.reset_env(key, p)[1].replace(target_CA=t),
        state_index=0, setpoint_index=2,
        target_range=(0.84, 0.91),
        n_points=8,
        sign=-1,
        tuning_rule="amigo",
    )
    # result["operating_points"], result["Kp"], result["Ki"], result["Kd"]
"""

from __future__ import annotations

import warnings
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

# ─── Tuning rules ────────────────────────────────────────────────────────────


def ziegler_nichols(Ku: float, Tu: float) -> tuple[float, float, float]:
    """Classic Ziegler–Nichols PID tuning from ultimate gain and period."""
    Kp = 0.6 * Ku
    Ti = Tu / 2.0
    Td = Tu / 8.0
    Ki = Kp / Ti
    Kd = Kp * Td
    return Kp, Ki, Kd


def tyreus_luyben(Ku: float, Tu: float) -> tuple[float, float, float]:
    """Tyreus–Luyben: less aggressive than Z-N, better robustness margins."""
    Kp = Ku / 3.2
    Ti = 2.2 * Tu
    Td = Tu / 6.3
    Ki = Kp / Ti
    Kd = Kp * Td
    return Kp, Ki, Kd


def amigo(Ku: float, Tu: float) -> tuple[float, float, float]:
    """AMIGO (Approximate M-constrained Integral Gain Optimization).

    Good trade-off between performance and robustness. Recommended default.
    """
    Kp = 0.45 * Ku
    Ti = 0.85 * Tu
    Td = 0.27 * Tu
    Ki = Kp / Ti
    Kd = Kp * Td
    return Kp, Ki, Kd


TUNING_RULES = {
    "ziegler_nichols": ziegler_nichols,
    "tyreus_luyben": tyreus_luyben,
    "amigo": amigo,
}


# ─── Relay experiment ─────────────────────────────────────────────────────────


def relay_experiment(
    env,
    params,
    reset_fn: Callable,
    state_index: int,
    setpoint_index: int,
    operating_point: float,
    relay_amplitude: float = 0.5,
    hysteresis: float = 0.0,
    n_settle_cycles: int = 2,
    n_measure_cycles: int = 3,
    max_steps: int = 10_000,
    seed: int = 0,
    action_dim: int = 1,
    action_index: int = 0,
    action_bias: float | None = None,
    plant_sign: int = 1,
    # ── Cascaded-loop extensions ──────────────────────────────
    fixed_setpoint: float | None = None,
    build_action: Callable | None = None,
    measure_fn: Callable | None = None,
    error_fn: Callable | None = None,
) -> dict:
    """Run a relay feedback experiment and extract Ku, Tu.

    Parameters
    ----------
    env : gymnax-style environment
    params : environment parameter dataclass
    reset_fn : callable (key, params, target) -> initial state
        Resets the environment at the given operating point.
    state_index : int
        Index of the process variable in the observation vector.
    setpoint_index : int
        Index of the setpoint in the observation vector.
    operating_point : float
        Target setpoint value to tune at.
    relay_amplitude : float
        Half-amplitude of the relay signal (in action units, [-1, 1]).
    hysteresis : float
        Dead-band half-width for the relay switch (prevents chattering).
    n_settle_cycles : int
        Number of oscillation cycles to discard (transient settling).
    n_measure_cycles : int
        Number of oscillation cycles to average over for Ku/Tu extraction.
    max_steps : int
        Maximum simulation steps (safety limit).
    seed : int
        Random seed for environment reset.
    action_dim : int
        Total number of action dimensions (1 for SISO).
    action_index : int
        Which action dimension this relay controls (for MIMO, others = 0).
    action_bias : float or None
        Steady-state action at the operating point. If None, estimated via
        a short settling experiment.  Defaults to 0.0 when *build_action*
        is provided.
    plant_sign : int
        +1 for positive-gain plants (more action → higher PV), -1 for
        negative-gain plants (more action → lower PV, e.g. CSTR). This
        flips the relay direction so the feedback loop is always negative.
    fixed_setpoint : float or None
        If provided, use this constant setpoint instead of reading
        ``obs[setpoint_index]``.
    build_action : callable or None
        ``(relay_output, obs) → action``.  When provided, the relay output
        is passed through this callback to construct the full action vector
        (e.g. for cascaded loops where the relay drives an intermediate
        signal, not a direct action).  The relay output is *not* clipped
        to [-1, 1] — the callback is responsible for any clipping.
    measure_fn : callable or None
        ``(obs) → float``.  Custom measurement extraction.  Defaults to
        ``obs[state_index]``.
    error_fn : callable or None
        ``(measured, setpoint) → float``.  Custom error computation (e.g.
        angle wrapping).  Defaults to ``setpoint - measured``.

    Returns
    -------
    dict with keys: Ku, Tu, amplitude, raw_periods, success.
    """
    key = jax.random.PRNGKey(seed)

    # Override episode length — relay experiments need more steps than a
    # typical RL episode (we're measuring oscillation, not training).
    params = params.replace(
        max_steps_in_episode=max(max_steps * 2, int(params.max_steps_in_episode))
    )

    # Reset at operating point
    state = reset_fn(key, params, operating_point)
    obs = (
        env.get_obs(state, params)
        if _accepts_params(env.get_obs)
        else env.get_obs(state)
    )

    # ── Find steady-state action bias if not provided ──
    if action_bias is None:
        if build_action is not None:
            # Cannot auto-detect bias with a custom action builder; default 0.
            action_bias = 0.0
        else:
            action_bias = _find_action_bias(
                env,
                params,
                reset_fn,
                state_index,
                setpoint_index,
                operating_point,
                action_dim=action_dim,
                action_index=action_index,
                seed=seed,
            )

    # ── Run relay ──
    # Track error zero-crossings (+ → - or - → +) to segment cycles.
    error_history = []
    crossing_steps = []  # steps where error crosses zero
    prev_sign = 0.0

    relay_state = 1.0  # start with positive relay

    for step in range(max_steps):
        measured = (
            float(measure_fn(obs))
            if measure_fn is not None
            else float(obs[state_index])
        )
        sp = (
            fixed_setpoint if fixed_setpoint is not None else float(obs[setpoint_index])
        )
        error = float(error_fn(measured, sp)) if error_fn is not None else sp - measured
        error_history.append(error)

        # Relay with hysteresis
        if error > hysteresis:
            relay_state = 1.0
        elif error < -hysteresis:
            relay_state = -1.0
        # else: keep previous relay_state (inside dead band)

        # Build action (plant_sign flips direction for negative-gain plants)
        u = action_bias + relay_amplitude * relay_state * plant_sign
        if build_action is not None:
            action = build_action(u, obs)
        else:
            u = float(np.clip(u, -1.0, 1.0))
            if action_dim == 1:
                action = u
            else:
                action_arr = np.zeros(action_dim)
                action_arr[action_index] = u
                action = jnp.array(action_arr)

        # Step environment
        obs, state, _, done, _ = env.step_env(key, state, action, params)
        if hasattr(obs, "shape") and obs.shape == ():
            obs = (
                env.get_obs(state, params)
                if _accepts_params(env.get_obs)
                else env.get_obs(state)
            )

        if bool(done):
            break

        # Detect zero-crossings of error
        sign = 1.0 if error > 0 else (-1.0 if error < 0 else 0.0)
        if prev_sign != 0 and sign != 0 and sign != prev_sign:
            crossing_steps.append(step)
        if sign != 0:
            prev_sign = sign

        # Have enough cycles?
        total_cycles_needed = n_settle_cycles + n_measure_cycles
        # Each full cycle = 2 zero-crossings
        if len(crossing_steps) >= 2 * total_cycles_needed + 1:
            break

    # ── Extract Ku and Tu ──
    if len(crossing_steps) < 2 * (n_settle_cycles + n_measure_cycles):
        # Not enough oscillation — system may not respond to relay
        warnings.warn(
            f"Relay experiment at operating_point={operating_point:.4f}: "
            f"only {len(crossing_steps)} zero-crossings detected "
            f"(need {2 * (n_settle_cycles + n_measure_cycles)}). "
            f"Ku/Tu may be unreliable."
        )
        if len(crossing_steps) < 4:
            return {
                "Ku": None,
                "Tu": None,
                "amplitude": None,
                "raw_periods": [],
                "success": False,
            }

    # Discard settling cycles (each cycle = 2 crossings)
    measure_crossings = crossing_steps[2 * n_settle_cycles :]
    errors_arr = np.array(error_history)

    # Half-periods: time between consecutive zero-crossings
    half_periods = np.diff(measure_crossings)
    # Full periods: sum consecutive pairs of half-periods
    full_periods = []
    for i in range(0, len(half_periods) - 1, 2):
        full_periods.append(half_periods[i] + half_periods[i + 1])
    if not full_periods:
        # Fall back to using half-periods
        full_periods = [2.0 * hp for hp in half_periods]

    Tu_steps = float(np.mean(full_periods))
    Tu = Tu_steps * float(params.delta_t)  # convert to seconds

    # Oscillation amplitude: peak-to-peak / 2 in the measurement region
    measure_start = measure_crossings[0]
    measure_end = (
        measure_crossings[-1] if len(measure_crossings) > 1 else len(errors_arr)
    )
    measured_errors = errors_arr[measure_start:measure_end]
    amplitude = (float(np.max(measured_errors)) - float(np.min(measured_errors))) / 2.0

    if amplitude < 1e-12:
        warnings.warn(
            f"Relay experiment at {operating_point:.4f}: zero oscillation amplitude."
        )
        return {
            "Ku": None,
            "Tu": None,
            "amplitude": 0.0,
            "raw_periods": full_periods,
            "success": False,
        }

    # Describing function: Ku = 4d / (πA)
    Ku = 4.0 * relay_amplitude / (np.pi * amplitude)

    return {
        "Ku": Ku,
        "Tu": Tu,
        "amplitude": amplitude,
        "raw_periods": [float(p) for p in full_periods],
        "success": True,
    }


# ─── Steady-state action finder ──────────────────────────────────────────────


def _find_action_bias(
    env,
    params,
    reset_fn,
    state_index,
    setpoint_index,
    operating_point,
    action_dim=1,
    action_index=0,
    seed=0,
    n_bisect=15,
    settle_steps=500,
) -> float:
    """Binary search for the steady-state action that holds the plant at *operating_point*.

    Runs ``settle_steps`` simulation steps with a constant action, reads
    the final process variable, and bisects to find the action that makes
    the PV converge to the setpoint.
    """
    key = jax.random.PRNGKey(seed)
    # Extend episode so settling doesn't hit truncation
    long_params = params.replace(
        max_steps_in_episode=max(settle_steps * 2, int(params.max_steps_in_episode))
    )

    def settle(action_val: float) -> float:
        state = reset_fn(key, long_params, operating_point)
        for _ in range(settle_steps):
            if action_dim == 1:
                action = action_val
            else:
                a = np.zeros(action_dim)
                a[action_index] = action_val
                action = jnp.array(a)
            _, state, _, done, _ = env.step_env(key, state, action, long_params)
            if bool(done):
                break
        obs = (
            env.get_obs(state, long_params)
            if _accepts_params(env.get_obs)
            else env.get_obs(state)
        )
        return float(obs[state_index])

    lo, hi = -1.0, 1.0
    # Determine direction: does increasing action increase or decrease PV?
    pv_lo = settle(lo)
    pv_hi = settle(hi)
    target = operating_point

    if abs(pv_hi - pv_lo) < 1e-10:
        # System doesn't respond — return midpoint
        return 0.0

    # Ensure lo < hi in terms of PV
    if pv_lo > pv_hi:
        lo, hi = hi, lo
        pv_lo, pv_hi = pv_hi, pv_lo

    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        pv_mid = settle(mid)
        if pv_mid < target:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


# ─── Multi-operating-point sweep ─────────────────────────────────────────────


def relay_sweep(
    env,
    params,
    reset_fn: Callable,
    state_index: int,
    setpoint_index: int,
    target_range: tuple[float, float],
    n_points: int = 8,
    sign: int = 1,
    tuning_rule: str = "amigo",
    relay_amplitude: float = 0.5,
    hysteresis: float = 0.0,
    n_settle_cycles: int = 2,
    n_measure_cycles: int = 3,
    max_steps: int = 10_000,
    action_dim: int = 1,
    action_index: int = 0,
    verbose: bool = True,
) -> dict:
    """Run relay experiments at N operating points and return a gain schedule.

    Parameters
    ----------
    sign : int
        +1 if increasing action increases the PV (positive gain plant),
        -1 if increasing action decreases the PV (negative gain, e.g. CSTR).
        Applied to the final (Kp, Ki, Kd).
    tuning_rule : str
        One of "amigo", "tyreus_luyben", "ziegler_nichols".

    Returns
    -------
    dict with keys:
        operating_points: list[float]
        Kp, Ki, Kd: list[float] — gains at each operating point
        relay_data: list[dict] — raw relay experiment results
        tuning_rule: str
    """
    rule_fn = TUNING_RULES[tuning_rule]
    targets = np.linspace(target_range[0], target_range[1], n_points)
    results = []
    Kp_list, Ki_list, Kd_list = [], [], []

    for i, target in enumerate(targets):
        if verbose:
            print(f"  Operating point {i+1}/{n_points}: target={target:.4f}")

        res = relay_experiment(
            env,
            params,
            reset_fn,
            state_index=state_index,
            setpoint_index=setpoint_index,
            operating_point=float(target),
            relay_amplitude=relay_amplitude,
            hysteresis=hysteresis,
            n_settle_cycles=n_settle_cycles,
            n_measure_cycles=n_measure_cycles,
            max_steps=max_steps,
            seed=i,
            action_dim=action_dim,
            action_index=action_index,
            plant_sign=sign,
        )
        results.append(res)

        if res["success"] and res["Ku"] is not None:
            Kp, Ki, Kd = rule_fn(res["Ku"], res["Tu"])
            Kp *= sign
            Ki *= sign
            Kd *= sign
            if verbose:
                print(
                    f"    Ku={res['Ku']:.4f}, Tu={res['Tu']:.4f}s → "
                    f"Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}"
                )
        else:
            # Relay failed at this point — interpolate later
            Kp, Ki, Kd = None, None, None
            if verbose:
                print(f"    Relay failed — will interpolate from neighbours")

        Kp_list.append(Kp)
        Ki_list.append(Ki)
        Kd_list.append(Kd)

    # Interpolate any failed points from successful neighbours
    _interpolate_gaps(Kp_list, targets)
    _interpolate_gaps(Ki_list, targets)
    _interpolate_gaps(Kd_list, targets)

    return {
        "operating_points": [float(t) for t in targets],
        "Kp": [float(k) for k in Kp_list],
        "Ki": [float(k) for k in Ki_list],
        "Kd": [float(k) for k in Kd_list],
        "relay_data": results,
        "tuning_rule": tuning_rule,
    }


def _interpolate_gaps(values: list, xs: np.ndarray):
    """Fill None entries by linear interpolation from non-None neighbours.

    Modifies *values* in-place. If all entries are None, fills with 0.
    """
    good = [(i, v) for i, v in enumerate(values) if v is not None]
    if not good:
        for i in range(len(values)):
            values[i] = 0.0
        return
    if len(good) == len(values):
        return
    good_idx = np.array([g[0] for g in good])
    good_val = np.array([g[1] for g in good])
    for i, v in enumerate(values):
        if v is None:
            values[i] = float(np.interp(i, good_idx, good_val))


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _accepts_params(fn) -> bool:
    """Check if a get_obs function accepts a params argument."""
    import inspect

    sig = inspect.signature(fn)
    return "params" in sig.parameters
