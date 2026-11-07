"""The expert (the PID slot) for the CSTR task: a cascade.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

The expert is the cascade reactor practice runs: concentration analysers are
too slow to close on the coolant directly, so an outer PI on C_a sets the
temperature an inner PI holds with the coolant. Both are SIMC-tuned from step
tests (``scripts/tune_process_experts.py``, PHYSICS.md section 6).
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CascadeParams:
    """Gains of the CSTR cascade, in the env's units (K, mol/L, min), and the
    design constants of the concentration balance.

    ``Kp`` and ``Ti`` are the outer PI on C_a, whose output is the temperature
    setpoint in K; ``Kp`` is negative, since a hotter reactor converts more and
    leaves less C_a. ``Kp_T`` and ``Ti_T`` are the inner PI on T, whose output
    is the raw coolant command. ``alpha_C`` and ``alpha_T`` are how far each
    reading's filter moves per step. ``q_V`` (1/min), ``Caf`` (mol/L), ``k0``
    (1/min) and ``EA_over_R`` (K) give the outer PI its bias.
    """

    Kp: float
    Ti: float
    Kp_T: float
    Ti_T: float
    alpha_C: float
    alpha_T: float
    dt: float
    q_V: float
    Caf: float
    k0: float
    EA_over_R: float


@struct.dataclass
class CascadeState:
    integral_C: float
    integral_T: float
    C_a: float
    T: float
    started: float


def steady_temperature(C_a, params: CascadeParams, xp=jnp):
    """The reactor temperature at which ``C_a`` is steady, from the
    concentration balance alone: (q/V)(Caf - C_a) = k0 exp(-E/RT) C_a."""
    rate = params.q_V * (params.Caf - C_a) / (params.k0 * C_a)
    return -params.EA_over_R / xp.log(rate)


def cascade_step(params: CascadeParams, state: CascadeState, obs, xp=jnp):
    """One update of the cascade on ``obs = [C_a, T, target_CA]``.

    The outer PI's output is biased by the target's steady temperature, the
    design value a commissioning engineer starts the reactor at, so its
    integral only trims the model's error. Returns ``(action [shape (1,)],
    new_state)``. Both integrals hold while the coolant is saturated, so
    neither winds up a term it cannot spend. Each reading passes its filter,
    which starts at the first reading.
    """
    target, first = obs[2], state.started > 0
    C_a = xp.where(first, state.C_a + params.alpha_C * (obs[0] - state.C_a), obs[0])
    T = xp.where(first, state.T + params.alpha_T * (obs[1] - state.T), obs[1])
    e_C = target - C_a
    integral_C = state.integral_C + e_C * params.dt
    T_sp = steady_temperature(target, params, xp) + params.Kp * (
        e_C + integral_C / params.Ti
    )
    e_T = T_sp - T
    integral_T = state.integral_T + e_T * params.dt
    u = params.Kp_T * (e_T + integral_T / params.Ti_T)
    u_clipped = xp.clip(u, -1.0, 1.0)
    free = u == u_clipped
    return xp.array([u_clipped]), CascadeState(
        integral_C=xp.where(free, integral_C, state.integral_C),
        integral_T=xp.where(free, integral_T, state.integral_T),
        C_a=C_a,
        T=T,
        started=xp.ones_like(T),
    )


def cstr_cascade(
    Kp: float,
    Ti: float,
    Kp_T: float,
    Ti_T: float,
    params=None,
    Tf_C: float = 0.0,
    Tf_T: float = 0.0,
):
    """The cascade with these gains on ``params``' reactor (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`. ``Tf_C`` and ``Tf_T`` are the
    filters on the readings, in minutes."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.cstr.env import CSTRParams

    p = params if params is not None else CSTRParams()
    dt = float(p.delta_t)
    return Controller(
        CascadeParams(
            Kp=Kp,
            Ti=Ti,
            Kp_T=Kp_T,
            Ti_T=Ti_T,
            alpha_C=-math.expm1(-dt / Tf_C) if Tf_C else 1.0,
            alpha_T=-math.expm1(-dt / Tf_T) if Tf_T else 1.0,
            dt=dt,
            q_V=float(p.q / p.V),
            Caf=float(p.Caf),
            k0=float(p.k0),
            EA_over_R=float(p.EA_over_R),
        ),
        CascadeState(integral_C=0.0, integral_T=0.0, C_a=0.0, T=0.0, started=0.0),
        cascade_step,
    )


def make_cstr_pid_controller():
    """The scored expert, the cascade, as a pure
    :class:`~target_gym.experts.pid.Controller`; ``.host()`` is what
    ``make_pid()`` returns. Gains from src/target_gym/data/pid_gains.json,
    written by ``scripts/tune_process_experts.py``."""
    from target_gym.experts.pid import gain

    return cstr_cascade(
        Kp=gain("cstr", "Kp"),
        Ti=gain("cstr", "Ti"),
        Kp_T=gain("cstr", "Kp_T"),
        Ti_T=gain("cstr", "Ti_T"),
        Tf_C=gain("cstr", "Tf_C"),
        Tf_T=gain("cstr", "Tf_T"),
    )
