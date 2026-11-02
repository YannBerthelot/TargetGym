"""Controllers for the CSTR task: the expert's cascade and the oracle.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
NewtonMPC, SamplingMPC and the v2 objective helpers)
stays in ``target_gym.experts.mpc``, which is in every task's baseline
fingerprint.

The expert is the cascade reactor practice runs: concentration analysers are
too slow to close on the coolant directly, so an outer PI on C_a sets the
temperature an inner PI holds with the coolant. Both are SIMC-tuned from step
tests (``scripts/tune_process_experts.py``, PHYSICS.md section 6).
"""

import jax.numpy as jnp
from flax import struct

from target_gym.experts.mpc import NewtonMPC, protocol_burn_in, window_weights

# ---------------------------------------------------------------------------
# CSTR
# ---------------------------------------------------------------------------


def make_cstr_mpc(
    env,
    params,
    window_start: int | None = None,
    resolve_every: int = 1,
    pre_weight: float = 1e-3,
    horizon: int | None = None,
    n_iter: int = 20,
):
    """The CSTR's oracle: ``NewtonMPC`` over the rest of the episode.

    It re-solves every step on the env's own step and reward, the
    concentration's tracking cost in floor units. ``window_start`` is where
    the protocol starts scoring, read by default from its burn-in
    (``protocol_burn_in``: 12 steps of the registered 150), and the steps
    before it weigh ``pre_weight`` (``window_weights``). The plant is
    deterministic, so the closed loop reaches the plant's optimum to float32
    rounding: the returns of the IPOPT shrinking-horizon NLP it replaced, to
    the unit, at a quarter of its time.
    """
    if window_start is None:
        window_start = protocol_burn_in("cstr", params)
    steps = int(params.max_steps_in_episode)
    horizon = steps if horizon is None else int(horizon)
    return NewtonMPC(
        env,
        params,
        horizon=horizon,
        n_iter=n_iter,
        weights=window_weights(steps, window_start, pre_weight, horizon),
        resolve_every=resolve_every,
    )


@struct.dataclass
class CascadeParams:
    """Gains of the CSTR cascade, in the env's units (K, mol/L, min), and the
    design constants of the concentration balance.

    ``Kp`` and ``Ti`` are the outer PI on C_a, whose output is the temperature
    setpoint in K; ``Kp`` is negative, since a hotter reactor converts more and
    leaves less C_a. ``Kp_T`` and ``Ti_T`` are the inner PI on T, whose output
    is the raw coolant command. ``q_V`` (1/min), ``Caf`` (mol/L), ``k0``
    (1/min) and ``EA_over_R`` (K) give the outer PI its bias.
    """

    Kp: float
    Ti: float
    Kp_T: float
    Ti_T: float
    dt: float
    q_V: float
    Caf: float
    k0: float
    EA_over_R: float


@struct.dataclass
class CascadeState:
    integral_C: float
    integral_T: float


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
    neither winds up a term it cannot spend.
    """
    C_a, T, target = obs[0], obs[1], obs[2]
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
    )


def cstr_cascade(Kp: float, Ti: float, Kp_T: float, Ti_T: float, params=None):
    """The cascade with these gains on ``params``' reactor (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.cstr.env import CSTRParams

    p = params if params is not None else CSTRParams()
    return Controller(
        CascadeParams(
            Kp=Kp,
            Ti=Ti,
            Kp_T=Kp_T,
            Ti_T=Ti_T,
            dt=float(p.delta_t),
            q_V=float(p.q / p.V),
            Caf=float(p.Caf),
            k0=float(p.k0),
            EA_over_R=float(p.EA_over_R),
        ),
        CascadeState(integral_C=0.0, integral_T=0.0),
        cascade_step,
    )


def make_cstr_pid_controller():
    """The scored expert, the cascade, as a pure
    :class:`~target_gym.experts.pid.Controller`; ``.host()`` is what
    ``make_pid()`` returns. Gains from src/target_gym/data/pid_gains.json,
    written by ``scripts/tune_process_experts.py``."""
    from target_gym.experts.pid import gain

    return cstr_cascade(
        Kp=gain("cstr", "Kp", -132.2),
        Ti=gain("cstr", "Ti", 1.074),
        Kp_T=gain("cstr", "Kp_T", 0.2652),
        Ti_T=gain("cstr", "Ti_T", 0.6350),
    )
