"""The expert (the PID slot) for the distillation task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

The expert is the base layer practice runs under its MPC: the
LV-configuration with a temperature loop, boilup holding the stage-15
temperature, and the two composition loops on top, reflux on the distillate
and the temperature setpoint on the bottoms. SIMC-tuned from step tests
(``scripts/tune_process_experts.py``, PHYSICS.md section 8).
"""

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class BaseLayerParams:
    """The base layer's three PI loops, in the env's units (kmol/min, K,
    mole fractions, min).

    ``Kp_T``, ``Ti_T``: boilup on the stage-15 temperature. ``Kp_D``,
    ``Ti_D``: reflux on yD. ``Kp_B``, ``Ti_B``: the temperature setpoint on
    xB, negative, since a hotter stage 15 strips the bottoms cleaner.
    """

    Kp_T: float
    Ti_T: float
    Kp_D: float
    Ti_D: float
    Kp_B: float
    Ti_B: float
    dt: float
    L_min: float
    L_max: float
    V_min: float
    V_max: float


@struct.dataclass
class BaseLayerState:
    """The loops' memory. The biases are the flows and the stage-15
    temperature read on the first step, so the switch to automatic is
    bumpless."""

    started: float
    L_bias: float
    V_bias: float
    T_bias: float
    integral_T: float
    integral_D: float
    integral_B: float


def base_layer_step(prm: BaseLayerParams, s: BaseLayerState, obs, xp=jnp):
    """One update on ``obs = [yD, xB, L_pct, V_pct, target_yD, target_xB,
    T_15]``. Returns ``(action [shape (2,)], new_state)``. A loop's integral
    holds while its flow is saturated, and the bottoms loop's while boilup
    is, since it acts through the temperature loop."""
    yD, xB, T = obs[0], obs[1], obs[6]
    L_now = prm.L_min + obs[2] * (prm.L_max - prm.L_min)
    V_now = prm.V_min + obs[3] * (prm.V_max - prm.V_min)
    first = s.started > 0
    L_bias = xp.where(first, s.L_bias, L_now)
    V_bias = xp.where(first, s.V_bias, V_now)
    T_bias = xp.where(first, s.T_bias, T)

    e_D, e_B = obs[4] - yD, obs[5] - xB
    integral_D = s.integral_D + e_D * prm.dt
    integral_B = s.integral_B + e_B * prm.dt
    L = L_bias + prm.Kp_D * (e_D + integral_D / prm.Ti_D)
    T_sp = T_bias + prm.Kp_B * (e_B + integral_B / prm.Ti_B)
    e_T = T_sp - T
    integral_T = s.integral_T + e_T * prm.dt
    V = V_bias + prm.Kp_T * (e_T + integral_T / prm.Ti_T)

    L_c = xp.clip(L, prm.L_min, prm.L_max)
    V_c = xp.clip(V, prm.V_min, prm.V_max)
    L_free, V_free = L == L_c, V == V_c
    action = xp.stack(
        [
            2.0 * (L_c - prm.L_min) / (prm.L_max - prm.L_min) - 1.0,
            2.0 * (V_c - prm.V_min) / (prm.V_max - prm.V_min) - 1.0,
        ]
    )
    return action, BaseLayerState(
        started=xp.ones_like(T),
        L_bias=L_bias,
        V_bias=V_bias,
        T_bias=T_bias,
        integral_T=xp.where(V_free, integral_T, s.integral_T),
        integral_D=xp.where(L_free, integral_D, s.integral_D),
        integral_B=xp.where(V_free, integral_B, s.integral_B),
    )


def distillation_base_layer(Kp_T, Ti_T, Kp_D, Ti_D, Kp_B, Ti_B, params=None):
    """The base layer with these gains on ``params``' column (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.distillation.env import DistillationParams

    p = params if params is not None else DistillationParams()
    return Controller(
        BaseLayerParams(
            Kp_T=float(Kp_T),
            Ti_T=float(Ti_T),
            Kp_D=float(Kp_D),
            Ti_D=float(Ti_D),
            Kp_B=float(Kp_B),
            Ti_B=float(Ti_B),
            dt=float(p.delta_t),
            **{f: float(getattr(p, f)) for f in ("L_min", "L_max", "V_min", "V_max")},
        ),
        BaseLayerState(
            started=0.0,
            L_bias=0.0,
            V_bias=0.0,
            T_bias=0.0,
            integral_T=0.0,
            integral_D=0.0,
            integral_B=0.0,
        ),
        base_layer_step,
    )


def make_distillation_pid_controller():
    """The scored expert as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains from
    src/target_gym/data/pid_gains.json, written by
    ``scripts/tune_process_experts.py``."""
    from target_gym.experts.pid import gain

    return distillation_base_layer(
        Kp_T=gain("distillation", "Kp_T", 0.133),
        Ti_T=gain("distillation", "Ti_T", 24.2),
        Kp_D=gain("distillation", "Kp_D", 19.9),
        Ti_D=gain("distillation", "Ti_D", 1.5),
        Kp_B=gain("distillation", "Kp_B", -200.0),
        Ti_B=gain("distillation", "Ti_B", 2.8),
    )
