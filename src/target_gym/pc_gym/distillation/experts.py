"""The expert (the PID slot) for the distillation task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

The expert is the base layer practice runs under its MPC: the
LV-configuration with a tight temperature loop, boilup holding the stage-15
temperature, and two composition loops on top, reflux on the distillate and the
temperature setpoint on the bottoms. The composition loops sit on chromatographs,
so they count updates and act only on a move past the noise band, then wait for
the update that can show it. The temperature setpoint ramps, so the tight loop
never sees a step. Tuned from bump tests (``scripts/tune_process_experts.py``,
PHYSICS.md section 8).
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class BaseLayerParams:
    """The base layer's three PI loops, in the env's units (kmol/min, K,
    mole fractions, min).

    ``Kp_T``, ``Ti_T``: boilup on the stage-15 temperature, read through a
    filter that moves ``alpha_T`` per step. ``Kp_D``, ``Ti_D``: reflux on yD.
    ``Kp_B``, ``Ti_B``: the temperature setpoint on xB, negative, since a hotter
    stage 15 strips the bottoms cleaner. ``band_D``, ``band_B``: how far an
    analyser's update must move to count, ``every`` the steps between updates,
    ``wait`` the updates a loop sits out after acting. ``sp_rate`` is the
    temperature setpoint's ramp, K per min.
    """

    Kp_T: float
    Ti_T: float
    alpha_T: float
    Kp_D: float
    Ti_D: float
    Kp_B: float
    Ti_B: float
    band_D: float
    band_B: float
    every: int
    wait: int
    sp_rate: float
    dt: float
    L_min: float
    L_max: float
    V_min: float
    V_max: float


@struct.dataclass
class AnalyserLoop:
    """One composition loop's memory: ``held`` the update it last acted on,
    ``left`` the updates it still sits out, ``output`` its last move."""

    held: float
    left: float
    integral: float
    output: float


@struct.dataclass
class BaseLayerState:
    """The loops' memory. The biases are the flows and the stage-15
    temperature read on the first step, so the switch to automatic is
    bumpless. ``clock`` counts steps, so the loops know their updates."""

    started: float
    clock: float
    L_bias: float
    V_bias: float
    T_bias: float
    T: float
    T_sp: float
    integral_T: float
    top: AnalyserLoop
    bottom: AnalyserLoop


def analyser_update(prm, loop, y, target, Kp, Ti, band, update, stop, xp):
    """N8 on one composition loop: on an update it is free to read, it acts if
    the reading has moved past ``band`` from the one it held, or sits past it
    from the target. It then waits ``prm.wait`` updates, integrating over that
    time. ``stop`` holds the integral while its flow is saturated."""
    left = xp.where(update, xp.maximum(loop.left - 1, 0), loop.left)
    moved = (xp.abs(y - loop.held) > band) | (xp.abs(target - loop.held) > band)
    act = update & (left <= 0) & moved
    held = xp.where(act, y, loop.held)
    e = target - held
    integral = loop.integral + e * prm.wait * prm.every * prm.dt
    integral = xp.where(act & ~stop, integral, loop.integral)
    output = xp.where(act, Kp * (e + integral / Ti), loop.output)
    return AnalyserLoop(
        held=held,
        left=xp.where(act, prm.wait, left),
        integral=integral,
        output=output,
    )


def base_layer_step(prm: BaseLayerParams, s: BaseLayerState, obs, xp=jnp):
    """One update on ``obs = [yD, xB, L_pct, V_pct, target_yD, target_xB,
    T_15]``. Returns ``(action [shape (2,)], new_state)``. A composition loop's
    integral holds while its flow is saturated, and the bottoms loop's while
    boilup is, since it acts through the temperature loop."""
    first = s.started > 0
    L_now = prm.L_min + obs[2] * (prm.L_max - prm.L_min)
    V_now = prm.V_min + obs[3] * (prm.V_max - prm.V_min)
    L_bias = xp.where(first, s.L_bias, L_now)
    V_bias = xp.where(first, s.V_bias, V_now)
    T_bias = xp.where(first, s.T_bias, obs[6])
    T = xp.where(first, s.T + prm.alpha_T * (obs[6] - s.T), obs[6])
    T_sp_last = xp.where(first, s.T_sp, obs[6])
    # The first call holds what it reads, so it acts only on the target.
    top = s.top.replace(held=xp.where(first, s.top.held, obs[0]))
    bottom = s.bottom.replace(held=xp.where(first, s.bottom.held, obs[1]))

    update = xp.mod(s.clock, prm.every) == 0
    L_stop = (L_now <= prm.L_min) | (L_now >= prm.L_max)
    V_stop = (V_now <= prm.V_min) | (V_now >= prm.V_max)
    top = analyser_update(
        prm, top, obs[0], obs[4], prm.Kp_D, prm.Ti_D, prm.band_D, update, L_stop, xp
    )
    bottom = analyser_update(
        prm, bottom, obs[1], obs[5], prm.Kp_B, prm.Ti_B, prm.band_B, update, V_stop, xp
    )

    ramp = prm.sp_rate * prm.dt
    T_sp = xp.clip(T_bias + bottom.output, T_sp_last - ramp, T_sp_last + ramp)
    e_T = T_sp - T
    integral_T = s.integral_T + e_T * prm.dt
    L = L_bias + top.output
    V = V_bias + prm.Kp_T * (e_T + integral_T / prm.Ti_T)

    L_c = xp.clip(L, prm.L_min, prm.L_max)
    V_c = xp.clip(V, prm.V_min, prm.V_max)
    action = xp.stack(
        [
            2.0 * (L_c - prm.L_min) / (prm.L_max - prm.L_min) - 1.0,
            2.0 * (V_c - prm.V_min) / (prm.V_max - prm.V_min) - 1.0,
        ]
    )
    return action, BaseLayerState(
        started=xp.ones_like(T),
        clock=s.clock + 1,
        L_bias=L_bias,
        V_bias=V_bias,
        T_bias=T_bias,
        T=T,
        T_sp=T_sp,
        integral_T=xp.where(V == V_c, integral_T, s.integral_T),
        top=top,
        bottom=bottom,
    )


def distillation_base_layer(
    Kp_T,
    Ti_T,
    Kp_D,
    Ti_D,
    Kp_B,
    Ti_B,
    Tf_T=0.0,
    band_D=0.0,
    band_B=0.0,
    every=1,
    wait=1,
    sp_rate=1e6,
    params=None,
):
    """The base layer with these gains on ``params``' column (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.distillation.env import DistillationParams

    p = params if params is not None else DistillationParams()
    dt = float(p.delta_t)
    fresh = AnalyserLoop(held=0.0, left=0.0, integral=0.0, output=0.0)
    return Controller(
        BaseLayerParams(
            Kp_T=float(Kp_T),
            Ti_T=float(Ti_T),
            alpha_T=-math.expm1(-dt / Tf_T) if Tf_T else 1.0,
            Kp_D=float(Kp_D),
            Ti_D=float(Ti_D),
            Kp_B=float(Kp_B),
            Ti_B=float(Ti_B),
            band_D=float(band_D),
            band_B=float(band_B),
            every=int(every),
            wait=int(wait),
            sp_rate=float(sp_rate),
            dt=dt,
            **{f: float(getattr(p, f)) for f in ("L_min", "L_max", "V_min", "V_max")},
        ),
        BaseLayerState(
            started=0.0,
            clock=0.0,
            L_bias=0.0,
            V_bias=0.0,
            T_bias=0.0,
            T=0.0,
            T_sp=0.0,
            integral_T=0.0,
            top=fresh,
            bottom=fresh,
        ),
        base_layer_step,
    )


def make_distillation_pid_controller():
    """The scored expert as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains from
    src/target_gym/data/pid_gains.json, written by
    ``scripts/tune_process_experts.py``."""
    from target_gym.experts.pid import gain

    keys = "Kp_T Ti_T Kp_D Ti_D Kp_B Ti_B Tf_T band_D band_B every wait sp_rate"
    return distillation_base_layer(
        **{key: gain("distillation", key) for key in keys.split()}
    )
