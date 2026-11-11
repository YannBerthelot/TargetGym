"""The expert (the PID slot) for the HVAC task: a heating curve.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

The curve is how slab buildings are run (REALITY.md sec. 2): supply water
temperature from the outdoor temperature's 24 h mean [9, 13], a straight line
from the heating limit, where the supply equals the room setpoint, to the
design point, where it is the design supply [12, p. 7], computed from the
building at the design outdoor temperature. No room feedback: pure weather
compensation is the first of the modes a controller offers [16, p. 80].
"""

import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import Controller


@struct.dataclass
class CurveParams:
    """The heating curve's two points and the valve's range."""

    T_set: float  # C, room setpoint, the band's heating edge [9]
    T_out_design: float  # C, design outdoor temperature
    T_supply_design: float  # C, the supply that holds T_set there
    T_supply_min: float
    T_supply_max: float
    #: A parallel shift of the curve, K (0 for the expert).
    offset: float = 0.0


@struct.dataclass
class CurveState:
    """The curve has no memory; the 24 h mean is the plant's reading."""

    started: float = 0.0


def curve_supply(params, T_out_24h, xp=jnp):
    """Supply water on the curve, C, before the valve's range."""
    p = params
    slope = (p.T_supply_design - p.T_set) / (p.T_set - p.T_out_design)
    return p.T_set + slope * (p.T_set - T_out_24h) + p.offset


def curve_step(params, state, obs, xp=jnp):
    """obs: ``[T_air, T_out, T_out_24h, T_core, T_return, T_supply, sin_h,
    cos_h, band_centre]``."""
    p = params
    T_sup = xp.clip(curve_supply(p, obs[..., 2], xp), p.T_supply_min, p.T_supply_max)
    raw = 2.0 * (T_sup - p.T_supply_min) / (p.T_supply_max - p.T_supply_min) - 1.0
    return xp.stack([raw], axis=-1), CurveState(started=1.0 + 0.0 * raw)


def make_hvac_curve(offset: float = 0.0, T_set: float | None = None) -> CurveParams:
    """The curve's parameters, from the building's design calculation."""
    from target_gym.hvac.env import HVACParams, steady_state

    pr = HVACParams()
    T_set = pr.band_low if T_set is None else T_set
    T_out_design = -10.0  # C, the building's design point (PHYSICS.md)
    *_, T_supply_design = steady_state(pr, T_out_design, T_set)
    return CurveParams(
        T_set=float(T_set),
        T_out_design=T_out_design,
        T_supply_design=float(T_supply_design),
        T_supply_min=float(pr.T_supply_min),
        T_supply_max=float(pr.T_supply_max),
        offset=float(offset),
    )


def make_hvac_pid_controller() -> Controller:
    """The expert: the heating curve as a :class:`Controller`."""
    return Controller(make_hvac_curve(), CurveState(), curve_step)
