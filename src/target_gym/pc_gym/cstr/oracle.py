"""The ceiling's pieces for the CSTR task (``pc_gym.foresight``).

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). Version 8 ships no oracle on the
shared solver yet: the planner model of versions 5 to 7 moved one valve to
track a setpoint, and one for two moves and the production score is not
written (registry, ``baselines_note``).
"""

import jax.numpy as jnp

#: The reward terms the ceiling's solver keeps (``pc_gym.foresight``): the
#: smooth one. The band and the trip are steps, so they are margins.
CEILING_TERMS = ("lost_production",)


def ceiling_margins(state, p):
    """The ceiling's margins (``pc_gym.foresight``), in tenths of a kelvin (the
    thermometry's resolution) backed off one tenth: the band's two edges and
    the interlock's temperature. The price is a step's product, what leaving
    the band costs: priced at the trip's cost, 960 times steeper, L-BFGS-B
    stalled against the wall (measured 2026-11-23)."""
    off = state.T - state.target_T
    edges = [p.T_band - off, p.T_band + off, p.T_trip - state.T]
    return jnp.stack([e / p.e_floor - 1.0 for e in edges]), 1.0
