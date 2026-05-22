"""Principled expert degradation: per-parameter perturbation schemes.

Two perturbation kinds:
  - log-normal scale: p' = p * exp(sigma * N(0, 1))     (positive-magnitude params)
  - additive wrapped: p' = wrap_pi(p + sigma * N(0, 1)) (phase-like params)

We register one *degradation schema* per step-fn, mapping each perturbable
field to its kind. This is orthogonal to ``_LEARNABLE_GAINS_BY_STEP_FN``:
learnable fields are the ones the RL actor modulates at runtime; degradable
fields are the ones we corrupt at expert-construction time to simulate a
poorly-tuned controller.

The degraded expert is a fresh ``FunctionalExpertPolicy`` that looks and
behaves exactly like the original but with shifted parameters. Nothing about
the action interface changes.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from target_gym.experts.pid import FunctionalExpertPolicy

# Per-step-fn schema: field name -> "logscale" or "phase"
_DEGRADATION_SCHEMA: dict[str, dict[str, str]] = {}


def register_degradation_schema(step_fn: Callable, schema: dict[str, str]) -> None:
    """Register the degradation kinds for each perturbable field of step_fn."""
    for kind in schema.values():
        if kind not in ("logscale", "phase"):
            raise ValueError(f"Unknown degradation kind {kind!r}")
    _DEGRADATION_SCHEMA[step_fn.__qualname__] = dict(schema)


def _wrap_angle(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _perturb_params(params, schema: dict[str, str], sigma: float, key: PRNGKey):
    """Apply per-field perturbation to a flat params struct, return a new struct.

    Handles array-shaped anchors (e.g. gain-schedule tables Kp_table[N])
    naturally via ``shape=anchor.shape``.
    """
    perturbed = {}
    subkeys = jax.random.split(key, max(len(schema), 1))
    for (field, kind), subkey in zip(schema.items(), subkeys):
        anchor = jnp.asarray(getattr(params, field))
        noise = jax.random.normal(subkey, shape=anchor.shape)
        if kind == "logscale":
            perturbed[field] = anchor * jnp.exp(sigma * noise)
        elif kind == "phase":
            perturbed[field] = _wrap_angle(anchor + sigma * noise)
        else:
            raise AssertionError(f"unreachable: kind={kind!r}")
    return params.replace(**perturbed)


def degrade_expert(
    expert: FunctionalExpertPolicy,
    sigma: float,
    key: PRNGKey,
) -> FunctionalExpertPolicy:
    """Return a new expert with its parameters perturbed.

    Parameters
    ----------
    expert : FunctionalExpertPolicy
        The anchor (tuned) expert.
    sigma : float
        Perturbation magnitude. ``sigma = 0`` returns an expert with identical
        parameters (up to JAX floating-point conversion); typical values are
        in [0.25, 0.75]. For phase-like parameters the same ``sigma`` is
        interpreted in radians.
    key : jax.random.PRNGKey
        Reproducibility seed for the perturbation noise.

    Returns
    -------
    FunctionalExpertPolicy
        A fresh expert with perturbed parameters, same step-fn and zero-state.
    """
    params = expert._params

    # MIMO case (FourTank, Plane 2D): the params dataclass wraps two
    # nested SISO param structs ``pid1`` and ``pid2``. Detect by structure
    # and recurse into each sub-struct with the appropriate per-loop schema.
    # Supports both vanilla MIMO PID (PIDParams sub-structs) and gain-
    # scheduled MIMO (GainSchedulePIDParams sub-structs with array tables).
    if hasattr(params, "pid1") and hasattr(params, "pid2"):
        sub = params.pid1
        if hasattr(sub, "Kp_table"):
            inner_schema = {
                "Kp_table": "logscale",
                "Ki_table": "logscale",
                "Kd_table": "logscale",
            }
        else:
            inner_schema = {"Kp": "logscale", "Ki": "logscale", "Kd": "logscale"}
        k1, k2 = jax.random.split(key)
        new_pid1 = _perturb_params(params.pid1, inner_schema, sigma, k1)
        new_pid2 = _perturb_params(params.pid2, inner_schema, sigma, k2)
        new_params = params.replace(pid1=new_pid1, pid2=new_pid2)
        return FunctionalExpertPolicy(new_params, expert._zero_state, expert._step_fn)

    step_key = expert._step_fn.__qualname__
    if step_key not in _DEGRADATION_SCHEMA:
        raise ValueError(
            f"No degradation schema registered for step_fn {step_key}. "
            "Register with register_degradation_schema()."
        )
    schema = _DEGRADATION_SCHEMA[step_key]
    new_params = _perturb_params(params, schema, sigma, key)
    return FunctionalExpertPolicy(new_params, expert._zero_state, expert._step_fn)


# ---------------------------------------------------------------------------
# Register schemas for the experts we ship
# ---------------------------------------------------------------------------


def _register_all():
    # Import here to avoid circular imports at module load.
    from target_gym.experts.pid import (
        pid_step,
        plane3d_heading_pid_step,
        plane3d_circle_pid_step,
        plane3d_figure8_pid_step,
        mimo_pid_step,
    )
    from target_gym.experts.cpg import cheetah_cpg_step

    # Generic single-loop PID: (Kp, Ki, Kd) all log-scale.
    register_degradation_schema(
        pid_step,
        {"Kp": "logscale", "Ki": "logscale", "Kd": "logscale"},
    )

    # Reacher PD-on-Jacobian: degrade the two controller gains. L1/L2 are
    # physical link lengths, not controller tuning, so they are left out.
    from target_gym.experts.pd import reacher_pd_step

    register_degradation_schema(
        reacher_pd_step,
        {"kp": "logscale", "kd": "logscale"},
    )

    # MIMO PID (Plane 2D, FourTank): pid1 and pid2 each have (Kp, Ki, Kd).
    # Its params dataclass nests PIDParams, which flax.struct can't replace
    # field-by-field the same way; degrade via a custom path below.
    # We skip registration here and handle MIMO in a dedicated helper.

    # Plane3DHeading PID.
    register_degradation_schema(
        plane3d_heading_pid_step,
        {
            "Kp_alt": "logscale",
            "Ki_alt": "logscale",
            "Kd_alt": "logscale",
            "Kp_hdg": "logscale",
            "Ki_hdg": "logscale",
            "Kd_hdg": "logscale",
            "Kp_bank": "logscale",
            "Kp_power": "logscale",
            "Ki_power": "logscale",
            "Kd_power": "logscale",
        },
    )

    # Plane3DCircle PID.
    register_degradation_schema(
        plane3d_circle_pid_step,
        {
            "Kp_alt": "logscale",
            "Ki_alt": "logscale",
            "Kd_alt": "logscale",
            "Kp_rad": "logscale",
            "Ki_rad": "logscale",
            "Kd_rad": "logscale",
            "Kp_bank": "logscale",
            "Kp_power": "logscale",
            "Ki_power": "logscale",
            "Kd_power": "logscale",
        },
    )

    # Plane3DFigureEight uses the Heading params dataclass.
    register_degradation_schema(
        plane3d_figure8_pid_step,
        {
            "Kp_alt": "logscale",
            "Ki_alt": "logscale",
            "Kd_alt": "logscale",
            "Kp_hdg": "logscale",
            "Ki_hdg": "logscale",
            "Kd_hdg": "logscale",
            "Kp_bank": "logscale",
            "Kp_power": "logscale",
            "Ki_power": "logscale",
            "Kd_power": "logscale",
        },
    )

    # CheetahRun CPG.
    register_degradation_schema(
        cheetah_cpg_step,
        {
            "frequency": "logscale",
            "amp_0": "logscale",
            "amp_1": "logscale",
            "amp_2": "logscale",
            "amp_3": "logscale",
            "amp_4": "logscale",
            "amp_5": "logscale",
            "phase_0": "phase",
            "phase_1": "phase",
            "phase_2": "phase",
            "phase_3": "phase",
            "phase_4": "phase",
            "phase_5": "phase",
        },
    )

    # BarkourJoystick CPG.
    from target_gym.experts.cpg import barkour_cpg_step

    register_degradation_schema(
        barkour_cpg_step,
        {
            "frequency": "logscale",
            **{f"amp_{i}": "logscale" for i in range(12)},
            **{f"phase_{i}": "phase" for i in range(12)},
        },
    )

    # WalkerRun CPG (6 actuators). dt is the timestep, not a tuning param,
    # so it is excluded from degradation.
    from target_gym.experts.cpg import walker_cpg_step

    register_degradation_schema(
        walker_cpg_step,
        {
            "frequency": "logscale",
            **{f"amp_{i}": "logscale" for i in range(6)},
            **{f"phase_{i}": "phase" for i in range(6)},
        },
    )

    # HopperHop CPG (4 actuators).
    from target_gym.experts.cpg import hopper_cpg_step

    register_degradation_schema(
        hopper_cpg_step,
        {
            "frequency": "logscale",
            **{f"amp_{i}": "logscale" for i in range(4)},
            **{f"phase_{i}": "phase" for i in range(4)},
        },
    )


_register_all()
