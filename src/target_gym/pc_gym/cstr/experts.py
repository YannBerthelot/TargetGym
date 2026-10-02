"""Reference controller (the MPC slot) for the CSTR task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, ShrinkingHorizonNLP, SamplingMPC and the v2 objective helpers)
stays in ``target_gym.experts.mpc``, which is in every task's baseline
fingerprint.
"""

from target_gym.experts.mpc import (
    ShrinkingHorizonNLP,
    casadi_step_map,
    protocol_burn_in,
)

try:
    import casadi
except ImportError:  # ShrinkingHorizonNLP refuses to build without it
    pass


# ---------------------------------------------------------------------------
# CSTR
# ---------------------------------------------------------------------------

#: Typical magnitude of [C_a, T], which scales the NLP's state variables.
_X_SCALE = (1.0, 300.0)


def cstr_step_map(env, params):
    """The CSTR's discrete step, ``F(x, u) -> x_next``, as a CasADi Function.

    ``x = [C_a, T]`` and ``u`` the raw coolant command in ``[-1, 1]``. The
    right-hand side is ``env.compute_velocity`` term for term, and the
    integration is the environment's own (``env.integration_method``,
    ``rk4_1`` as registered), so on the action box this is ``step_env`` in
    float64. The env clips the command to the box first; the NLP bounds it
    there, so the clip is left out to keep the map smooth at the bounds.
    """
    p = params

    def velocity(x, u):
        C_a, T = x[0], x[1]
        T_c = p.T_c_min + 0.5 * (u[0] + 1.0) * (p.T_c_max - p.T_c_min)
        rA = p.k0 * casadi.exp(-p.EA_over_R / T) * C_a
        return casadi.vertcat(
            p.q / p.V * (p.Caf - C_a) - rA,
            p.q / p.V * (p.Ti - T)
            + ((-p.deltaHr) * rA) * (1 / (p.rho * p.C))
            + p.UA * (T_c - T) * (1 / (p.rho * p.C * p.V)),
        )

    method = getattr(env, "integration_method", "rk4_1")
    return casadi_step_map(velocity, 2, 1, float(p.delta_t), method)


def make_cstr_mpc(
    env,
    params,
    window_start: int | None = None,
    resolve_every: int = 1,
    pre_weight: float = 1e-3,
):
    """The CSTR's oracle: a shrinking-horizon NLP over the rest of the episode.

    It re-solves every step on the env's own step (``cstr_step_map``), with
    the concentration's tracking cost in floor units. ``window_start`` is
    where the protocol starts scoring, read by default from its burn-in
    (``protocol_burn_in``: 12 steps of the registered 100), and the steps
    before it weigh ``pre_weight``. The plant is deterministic, so the
    closed loop reaches the plant's optimum to float32 rounding.

    It replaced a do-mpc controller on PC-gym's 5-step horizon (oracle audit,
    2026-10), whose move penalty, in raw mol/L, was 1e4 in floor units and
    alone made its 0.317 per step on the protocol seeds. This one gives
    9.4e-9 there with zero trips (``tests/pc_gym/cstr``).
    """
    if window_start is None:
        window_start = protocol_burn_in("cstr", params)
    return ShrinkingHorizonNLP(
        env,
        params,
        step_map=cstr_step_map(env, params),
        state_fields=("C_a", "T"),
        target_fields=("target_CA",),
        tracked=(0,),
        window_start=window_start,
        x_scale=_X_SCALE,
        pre_weight=pre_weight,
        resolve_every=resolve_every,
    )
