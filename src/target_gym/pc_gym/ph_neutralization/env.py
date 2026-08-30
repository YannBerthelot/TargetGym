"""
pH neutralisation — CSTR with acid, buffer and base streams.

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Model
-----
The classic pH neutralisation benchmark, in the **reaction-invariant**
formulation (Gustafsson & Waller; Henson & Seborg). Acid-base reactions are
fast enough to be at equilibrium, so the thermodynamic state is fully
determined by two invariants that are *conserved* by the reaction and
therefore obey plain CSTR mixing::

    V dWa/dt = q1 (Wa1 - Wa) + q2 (Wa2 - Wa) + q3 (Wa3 - Wa)
    V dWb/dt = q1 (Wb1 - Wb) + q2 (Wb2 - Wb) + q3 (Wb3 - Wb)

    Wa = [H+] - [OH-] - [HCO3-] - 2[CO3--]     (charge-related invariant)
    Wb = [H2CO3] + [HCO3-] + [CO3--]           (total carbonate)

pH is then the root of the charge balance, an *implicit algebraic* equation::

    Wa + 10^(pH-14) - 10^(-pH)
       + Wb (1 + 2*10^(pH-pK2)) / (1 + 10^(pK1-pH) + 10^(pH-pK2)) = 0

That separation is what makes the model both cheap and brutal: two linear
mixing states, and all the nonlinearity in a scalar root-find.

Why this is a hard target MDP
-----------------------------
* **The titration curve is savagely nonlinear.** Process gain varies ~45x
  across the operating range at nominal buffering, and ~460x with none. A
  fixed-gain controller is either sluggish on the flat shoulders or unstable
  through the steep middle.
* **Buffering is the disturbance, and it is unmeasured.** Buffer flow shifts
  the operating point (pH 4.2 unbuffered to 7.9 heavily buffered) *and*
  flattens the curve by an order of magnitude. The controller sees neither the
  buffer flow nor the invariants -- only pH.
* **pH does not determine the state.** The same pH can arise from different
  (Wa, Wb) pairs with different local gain, so the plant is genuinely
  partially observed rather than merely noisy.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

# Bisection steps for the pH root-find. The bracket is [0, 14], so 44 halvings
# resolve to ~1e-12 pH units -- far below any physical relevance, and cheap
# because each step is a handful of scalar ops. Bisection rather than Newton:
# the residual is near-vertical at the equivalence point, where Newton
# overshoots out of the bracket.
PH_BISECTION_STEPS = 44

# Ornstein-Uhlenbeck buffer-flow disturbance: mean-reversion rate (1/s).
# 1/theta ~ 500 s, several residence times, so buffering drifts as a changing
# operating condition rather than as noise.
BUFFER_OU_THETA = 2.0e-3


@struct.dataclass
class PHParams(EnvParams):
    # ---- Reactor ----
    V: float = 2900.0  # mL, well-mixed tank volume

    # ---- Acid stream (HNO3), fixed ----
    q1: float = 16.6  # mL/s
    Wa1: float = 3.0e-3  # M
    Wb1: float = 0.0

    # ---- Buffer stream (NaHCO3) -- the hidden disturbance ----
    q2_nominal: float = 0.55  # mL/s
    q2_noise_std: float = 0.35  # mL/s, OU stationary std
    q2_min: float = 0.0
    q2_max: float = 2.5
    Wa2: float = -3.0e-2  # M
    Wb2: float = 3.0e-2  # M

    # ---- Base stream (NaOH + NaHCO3) -- the manipulated variable ----
    q3_min: float = 10.0  # mL/s
    q3_max: float = 22.0  # mL/s
    Wa3: float = -3.05e-3  # M
    Wb3: float = 5.0e-5  # M

    # ---- Carbonic acid dissociation ----
    pK1: float = 6.35
    pK2: float = 10.25

    # ---- Operating / termination bounds ----
    pH_min: float = 2.0  # grossly acidic -- off spec
    pH_max: float = 12.0  # grossly alkaline -- off spec

    # ---- Reward shaping ----
    tracking_band: float = 1.0  # pH units at which tracking reward reaches 0
    reagent_cost_weight: float = 0.05

    # ---- Initial / target ranges ----
    # Targets sit around neutrality, which is where the titration curve is
    # steepest and the control problem is genuinely hard.
    target_pH_range: Tuple[float, float] = (6.5, 8.0)
    initial_q3_range: Tuple[float, float] = (14.5, 16.5)

    # ---- Time discretization ----
    # Residence time V/q_total ~ 88 s, so dt = 5 s gives ~18 steps per
    # residence time. 600 steps = 50 min ~ 34 residence times.
    delta_t: float = 5.0
    max_steps_in_episode: int = 600


@struct.dataclass
class PHState(EnvState):
    Wa: float  # charge-related reaction invariant (HIDDEN)
    Wb: float  # total carbonate invariant (HIDDEN)
    q2: float  # buffer flow, the unmeasured disturbance (HIDDEN)

    pH: float  # the single measurement
    q3: float  # commanded base flow
    target_pH: float


def titration_residual(pH, Wa, Wb, params: PHParams):
    """Charge balance. Its root in ``pH`` is the equilibrium pH.

    Strictly increasing in ``pH``, which is what makes bisection safe.
    """
    p = params
    carbonate = (1.0 + 2.0 * 10.0 ** (pH - p.pK2)) / (
        1.0 + 10.0 ** (p.pK1 - pH) + 10.0 ** (pH - p.pK2)
    )
    return Wa + 10.0 ** (pH - 14.0) - 10.0 ** (-pH) + Wb * carbonate


def solve_pH(Wa, Wb, params: PHParams):
    """pH from the invariants, by bisection on [0, 14].

    The residual is monotone in pH, so bisection cannot fail; a fixed step
    count keeps it jit- and vmap-friendly with no data-dependent control flow.
    """

    def body(bounds, _):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        negative = titration_residual(mid, Wa, Wb, params) < 0.0
        # Residual increases with pH: if it is still negative at mid, the root
        # lies above.
        return (jnp.where(negative, mid, lo), jnp.where(negative, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(
        body,
        (jnp.zeros_like(Wa), jnp.full_like(Wa, 14.0)),
        xs=None,
        length=PH_BISECTION_STEPS,
    )
    return 0.5 * (lo + hi)


def compute_velocity(position, action, q2, params: PHParams):
    """RHS for the two reaction invariants. ``action`` is the base flow q3."""
    p = params
    Wa, Wb = position[0], position[1]
    q3 = action
    dWa = (p.q1 * (p.Wa1 - Wa) + q2 * (p.Wa2 - Wa) + q3 * (p.Wa3 - Wa)) / p.V
    dWb = (p.q1 * (p.Wb1 - Wb) + q2 * (p.Wb2 - Wb) + q3 * (p.Wb3 - Wb)) / p.V
    return jnp.array([dWa, dWb]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: float,
    state: PHState,
    params: PHParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` in [-1, 1] maps to base flow in [q3_min, q3_max]."""
    p = params
    q3 = convert_raw_action_to_range(
        action_raw, min_action=p.q3_min, max_action=p.q3_max
    )

    # OU buffer-flow disturbance. The innovation is drawn from a key folded
    # with ``state.time`` so a caller passing a constant key -- which every
    # rollout helper here does -- still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.q2_noise_std * jnp.sqrt(2.0 * BUFFER_OU_THETA * p.delta_t)
    q2 = (
        state.q2
        + BUFFER_OU_THETA * (p.q2_nominal - state.q2) * p.delta_t
        + sigma * noise
    )
    q2 = jnp.clip(q2, p.q2_min, p.q2_max)

    _compute_velocity = partial(compute_velocity, action=q3, q2=q2, params=params)
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.Wa, state.Wb]),
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    Wa, Wb = new_positions[0], new_positions[1]
    return (
        state.replace(
            Wa=Wa,
            Wb=Wb,
            q2=q2,
            pH=solve_pH(Wa, Wb, params),
            q3=q3,
            time=state.time + 1,
        ),
        None,
    )


@partial(jax.jit, static_argnames=["params"])
def get_obs(state: PHState, params: PHParams):
    """``[pH, q3_pct, target_pH]`` -- a pH probe and the operator's own valve.

    The reaction invariants and the buffer flow are hidden. That is not an
    artificial restriction: a plant has a pH electrode, not an on-line assay
    of carbonate speciation. It also makes the environment a genuine POMDP,
    because the same pH can arise from different (Wa, Wb) pairs whose local
    process gain differs by an order of magnitude.
    """
    q3_pct = 100.0 * (state.q3 - params.q3_min) / (params.q3_max - params.q3_min)
    return jnp.array([state.pH, q3_pct, state.target_pH])


def check_is_terminal(state: PHState, params: PHParams, xp=jnp):
    terminated = xp.logical_or(state.pH <= params.pH_min, state.pH >= params.pH_max)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: PHState, params: PHParams, xp=jnp):
    """pH tracking minus a small reagent cost."""
    err = xp.abs(state.target_pH - state.pH)
    tracking = xp.clip(1.0 - err / params.tracking_band, 0.0, 1.0) ** 2
    reagent = (state.q3 - params.q3_min) / (params.q3_max - params.q3_min)
    return tracking - params.reagent_cost_weight * reagent


def steady_state_invariants(q3, q2, params: PHParams):
    """Invariants a given pair of flows settles to -- used for reset and tests."""
    p = params
    q_total = p.q1 + q2 + q3
    Wa = (p.q1 * p.Wa1 + q2 * p.Wa2 + q3 * p.Wa3) / q_total
    Wb = (p.q1 * p.Wb1 + q2 * p.Wb2 + q3 * p.Wb3) / q_total
    return Wa, Wb


def process_gain(q3, q2, params: PHParams, dq: float = 0.01):
    """Local steady-state gain dpH/dq3 -- the figure of merit for difficulty."""
    lo = solve_pH(*steady_state_invariants(q3 - dq, q2, params), params)
    hi = solve_pH(*steady_state_invariants(q3 + dq, q2, params), params)
    return (hi - lo) / (2.0 * dq)
