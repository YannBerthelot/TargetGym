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

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

# Bisection steps for the pH root-find. The bracket is [0, 14], so 20 halvings
# resolve to ~1.3e-5 pH units -- three orders of magnitude finer than a pH
# electrode, which reads to about 0.01. Bisection rather than Newton: the
# residual is near-vertical at the equivalence point, where Newton overshoots
# out of the bracket.
#
# This was 44, on the reasoning that each halving is "a handful of scalar ops
# and cheap". Measured, it is the whole cost of the environment: the scan is
# sequential, and throughput is exactly linear in the count -- 44 halvings give
# 2.03 M steps/s, 20 give 4.05, 8 give 8.09. At 44 this environment was ten
# times slower than its compiled graph size predicts, and the extra 24 halvings
# were resolving the pH to 1e-13, which nothing downstream can see.
PH_BISECTION_STEPS = 20

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
    precision_floor: float = 1e-2  # pH units, glass electrode resolution
    pH_max: float = 12.0  # grossly alkaline -- off spec

    # ---- Reward shaping ----
    # Error scale for the MPC's tracking term, not read by ``compute_reward``.
    # See "Why the MPC does not minimise the reward" in docs/baselines.md.
    tracking_band: float = 1.0  # pH units
    # Zeroed for the 0.6 line: this phase scores setpoint tracking alone.
    # A running cost is a real part of every one of these plants, but its
    # weight against tracking accuracy is a design decision this library has
    # not earned yet, and an arbitrary one turns a tracking benchmark into a
    # multi-objective problem whose Pareto point nobody chose. The glass
    # furnace showed the cost of getting it wrong: its MPC sat 6 K cold with
    # fuel at minimum 80% of the time, because a 0.1 fuel weight against a
    # quadratic tracking surrogate made that the optimum of the objective it
    # was given. The field and the term stay, so a weight can be restored
    # once there is a defensible way to set it. See docs/roadmap.md.
    reagent_cost_weight: float = 0.0  # was 0.05

    # ---- Initial / target ranges ----
    # Targets sit around neutrality, which is where the titration curve is
    # steepest and the control problem is genuinely hard.
    target_pH_range: Tuple[float, float] = (6.5, 8.0)
    initial_q3_range: Tuple[float, float] = (14.5, 16.5)

    # ---- Time discretization ----
    # Residence time V/q_total ~ 88 s, so dt = 5 s gives ~18 steps per
    # residence time. 600 steps = 50 min ~ 34 residence times.
    delta_t: float = 5.0
    max_steps_in_episode: int = 300

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # Floor: the shipped MPC's long-run mean |pH error| under the shipped
    # buffer-flow disturbance (q2 noise), 0.0080 pH (the lowest of three seeds: 0.0139 / 0.0220 / 0.0080; PID 0.016-0.054) over 900 hold steps after a
    # 108-step burn-in, `scripts/measure_hold.py` -- an upper bound on the
    # achievable floor (no reduced-model optimum exists for this plant).
    # e_tol = 0 provisionally: the discharge permit band a plant would use
    # here is a regulatory number to be supplied (typically pH 6-9 on the
    # outfall). Reagent above the hold-phase flow (16.24 mL/s, the same for
    # PID and MPC) is charged at weight 1: one floor-width of pH error is
    # worth the whole hold-phase reagent flow again. The span costs
    # (10 / 0.0080)^2 = 1.6e6 per step; off-spec termination twice that.
    reward_version: int = 2
    e_floor: float = 0.0080  # pH, lowest per-seed MPC hold error (upper bound)
    e_tol: float = 0.0  # provisional; permit band to be supplied
    tracking_exponent: float = 2.0
    c_hold: float = 16.24  # mL/s reagent while holding (PID = MPC)
    running_weight: float = 1.0
    failure_cost: float = 3.1e6
    #: Restart time priced into a trip (``reward.trip_cost``; 1 h at 5 s steps: flush the tank after a gross excursion, provisional).
    restart_steps: int = 720
    #: Tracking cost per step at the floor, in the reward's units; the NEA floor.
    rho_floor_tracking: float = 1.0
    rho_floor: float = 1.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False


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


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
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
    # No trip: the effluent is a convex mix of the inlet streams, so its pH
    # stays within about 3.1-10.6 whatever the valves do, and the 2 / 12
    # limits are unreachable. Kept as documentation of the off-spec range.
    terminated = xp.zeros((), dtype=bool)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: PHState, params: PHParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    terminated, _ = check_is_terminal(state, params, xp)
    terms = {
        "tracking": R.tracking_cost(
            state.target_pH - state.pH,
            params.e_floor,
            params.e_tol,
            params.tracking_exponent,
            xp,
        ),
        "running": R.running_cost(state.q3, params.c_hold, params.running_weight, xp),
    }
    return R.with_trip(terms, terminated, R.trip_cost(params), xp)


def compute_reward_v1(state: PHState, params: PHParams, xp=jnp):
    """pH tracking minus a small reagent cost."""
    err = xp.abs(state.target_pH - state.pH)
    tracking = log_scaled_reward(
        err, params.precision_floor, params.pH_max - params.pH_min, xp
    )
    reagent = (state.q3 - params.q3_min) / (params.q3_max - params.q3_min)
    return tracking * (1.0 - params.reagent_cost_weight * reagent)


def compute_reward(state: PHState, params: PHParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


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
