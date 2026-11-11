"""
pH neutralisation — CSTR with acid, buffer and base streams.

See ``PHYSICS.md`` in this directory for the sources, the limits and the
score, and ``REALITY.md`` for the plant record they come from.

The job (charter Amendments 2 and 3): keep the discharge of a two-stage
neutralization inside its permit, pH 6 to 8, at the least reagent. An
excursion past 60 min is a violation; effluent below pH 5 is diverted into a
finite basin and pumped back, and an overrun basin is a violation too.

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

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
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
    # ---- Stages (charter Amendment 3) ----
    # Two stirred tanks in series, as strong acid neutralization is staged
    # [P3; P5, p. 2; P7, p. 1], each holding more than 10 min of flow at the
    # nominal steady state (about 32.8 mL/s), the retention [P4, p. 2] asks of
    # a continuous tank. Ours, family member: 20 L each, about 10.2 min.
    V_s1: float = 20000.0  # mL, stage 1
    V: float = 20000.0  # mL, stage 2, whose outlet is the discharge

    # ---- Acid stream (HNO3) ----
    q1: float = 16.6  # mL/s, nominal influent
    Wa1: float = 3.0e-3  # M
    Wb1: float = 0.0

    # ---- Buffer stream (NaHCO3) -- the hidden disturbance ----
    q2_nominal: float = 0.55  # mL/s
    q2_noise_std: float = 0.35  # mL/s, OU stationary std
    q2_min: float = 0.0
    q2_max: float = 2.5
    Wa2: float = -3.0e-2  # M
    Wb2: float = 3.0e-2  # M

    # ---- Base reagent (NaOH + NaHCO3) -- the manipulated variables ----
    # One valve per stage [P3, pp. 4-5]. No source gives a floor on reagent
    # flow: a dosing pump or valve can stop. The stage-1 top is ours (the rig's
    # valve size was not found). The stage-2 trim valve is a tenth of it, as
    # the two final elements of [P4, p. 2] (1 and 0.1 GPM).
    q3_min: float = 0.0  # mL/s
    q3_max: float = 22.0  # mL/s
    q3_trim_max: float = 2.2  # mL/s
    Wa3: float = -3.05e-3  # M
    Wb3: float = 5.0e-5  # M

    # ---- Carbonic acid dissociation ----
    pK1: float = 6.35
    pK2: float = 10.25

    # ---- Influent upsets (REALITY.md, "A normal day and a bad day") ----
    # Large unknown changes in the influent's strong acid concentration from
    # production units [P3, p. 4]: two an episode, within +-20 % of nominal,
    # ramped over 10 min (ours, the "calm to rocking and reeling during one
    # lunch break" of [P3, p. 8]).
    acid_upset_size: float = 0.2  # fraction of Wa1
    acid_upset_ramp: float = 600.0  # s
    # Influent flow swings (Amendment 3), metered [P3, Figure 2]: two an
    # episode, within +-flow_swing of nominal, ramped like the acid. Ours.
    flow_swing: float = 0.2  # fraction of q1

    # ---- Limits (REALITY.md, "Limits"; charter rule 14) ----
    # Zone: the permit range, which the guideline sets [P8]. Ours, from the one
    # range a source prints: the example spec pH 7 +- 1 [P7, p. 2].
    permit_low: float = 6.0  # pH
    permit_high: float = 8.0  # pH
    # Timed: no single excursion outside the range over 60 min [P8], diverted
    # or not (Amendment 3). Past it is a permit violation, the task's trip.
    excursion_limit: float = 3600.0  # s
    # Protected: no discharge below pH 5.0 to a public sewer [P9]; the field
    # plant diverts effluent when the tank is too low [P3, pp. 4-5], into a
    # finite basin (Amendment 3) that is pumped back to stage 1 for treatment.
    # Overrunning the basin is a violation. Ours, family range "hours of
    # flow": 2 h of nominal discharge; pump-back at a quarter of the influent.
    sewer_floor: float = 5.0  # pH
    basin_volume: float = 2.0 * 3600.0 * 32.8 / 1000.0  # L
    pumpback_flow: float = 4.15  # mL/s

    # ---- Retired version-1 reward fields, read only by reward_version 1 ----
    pH_min: float = 2.0
    precision_floor: float = 1e-2  # pH units, glass electrode resolution
    pH_max: float = 12.0
    reagent_cost_weight: float = 0.0

    # ---- Episode (charter rules 17 and 20) ----
    # Established operation: both tanks on the steady state at the operator's
    # set point, pH 7, the centre of the permit range. The set point is the
    # operator's choice [P3, p. 4] and is not scored. 2 h at 5 s: longer than
    # the 60 min excursion the permit counts [P8], with room for the upsets.
    # Ours. The scenario suite (scenarios.py) sets which upsets run.
    setpoint_pH: float = 7.0
    delta_t: float = 5.0
    max_steps_in_episode: int = 1440

    # ---- Score (charter rule 15; REALITY.md, "The operator's job") ----
    # Keep the effluent inside the permit without overdosing reagent [P5, p. 2].
    # In litres of reagent, the quantity the field plant tracks per shift
    # [P3, p. 9]. No price was found, so each charge is in reagent litres, ours:
    # - excursion_price: a minute outside the permit costs 100 minutes of the
    #   reagent a hold uses (about 1 L/min), so leaving the zone never pays;
    # - diversion_price: a diverted litre is pumped back and treated again,
    #   priced above reagent (Amendment 3): 2 L per L, the reagent it takes
    #   again (about 1 L per L with this dilute base) and the pumping;
    # - a violation is charged as another hour at the worst surviving step's
    #   cost, so tripping on purpose never pays.
    reward_version: int = 4
    reagent_price: float = 1.0  # per L of reagent
    excursion_price: float = 100.0 / 60.0  # L per s outside the permit
    diversion_price: float = 2.0  # L per L diverted
    failure_cost: float = 9.0  # L per step, above any surviving step
    #: A violation is charged as another hour (720 steps of 5 s).
    restart_steps: int = 720
    # Read by the benchmark's time-in-band KPI, not by the reward: the permit's
    # half-width around the set point, so "in band" means inside the permit.
    e_floor: float = 1.0  # pH


#: Units of the PHParams fields, for the generated environment pages.
PARAM_UNITS = {
    "V_s1": "mL",
    "V": "mL",
    "q1": "mL/s",
    "Wa1": "mol/L",
    "Wb1": "mol/L",
    "q2_nominal": "mL/s",
    "q2_noise_std": "mL/s",
    "q2_min": "mL/s",
    "q2_max": "mL/s",
    "Wa2": "mol/L",
    "Wb2": "mol/L",
    "q3_min": "mL/s",
    "q3_max": "mL/s",
    "q3_trim_max": "mL/s",
    "Wa3": "mol/L",
    "Wb3": "mol/L",
    "pK1": "-",
    "pK2": "-",
    "acid_upset_size": "-",
    "acid_upset_ramp": "s",
    "flow_swing": "-",
    "permit_low": "pH",
    "permit_high": "pH",
    "excursion_limit": "s",
    "sewer_floor": "pH",
    "basin_volume": "L",
    "pumpback_flow": "mL/s",
    "pH_min": "pH",
    "precision_floor": "pH",
    "pH_max": "pH",
    "reagent_cost_weight": "-",
    "setpoint_pH": "pH",
    "reward_version": "-",
    "reagent_price": "L/L",
    "excursion_price": "L/s",
    "diversion_price": "L/L",
    "failure_cost": "L",
    "restart_steps": "steps",
    "e_floor": "pH",
}


@struct.dataclass
class PHState(EnvState):
    Wa: float  # stage 2 (discharge) charge invariant (HIDDEN)
    Wb: float  # stage 2 total carbonate (HIDDEN)
    q2: float  # buffer flow, the unmeasured disturbance (HIDDEN)

    pH: float  # discharge pH, stage 2's probe
    q3: float  # stage-1 reagent valve, mL/s
    target_pH: float  # the operator's set point, informational, not scored

    # Influent acid strength and flow, as multiples of nominal (acid HIDDEN,
    # flow metered): each ramps from ``levels[i]`` to ``levels[i + 1]`` from
    # ``times[i]`` (steps).
    upset_times: jax.Array
    upset_levels: jax.Array
    flow_times: jax.Array
    flow_levels: jax.Array
    # Seconds the discharge has been outside the permit in this excursion, the
    # clock the continuous monitor keeps [P8]. Read by the operator.
    excursion: float

    # Stage 1 (HIDDEN invariants, its probe read), the trim valve, the basin.
    Wa_s1: float = struct.field(default=0.0, kw_only=True)
    Wb_s1: float = struct.field(default=0.0, kw_only=True)
    pH_s1: float = struct.field(default=7.0, kw_only=True)
    q3_trim: float = struct.field(default=0.0, kw_only=True)
    basin: float = struct.field(default=0.0, kw_only=True)  # L held
    q1: float = struct.field(default=16.6, kw_only=True)  # influent, mL/s, metered
    Wa_basin: float = struct.field(default=0.0, kw_only=True)
    Wb_basin: float = struct.field(default=0.0, kw_only=True)

    actuator_held: jax.Array = actuator_field(2)


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


def ramp_schedule(times, levels, state, params: PHParams, xp=jnp):
    """A multiple of nominal at ``state.time``: piecewise ramps over
    ``acid_upset_ramp`` from ``levels[i]`` to ``levels[i + 1]`` at
    ``times[i]``."""
    t = state.time * params.delta_t
    starts = times * params.delta_t
    ramp = jnp.maximum(params.acid_upset_ramp, params.delta_t)
    frac = xp.clip((t - starts) / ramp, 0.0, 1.0)
    return levels[0] + xp.sum(frac * (levels[1:] - levels[:-1]))


def acid_scale(state, params: PHParams, xp=jnp):
    """The influent's strong acid strength now, as a multiple of ``Wa1``."""
    return ramp_schedule(state.upset_times, state.upset_levels, state, params, xp)


def influent_flow(state, params: PHParams, xp=jnp):
    """The influent flow now, mL/s, as its meter reads it."""
    return params.q1 * ramp_schedule(
        state.flow_times, state.flow_levels, state, params, xp
    )


def compute_velocity(position, action, q2, params: PHParams, Wa1=None):
    """RHS for one tank fed by the acid, buffer and base streams, the single
    stage of the steady-state helpers. ``action`` is the base flow q3."""
    p = params
    Wa1 = p.Wa1 if Wa1 is None else Wa1
    Wa, Wb = position[0], position[1]
    q3 = action
    dWa = (p.q1 * (Wa1 - Wa) + q2 * (p.Wa2 - Wa) + q3 * (p.Wa3 - Wa)) / p.V_s1
    dWb = (p.q1 * (p.Wb1 - Wb) + q2 * (p.Wb2 - Wb) + q3 * (p.Wb3 - Wb)) / p.V_s1
    return jnp.array([dWa, dWb]), None


def staged_velocity(position, flows, params: PHParams):
    """Both tanks' invariants. ``flows`` holds the influent ``q1`` and its acid
    ``Wa1``, the buffer ``q2``, both valves, and the pump-back ``qp`` with the
    basin's invariants."""
    p = params
    q1, Wa1, q2, q3, q3t, qp, Wa_b, Wb_b = flows
    Wa_1, Wb_1, Wa_2, Wb_2 = position
    dWa_1 = (
        q1 * (Wa1 - Wa_1)
        + q2 * (p.Wa2 - Wa_1)
        + q3 * (p.Wa3 - Wa_1)
        + qp * (Wa_b - Wa_1)
    ) / p.V_s1
    dWb_1 = (
        q1 * (p.Wb1 - Wb_1)
        + q2 * (p.Wb2 - Wb_1)
        + q3 * (p.Wb3 - Wb_1)
        + qp * (Wb_b - Wb_1)
    ) / p.V_s1
    Q1 = q1 + q2 + q3 + qp
    dWa_2 = (Q1 * (Wa_1 - Wa_2) + q3t * (p.Wa3 - Wa_2)) / p.V
    dWb_2 = (Q1 * (Wb_1 - Wb_2) + q3t * (p.Wb3 - Wb_2)) / p.V
    return jnp.array([dWa_1, dWb_1, dWa_2, dWb_2]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw,
    state: PHState,
    params: PHParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` in [-1, 1]^2 maps to the stage-1 base flow in
    [q3_min, q3_max] and the stage-2 trim in [0, q3_trim_max]."""
    p = params
    action_raw = jnp.reshape(action_raw, (2,))
    q3 = convert_raw_action_to_range(
        action_raw[0], min_action=p.q3_min, max_action=p.q3_max
    )
    q3t = convert_raw_action_to_range(
        action_raw[1], min_action=0.0, max_action=p.q3_trim_max
    )

    # OU buffer-flow disturbance, keyed on ``state.time`` so a constant key
    # still gives a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.q2_noise_std * jnp.sqrt(2.0 * BUFFER_OU_THETA * p.delta_t)
    q2 = (
        state.q2
        + BUFFER_OU_THETA * (p.q2_nominal - state.q2) * p.delta_t
        + sigma * noise
    )
    q2 = jnp.clip(q2, p.q2_min, p.q2_max)

    q1 = influent_flow(state, p)
    Wa1 = p.Wa1 * acid_scale(state, p)
    # The interlock reads the discharge at the step's start: below the sewer
    # floor the outflow goes to the basin, else the basin is pumped back.
    diverting = state.pH < p.sewer_floor
    qp = jnp.where(
        diverting,
        0.0,
        jnp.minimum(p.pumpback_flow, 1000.0 * state.basin / p.delta_t),
    )
    flows = (q1, Wa1, q2, q3, q3t, qp, state.Wa_basin, state.Wb_basin)
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.Wa_s1, state.Wb_s1, state.Wa, state.Wb]),
        delta_t=p.delta_t,
        compute_velocity=partial(staged_velocity, flows=flows, params=p),
        method=integration_method,
    )
    Wa_1, Wb_1, Wa, Wb = (new_positions[i] for i in range(4))

    # The basin mixes what it receives and loses what is pumped back.
    outflow_L = (q1 + q2 + q3 + qp + q3t) * p.delta_t / 1000.0
    into = jnp.where(diverting, outflow_L, 0.0)
    held = state.basin + into
    weight = jnp.where(held > 0.0, into / jnp.maximum(held, 1e-9), 0.0)
    Wa_basin = state.Wa_basin + weight * (state.Wa - state.Wa_basin)
    Wb_basin = state.Wb_basin + weight * (state.Wb - state.Wb_basin)
    basin = jnp.maximum(held - qp * p.delta_t / 1000.0, 0.0)

    pH = solve_pH(Wa, Wb, params)
    # The excursion clock runs while the discharge is outside the permit,
    # diverted or not (Amendment 3), and resets when it is back inside.
    outside = jnp.logical_or(pH < p.permit_low, pH > p.permit_high)
    excursion = jnp.where(outside, state.excursion + p.delta_t, 0.0)
    return (
        state.replace(
            Wa=Wa,
            Wb=Wb,
            Wa_s1=Wa_1,
            Wb_s1=Wb_1,
            q2=q2,
            pH=pH,
            pH_s1=solve_pH(Wa_1, Wb_1, params),
            q3=q3,
            q3_trim=q3t,
            q1=q1,
            basin=basin,
            Wa_basin=Wa_basin,
            Wb_basin=Wb_basin,
            excursion=excursion,
            time=state.time + 1,
        ),
        None,
    )


# Deliberately not jitted (see the other process tasks): callers that want it
# fused jit ``step_env``, which traces this inline.
def get_obs(state: PHState, params: PHParams):
    """``[pH, q3_pct, setpoint_pH, excursion_min, pH_s1, trim_pct, q1,
    basin_pct]``: the discharge probe on stage 2 [P5, p. 1; P6], the stage-1
    valve, the set point, the minutes outside the permit in this excursion,
    which the monitor records [P5, p. 2; P8], the stage-1 probe [P3, pp. 4-5],
    the trim valve, the metered influent flow [P3, Figure 2], and the basin's
    level, which the operator reads (ours).

    The invariants, the buffer flow and the influent's acid strength are
    hidden: a plant has pH electrodes, not an assay of carbonate speciation,
    and the acid changes come unannounced [P3, p. 4].
    """
    q3_pct = 100.0 * (state.q3 - params.q3_min) / (params.q3_max - params.q3_min)
    trim_pct = 100.0 * state.q3_trim / params.q3_trim_max
    return jnp.array(
        [
            state.pH,
            q3_pct,
            state.target_pH,
            state.excursion / 60.0,
            state.pH_s1,
            trim_pct,
            state.q1,
            100.0 * state.basin / params.basin_volume,
        ]
    )


def check_is_terminal(state: PHState, params: PHParams, xp=jnp):
    """The trip is a permit violation: one excursion past the allowed 60 min
    [P8], or an overrun diversion basin (Amendment 3)."""
    terminated = xp.logical_or(
        state.excursion > params.excursion_limit, state.basin > params.basin_volume
    )
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: PHState, params: PHParams, xp=jnp, tripped=None):
    """The reward's additive cost terms, each >= 0, in litres of reagent
    (PHParams, "Score").

    - ``reagent``: the base both valves used this step [P5, p. 2; P3, p. 9];
    - ``excursion``: the discharge outside the permit, per second [P8];
    - ``diversion``: effluent diverted below the sewer floor, to be pumped back
      and treated again [P9; P3, pp. 4-5];
    - ``failure``: the violation, through ``reward.with_trip``.

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    diverted = state.pH < p.sewer_floor
    outside = xp.logical_or(state.pH < p.permit_low, state.pH > p.permit_high)
    outflow_L = (state.q1 + state.q2 + state.q3 + state.q3_trim) * p.delta_t / 1000.0
    reagent_L = (state.q3 + state.q3_trim) * p.delta_t / 1000.0
    terms = {
        "reagent": R.priced_cost(reagent_L, p.reagent_price, xp),
        "excursion": xp.where(outside, p.excursion_price * p.delta_t, 0.0),
        "diversion": xp.where(diverted, p.diversion_price * outflow_L, 0.0),
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


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


def steady_base_flow(pH, q2, params: PHParams, Wa1=None):
    """The base flow q3 whose steady state sits at ``pH``, for the acid
    strength ``Wa1`` (``params.Wa1`` by default). At steady state the
    invariants are the streams' flow-weighted means, so the charge balance is
    linear in q3 and the inverse titration curve is closed-form."""
    p = params
    Wa1 = p.Wa1 if Wa1 is None else Wa1
    f = 10.0 ** (pH - 14.0) - 10.0 ** (-pH)
    c = (1.0 + 2.0 * 10.0 ** (pH - p.pK2)) / (
        1.0 + 10.0 ** (p.pK1 - pH) + 10.0 ** (pH - p.pK2)
    )
    r1 = Wa1 + f + p.Wb1 * c
    r2 = p.Wa2 + f + p.Wb2 * c
    r3 = p.Wa3 + f + p.Wb3 * c
    return -(p.q1 * r1 + q2 * r2) / r3
