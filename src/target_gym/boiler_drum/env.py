"""
Boiler drum — natural-circulation drum boiler with shrink-and-swell.

See ``PHYSICS.md`` in this directory for provenance, the parameter table and
the validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Topology::

                    steam out q_s
                         ^
                    +----|----------------+
       feedwater    |    steam space      |
       q_f -------> |=====================|  <- drum water level
                    |  water + bubbles    |
                    +--+--------------+---+
                       |              ^
              downcomer|              | riser (two-phase, heated)
                       v              |
                       +----[ Q ]-----+

Four states: drum pressure ``p``, total water volume ``V_wt``, and the steam
mass held in the risers (``m_sr``) and as bubbles below the drum water level
(``m_sd``). The first two carry the global inventory; the last two say where
the steam *is*, which is what fixes the water surface.

Why this environment exists
---------------------------
Drum level is the classic **non-minimum-phase** control problem, and nothing
else in this suite has that shape:

* **Swell.** Open the turbine valve and pressure falls, so every bubble in the
  system expands and the hot water inventory flashes. The level goes *up* even
  though mass is leaving. A controller that reacts to the level it sees cuts
  feedwater exactly when it should be adding it, and the level then collapses.
* **Shrink.** Subcooled feedwater collapses bubbles, so adding water makes the
  level *fall* before it rises.
* **Level is an integrator.** It has no natural steady state -- a 2 % feedwater
  bias walks the level 9 cm in ten minutes -- so the controller cannot simply
  settle on a bias.
* **Both limits are irrecoverable.** High level carries water into the turbine;
  low level uncovers the tubes. Real boilers trip on both.
* **Voidage is unmeasurable.** ``m_sr`` and ``m_sd`` are hidden. No plant
  instrument reads riser void fraction, and it is precisely the state driving
  the inverse response.

The controller sets firing rate and feedwater flow; steam demand is the
turbine's, and is a measured disturbance (as in three-element control) whose
*future* is unknown.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

PA_PER_BAR = 1e5

# Ornstein-Uhlenbeck steam demand. 1/theta is about 200 s, so a load swing
# persists over several drum time constants rather than flickering step to step.
STEAM_OU_THETA = 5.0e-3


@struct.dataclass
class BoilerDrumParams(EnvParams):
    delta_t: float = 2.0
    max_steps_in_episode: int = 1_800  # 1 hour

    # -- geometry (Astrom & Bell P16-G16, Oresundsverket 160 MW unit) --------
    V_t: float = 88.0  # m3   total water + steam volume
    V_d: float = 40.0  # m3   drum
    V_r: float = 37.0  # m3   risers
    V_dc: float = 11.0  # m3   downcomers
    A_d: float = 20.0  # m2   drum area at normal water level
    A_dc: float = 0.355  # m2   downcomer flow area
    L_r: float = 11.0  # m    riser height
    m_metal: float = 300_000.0  # kg   tube + drum metal
    C_metal: float = 550.0  # J/(kg K)

    # -- saturated-steam property fits, quadratic in pressure [bar] ----------
    # Least-squares fits to IAPWS saturation values at 60-110 bar; residuals
    # are asserted in the tests (all within 0.5 %).
    ts_coef: Tuple[float, float, float] = (-3.8655462e-3, 1.504000e0, 1.9935496e2)
    rho_w_coef: Tuple[float, float, float] = (3.6974790e-3, -2.3474286e0, 8.8550706e2)
    rho_s_coef: Tuple[float, float, float] = (1.9215686e-3, 3.0847619e-1, 5.4157983e0)
    h_w_coef: Tuple[float, float, float] = (-1.3899160e1, 7.0854286e3, 8.3899193e5)
    h_s_coef: Tuple[float, float, float] = (-9.8207283e0, 8.6952381e1, 2.8145190e6)

    # -- circulation and voidage --------------------------------------------
    k_friction: float = 25.0  # -    downcomer/riser loop friction; sets the
    #                                 circulation ratio (validated at 5-15)
    tau_sr: float = 8.0  # s    steam transit time through the risers
    T_d: float = 15.0  # s    bubble residence below the drum water level
    f_carry: float = 0.10  # -    riser steam passing below the water level;
    #                                 the rest is separated above it
    f_cd: float = 0.25  # -    share of the feedwater heat deficit paid by
    #                                 collapsing drum bubbles (see PHYSICS.md)
    c_pw: float = 5700.0  # J/(kg K) saturated water near 300 C
    gravity: float = 9.81

    # -- operating point ------------------------------------------------------
    h_feedwater: float = 1.037e6  # J/kg feedwater at 240 C
    Q_max: float = 2.4e8  # W    maximum firing rate (150 % of nominal)
    q_feed_max: float = 200.0  # kg/s maximum feedwater flow
    Q_nominal: float = 1.6e8  # W
    q_steam_nominal: float = 93.35  # kg/s
    p_nominal: float = 85.0  # bar
    V_wt_nominal: float = 58.0  # m3   water inventory at normal water level

    # -- disturbance ----------------------------------------------------------
    q_steam_noise_std: float = 8.0  # kg/s stationary std of the load swing

    # -- task -----------------------------------------------------------------
    target_pressure_range: Tuple[float, float] = (82.0, 88.0)
    initial_level_range: Tuple[float, float] = (-0.05, 0.05)
    level_band: float = 0.10  # m    tracking band for the reward
    pressure_band: float = 2.0  # bar
    level_trip: float = 0.25  # m    carryover / dryout, both irrecoverable
    pressure_min: float = 65.0  # bar
    pressure_max: float = 105.0  # bar
    fuel_weight: float = 0.05


@struct.dataclass
class BoilerDrumState(EnvState):
    pressure: jnp.ndarray  # bar
    V_wt: jnp.ndarray  # m3   total water volume
    m_sr: jnp.ndarray  # kg   steam held in the risers
    m_sd: jnp.ndarray  # kg   bubbles below the drum water level
    level: jnp.ndarray  # m    deviation from normal water level
    q_steam: jnp.ndarray  # kg/s turbine draw (the disturbance)
    Q_fuel: jnp.ndarray  # W
    q_feed: jnp.ndarray  # kg/s
    target_pressure: jnp.ndarray  # bar
    level_ref: jnp.ndarray  # m3   water volume that reads as zero level


# ---------------------------------------------------------------------------
# Saturated steam properties
# ---------------------------------------------------------------------------


def _poly(coef, p):
    return coef[0] * p * p + coef[1] * p + coef[2]


def _dpoly(coef, p):
    """Derivative with respect to pressure, per Pa (the state is in bar)."""
    return (2.0 * coef[0] * p + coef[1]) / PA_PER_BAR


def saturation_temperature(p_bar, params: BoilerDrumParams):
    return _poly(params.ts_coef, p_bar)


def water_density(p_bar, params: BoilerDrumParams):
    return _poly(params.rho_w_coef, p_bar)


def steam_density(p_bar, params: BoilerDrumParams):
    return _poly(params.rho_s_coef, p_bar)


def water_enthalpy(p_bar, params: BoilerDrumParams):
    return _poly(params.h_w_coef, p_bar)


def steam_enthalpy(p_bar, params: BoilerDrumParams):
    return _poly(params.h_s_coef, p_bar)


def latent_heat(p_bar, params: BoilerDrumParams):
    return steam_enthalpy(p_bar, params) - water_enthalpy(p_bar, params)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def void_fraction(m_sr, p_bar, params: BoilerDrumParams):
    """Mean volumetric void fraction in the risers.

    Steam is tracked as *mass*, so a falling pressure raises the void fraction
    instantly at constant mass. That is the mechanism behind the swell.
    """
    rho_s = steam_density(p_bar, params)
    return jnp.clip(m_sr / (rho_s * params.V_r), 0.0, 0.95)


def circulation_flow(alpha_v, p_bar, params: BoilerDrumParams):
    """Natural circulation from the downcomer/riser density difference.

    The all-water downcomer is heavier than the two-phase riser; the imbalance
    drives the loop and is balanced by friction.
    """
    rho_w = water_density(p_bar, params)
    rho_s = steam_density(p_bar, params)
    driving = (rho_w - rho_s) * alpha_v * params.A_dc * params.L_r * params.gravity
    return jnp.sqrt(
        jnp.clip(2.0 * rho_w * driving / params.k_friction, 0.0, jnp.inf) + 1e-9
    )


def drum_level(V_wt, m_sr, m_sd, level_ref, p_bar, params: BoilerDrumParams):
    """Level as a deviation from normal water level, which is what the gauge reads.

    Water in the drum is what the total inventory leaves over once the
    downcomers and the water fraction of the risers are accounted for -- so
    riser voidage moves the level directly, without any mass entering or
    leaving the drum.
    """
    rho_s = steam_density(p_bar, params)
    alpha_v = void_fraction(m_sr, p_bar, params)
    V_wd = V_wt - params.V_dc - (1.0 - alpha_v) * params.V_r
    return (V_wd + m_sd / rho_s - level_ref) / params.A_d


def compute_velocity(position, action, q_steam, params: BoilerDrumParams):
    """Time derivatives of ``[pressure, V_wt, m_sr, m_sd]``.

    ``action`` is ``(Q_fuel [W], q_feed [kg/s])``.
    """
    p_bar, V_wt, m_sr, m_sd = position[0], position[1], position[2], position[3]
    Q, q_f = action
    pr = params

    p_bar = jnp.clip(p_bar, 20.0, 200.0)
    rho_w = water_density(p_bar, pr)
    rho_s = steam_density(p_bar, pr)
    h_w = water_enthalpy(p_bar, pr)
    h_s = steam_enthalpy(p_bar, pr)
    h_c = jnp.maximum(h_s - h_w, 1e4)
    d_rho_w = _dpoly(pr.rho_w_coef, p_bar)
    d_rho_s = _dpoly(pr.rho_s_coef, p_bar)
    d_h_w = _dpoly(pr.h_w_coef, p_bar)
    d_h_s = _dpoly(pr.h_s_coef, p_bar)
    d_ts = _dpoly(pr.ts_coef, p_bar)

    V_st = pr.V_t - V_wt
    alpha_v = void_fraction(m_sr, p_bar, pr)
    q_dc = circulation_flow(alpha_v, p_bar, pr)

    # -- global mass and energy balances, solved jointly for (dp, dV_wt) -----
    #   d/dt[rho_s V_st + rho_w V_wt]                        = q_f - q_s
    #   d/dt[rho_s h_s V_st + rho_w h_w V_wt - p V_t + metal] = Q + q_f h_f - q_s h_s
    a11 = V_st * d_rho_s + V_wt * d_rho_w
    a12 = rho_w - rho_s
    a21 = (
        V_st * (rho_s * d_h_s + h_s * d_rho_s)
        + V_wt * (rho_w * d_h_w + h_w * d_rho_w)
        + pr.m_metal * pr.C_metal * d_ts
        - pr.V_t
    )
    a22 = rho_w * h_w - rho_s * h_s
    b1 = q_f - q_steam
    b2 = Q + q_f * pr.h_feedwater - q_steam * h_s
    det = a11 * a22 - a12 * a21
    det = jnp.where(jnp.abs(det) < 1e-6, 1e-6, det)
    dp = (b1 * a22 - a12 * b2) / det  # Pa/s
    dV_wt = (a11 * b2 - b1 * a21) / det

    # -- flashing -------------------------------------------------------------
    # A falling pressure lowers the saturation temperature, leaving the water
    # inventory superheated so that it boils. This is fast, and it is the
    # dominant swell mechanism.
    V_wr = pr.V_r * (1.0 - alpha_v)
    V_wd = jnp.maximum(V_wt - pr.V_dc - V_wr, 0.0)
    flash_gain = -(rho_w * pr.c_pw * d_ts / h_c) * dp
    q_flash_r = V_wr * flash_gain
    q_flash_d = V_wd * flash_gain

    # -- riser steam mass -----------------------------------------------------
    # The feedwater heat deficit is paid ONCE: (1 - f_cd) of it arrives at the
    # risers as subcooling and suppresses boiling there, and f_cd collapses
    # bubbles in the drum below. Charging it to both drives m_sd negative.
    subcool = (1.0 - pr.f_cd) * (q_f / jnp.maximum(q_dc, 1e-6)) * (h_w - pr.h_feedwater)
    q_ct = jnp.maximum(Q - q_dc * subcool, 0.0) / h_c
    carry_out = m_sr / pr.tau_sr
    dm_sr = q_ct + q_flash_r - carry_out

    # -- drum bubbles ---------------------------------------------------------
    q_cd = pr.f_cd * q_f * (h_w - pr.h_feedwater) / h_c
    dm_sd = pr.f_carry * carry_out + q_flash_d - m_sd / pr.T_d - q_cd

    # dp falls out of the balance in Pa/s; the state is carried in bar.
    return jnp.array([dp / PA_PER_BAR, dV_wt, dm_sr, dm_sd]), None


def compute_next_state(
    action_raw,
    state: BoilerDrumState,
    params: BoilerDrumParams,
    key: jax.Array,
    integration_method: str = "rk4_1",
):
    """``action_raw`` is ``[fuel, feedwater]``, each in [-1, 1]."""
    pr = params
    action_raw = jnp.atleast_1d(action_raw)
    Q_fuel = convert_raw_action_to_range(
        action_raw[0], min_action=0.0, max_action=pr.Q_max
    )
    q_feed = convert_raw_action_to_range(
        action_raw[1], min_action=0.0, max_action=pr.q_feed_max
    )

    # OU steam demand. The innovation is drawn from a key folded with
    # ``state.time`` so a caller passing a constant key -- which every rollout
    # helper in this repo does -- still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = pr.q_steam_noise_std * jnp.sqrt(2.0 * STEAM_OU_THETA * pr.delta_t)
    q_steam = (
        state.q_steam
        - STEAM_OU_THETA * (state.q_steam - pr.q_steam_nominal) * pr.delta_t
        + sigma * noise
    )
    q_steam = jnp.clip(q_steam, 0.0, 2.0 * pr.q_steam_nominal)

    _compute_velocity = partial(
        compute_velocity, action=(Q_fuel, q_feed), q_steam=q_steam, params=pr
    )
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.pressure, state.V_wt, state.m_sr, state.m_sd]),
        delta_t=pr.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    pressure = jnp.clip(new_positions[0], 20.0, 200.0)
    V_wt = jnp.clip(new_positions[1], 1.0, pr.V_t - 1.0)
    m_sr = jnp.maximum(new_positions[2], 0.0)
    m_sd = jnp.maximum(new_positions[3], 0.0)

    level = drum_level(V_wt, m_sr, m_sd, state.level_ref, pressure, pr)
    return (
        state.replace(
            pressure=pressure,
            V_wt=V_wt,
            m_sr=m_sr,
            m_sd=m_sd,
            level=level,
            q_steam=q_steam,
            Q_fuel=Q_fuel,
            q_feed=q_feed,
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
def get_obs(state: BoilerDrumState, params: BoilerDrumParams):
    """Plant instrumentation only.

    ``[level, pressure, q_steam, fuel_pct, feed_pct, target_level, target_pressure]``

    A boiler has a drum level gauge, a drum pressure transmitter and a steam
    flow meter -- the three measurements of classic three-element control --
    and knows its own fuel and feedwater demands. It does **not** measure riser
    void fraction or total water inventory, so ``m_sr``, ``m_sd`` and ``V_wt``
    are hidden. The void distribution is exactly what drives the inverse
    response the controller has to anticipate.

    The level target is a constant zero (normal water level) and is carried in
    the observation so that both targets are addressable the same way.
    """
    return jnp.array(
        [
            state.level,
            state.pressure,
            state.q_steam,
            100.0 * state.Q_fuel / params.Q_max,
            100.0 * state.q_feed / params.q_feed_max,
            jnp.zeros_like(state.level),
            state.target_pressure,
        ]
    )


def check_is_terminal(state: BoilerDrumState, params: BoilerDrumParams, xp=jnp):
    """High level carries water to the turbine; low level uncovers the tubes."""
    level_trip = xp.abs(state.level) >= params.level_trip
    pressure_trip = xp.logical_or(
        state.pressure <= params.pressure_min, state.pressure >= params.pressure_max
    )
    terminated = xp.logical_or(level_trip, pressure_trip)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: BoilerDrumState, params: BoilerDrumParams, xp=jnp):
    """Level and pressure tracking, minus fuel.

    Both terms are squared-normalised bands rather than Gaussians: they stay
    informative well outside the band, so a controller that is far off still
    sees a gradient back toward the setpoint.
    """
    level_err = xp.abs(state.level)
    level_score = xp.clip(1.0 - level_err / params.level_band, 0.0, 1.0) ** 2
    p_err = xp.abs(state.pressure - state.target_pressure)
    pressure_score = xp.clip(1.0 - p_err / params.pressure_band, 0.0, 1.0) ** 2
    fuel = state.Q_fuel / params.Q_max
    return 0.5 * level_score + 0.5 * pressure_score - params.fuel_weight * fuel


def steady_state(params: BoilerDrumParams, p_bar=None, q_steam=None):
    """Solve for the operating point that balances all four derivatives.

    Used by ``reset_env`` and by the tests. At steady state ``dp = 0``, so the
    flashing terms vanish and the problem collapses: ``m_sr`` is the root of a
    *scalar* balance (steam generated = steam carried out) and ``m_sd`` then
    follows explicitly. Solving the full 4-vector with Newton would also work
    but is far too expensive to sit in a reset path.

    ``V_wt`` is not solved for: the global balances vanish identically here for
    any water inventory, so where normal water level sits is a design choice,
    not something the steady state determines.
    """
    pr = params
    p_bar = pr.p_nominal if p_bar is None else p_bar
    q_steam = pr.q_steam_nominal if q_steam is None else q_steam
    h_s = steam_enthalpy(p_bar, pr)
    h_w = water_enthalpy(p_bar, pr)
    h_c = h_s - h_w
    rho_s = steam_density(p_bar, pr)

    # Firing that sustains this steam flow, with feedwater matching it.
    Q = q_steam * (h_s - pr.h_feedwater)
    q_f = q_steam
    V_wt = pr.V_wt_nominal

    def residual(m_sr):
        alpha_v = void_fraction(m_sr, p_bar, pr)
        q_dc = circulation_flow(alpha_v, p_bar, pr)
        subcool = (
            (1.0 - pr.f_cd) * (q_f / jnp.maximum(q_dc, 1e-6)) * (h_w - pr.h_feedwater)
        )
        q_ct = jnp.maximum(Q - q_dc * subcool, 0.0) / h_c
        return q_ct - m_sr / pr.tau_sr

    m_sr = pr.tau_sr * q_steam
    grad = jax.grad(residual)
    for _ in range(15):
        m_sr = m_sr - residual(m_sr) / grad(m_sr)

    alpha_v = void_fraction(m_sr, p_bar, pr)
    q_cd = pr.f_cd * q_f * (h_w - pr.h_feedwater) / h_c
    m_sd = pr.T_d * (pr.f_carry * m_sr / pr.tau_sr - q_cd)

    V_wd = V_wt - pr.V_dc - (1.0 - alpha_v) * pr.V_r
    level_ref = V_wd + m_sd / rho_s
    return V_wt, m_sr, m_sd, level_ref, Q


def circulation_ratio(params: BoilerDrumParams, p_bar=None, q_steam=None):
    """Circulation flow divided by steam flow -- 5-15 for a natural-circulation
    drum boiler, and the figure of merit that sets ``k_friction``."""
    pr = params
    p_bar = pr.p_nominal if p_bar is None else p_bar
    q_steam = pr.q_steam_nominal if q_steam is None else q_steam
    _, m_sr, _, _, _ = steady_state(pr, p_bar, q_steam)
    alpha_v = void_fraction(m_sr, p_bar, pr)
    return circulation_flow(alpha_v, p_bar, pr) / q_steam
