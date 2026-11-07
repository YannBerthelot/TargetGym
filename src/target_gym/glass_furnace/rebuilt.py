"""The rebuilt furnace: roof by conduction, a bath with currents, a real blanket.

docs/proposals/glass-furnace-task.md, "The rebuild". Sits beside the shipped plant
and reuses its firing, regenerators and clocks. Not registered: the shipped task and
its records stay as they are until the owner reads the judging.

Every number below is cited or labelled ours in PHYSICS.md, section "Rebuilt plant".

The roof is a five-cell brick, hot face first, finer near the hot face because that
is where a reversal's heat goes. The bath has three zones along the tank, blanket,
hot spot and throat side, each a surface and a deep layer split at the current's
reversal height, and the working end after them. The pull runs through the zones and
two buoyant cells run in them, sized by Poles's parallel-flow solution with no
calibration:

    hot surface -> blanket surface -> blanket deep -> hot deep -> hot surface
    hot surface -> throat surface  -> throat deep  -> hot deep -> hot surface
"""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import (
    EnvState,
    FieldReading,
    actuate,
    actuator_field,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.glass_furnace.env import (
    _GAS,
    FUEL_DEAD_TIME_STEPS,
    KELVIN_OFFSET,
    LHV_AR_RHO,
    M_PULL_AR_RHO,
    N_REGEN_NODES,
    N_SETPOINTS,
    SIGMA_SB,
    GlassFurnaceParams,
    _regenerator_flows,
    charge_rate_now,
    firing_fraction,
    get_target_from_schedule,
    glass_c_p,
    pull_rate,
    reversal_phase,
)
from target_gym.glass_furnace.env_jax import GlassFurnace
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

#: Roof cells, hot face first, m. Sums to the 450 mm crown.
ROOF_CELLS = (0.015, 0.030, 0.060, 0.120, 0.225)
N_ROOF = len(ROOF_CELLS)
#: Positions: roof, six bath layers (bs, bd, hs, hd, ts, td), working end, blanket
#: mass, then the regenerators.
BS, BD, HS, HD, TS, TD = range(N_ROOF, N_ROOF + 6)
WORK = N_ROOF + 6
BATCH = N_ROOF + 7
RA = N_ROOF + 8
RB = RA + N_REGEN_NODES
N_POS = RB + N_REGEN_NODES

#: Moves in an episode, of each kind.
N_MOVES = 6


@struct.dataclass
class RebuiltParams(GlassFurnaceParams):
    # ---- Roof ----
    k_brick: float = 2.48  # W/(m K), silica at service temperature
    rho_brick: float = 1870.0  # kg/m^3
    cp_brick: float = 1245.0  # J/(kg K), 3R per atom on SiO2
    U_roof_back: float = 1.70  # W/(m^2 K), 2 kW/m^2 at the 1200 C cold face
    tc_depth: float = 0.036  # m below the hot face, ours: a 10 min reading
    #: Added to the thermocouple to put it on the hot face's scale, K: the settled
    #: orbit's mean difference at the design firing (``settled``).
    tc_offset: float = 23.5

    # ---- Bath ----
    glass_mass: float = 625.0e3  # kg in the three melter zones
    tank_depth: float = 1.35  # m, 625 t over the melter at 2319 kg/m^3
    tank_width: float = 11.0  # m
    surface_fraction: float = 0.422  # above the reversal height z = 0.578
    rho_glass: float = 2319.0  # kg/m^3 at 1400 C
    beta_glass: float = 5.95e-5  # 1/K, cubical
    g_accel: float = 9.81
    U_bottom: float = 1.0  # W/(m^2 K)
    gas_view: float = 0.5  # what the floor sees of the flame, the rest crown
    current_scale: float = 0.05  # on Poles's flow: the first loop's 600 %
    back_flow_ratio: float = 2.6  # conditioning return over the pull

    # ---- Blanket ----
    rho_batch: float = 1400.0  # kg/m^3
    blanket_thickness: float = 0.20  # m
    blanket_solid: float = 0.85  # batch fraction of the blanket's footprint
    T_batch_top: float = 1200.0  # C, reactions complete
    H_melt: float = 2.2e6  # J/kg batch, heating and reactions to 1450 K
    cullet_gain: float = 0.352  # fraction of H_melt per unit of cullet ratio

    # ---- Moves, hidden ----
    cullet_move: float = 0.10  # largest change of the cullet ratio per move
    cullet_band: float = 0.05  # off the middle of the 20 to 30 % ratio
    thickness_move: float = 0.20  # largest fractional change per move
    thickness_band: float = 0.20
    move_ramp: float = 3600.0  # s

    # ---- Pull, the sourced +-2 % as two sigma ----
    m_pull_noise_std: float = 0.058

    max_steps_in_episode: int = 8640  # 3 days at 30 s


@struct.dataclass
class RebuiltState(EnvState):
    T_gas: float
    T_roof: jnp.ndarray
    T_bath: jnp.ndarray  # bs, bd, hs, hd, ts, td
    T_work: float
    T_rA: jnp.ndarray
    T_rB: jnp.ndarray
    m_batch: float

    T_crown: float  # the hot face, scored
    target_T_crown: float
    target_schedule: jnp.ndarray
    m_pull_disturbance: float
    lhv_deviation: float
    move_times: jnp.ndarray  # (2, N_MOVES) s, cullet then thickness
    move_sizes: jnp.ndarray  # (2, N_MOVES)

    T_crown_meas: float
    fuel_pipeline: jnp.ndarray
    fuel_flow: float
    T_air_preheat: float
    T_stack: float

    actuator_held: jax.Array = actuator_field(1)


def brick_temperature(T_roof, depth):
    """The brick at ``depth`` below the hot face, between cell centres. The hot
    face is the first cell, 15 mm, the finest this roof resolves."""
    cells = jnp.array(ROOF_CELLS)
    centres = jnp.cumsum(cells) - 0.5 * cells
    return jnp.interp(depth, centres, T_roof)


def glass_viscosity(T):
    """Pa s, the DGG X standard's VFT fit, T in C."""
    return 0.1 * 10.0 ** (-1.56151 + 4289.18 / (T - 250.737))


def k_eff_glass(T):
    """Effective conductivity of the melt, W/(m K), Pilon's fit, T in C."""
    K = T + KELVIN_OFFSET
    return 213.0 - 0.3698 * K + 1.658e-4 * K**2


def cell_flow(dT, T_deep, p):
    """A buoyant cell's return flow, kg/s: Poles's free-top core solution,
    0.00542 g beta (dT / l) h^4 / nu per unit width, ``l`` the zone's length."""
    length = p.A_melt / p.tank_width / 3.0
    nu = glass_viscosity(T_deep) / p.rho_glass
    per_width = (
        0.00542 * p.g_accel * p.beta_glass * (dT / length) * p.tank_depth**4 / nu
    )
    return p.current_scale * per_width * p.tank_width * p.rho_glass


def moved(time_s, times, sizes, ramp, band):
    """A hidden quantity's offset: each move a ramp of ``ramp`` seconds, the
    total held inside ``band``."""
    total = jnp.sum(sizes * jnp.clip((time_s - times) / ramp, 0.0, 1.0))
    return jnp.clip(total, -band, band)


def footprint(m_batch, thickness, p):
    """The blanket's footprint, m^2, batch and the free glass between its logs."""
    return jnp.maximum(m_batch, 0.0) / (p.rho_batch * thickness * p.blanket_solid)


def surfaces(m_batch, thickness, p):
    """Open glass in each zone and the batch's area, m^2. The blanket fills the
    blanket zone first and runs into the hot-spot zone past it."""
    zone = p.A_melt / 3.0
    foot = footprint(m_batch, thickness, p)
    in_b = jnp.minimum(foot, zone)
    in_h = jnp.clip(foot - zone, 0.0, zone)
    open_b = zone - p.blanket_solid * in_b
    open_h = zone - p.blanket_solid * in_h
    return open_b, open_h, zone, p.blanket_solid * (in_b + in_h)


def flame(T_guess, x, m_fuel, T_air, thickness, p):
    """Quasi-steady flame temperature, as ``env.solve_T_gas``, on this floor."""
    m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
    m_gas = m_fuel + m_air
    Q_in = m_fuel * p.LHV + m_air * p.c_p_air * (T_air - p.T_ambient)
    open_b, open_h, open_t, batch = surfaces(x[BATCH], thickness, p)
    areas = jnp.array([p.A_crown, open_b, open_h, open_t, batch, p.A_work])
    temps = jnp.stack(
        [x[0], x[BS], x[HS], x[TS], jnp.asarray(p.T_batch_top, x.dtype), x[WORK]]
    )
    # The flame radiates to the crown in full and to a floor by what the floor sees.
    views = jnp.array([1.0, p.gas_view, p.gas_view, p.gas_view, p.gas_view, p.gas_view])
    A_rad = jnp.sum(views * areas)
    A_conv = areas.sum()
    sink_K4 = jnp.sum(views * areas * (temps + KELVIN_OFFSET) ** 4)
    sink_T = jnp.sum(areas * temps)

    def body(T, _):
        T_K = T + KELVIN_OFFSET
        f = (
            Q_in
            - p.eps_rad * SIGMA_SB * (A_rad * T_K**4 - sink_K4)
            - p.h_conv * (A_conv * T - sink_T)
            - m_gas * p.c_p_gas * (T - p.T_ambient)
        )
        df = (
            -4.0 * p.eps_rad * SIGMA_SB * A_rad * T_K**3
            - p.h_conv * A_conv
            - m_gas * p.c_p_gas
        )
        return T - f / df, None

    T_gas, _ = jax.lax.scan(body, T_guess, xs=None, length=6)
    return jnp.clip(T_gas, p.T_ambient, 4000.0)


def regenerators(x, T_gas, m_fuel, phase, p):
    m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
    m_gas = m_fuel + m_air
    A_is_air = phase < 0.5
    T_rA, T_rB = x[RA:RB], x[RB:]
    QA, T_air_A, T_st_A = _regenerator_flows(T_rA, T_gas, m_air, m_gas, A_is_air, p)
    QB, T_air_B, T_st_B = _regenerator_flows(
        T_rB, T_gas, m_air, m_gas, jnp.logical_not(A_is_air), p
    )
    return (
        QA,
        QB,
        jnp.where(A_is_air, T_air_A, T_air_B),
        jnp.where(A_is_air, T_st_B, T_st_A),
    )


def velocity(x, m_fuel, m_pull, phase, T_gas, H_melt, thickness, params, position_time):
    """Right-hand side, ``x`` the positions."""
    p = params
    QA, QB, _, _ = regenerators(x, T_gas, m_fuel, phase, p)

    def floor(area, T):
        """What a floor area takes from the combustion space, W."""
        K4 = (T + KELVIN_OFFSET) ** 4
        flame_K4 = (T_gas + KELVIN_OFFSET) ** 4
        roof_K4 = (x[0] + KELVIN_OFFSET) ** 4
        from_gas = p.eps_rad * SIGMA_SB * p.gas_view * (flame_K4 - K4)
        from_gas = from_gas + p.h_conv * (T_gas - T)
        from_roof = p.eps_rad * SIGMA_SB * (1.0 - p.gas_view) * (roof_K4 - K4)
        return area * from_gas, area * from_roof

    open_b, open_h, open_t, batch = surfaces(x[BATCH], thickness, p)
    gas_b, roof_b = floor(open_b, x[BS])
    gas_h, roof_h = floor(open_h, x[HS])
    gas_t, roof_t = floor(open_t, x[TS])
    gas_w, roof_w = floor(p.A_work, x[WORK])
    gas_batch, roof_batch = floor(batch, p.T_batch_top)

    # ---- Roof: conduction between cells, the back loses through its insulation
    cells = jnp.array(ROOF_CELLS)
    gaps = 0.5 * (cells[1:] + cells[:-1])
    across = p.k_brick * (x[: N_ROOF - 1] - x[1:N_ROOF]) / gaps
    A_flame = (
        p.eps_rad
        * SIGMA_SB
        * ((T_gas + KELVIN_OFFSET) ** 4 - (x[0] + KELVIN_OFFSET) ** 4)
    )
    into_face = (
        A_flame * p.A_crown
        + p.h_conv * p.A_crown * (T_gas - x[0])
        - roof_b - roof_h - roof_t - roof_w - roof_batch
    ) / p.A_crown  # fmt: skip
    # The back cell's surface sits half a cell past its centre.
    back = (x[N_ROOF - 1] - p.T_ambient) / (
        1.0 / p.U_roof_back + 0.5 * cells[-1] / p.k_brick
    )
    fluxes = jnp.concatenate([into_face[None], across, back[None]])
    d_roof = (fluxes[:-1] - fluxes[1:]) / (p.rho_brick * p.cp_brick * cells)

    # ---- Bath: the pull and two cells, upwind
    r1 = cell_flow(x[HS] - x[BS], 0.5 * (x[BD] + x[HD]), p)
    r2 = cell_flow(x[HS] - x[TS], 0.5 * (x[TD] + x[HD]), p)

    # ---- Blanket: fed by the charger, melted from above and below
    interface = k_eff_glass(0.5 * (x[BS] + p.T_batch_top)) / (
        0.5 * p.surface_fraction * p.tank_depth
    )
    zone = p.A_melt / 3.0
    in_b = jnp.minimum(batch, p.blanket_solid * zone)
    Q_under_b = interface * in_b * (x[BS] - p.T_batch_top)
    Q_under_h = interface * (batch - in_b) * (x[HS] - p.T_batch_top)
    Q_batch = gas_batch + roof_batch + Q_under_b + Q_under_h
    melt_rate = jnp.maximum(Q_batch, 0.0) / H_melt
    d_batch = charge_rate_now(m_pull, position_time, p) - melt_rate
    glass_in = melt_rate * p.batch_yield

    def cp(T):
        return glass_c_p(T, p)

    def exchange(i, j):
        """Heat rising by conduction between the layers of a zone, W."""
        mean = 0.5 * (x[i] + x[j])
        return k_eff_glass(mean) * zone * (x[j] - x[i]) / (0.5 * p.tank_depth)

    hs_b, hs_h, hs_t = exchange(BS, BD), exchange(HS, HD), exchange(TS, TD)
    bottom = p.U_bottom * zone

    thru = jnp.maximum(m_pull, 0.0)
    surface_mass = p.glass_mass * p.surface_fraction / 3.0
    deep_mass = p.glass_mass * (1.0 - p.surface_fraction) / 3.0

    # ---- Directed flows, kg/s, each carrying its source's enthalpy to its target.
    # The conditioning return comes back along the bottom past the waist, so the
    # throat zone's layers trade whatever the two currents leave over. A cell whose
    # far end is the hotter runs the other way.
    r1, r2_back = jnp.maximum(r1, 0.0), jnp.maximum(-r2, 0.0)
    r2 = jnp.maximum(r2, 0.0)
    r3 = p.back_flow_ratio * thru
    down, up = jnp.maximum(r2 - r3, 0.0), jnp.maximum(r3 - r2, 0.0)
    flows = (
        (HS, BS, r1),
        (BS, BD, r1 + thru),
        (BD, HD, r1 + thru),
        (HD, HS, r1 + r2 + thru),
        (HS, TS, r2 + thru),
        (TS, TD, down),
        (TD, TS, up),
        (TD, HD, r2),
        (TS, HS, r2_back),
        (HS, HD, r2_back),
        (HD, TD, r2_back),
        (TD, TS, r2_back),
        (TS, WORK, thru + r3),
        (WORK, TD, r3),
    )
    heat = jnp.zeros(N_POS, x.dtype)
    for source, sink, flow in flows:
        heat = heat.at[sink].add(flow * cp(x[source]) * (x[source] - x[sink]))

    dbs = (
        gas_b + roof_b + hs_b - Q_under_b + heat[BS]
        + glass_in * cp(p.T_batch_top) * (p.T_batch_top - x[BS])
    ) / (surface_mass * cp(x[BS]))  # fmt: skip
    dbd = (-hs_b - bottom * (x[BD] - p.T_ambient) + heat[BD]) / (deep_mass * cp(x[BD]))
    dhs = (gas_h + roof_h + hs_h - Q_under_h + heat[HS]) / (surface_mass * cp(x[HS]))
    dhd = (-hs_h - bottom * (x[HD] - p.T_ambient) + heat[HD]) / (deep_mass * cp(x[HD]))
    dts = (gas_t + roof_t + hs_t + heat[TS]) / (surface_mass * cp(x[TS]))
    dtd = (-hs_t - bottom * (x[TD] - p.T_ambient) + heat[TD]) / (deep_mass * cp(x[TD]))
    dwork = (
        gas_w + roof_w
        - p.U_wall * p.A_wall_work * (x[WORK] - p.T_ambient)
        - p.UA_work_cooling * (x[WORK] - p.T_ambient)
        + heat[WORK]
    ) / (p.C_work / p.c_p_glass_a * cp(x[WORK]))  # fmt: skip

    UA_node = p.U_regen * p.A_regen / N_REGEN_NODES
    d_rA = (QA - UA_node * (x[RA:RB] - p.T_ambient)) / p.C_regen_node
    d_rB = (QB - UA_node * (x[RB:] - p.T_ambient)) / p.C_regen_node
    return (
        jnp.concatenate(
            [
                d_roof,
                jnp.stack([dbs, dbd, dhs, dhd, dts, dtd, dwork, d_batch]),
                d_rA,
                d_rB,
            ]
        ),
        None,
    )


def positions_of(s):
    return jnp.concatenate(
        [s.T_roof, s.T_bath, jnp.stack([s.T_work, s.m_batch]), s.T_rA, s.T_rB]
    )


def state_moves(state, time, p):
    """The hidden batch's melting heat and thickness at a step."""
    t = time * p.delta_t
    cullet = moved(
        t, state.move_times[0], state.move_sizes[0], p.move_ramp, p.cullet_band
    )
    thick = moved(
        t, state.move_times[1], state.move_sizes[1], p.move_ramp, p.thickness_band
    )
    return p.H_melt * (1.0 - p.cullet_gain * cullet), p.blanket_thickness * (
        1.0 + thick
    )


def plant_step(state, m_fuel, new_disturbance, new_lhv, p, method):
    """The physics of one step from ``state``: ``(positions, T_gas, phase, T_air,
    T_stack)``. Kept apart so a planner can run the same step on its own paths."""
    phase = reversal_phase(state.time, p)
    x = positions_of(state)
    H_melt, thickness = state_moves(state, state.time, p)
    _, _, T_air, _ = regenerators(x, state.T_gas, m_fuel, phase, p)
    T_gas = flame(
        state.T_gas, x, m_fuel, T_air, thickness,
        p.replace(LHV=p.LHV * (1.0 + new_lhv)),
    )  # fmt: skip
    new, _ = integrate_dynamics(
        positions=x,
        delta_t=p.delta_t,
        compute_velocity=partial(
            velocity,
            m_fuel=m_fuel,
            m_pull=pull_rate(new_disturbance, p),
            phase=phase,
            T_gas=T_gas,
            H_melt=H_melt,
            thickness=thickness,
            params=p,
            position_time=state.time,
        ),
        method=method,
    )
    _, _, T_air_new, T_stack_new = regenerators(new, T_gas, m_fuel, phase, p)
    return new, T_gas, phase, T_air_new, T_stack_new


def advance(
    state, new, T_gas, m_fuel, new_disturbance, new_lhv, T_air, T_stack, cmd, p
):
    """The state after ``plant_step``."""
    alpha = 1.0 - jnp.exp(-p.delta_t / jnp.maximum(p.tau_thermocouple, 1e-6))
    tc = brick_temperature(new[:N_ROOF], p.tc_depth) + p.tc_offset
    new_time = state.time + 1
    return state.replace(
        T_gas=T_gas,
        T_roof=new[:N_ROOF],
        T_bath=new[BS:WORK],
        T_work=new[WORK],
        m_batch=jnp.maximum(new[BATCH], 0.0),
        T_rA=new[RA:RB],
        T_rB=new[RB:],
        T_crown=new[0],
        target_T_crown=get_target_from_schedule(state.target_schedule, new_time, p),
        m_pull_disturbance=new_disturbance,
        lhv_deviation=new_lhv,
        T_crown_meas=state.T_crown_meas + alpha * (tc - state.T_crown_meas),
        fuel_pipeline=jnp.concatenate([state.fuel_pipeline[1:], cmd[None]]),
        fuel_flow=m_fuel,
        T_air_preheat=T_air,
        T_stack=T_stack,
        time=new_time,
    )


def next_noise(state, key, p):
    """The pull and the gas after this step, drawn as ``env.compute_next_state``."""
    innovation = (
        jax.random.normal(jax.random.fold_in(key, state.time))
        * p.m_pull_noise_std
        * jnp.sqrt(1.0 - M_PULL_AR_RHO**2)
    )
    gas_key = jax.random.fold_in(jax.random.fold_in(key, _GAS), state.time)
    lhv = LHV_AR_RHO * state.lhv_deviation + (
        jax.random.normal(gas_key) * p.lhv_noise_std * jnp.sqrt(1.0 - LHV_AR_RHO**2)
    )
    return M_PULL_AR_RHO * state.m_pull_disturbance + innovation, lhv


def next_state(raw, state, p, key, method):
    cmd = convert_raw_action_to_range(raw, min_action=p.fuel_min, max_action=p.fuel_max)
    m_fuel = state.fuel_pipeline[0] * firing_fraction(state.time, p)
    disturbance, lhv = next_noise(state, key, p)
    new, T_gas, _, T_air, T_stack = plant_step(
        state, m_fuel, disturbance, lhv, p, method
    )
    return advance(state, new, T_gas, m_fuel, disturbance, lhv, T_air, T_stack, cmd, p)


def is_tripped(state, p, xp=jnp):
    crown = xp.logical_or(
        state.T_crown <= p.T_crown_min, state.T_crown >= p.T_crown_max
    )
    bath = xp.stack([state.T_bath.min(), state.T_work])
    low = xp.any(bath <= p.T_glass_min)
    high = xp.logical_or(
        state.T_bath.max() >= p.T_glass_max, state.T_work >= p.T_glass_max
    )
    return xp.logical_or(crown, xp.logical_or(low, high))


def reward_terms(state, p, xp=jnp, tripped=None):
    crown = state.T_crown
    if tripped is None:
        tripped = is_tripped(state, p, xp)
    else:
        crown = xp.where(p.sensor_noise_scale > 0, state.T_crown_meas, crown)
    terms = {
        "tracking": R.tracking_cost(
            state.target_T_crown - crown, p.e_floor, p.e_tol, p.tracking_exponent, xp
        ),
        "running": R.running_cost(state.fuel_flow, p.c_hold, p.running_weight, xp),
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


#: The orbit at two firings, from ``scripts/glass_rebuilt_settle.py``: the hot
#: face's mean over a cycle, the raw action, then the positions where an episode
#: starts, disturbances off.
SETTLED_LOW = (1568.0, -0.05, 1532.37, 1548.56, 1513.39, 1440.3, 1300.42, 1510.07, 1508.65, 1553.58, 1514.39, 1555.55, 1480.36, 1452.7, 10872.2, 1370.42, 1110.46, 830.94, 538.1, 1198.24, 926.31, 636.25, 334.69)  # fmt: skip
SETTLED_HIGH = (1623.51, 0.2, 1588.69, 1603.33, 1566.83, 1490.99, 1345.84, 1572.63, 1571.57, 1609.69, 1576.45, 1611.66, 1540.22, 1508.32, 8928.7, 1419.08, 1151.05, 862.33, 559.94, 1234.42, 953.29, 653.21, 341.67)  # fmt: skip


class RebuiltFurnace(GlassFurnace):
    """Actions and observations as the shipped furnace."""

    registry_name: str = "glass_furnace"
    readings: tuple = (
        FieldReading("T_crown_meas", sigma=0.36),
        FieldReading("T_air_preheat", sigma=0.32),
        FieldReading("fuel_flow", sigma=3e-4),
    )

    @property
    def default_params(self) -> RebuiltParams:
        return RebuiltParams()

    def compute_reward(self, state, params):
        return R.total(reward_terms(state, params))

    def reward_terms(self, state, params, tripped=None):
        return reward_terms(state, params, tripped=tripped)

    def is_terminated(self, state, params):
        return is_tripped(state, params)

    def step_env(self, key, state, action, params=None):
        params = self.default_params if params is None else params
        action, state = actuate(self, state, jnp.reshape(action, (1,)), params)
        new = next_state(
            action.reshape(()), state, params, key, self.integration_method
        )
        new, reward, terminated, info = finish_step(self, key, state, new, params)
        return self.get_obs(new, params), new, reward, terminated, info

    @measured_obs
    def get_obs(self, state, params=None):
        params = self.default_params if params is None else params
        return jnp.array(
            [
                state.T_crown_meas,
                state.T_air_preheat,
                100.0 * state.fuel_flow / params.fuel_max,
                reversal_phase(state.time, params),
                state.target_T_crown,
                pull_rate(state.m_pull_disturbance, params),
            ]
        )

    def settled(self, T_crown):
        """The orbit at a hot face's mean: ``(raw action, positions)``."""
        low, high = jnp.array(SETTLED_LOW), jnp.array(SETTLED_HIGH)
        point = low + (T_crown - low[0]) / (high[0] - low[0]) * (high - low)
        return point[1], point[2:]

    @canonical_reset
    def reset_env(self, key, params=None):
        p = self.default_params if params is None else params
        key, schedule_key, crown_key, gas_key, move_key = jax.random.split(key, 5)
        start_key, trim_key = jax.random.split(schedule_key)
        start = jax.random.uniform(
            start_key,
            minval=p.target_T_crown_start_range[0],
            maxval=p.target_T_crown_start_range[1],
        )
        trims = jax.random.uniform(
            trim_key, (N_SETPOINTS,), minval=-p.target_T_crown_trim,
            maxval=p.target_T_crown_trim,
        )  # fmt: skip
        schedule = jnp.clip(
            start + jnp.cumsum(trims.at[0].set(0.0)),
            p.target_T_crown_range[0],
            p.target_T_crown_range[1],
        )
        target = get_target_from_schedule(schedule, 0, p)
        crown = jnp.clip(
            target
            + jax.random.uniform(
                crown_key,
                minval=-p.initial_T_crown_offset,
                maxval=p.initial_T_crown_offset,
            ),
            p.initial_T_crown_range[0],
            p.initial_T_crown_range[1],
        )
        k1, k2 = jax.random.split(move_key)
        horizon = p.max_steps_in_episode * p.delta_t
        times = jax.random.uniform(k1, (2, N_MOVES), maxval=horizon)
        scale = jnp.array([[p.cullet_move], [p.thickness_move]])
        sizes = scale * jax.random.uniform(k2, (2, N_MOVES), minval=-1.0, maxval=1.0)

        lhv = p.lhv_noise_std * jax.random.normal(gas_key)
        raw, x = self.settled(crown)
        fuel = convert_raw_action_to_range(
            raw, min_action=p.fuel_min, max_action=p.fuel_max
        ) / (1.0 + lhv)
        T_gas = x[0] + p.initial_T_gas_offset
        _, _, T_air, T_stack = regenerators(x, T_gas, fuel, 0.0, p)
        tc = brick_temperature(x[:N_ROOF], p.tc_depth) + p.tc_offset
        state = RebuiltState(
            time=0,
            T_gas=T_gas,
            T_roof=x[:N_ROOF],
            T_bath=x[BS:WORK],
            T_work=x[WORK],
            T_rA=x[RA:RB],
            T_rB=x[RB:],
            m_batch=x[BATCH],
            T_crown=x[0],
            target_T_crown=target,
            target_schedule=schedule,
            m_pull_disturbance=jnp.zeros(()),
            lhv_deviation=lhv,
            move_times=times,
            move_sizes=sizes,
            T_crown_meas=tc,
            fuel_pipeline=jnp.full((FUEL_DEAD_TIME_STEPS,), fuel),
            fuel_flow=fuel,
            T_air_preheat=T_air,
            T_stack=T_stack,
        )
        return self.get_obs(state), state

    def observation_space(self, params) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)
