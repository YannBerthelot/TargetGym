"""Attempt 3's environment: the slow plant, the refiner's own cooling loop, the walks that
move the plant with the valve held, and a reward in money. Functional and unregistered.

The score is the operator's job (REALITY.md, section 6): fuel at its price; glass charged
while the canal is under its target, or not fined, is lost to cullet at its price, on the
pull actually charged. The silica crown's 1620 C is the plant's own override (owner,
2026-11-08; a known deviation, since no source shows such a protection): a limit control
trims the valve so the hottest face stays under it, and the agent sees the valve it got.
The refiner's cooling loop holds the canal just above its target, as a plant's local loop
does, and the agent sees how hard it works.

The agent reads what operators read: crown and bottom couples, the canal, the fining
share only at the cold end a transit later, the cullet ratio the batch plant holds, the
batch line on its camera's cadence, and the pull plan as far ahead as its notice. Gas
quality, blanket thickness and sulfate stay hidden. The episode is established operation:
the walks start where the seed puts them, and a planned pull change or a gas swing comes
at a time the seed draws inside its window.
"""

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.glass_furnace import physical as P

WALKS = ("gas", "pull", "cullet", "thickness", "sulfate")
OBS_NAMES = (
    "crown couple, K",
    "floor under bay 3, K",
    "floor at the melter's end, K",
    "canal, K",
    "fining share read at the cold end",
    "refiner cooling, share of its capacity",
    "air preheat, C",
    "stack, C",
    "fuel, kg/s",
    "pull, kg/s",
    "pull planned, kg/s",
    "cullet ratio",
    "batch line, m",
)
TEMPERATURES = (0, 1, 2, 3, 6, 7)  # the readings that carry the couples' noise


@struct.dataclass
class EnvParams:
    plant: P.PhysicalParams = P.PhysicalParams()
    grid: P.Grid = struct.field(pytree_node=False, default=P.Grid())
    gas_price: float = 60.5  # EUR/MWh, Eurostat, EU non-household, H2 2025
    lost_glass_price: float = 600.0  # EUR/t, ours (owner's pick); no source found
    pull: float = 5.79  # kg/s, 500 t/d, W2000b
    cullet: float = 0.25
    thickness: float = 0.2  # m, blanket as laid
    sulfate: float = 0.01  # Na2SO4 per kg batch, Laimbock
    canal_target: float = 1373.15  # K: "minimum 1100 C before entering the bath"
    # Fixed before any run: glass passes when, flow-weighted at the canal, it has on
    # average passed the middle of Beerkens' sulfate fining band, 1420 to 1500 C [S31].
    # Ours, inside that band.
    fining_pass: float = 0.5
    crown_high: float = 1893.15  # K, 1620 C: silica degrades above, SEFPRO
    fuel_min: float = 0.25  # kg/s, the valve's range
    fuel_max: float = 0.75
    # The crown override: PI on the hottest face held under the limit by a margin (ours),
    # by SIMC on the fuel step's 808 K per kg/s and 4.1 h lag, closed loop in 30 min.
    override_margin: float = 5.0
    override_gain: float = 0.0101  # kg/s per K
    override_ti: float = 7200.0  # s
    # No capacity was found. Ours: twice what holds the canal at the design point, so
    # the coolers bind only far from it. Set from the settle script's output.
    cooling_max: float = 1.54e6
    cooling_above: float = 1.0  # K over the target the loop holds, ours
    cooling_gain: float = 2.619e5  # W/K, SIMC on a 5 % cooling step (scripts)
    cooling_ti: float = 7680.0  # s, same recipe
    # gas: ENTSOG's 4 to 6 MJ/m3 swings on about 50, read as four spreads, daily;
    # pull: +-2 % wander (US11261117B2) as two spreads; cullet, blanket: the owner's
    # +-10 points and +-20 % as two spreads; sulfate: ours.
    walk_sd: tuple = (0.02, 0.01, 0.05, 0.10, 0.17)
    walk_hours: tuple = (24.0, 12.0, 12.0, 12.0, 12.0)
    noise_sd: float = 0.36  # K, the shipped crown couple's; ours for the others
    # The cold-end reading: the fining share a transit late, with noise. Ours, family
    # range 14 to 20 min: a 50 m bath and a 100 m lehr at 7.5 to 10.4 m/min of ribbon
    # (glass-sources-6). Its noise is ours; no scanner's counting noise was found.
    quality_delay_h: float = 0.3
    quality_noise: float = 0.005
    batch_line_every_h: float = 0.5  # a camera every 30 min, Klemiato 2021 [S23]
    # A planned pull change: its size (fraction, 0 in the normal scenario), its window
    # for the start time the seed draws, its ramp and its notice. Float sizes, notice and
    # ramps were not found; sizes are ours inside the family's 2 % float wander to the
    # 29 to 42 % of a container plant's two pulls (glass-sources-6); notice and ramp ours.
    pull_change: float = 0.0
    pull_change_window_h: tuple = (24.0, 96.0)
    pull_ramp_h: float = 2.0
    pull_notice_h: float = 8.0
    # An unannounced gas quality swing (fraction of heating value; 0 in the normal
    # scenario), over a ramp, at a time the seed draws in the same window.
    gas_swing: float = 0.0
    gas_swing_h: float = 6.0


@struct.dataclass
class EnvState:
    fields: P.Fields
    walk: jnp.ndarray  # deviations, in WALKS order: hidden
    cooling: jnp.ndarray  # W, the refiner loop's output
    integral: jnp.ndarray  # W, its integral
    fuel: jnp.ndarray  # kg/s on the valve after the override, last step
    limit: jnp.ndarray  # the override's integral, kg/s, reset to the valve
    pull: jnp.ndarray  # kg/s charged, last step: measured
    cullet: jnp.ndarray  # cullet ratio charged, last step: measured
    quality: jnp.ndarray  # the fining shares on their way to the cold end, oldest first
    line: jnp.ndarray  # m, the batch line as last read
    events: jnp.ndarray  # h: when the pull change and the gas swing start
    time: jnp.ndarray


def _decay(ep):
    dt = ep.plant.delta_t
    phi = jnp.exp(-dt / (3600.0 * jnp.asarray(ep.walk_hours)))
    return phi, jnp.asarray(ep.walk_sd) * jnp.sqrt(1.0 - phi**2)


def walk_step(walk, key, ep):
    """One step of each walk: first order, stationary at its spread."""
    phi, kick = _decay(ep)
    return phi * walk + kick * jax.random.normal(key, walk.shape)


def walks(key, n, ep):
    """``n`` steps of the walks from a stationary start, ``(n, len(WALKS))``."""
    k0, k1 = jax.random.split(key)
    start = jnp.asarray(ep.walk_sd) * jax.random.normal(k0, (len(WALKS),))
    keys = jax.random.split(k1, n)
    return jax.lax.scan(lambda w, k: (walk_step(w, k, ep),) * 2, start, keys)[1]


def reward_terms(fuel, canal, fining, pull, ep):
    """Costs of one step, EUR: ``fuel`` kg/s burnt at nominal heating value, and the
    step's charged ``pull`` if the canal is under its target or the glass under-fined.
    """
    dt = ep.plant.delta_t
    lost = (canal < ep.canal_target) | (fining < ep.fining_pass)
    return dict(
        fuel=fuel * dt * ep.plant.LHV / 3.6e9 * ep.gas_price,
        lost_glass=jnp.where(lost, pull * dt / 1000.0 * ep.lost_glass_price, 0.0),
    )


def override(asked, hot, limit, ep):
    """The plant's crown protection: ``(valve, integral)``. A PI loop on the hottest face,
    held under the top limit, selects against the asked valve; its integral is reset
    toward the valve chosen, so it does not wind up."""
    cap = limit + ep.override_gain * (ep.crown_high - ep.override_margin - hot)
    u = jnp.clip(jnp.minimum(asked, cap), ep.fuel_min, ep.fuel_max)
    return u, limit + ep.plant.delta_t / ep.override_ti * (u - limit)


def _ramp(hours, start, length):
    return jnp.clip((hours - start) / jnp.maximum(length, 1e-9), 0.0, 1.0)


def planned_pull(hours, events, ep):
    """The pull the plan sets at ``hours``, kg/s, before the wander."""
    return ep.pull * (1.0 + ep.pull_change * _ramp(hours, events[0], ep.pull_ramp_h))


def batch_line(fields, ep):
    """m from the back wall that the blanket covers, as a camera would read it."""
    a = P.layout(ep.grid)
    return jnp.sum(P.cover(fields.m) * a.dx[: a.nm])


def observe(state, ep):
    """The readings, noise-free. Nothing hidden: no walk, no glass inside the tank."""
    p, s = ep.plant, state.fields
    read = P.sensors(s, ep.grid, p)
    fired = state.fuel * P.mean_firing(p)
    m_air = p.AFR * (1.0 + p.excess_air) * fired
    flue = jnp.mean(s.T_gas) - P.KELVIN_OFFSET
    _, air, stack = P.averaged_stack(s.T_r, flue, m_air, fired + m_air, p)
    hours = state.time * p.delta_t / 3600.0
    return jnp.stack(
        [
            read["crown"],
            read["floor_bay"],
            read["floor"],
            read["canal"],
            state.quality[0],
            state.cooling / ep.cooling_max,
            air,
            stack,
            state.fuel,
            state.pull,
            planned_pull(hours + ep.pull_notice_h, state.events, ep),
            state.cullet,
            state.line,
        ]
    )


def _delay_steps(ep):
    """Steps a share takes to reach the cold end; the queue holds one more, the oldest
    being what the scanner reads now."""
    return max(int(round(ep.quality_delay_h * 3600.0 / ep.plant.delta_t)), 1) + 1


def reset(key, fields, ep, fuel=0.45, cooling=0.0):
    """Start in established operation on a settled plant: the walks where the seed puts
    them at their stationary spread, the loops where they were, the events' times drawn
    inside their window."""
    k_walk, k_events = jax.random.split(key)
    walk = jnp.asarray(ep.walk_sd) * jax.random.normal(k_walk, (len(WALKS),))
    lo, hi = ep.pull_change_window_h
    events = jax.random.uniform(k_events, (2,), minval=lo, maxval=hi)
    c = jnp.asarray(cooling, float)
    share = P.sensors(fields, ep.grid, ep.plant)["fining"]
    state = EnvState(
        fields=fields,
        walk=walk,
        cooling=c,
        integral=c,
        fuel=jnp.asarray(fuel, float),
        limit=jnp.asarray(fuel, float),
        pull=planned_pull(0.0, events, ep) * (1.0 + walk[WALKS.index("pull")]),
        cullet=ep.cullet + walk[WALKS.index("cullet")],
        quality=jnp.full((_delay_steps(ep),), share),
        line=batch_line(fields, ep),
        events=events,
        time=jnp.asarray(0),
    )
    return observe(state, ep), state


def step(key, state, fuel, ep):
    """One step: ``(obs, state, reward, costs)``. ``fuel`` kg/s asked of the valve."""
    k_walk, k_noise = jax.random.split(key)
    hot = P.sensors(state.fields, ep.grid, ep.plant)["hot"]
    fuel, limit = override(jnp.asarray(fuel, float), hot, state.limit, ep)
    walk = walk_step(state.walk, k_walk, ep)
    gas, pull, cullet, thick, sulfate = (walk[i] for i in range(len(WALKS)))
    hours = (state.time + 1) * ep.plant.delta_t / 3600.0
    charge = planned_pull(hours, state.events, ep) * (1.0 + pull)
    gas = gas + ep.gas_swing * _ramp(hours, state.events[1], ep.gas_swing_h)
    p = ep.plant.replace(sulfate_batch=ep.sulfate * (1.0 + sulfate))
    fields, books = P.plant_step(
        state.fields, fuel, charge, ep.cullet + cullet, ep.thickness * (1.0 + thick),
        state.cooling, ep.grid, p, lhv=gas,
    )  # fmt: skip
    read = P.sensors(fields, ep.grid, p)

    # The refiner's own loop: PI on the canal, clamped to what the coolers can take.
    e = read["canal"] - (ep.canal_target + ep.cooling_above)
    integral = jnp.clip(
        state.integral + ep.cooling_gain * e * p.delta_t / ep.cooling_ti,
        0.0,
        ep.cooling_max,
    )
    cooling = jnp.clip(ep.cooling_gain * e + integral, 0.0, ep.cooling_max)

    costs = reward_terms(
        books["fuel"] * (1.0 + gas), read["canal"], read["fining"], charge, ep
    )
    every = max(int(round(ep.batch_line_every_h * 3600.0 / p.delta_t)), 1)
    k_temp, k_quality = jax.random.split(k_noise)
    late = read["fining"] + ep.quality_noise * jax.random.normal(k_quality)
    new = EnvState(
        fields=fields, walk=walk, cooling=cooling, integral=integral, fuel=fuel,
        limit=limit, pull=charge, cullet=ep.cullet + cullet,
        quality=jnp.concatenate([state.quality[1:], late[None]]),
        line=jnp.where(state.time % every == 0, batch_line(fields, ep), state.line),
        events=state.events, time=state.time + 1,
    )  # fmt: skip
    obs = observe(new, ep)
    noise = ep.noise_sd * jax.random.normal(k_temp, obs.shape)
    mask = jnp.zeros(obs.shape).at[jnp.asarray(TEMPERATURES)].set(1.0)
    return obs + mask * noise, new, -sum(costs.values()), costs
