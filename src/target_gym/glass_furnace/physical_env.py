"""Attempt 3's environment: the slow plant, the refiner's own cooling loop, the walks that
move the plant with the valve held, and a reward in money. Functional and unregistered.

The score is the owner's, 2026-11-07: fuel at its price; glass reaching the canal under
its target, or not fined, is lost to cullet at its price. The silica crown's limits are the
plant's own override (owner, 2026-11-08): a limit control trims the valve so the hottest
face stays inside them, and the agent sees the valve it got. The refiner's cooling loop
holds the canal just above its target, as a plant's local loop does, and the agent sees
how hard it works.
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
    "fining share at the canal",
    "refiner cooling, share of its capacity",
    "air preheat, C",
    "stack, C",
    "fuel, kg/s",
    "pull, kg/s",
)


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
    fining_pass: float = 0.507  # the owner's: the settled design point's own share
    crown_low: float = 1673.15  # K, 1400 C: sodium silicate forms below, SEFPRO
    crown_high: float = 1893.15  # K, 1620 C: silica degrades above, SEFPRO
    fuel_min: float = 0.25  # kg/s, the valve's range
    fuel_max: float = 0.75
    # The crown override: PI on the hottest face held 5 K inside each limit (ours), by
    # SIMC on the fuel step's 808 K per kg/s and 4.1 h lag, closed loop in 30 min.
    override_margin: float = 5.0
    override_gain: float = 0.0101  # kg/s per K
    override_ti: float = 7200.0  # s
    # No capacity was found. Ours: twice what holds the canal at the design point, so
    # the coolers bind only far from it. Set from the settle script's output.
    cooling_max: float = 2.04e6
    cooling_above: float = 1.0  # K over the target the loop holds, ours
    cooling_gain: float = 2.796e5  # W/K, SIMC on a 5 % cooling step (scripts)
    cooling_ti: float = 7680.0  # s, same recipe
    # gas: ENTSOG's 4 to 6 MJ/m3 swings on about 50, read as four spreads, daily;
    # pull: +-2 % wander (US11261117B2) as two spreads; cullet, blanket: the owner's
    # +-10 points and +-20 % as two spreads; sulfate: ours.
    walk_sd: tuple = (0.02, 0.01, 0.05, 0.10, 0.17)
    walk_hours: tuple = (24.0, 12.0, 12.0, 12.0, 12.0)
    noise_sd: float = 0.36  # K, the shipped crown couple's; ours for the others


@struct.dataclass
class EnvState:
    fields: P.Fields
    walk: jnp.ndarray  # deviations, in WALKS order: hidden
    cooling: jnp.ndarray  # W, the refiner loop's output
    integral: jnp.ndarray  # W, its integral
    fuel: jnp.ndarray  # kg/s on the valve after the override, last step
    limits: jnp.ndarray  # the override's two integrals, kg/s, reset to the valve
    pull: jnp.ndarray  # kg/s charged, last step: measured
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


def reward_terms(fuel, canal, fining, ep):
    """Costs of one step, EUR: ``fuel`` kg/s burnt at nominal heating value, and the
    step's glass if the canal is under its target or the glass under-fined."""
    dt = ep.plant.delta_t
    lost = (canal < ep.canal_target) | (fining < ep.fining_pass)
    return dict(
        fuel=fuel * dt * ep.plant.LHV / 3.6e9 * ep.gas_price,
        lost_glass=jnp.where(lost, ep.pull * dt / 1000.0 * ep.lost_glass_price, 0.0),
    )


def override(asked, hot, limits, ep):
    """The plant's crown protection: ``(valve, integrals)``. Two PI loops on the hottest
    face, one held under the top limit and one over the bottom, select against the asked
    valve; each integral is reset toward the valve chosen, so neither winds up."""
    lo = ep.crown_low + ep.override_margin
    hi = ep.crown_high - ep.override_margin
    cap = limits[0] + ep.override_gain * (hi - hot)
    floor = limits[1] + ep.override_gain * (lo - hot)
    u = jnp.clip(jnp.maximum(floor, jnp.minimum(asked, cap)), ep.fuel_min, ep.fuel_max)
    return u, limits + ep.plant.delta_t / ep.override_ti * (u - limits)


def observe(state, ep):
    """The readings, noise-free. Nothing hidden: no walk, no glass inside the tank."""
    p, s = ep.plant, state.fields
    read = P.sensors(s, ep.grid, p)
    fired = state.fuel * P.mean_firing(p)
    m_air = p.AFR * (1.0 + p.excess_air) * fired
    flue = jnp.mean(s.T_gas) - P.KELVIN_OFFSET
    _, air, stack = P.averaged_stack(s.T_r, flue, m_air, fired + m_air, p)
    return jnp.stack(
        [
            read["crown"],
            read["floor_bay"],
            read["floor"],
            read["canal"],
            read["fining"],
            state.cooling / ep.cooling_max,
            air,
            stack,
            state.fuel,
            state.pull,
        ]
    )


def reset(key, fields, ep, fuel=0.45, cooling=0.0):
    """Start on a settled plant: the walks at their design values, the loop where it was."""
    del key  # the furnace is found at its design point; the walks set what follows
    c = jnp.asarray(cooling, float)
    state = EnvState(
        fields=fields,
        walk=jnp.zeros(len(WALKS)),
        cooling=c,
        integral=c,
        fuel=jnp.asarray(fuel, float),
        limits=jnp.full((2,), fuel, float),
        pull=jnp.asarray(ep.pull, float),
        time=jnp.asarray(0),
    )
    return observe(state, ep), state


def step(key, state, fuel, ep):
    """One step: ``(obs, state, reward, costs)``. ``fuel`` kg/s asked of the valve."""
    k_walk, k_noise = jax.random.split(key)
    hot = P.sensors(state.fields, ep.grid, ep.plant)["hot"]
    fuel, limits = override(jnp.asarray(fuel, float), hot, state.limits, ep)
    walk = walk_step(state.walk, k_walk, ep)
    gas, pull, cullet, thick, sulfate = (walk[i] for i in range(len(WALKS)))
    charge = ep.pull * (1.0 + pull)
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

    costs = reward_terms(books["fuel"] * (1.0 + gas), read["canal"], read["fining"], ep)
    new = EnvState(
        fields=fields, walk=walk, cooling=cooling, integral=integral,
        fuel=fuel, limits=limits, pull=charge, time=state.time + 1,
    )  # fmt: skip
    obs = observe(new, ep)
    noise = ep.noise_sd * jax.random.normal(k_noise, obs.shape)
    obs = obs + noise.at[4:].set(0.0)  # temperatures only
    return obs, new, -sum(costs.values()), costs
