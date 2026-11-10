"""The glass furnace task (attempt 3): a float melter, its refiner's own cooling loop, the
walks that move it with the valve held, and a reward in money.

The score is the operator's job (REALITY.md, section 6): fuel at its price; glass charged
while the canal is under its target, or not fined, is lost to cullet at its price, on the
pull actually charged. The silica crown's 1620 C is the plant's own override (owner,
2026-11-08; a known deviation, since no source shows such a protection): a limit control
trims the valve so the hottest face stays under it, and the agent sees the valve it got.
The refiner's cooling loop holds the canal just above its target, as a plant's local loop
does, and the agent sees how hard it works.

The agent reads what operators read (REALITY.md, section 1): crown and bottom couples, the
canal, the fining share only at the cold end a transit later, the cullet ratio the batch
plant holds, the batch line on its camera's cadence, and the pull plan as far ahead as its
notice. Gas quality, blanket thickness and sulfate stay hidden. The episode is established
operation: it starts from the settled design state, the walks where the seed puts them,
and a planned pull change or a gas swing comes at a time the seed draws inside its window.
"""

import functools
import hashlib
import pathlib
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

from target_gym import reward as R
from target_gym.base import (
    ChannelReading,
    EnvParams,
    EnvState,
    actuate,
    actuator_field,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.expert_access import RegistryExperts
from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace.rendering import _render

WALKS = ("gas", "pull", "cullet", "thickness", "sulfate")
OBS_NAMES = (
    "crown couple, K",
    "floor under bay 3, K",
    "floor at the melter's end, K",
    "canal, K",
    "fining share read at the cold end",
    "fining pass mark",
    "refiner cooling, share of its capacity",
    "fuel, kg/s",
    "pull, kg/s",
    "pull planned, kg/s",
    "cullet ratio",
    "batch line, m",
)
#: The settled design point: crown face 1590 C in bay 3, canal 1101 C
#: (``scripts/glass_slow_settle.py``).
DESIGN_STATE = pathlib.Path(__file__).with_name("design_state.npz")
#: The file's digest, here so that the version stamp, which reads code, changes with it.
DESIGN_STATE_SHA256 = "55a6ac2ffb53d5d1b142758399532af3428263585290ab5b40f1ed2a2484bba6"
#: The fining share's reading scatter (ours), and the tolerance a controller is told.
FINING_SCATTER = 0.005


@functools.lru_cache(maxsize=1)
def design_state():
    """``(fuel kg/s, cooling W, plant fields)`` of the settled design point."""
    digest = hashlib.sha256(DESIGN_STATE.read_bytes()).hexdigest()
    if digest != DESIGN_STATE_SHA256:
        raise ValueError(f"{DESIGN_STATE.name} changed: update DESIGN_STATE_SHA256")
    d = np.load(DESIGN_STATE)
    fields = {k: d[k] for k in d.files if k not in ("fuel", "cooling")}
    return float(d["fuel"]), float(d["cooling"]), fields


@struct.dataclass
class GlassFurnaceParams(EnvParams):
    delta_t: float = 120.0  # s: the surface current's Courant number
    max_steps_in_episode: int = 5040  # seven days of established operation
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
    e_floor: float = FINING_SCATTER  # the tolerance on the fining share (benchmark)
    crown_high: float = 1893.15  # K, 1620 C: silica degrades above, SEFPRO
    fuel_min: float = 0.25  # kg/s, the valve's range
    fuel_max: float = 0.75
    # The crown override: PI on the hottest face held under the limit by a margin (ours),
    # by SIMC on the fuel step's 808 K per kg/s and 4.1 h lag, closed loop in 30 min.
    override_margin: float = 5.0
    override_gain: float = 0.0101  # kg/s per K
    override_ti: float = 7200.0  # s
    # No capacity was found. Ours: twice what holds the canal at the design point.
    cooling_max: float = 1.54e6
    cooling_above: float = 1.0  # K over the target the loop holds, ours
    cooling_gain: float = 2.619e5  # W/K, SIMC on a 5 % cooling step (scripts)
    cooling_ti: float = 7680.0  # s, same recipe
    # gas: ENTSOG's 4 to 6 MJ/m3 swings on about 50, read as four spreads, daily;
    # pull: +-2 % wander (US11261117B2) as two spreads; cullet, blanket: the owner's
    # +-10 points and +-20 % as two spreads; sulfate: ours.
    walk_sd: tuple = (0.02, 0.01, 0.05, 0.10, 0.17)
    walk_hours: tuple = (24.0, 12.0, 12.0, 12.0, 12.0)
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
    reward_version: int = 2
    # No trip: the crown is protected and the canal and fining are zones, priced.
    failure_cost: float = 0.0
    restart_steps: int = 0


PARAM_UNITS = {
    "gas_price": "EUR/MWh",
    "lost_glass_price": "EUR/t",
    "pull": "kg/s",
    "cullet": "-",
    "thickness": "m",
    "sulfate": "-",
    "canal_target": "K",
    "fining_pass": "-",
    "e_floor": "-",
    "crown_high": "K",
    "fuel_min": "kg/s",
    "fuel_max": "kg/s",
    "override_margin": "K",
    "override_gain": "kg/s per K",
    "override_ti": "s",
    "cooling_max": "W",
    "cooling_above": "K",
    "cooling_gain": "W/K",
    "cooling_ti": "s",
    "pull_change": "-",
    "pull_ramp_h": "h",
    "pull_notice_h": "h",
    "gas_swing": "-",
    "gas_swing_h": "h",
    "reward_version": "-",
    "failure_cost": "reward",
    "restart_steps": "steps",
}


@struct.dataclass
class GlassFurnaceState(EnvState):
    plant: P.Fields
    walk: jax.Array  # deviations, in WALKS order: hidden
    cooling: jax.Array  # W, the refiner loop's output
    integral: jax.Array  # W, its integral
    fuel: jax.Array  # kg/s on the valve after the override, last step
    limit: jax.Array  # the override's integral, kg/s, reset to the valve
    pull: jax.Array  # kg/s charged, last step: measured
    cullet: jax.Array  # cullet ratio charged, last step: measured
    events: jax.Array  # h: when the pull change and the gas swing start
    actuator_held: jax.Array = actuator_field(1)


def _decay(params):
    phi = jnp.exp(-params.delta_t / (3600.0 * jnp.asarray(params.walk_hours)))
    return phi, jnp.asarray(params.walk_sd) * jnp.sqrt(1.0 - phi**2)


def walk_step(walk, key, params):
    """One step of each walk: first order, stationary at its spread."""
    phi, kick = _decay(params)
    return phi * walk + kick * jax.random.normal(key, walk.shape)


def walks(key, n, params):
    """``n`` steps of the walks from a stationary start, ``(n, len(WALKS))``."""
    k0, k1 = jax.random.split(key)
    start = jnp.asarray(params.walk_sd) * jax.random.normal(k0, (len(WALKS),))
    keys = jax.random.split(k1, n)
    return jax.lax.scan(lambda w, k: (walk_step(w, k, params),) * 2, start, keys)[1]


def to_fuel(action, params):
    """kg/s for a raw action in [-1, 1]."""
    span = params.fuel_max - params.fuel_min
    return params.fuel_min + 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0) * span


def to_action(fuel, params):
    """The raw action that asks ``fuel`` kg/s, shape (1,)."""
    span = params.fuel_max - params.fuel_min
    return jnp.reshape(2.0 * (fuel - params.fuel_min) / span - 1.0, (1,))


def _ramp(hours, start, length):
    return jnp.clip((hours - start) / jnp.maximum(length, 1e-9), 0.0, 1.0)


def planned_pull(hours, events, params):
    """The pull the plan sets at ``hours``, kg/s, before the wander."""
    ramp = _ramp(hours, events[0], params.pull_ramp_h)
    return params.pull * (1.0 + params.pull_change * ramp)


def gas_factor(walk, time, events, params):
    """The gas's heating value off nominal, as a fraction, at step ``time``."""
    hours = time * params.delta_t / 3600.0
    return walk[0] + params.gas_swing * _ramp(hours, events[1], params.gas_swing_h)


def batch_line(plant, params):
    """m from the back wall that the blanket covers, as a camera would read it."""
    a = P.layout(params.grid)
    return jnp.sum(P.cover(plant.m) * a.dx[: a.nm])


def override(asked, hot, limit, params):
    """The plant's crown protection: ``(valve, integral)``. A PI loop on the hottest face,
    held under the top limit, selects against the asked valve; its integral is reset
    toward the valve chosen, so it does not wind up."""
    p = params
    cap = limit + p.override_gain * (p.crown_high - p.override_margin - hot)
    u = jnp.clip(jnp.minimum(asked, cap), p.fuel_min, p.fuel_max)
    return u, limit + p.delta_t / p.override_ti * (u - limit)


def costs(fuel, canal, fining, pull, params):
    """Costs of one step, EUR: ``running`` the ``fuel`` kg/s burnt at nominal heating
    value, ``tracking`` the step's charged ``pull`` if the canal is under its target or
    the glass under-fined."""
    dt = params.delta_t
    lost = (canal < params.canal_target) | (fining < params.fining_pass)
    return dict(
        running=fuel * dt * params.plant.LHV / 3.6e9 * params.gas_price,
        tracking=jnp.where(lost, pull * dt / 1000.0 * params.lost_glass_price, 0.0),
    )


def compute_reward_terms(state, params, tripped=None, xp=jnp):
    """The reward's additive cost terms in EUR per step (``target_gym.reward``)."""
    read = P.sensors(state.plant, params.grid, params.plant)
    gas = gas_factor(state.walk, state.time, state.events, params)
    fired = state.fuel * P.mean_firing(params.plant) * (1.0 + gas)
    terms = costs(fired, read["canal"], read["fining"], state.pull, params)
    tripped = False if tripped is None else tripped
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


def compute_reward_v1(state, params):
    """Version 1, kept constructible (``reward.select``): 1 for a step of good glass,
    0 for one lost."""
    lost = compute_reward_terms(state, params)["tracking"] > 0.0
    return jnp.where(lost, 0.0, 1.0)


def compute_reward(state, params):
    """Minus the step's cost in EUR: the fuel burnt, at its heating value and the gas
    price, plus the step's charged pull at the lost glass price when the canal is under
    its target or the glass leaving is under-fined. Nothing trips."""
    terms = compute_reward_terms(state, params)
    return R.select(
        params.reward_version, compute_reward_v1(state, params), R.total(terms)
    )


class GlassFurnace(
    RegistryExperts, environment.Environment[GlassFurnaceState, GlassFurnaceParams]
):
    """A regenerative float glass furnace, its melt in two dimensions.

    Observation (12,): OBS_NAMES.
    Action (1,): fuel valve, raw in [-1, 1] -> [fuel_min, fuel_max] kg/s.
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "walk",
        "pull",
        "cullet",
        "events",
    )

    registry_name: str = "glass_furnace"
    obs_value_index: int = 4  # the fining share read at the cold end
    obs_target_index: int = 5  # its pass mark
    tracked_names: tuple = ("fining share read at the cold end",)

    # Crown and bottom couples and the canal pyrometer at the shipped couple's 0.36 K
    # (ours for the three without a figure); the fining share at the cold end 0.3 h
    # late (ours, family range 14 to 20 min: a 50 m bath and a 100 m lehr at 7.5 to
    # 10.4 m/min), its noise ours; the batch line every 30 min (Klemiato 2021).
    readings: tuple = (
        ChannelReading(channel=0, sigma=0.36),
        ChannelReading(channel=1, sigma=0.36),
        ChannelReading(channel=2, sigma=0.36),
        ChannelReading(channel=3, sigma=0.36),
        ChannelReading(channel=4, sigma=FINING_SCATTER, dead_time=1080.0),
        ChannelReading(channel=11, sigma=0.0, interval=1800.0, sampled=True),
    )

    render_furnace = classmethod(_render)

    def __init__(self, integration_method: str = "euler_1"):
        # The plant steps its own explicit scheme at a step its advection's Courant
        # number allows (``physical.plant_step``); no other family is offered.
        if integration_method != "euler_1":
            raise ValueError(
                f"glass_furnace steps euler_1 only, not {integration_method}"
            )
        self.integration_method = integration_method
        self.obs_shape = (len(OBS_NAMES),)

    @property
    def default_params(self) -> GlassFurnaceParams:
        return GlassFurnaceParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: GlassFurnaceState,
        action: jnp.ndarray,
        params: GlassFurnaceParams = None,
    ):
        p = self.default_params if params is None else params
        action, state = actuate(self, state, jnp.reshape(action, (1,)), p)
        hot = P.sensors(state.plant, p.grid, p.plant)["hot"]
        fuel, limit = override(to_fuel(action[0], p), hot, state.limit, p)
        # Drawn from the key and the step, so a caller passing one key every step still
        # meets a fresh innovation (test_env_conformance).
        walk = walk_step(state.walk, jax.random.fold_in(key, state.time), p)
        _, pull, cullet, thick, sulfate = (walk[i] for i in range(len(WALKS)))
        time = state.time + 1
        hours = time * p.delta_t / 3600.0
        charge = planned_pull(hours, state.events, p) * (1.0 + pull)
        plant_p = p.plant.replace(
            delta_t=p.delta_t, sulfate_batch=p.sulfate * (1.0 + sulfate)
        )
        plant, _ = P.plant_step(
            state.plant, fuel, charge, p.cullet + cullet, p.thickness * (1.0 + thick),
            state.cooling, p.grid, plant_p, lhv=gas_factor(walk, time, state.events, p),
        )  # fmt: skip
        # The refiner's own loop: PI on the canal, clamped to what the coolers can take.
        e = (
            P.sensors(plant, p.grid, plant_p)["canal"]
            - p.canal_target
            - p.cooling_above
        )
        integral = jnp.clip(
            state.integral + p.cooling_gain * e * p.delta_t / p.cooling_ti,
            0.0,
            p.cooling_max,
        )
        cooling = jnp.clip(p.cooling_gain * e + integral, 0.0, p.cooling_max)
        new = state.replace(
            time=time, plant=plant, walk=walk, cooling=cooling, integral=integral,
            fuel=fuel, limit=limit, pull=charge, cullet=p.cullet + cullet,
        )  # fmt: skip
        new, reward, terminated, info = finish_step(self, key, state, new, p)
        return self.get_obs(new, p), new, reward, terminated, info

    @measured_obs
    def get_obs(self, state: GlassFurnaceState, params: GlassFurnaceParams = None):
        p = self.default_params if params is None else params
        read = P.sensors(state.plant, p.grid, p.plant)
        hours = state.time * p.delta_t / 3600.0
        values = (
            read["crown"],
            read["floor_bay"],
            read["floor"],
            read["canal"],
            read["fining"],
            p.fining_pass,
            state.cooling / p.cooling_max,
            state.fuel,
            state.pull,
            planned_pull(hours + p.pull_notice_h, state.events, p),
            state.cullet,
            batch_line(state.plant, p),
        )
        return jnp.stack([jnp.asarray(v, jnp.float32) for v in values])

    def render(self, screen, state, params, frames, clock):
        return self.render_furnace(screen, state, params, frames, clock)

    def is_terminated(
        self, state: GlassFurnaceState, params: GlassFurnaceParams
    ) -> jnp.ndarray:
        """Nothing trips: the crown is protected, and the zones are priced."""
        return jnp.asarray(False)

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: GlassFurnaceParams = None
    ) -> Tuple[jnp.ndarray, GlassFurnaceState]:
        p = self.default_params if params is None else params
        fuel, cooling, fields = design_state()
        k_walk, k_events = jax.random.split(key)
        walk = jnp.asarray(p.walk_sd) * jax.random.normal(k_walk, (len(WALKS),))
        lo, hi = p.pull_change_window_h
        events = jax.random.uniform(k_events, (2,), minval=lo, maxval=hi)
        state = GlassFurnaceState(
            time=0,
            plant=P.Fields(**{k: jnp.asarray(v) for k, v in fields.items()}),
            walk=walk,
            cooling=jnp.asarray(cooling, jnp.float32),
            integral=jnp.asarray(cooling, jnp.float32),
            fuel=jnp.asarray(fuel, jnp.float32),
            limit=jnp.asarray(fuel, jnp.float32),
            pull=planned_pull(0.0, events, p) * (1.0 + walk[WALKS.index("pull")]),
            cullet=p.cullet + walk[WALKS.index("cullet")],
            events=events,
        )
        return self.get_obs(state, params), state

    def initial_action(self, state: GlassFurnaceState, params: GlassFurnaceParams):
        """The raw action that keeps the design valve."""
        return to_action(state.fuel, params)

    def action_space(self, params: GlassFurnaceParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]), high=jnp.array([1.0]), shape=(1,), dtype=jnp.float32
        )

    def observation_space(self, params: GlassFurnaceParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: GlassFurnaceParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        n = len(GlassFurnaceState.__dataclass_fields__)
        return spaces.Box(-inf, inf, n, dtype=jnp.float32)
