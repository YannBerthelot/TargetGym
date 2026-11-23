"""Prototype, not a task: the jacketed CSTR scored on what it makes.

From the cstr review of 2026-11-23 (owner: build its two ends before deciding).
Nothing here is registered, recorded or fingerprinted.

    python scripts/prototypes/cstr_production.py OUT_DIR build    # ladder, set points
    python scripts/prototypes/cstr_production.py OUT_DIR judge COMMIT   # J2's runs
    python scripts/prototypes/cstr_production.py OUT_DIR measure  # reference, ceiling
    python scripts/prototypes/cstr_production.py OUT_DIR sweep    # B7's draws

What changes from ``pc_gym.cstr``, and why:

- A second handle, the feed, 40 to 110 g/s. The top is S1's feed range [S1,
  Table 1]; the floor is ours. When the cooling saturates the feed is what
  practice cuts: "Shinskey (1988) describes this solution for temperature
  control in a reactor, where the feed rate is reduced if the cooling
  saturates" [P1, p. 436]. The shipped task has no such handle (PHYSICS.md D3).
- The score is production lost against the nameplate (the top feed fully
  converted), in fractions of it per step. Product made with the reactor more
  than ``T_band`` (ours: 1 K) off its setpoint is off-spec and counts for
  nothing. A trip loses 4 h of nameplate production, as the shipped task prices
  it. One unit throughout, so no price is needed.
- The window starts in established production: the reactor on its setpoint, the
  water valve at ``valve_rest`` of its span (0.95, where [P1]'s example holds
  it) and the feed that takes, or the feed at its top and the water that takes.

The reference is the ladder's (``scripts/reference_ladder.py``, run on this plant
in this process): the temperature on the water by SIMC, and its valve position
rung holding the water valve by the feed, which is [P1]'s structure for a
temperature whose cooling may saturate while throughput is given up. [P1]'s two
other structures steer the temperature with the feed; measured here at 4acda1c,
they swing this reactor by 8 K at SIMC and at 10 and 30 times slower, because a
feed cut first heats it. So did the feedforward and cascade rungs under the valve
position loop, which is why the row leaves them out. Set points are placed on a
fixed grid by the owner's rule (2026-11-23): the cheapest at which no seed of the
normal day leaves the band, a seed no set point saves not deciding. The ladder's
own placement (PR #232) replaces ``place`` once it is merged.

The ceiling is ``pc_gym.foresight``'s, with the band's edge priced at what
leaving it costs (a step's product). Priced at the trip's cost, 960 times
steeper, L-BFGS-B stopped after 19 iterations short of the plan (projected
gradient 4.6e-2) and a closed loop beat it. It starts from the reset's hold and
from the reference's moves, restarts from its own end plan, and is never above the
reference: its moves are a candidate too.

[P1] A. Reyes-Lua, C. Zotica, S. Skogestad, "Optimal Operation with Changing
Active Constraint Regions using Classical Advanced Control", Preprints, 10th
IFAC ADCHEM, Shenyang, 25-27 July 2018, pp. 434-439.
https://skoge.folk.ntnu.no/prost/proceedings/ADCHEM-2018/media/files/0071.pdf
Fetched twice, the two copies byte-identical.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import struct  # noqa: E402
from gymnax.environments import spaces  # noqa: E402
from jax.tree_util import Partial as partial  # noqa: E402

from target_gym import reward as R  # noqa: E402
from target_gym.base import (  # noqa: E402
    actuate,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.benchmark import episodes_of, run_policy, task_info  # noqa: E402
from target_gym.integration import integrate_dynamics  # noqa: E402
from target_gym.pc_gym import foresight as F  # noqa: E402
from target_gym.pc_gym.cstr.env import (  # noqa: E402
    CP_WATER,
    CSTRParams,
    CSTRState,
    check_is_terminal,
    day_minute,
    event_sizes,
    events,
    feed_temperature,
    rate,
)
from target_gym.pc_gym.cstr.env_jax import CSTR  # noqa: E402


@struct.dataclass
class ProdParams(CSTRParams):
    q_min: float = 2.4  # L/min, 40 g/s (ours)
    q_max: float = 6.6  # L/min, 110 g/s [S1, Table 1]
    T_band: float = 1.0  # K either side of the setpoint (ours)
    #: The water valve at the reset, of its span: where [P1]'s example holds it, "some
    #: back-off" keeping it "within the controllable range" [P1, pp. 436, 438]. At
    #: its top (the first choice, ours) the reference had no cooling left at the
    #: first step and could not be placed in a band under 0.9 K.
    valve_rest: float = 0.95
    #: Minutes a setpoint move takes. Derived: S1's closed loop took about an hour
    #: to carry its 2 C move to its peak [S1, Fig. 2], and a band that jumped 2 K
    #: at once could not be kept by anything.
    setpoint_ramp: float = 60.0
    failure_cost: float = 1.0  # a step of nameplate production
    restart_steps: int = 960  # 4 h, as the shipped task


@struct.dataclass
class ProdState(CSTRState):
    q: float = struct.field(default=5.4, kw_only=True)  # L/min, the feed as it ran


def velocity(x, F_w, q, p, extra):
    """The shipped balances with the feed ``q`` (L/min) in place of ``p.q``."""
    T_f, Caf, k0 = extra
    C_a, T, T_j = x[0], x[1], x[2]
    r = rate(T, C_a, p, k0)
    dC = (Caf - C_a) * q / p.V - r
    dT = (T_f - T) * q / p.V + (-p.deltaHr) * r / (p.rho * p.C)
    dT = dT - p.UA * (T - T_j) / (p.rho * p.C * p.V)
    dTj = (F_w * CP_WATER * (p.Tw_in - T_j) + p.UA * (T - T_j)) / (p.M_j * CP_WATER)
    return jnp.array([dC, dT, dTj]), None


def rest(T, T_f, F_w, p):
    """The steady state at ``T`` with the water at ``F_w``: ``(C_A, T_j, q)``. The
    jacket's balance gives its temperature and duty; the reactor's is a quadratic
    in the feed, whose smaller root is the operating one."""
    T_j = (F_w * CP_WATER * p.Tw_in + p.UA * T) / (F_w * CP_WATER + p.UA)
    duty = p.UA * (T - T_j)
    k = p.k0 * jnp.exp(-p.EA_over_R / T)
    a, c = p.rho * p.C * (T - T_f), k * p.V
    b = (-p.deltaHr) * k * p.V * p.Caf
    h = b - a * c - duty
    q = (h - jnp.sqrt(h * h - 4.0 * a * duty * c)) / (2.0 * a)
    return p.Caf / (1.0 + c / q), T_j, q


def target_at(state, p, minute):
    """The setpoint at ``minute``: a move ramps over ``setpoint_ramp`` minutes."""
    gone = jnp.clip((minute - state.event_at) / p.setpoint_ramp, 0.0, 1.0)
    return p.T_setpoint + state.event_size[0] * gone


def made(state, p):
    """Product made now, mol/min, and the feed's reactant it is made from."""
    _, Caf, _, _ = events(state, p, state.time * p.delta_t)
    return state.q * (Caf - state.C_a)


class ProductionCSTR(CSTR):
    """Observation (5,): [T, T_j, F_w_pct, q_pct, T_setpoint]. Action (2,):
    cooling water and feed, raw in [-1, 1]."""

    obs_target_index: int = 4

    def __init__(self, integration_method: str = "rk4_1"):
        super().__init__(integration_method)
        self.obs_shape = (5,)

    @property
    def default_params(self):
        return ProdParams()

    def action_space(self, params=None):
        return spaces.Box(-jnp.ones(2), jnp.ones(2), (2,), jnp.float32)

    def initial_action(self, state, params):
        p = params
        w = 2.0 * (state.F_w - p.Fw_min) / (p.Fw_max - p.Fw_min) - 1.0
        f = 2.0 * (state.q - p.q_min) / (p.q_max - p.q_min) - 1.0
        return jnp.stack([w, f])

    @canonical_reset
    def reset_env(self, key, params=None):
        p = self.default_params if params is None else params
        feed_key, event_key = jax.random.split(key)
        phase = jax.random.uniform(feed_key, minval=0.0, maxval=2.0 * jnp.pi)
        T_feed = feed_temperature(phase, p, 0.0)
        T = jnp.asarray(p.T_setpoint)
        F_w = p.Fw_min + p.valve_rest * (p.Fw_max - p.Fw_min)
        q = jnp.clip(rest(T, T_feed, F_w, p)[2], p.q_min, p.q_max)
        # Where the feed's top binds, the water the balances then need.
        k = p.k0 * jnp.exp(-p.EA_over_R / T)
        C_a = p.Caf / (1.0 + k * p.V / q)
        duty = q * p.rho * p.C * (T_feed - T) + (-p.deltaHr) * k * C_a * p.V
        T_j = T - duty / p.UA
        F_w = jnp.clip(duty / (CP_WATER * (T_j - p.Tw_in)), p.Fw_min, p.Fw_max)
        lo, hi = p.event_window
        at = jax.random.uniform(event_key, minval=lo, maxval=hi)
        state = ProdState(
            time=0, C_a=C_a, T=T, target_T=T, F_w=F_w, T_j=T_j, T_feed=T_feed,
            feed_phase=phase, event_at=at, event_until=at + p.event_hold,
            event_size=event_sizes(p), q=q,
            actuator_held=jnp.zeros(2),
        )  # fmt: skip
        return self.get_obs(state), state

    def step_env(self, key, state, action, params=None):
        p = self.default_params if params is None else params
        action, state = actuate(self, state, jnp.reshape(action, (2,)), p)
        minute = state.time * p.delta_t
        u = jnp.clip(action, -1.0, 1.0)
        F_w = p.Fw_min + 0.5 * (u[0] + 1.0) * (p.Fw_max - p.Fw_min)
        q = p.q_min + 0.5 * (u[1] + 1.0) * (p.q_max - p.q_min)
        _, Caf, k0, cooling = events(state, p, minute)
        F_w = jnp.where(cooling, F_w, 0.0)
        (C_a, T, T_j), _ = integrate_dynamics(
            positions=jnp.array([state.C_a, state.T, state.T_j]),
            delta_t=p.delta_t,
            compute_velocity=partial(
                velocity, F_w=F_w, q=q, p=p, extra=(state.T_feed, Caf, k0)
            ),
            method=self.integration_method,
        )
        proposal = state.replace(
            C_a=C_a, T=T, T_j=T_j, F_w=F_w, q=q,
            target_T=target_at(state, p, minute + p.delta_t),
            T_feed=feed_temperature(state.feed_phase, p, day_minute(state, p)),
            time=state.time + 1,
        )  # fmt: skip
        new, reward, done, info = finish_step(self, key, state, proposal, p)
        return self.get_obs(new, p), new, reward, done, info

    @measured_obs
    def get_obs(self, state, params=None):
        p = self.default_params if params is None else params
        w = 100.0 * (state.F_w - p.Fw_min) / (p.Fw_max - p.Fw_min)
        f = 100.0 * (state.q - p.q_min) / (p.q_max - p.q_min)
        return jnp.array([state.T, state.T_j, w, f, state.target_T])

    def reward_terms(self, state, params, tripped=None):
        p = params
        if tripped is None:
            tripped, _ = check_is_terminal(state, p)
        nameplate = p.q_max * p.Caf
        product = made(state, p) / nameplate
        on_spec = jnp.abs(state.T - state.target_T) <= p.T_band
        terms = {
            "lost_production": 1.0 - product,
            "off_spec": jnp.where(on_spec, 0.0, product),
        }
        return R.with_trip(terms, tripped, R.trip_cost(p))

    def compute_reward(self, state, params):
        return R.total(self.reward_terms(state, params))


#: For ``foresight.solve``: the smooth term, and the steps as margins, in tenths
#: of a kelvin backed off one tenth: the band's two edges and the interlock. The
#: price is a step's product, what leaving the band costs.
CEILING_TERMS = ("lost_production",)


def ceiling_margins(state, p):
    off = state.T - state.target_T
    edges = [p.T_band - off, p.T_band + off, p.T_trip - state.T]
    return jnp.stack([e / p.e_floor - 1.0 for e in edges]), 1.0


def hold(o, xp):
    """The raw action holding both valves where the observation shows them."""
    return xp.stack([o[2] / 50.0 - 1.0, o[3] / 50.0 - 1.0])


SCENARIOS = {  # the shipped suite on this plant, with its weights
    "normal": ({}, 0.7),
    "setpoint_move": ({"setpoint_step": 2.0}, 0.1),  # [S1, Fig. 2], as a ramp
    "catalyst_loss": ({"k0_drop": 0.4, "k0_decay": 90.0}, 0.1),
    "feed_buildup": ({"feed_rise": 0.1}, 0.05),  # with a feed to cut, the 10 %
    "cooling_loss": ({"cooling_loss": 5.0}, 0.05),
}
SEEDS = tuple(range(100, 110))
#: Solves per start, each from the last one's end plan. Two settle every scenario
#: but the catalyst decay, whose starts ended up to 46 % apart; six brought them
#: within 12 % (1.5 % at the median) and its ceiling 2.4 % lower.
SOLVES = {"catalyst_loss": 6}
RECORD = {
    "made": lambda s, p, t: made(s, p),
    "T": lambda s, p, t: s.T,
    "off": lambda s, p, t: jnp.abs(s.T - s.target_T) > p.T_band,
    "Fw": lambda s, p, t: s.F_w,
    "q": lambda s, p, t: s.q,
}
#: The placement grid, fixed before any run: the temperature set point in tenths
#: of a kelvin under the band's top (the score's resolution) down to its centre,
#: and the valve's set point from [P1]'s 95 % toward its top and down to 90 %.
GRID_B = tuple(round(0.1 * k, 1) for k in range(1, 11))
GRID_VALVE = (0.99, 0.98, 0.95, 0.9)
#: The project's own numbers and the ranges B7 draws them from (cstr/PHYSICS.md,
#: "Production prototype"). The feed's floor runs down to S1's 0 [S1, Table 1].
OURS = {
    "T_band": (0.5, 2.0),
    "q_min": (0.0, 2.4),
    "setpoint_ramp": (30.0, 120.0),
    "feed_rise": (0.03, 0.10),
    "cooling_loss": (2.0, 8.0),
    "normal_weight": (0.5, 0.9),
}


def run(env, p, policy, seeds):
    """``policy`` on ``seeds``: a summary, its moves and the true traces."""
    task = task_info("cstr", env, p, len(seeds))
    ro = run_policy(policy, env, p, task, seeds=seeds, record=RECORD)
    eps, rec = episodes_of(ro), ro["records"]
    summary = {
        "cost": [float(e.cost.mean()) for e in eps],
        "trips": [bool(np.any(e.failed)) for e in eps],
        "on_spec": (rec["made"] * (1 - rec["off"])).mean(1).tolist(),
        "off_spec": rec["off"].mean(1).tolist(),
        "T_max": rec["T"].max(1).tolist(),
        "valve": ((rec["Fw"].mean(1) / 60 - 40) / 96).tolist(),
        "feed_gps": (rec["q"].mean(1) * 1000 / 60).tolist(),
    }
    return summary, np.asarray(ro["action"], float), rec


def show(name, label, r):
    print(
        f"{name:14s} {label:28s} cost {np.mean(r['cost']):.5f}"
        f" on-spec {np.mean(r['on_spec']):.3f} mol/min off {np.mean(r['off_spec']):.3f}"
        f" Tmax {max(r['T_max']) - 273.15:.2f} C trips {sum(r['trips'])}",
        flush=True,
    )


def ladder():
    """``reference_ladder`` with this plant standing in for cstr, this process only."""
    from target_gym import registry

    registry._ENV_CACHE["cstr"] = ProductionCSTR()
    sys.path.insert(0, str(ROOT / "scripts"))
    import reference_ladder as RL

    return RL


def gains(RL, test=None):
    """The ladder's row on this plant, built: the PI's and the valve position
    loop's SIMC gains, from bump tests at a reset with both valves in mid-range."""
    p = ProdParams(**(test or {}))
    lo, hi = (float(x) for x in p.feed_T_range)
    window = int(np.ceil(2 * 0.99 * 34.0 / float(p.delta_t)))
    T = RL.Cv("T on water", 0, lambda o, xp: o[0], lambda o, xp: o[4], "T", 0.1, window)
    # The valve answers the feed through the closed temperature loop, about half an
    # hour late (first measured at 4acda1c), so its test runs three hours.
    valve = RL.Valve("valve by feed", lambda o, xp: o[2] / 50.0 - 1.0, 0.9, 1, 0.1, 720)
    measured = (
        "leaves the band at every set point under the valve position loop (4acda1c)"
    )
    row = RL.Row(
        240, hold, (T,), test={"valve_rest": 0.8, **(test or {})}, valves=(valve,),
        points=({"feed_T_range": (lo, lo)}, {"feed_T_range": (hi, hi)}), bumpless=True,
        why={"feedforward": measured, "cascade": measured, "decoupler": "one loop",
             "override": "a feed cut first heats the reactor (4acda1c)"},
    )  # fmt: skip
    kept, build = {}, RL.ladder_controller
    RL.ladder_controller = lambda rung, *a: kept.update({rung.name: rung}) or build(
        rung, *a
    )
    try:
        RL.build("cstr", row)
    finally:
        RL.ladder_controller = build
    rung = kept["valve position"]
    pi, vp = rung.loops[0], rung.valve_positions[0]
    return {"water": [pi.kc, pi.ti, pi.alpha], "valve": [vp.kc, vp.ti]}


def reference(RL, p, g, b, valve):
    """The valve position rung on the stored gains, the temperature held ``b``
    under the band's top and the water valve at ``valve`` of its span."""
    from target_gym.experts.ladder import Pi, Rung, ValvePosition, ladder_controller

    band = float(p.T_band)
    position = ValvePosition(
        lambda o, xp: o[2] / 50.0 - 1.0, 2 * valve - 1, 1, *g["valve"]
    )
    rung = Rung(
        "valve position",
        (Pi(0, *g["water"]),),
        np.array([[1.0], [0.0]]),
        (),
        (),
        (position,),
    )
    law = ladder_controller(
        rung, lambda o, xp: xp.stack([o[0]]), lambda o, xp: xp.stack([o[4] + band - b]),
        lambda o, xp: xp.zeros(2), float(p.delta_t), hold,
    )  # fmt: skip
    return RL.Reference("reference", law)


def place(RL, env, p, g, seeds=SEEDS):
    """``(b, valve)`` by the owner's rule: on the fixed grid, the cheapest setting
    at which no deciding seed of the normal day leaves the band or trips. A seed
    that does so at every setting does not decide, and is returned by name."""
    cost, exits = {}, {}
    for valve in GRID_VALVE:
        for b in GRID_B:
            r, _, _ = run(env, p, reference(RL, p, g, b, valve), seeds)
            cost[b, valve] = float(np.mean(r["cost"]))
            exits[b, valve] = (np.array(r["off_spec"]) > 0) | np.array(r["trips"])
    left_out = np.all(list(exits.values()), axis=0)
    stays = [k for k in cost if not exits[k][~left_out].any()]
    best = min(stays, key=cost.get) if stays else None
    return best, [seeds[i] for i in np.flatnonzero(left_out)], cost


def scenarios(p, weights=None, **sizes):
    """``{name: (params, weight)}``, weights summing to 1."""
    out = {}
    for name, (over, w) in SCENARIOS.items():
        over = {k: sizes.get(k, v) for k, v in over.items()}
        out[name] = (p.replace(**over), (weights or {}).get(name, w))
    return out


class Constant:
    """A null: both valves held at one raw value."""

    oracle = False

    def __init__(self, raw):
        self.raw, self.name = raw, f"constant {(raw + 1) / 2:g}"

    def reset(self, key, obs, task):
        self.n = task.n_episodes

    def act(self, carry, obs, t):
        return np.full((self.n, 2), self.raw), carry


def nulls(RL):
    return [Constant(-1.0), Constant(0.0), Constant(1.0), RL.HoldReset()]


def load(out):
    ref = json.loads((out / "production_reference.json").read_text())
    return ref["gains"], ref["b"], ref["valve"]


def build(out):
    RL, env, p = ladder(), ProductionCSTR(), ProdParams(sensor_noise_scale=1.0)
    g = gains(RL)
    best, left_out, cost = place(RL, env, p, g)
    record = {"gains": g, "b": best[0], "valve": best[1], "left_out": left_out}
    record["grid"] = {f"{b}/{v}": c for (b, v), c in cost.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "production_reference.json").write_text(
        json.dumps(record, indent=1, default=float)
    )
    print(
        json.dumps({k: v for k, v in record.items() if k != "grid"}, default=float),
        flush=True,
    )


def judge(out):
    """J2's runs: the reference and the nulls on the hash seeds of ``COMMIT``."""
    from target_gym import charter

    RL, env, base = ladder(), ProductionCSTR(), ProdParams(sensor_noise_scale=1.0)
    g, b, valve = load(out)
    weights = {k: w for k, (_, w) in SCENARIOS.items()}
    seeds = charter.j2_seeds(sys.argv[3], weights)
    policies = {"reference": lambda p: reference(RL, p, g, b, valve)}
    policies |= {n.name: (lambda p, n=n: n) for n in nulls(RL)}
    costs, trips = {k: {} for k in policies}, 0
    for name, (p, _) in scenarios(base).items():
        for label, make in policies.items():
            policy = make(p)
            if hasattr(policy, "bind"):
                policy.bind(env, p, seeds[name])
            r, _, _ = run(env, p, policy, tuple(seeds[name]))
            costs[label][name] = r["cost"]
            trips += sum(r["trips"]) if label == "reference" else 0
    n = sum(len(v) for v in seeds.values())
    wins = {
        k: charter.null_wins(costs["reference"], v)
        for k, v in costs.items()
        if k != "reference"
    }
    result = {
        "commit": sys.argv[3],
        "seeds": seeds,
        "n": n,
        "reference_trips": trips,
        "wins": wins,
        "costs": costs,
    }
    (out / "production_j2.json").write_text(json.dumps(result, indent=1, default=float))
    print(
        json.dumps({k: result[k] for k in ("n", "reference_trips", "wins")}), flush=True
    )


def ceiling(env, p, name, seeds, moves, cost, solves):
    """Each seed's plan: L-BFGS-B from the reference's first move held and from
    its moves, restarted from its own end plan, the reference's moves a candidate."""
    plans, certs = [], []
    for i, seed in enumerate(seeds):
        first = moves[i]
        options = [(cost[i], first, "reference", 0.0)]
        for start, init in (
            ("hold", np.tile(first[:1], (len(first), 1))),
            ("moves", first),
        ):
            plan = init
            for _ in range(solves):
                plan, cert = F.solve(
                    env, p, seed, ceiling_margins, CEILING_TERMS, plan, maxiter=3000
                )
            if not cert["tripped"]:
                options.append(
                    (cert["planned_cost"], plan, start, cert["projected_gradient"])
                )
        solved = sorted(o[0] for o in options[1:])
        best = min(options, key=lambda o: o[0])
        plans.append(best[1])
        gap = (
            (solved[-1] - solved[0]) / max(solved[0], 1e-9) if len(solved) > 1 else None
        )
        certs.append(
            {"start": best[2], "starts_gap": gap, "projected_gradient": best[3]}
        )
        print(
            f"  {name} seed {seed}: {best[0]:.5f} from {best[2]}, gap {gap}", flush=True
        )
    row, _, _ = run(env, p, F.PlanPolicy(np.stack(plans)), seeds)
    return row | {"certs": certs}


def measure(out, seeds=SEEDS):
    """The placed reference and the ceiling on every scenario, per seed."""
    RL, env, base = ladder(), ProductionCSTR(), ProdParams(sensor_noise_scale=1.0)
    g, b, valve = load(out)
    result = {}
    for name, (p, weight) in scenarios(base).items():
        ref, moves, _ = run(env, p, reference(RL, p, g, b, valve), seeds)
        top = ceiling(env, p, name, seeds, moves, ref["cost"], SOLVES.get(name, 2))
        result[name] = {"weight": weight, "reference": ref, "ceiling": top}
        show(name, "reference", ref)
        show(name, "ceiling", top)
        (out / "production_measure.json").write_text(
            json.dumps(result, indent=1, default=float)
        )
    print("done", flush=True)


def sweep(out):
    """B7: the project's own numbers drawn jointly, 8 Latin-hypercube draws on
    fixed seeds. On each the ladder is rebuilt and placed, and the reference must
    cost less than every null on the median over seeds; the ceiling is solved on
    the two draws at the ends of the band, the number the room moves with most."""
    from scipy.stats import qmc

    RL, env = ladder(), ProductionCSTR()
    unit = qmc.LatinHypercube(d=len(OURS), seed=0).random(8)
    draws = [
        {k: lo + u * (hi - lo) for u, (k, (lo, hi)) in zip(row, OURS.items())}
        for row in unit
    ]
    bands = [d["T_band"] for d in draws]
    ends, result = {int(np.argmin(bands)), int(np.argmax(bands))}, []
    for i, d in enumerate(draws):
        plant = {k: d[k] for k in ("T_band", "q_min", "setpoint_ramp")}
        p = ProdParams(sensor_noise_scale=1.0, **plant)
        g = gains(RL, plant)
        best, left_out, _ = place(RL, env, p, g)
        other = (1.0 - d["normal_weight"]) / 0.3
        weights = {
            k: (d["normal_weight"] if k == "normal" else w * other)
            for k, (_, w) in SCENARIOS.items()
        }
        row = {"draw": d, "placed": best, "left_out": left_out, "J": {}, "trips": 0}
        if best is not None:
            suite = scenarios(
                p, weights, feed_rise=d["feed_rise"], cooling_loss=d["cooling_loss"]
            )
            J = {n.name: np.zeros(len(SEEDS)) for n in nulls(RL)} | {
                "reference": np.zeros(len(SEEDS))
            }
            top = np.zeros(len(SEEDS))
            for name, (q, w) in suite.items():
                ref, moves, _ = run(env, q, reference(RL, q, g, *best), SEEDS)
                J["reference"] += w * np.array(ref["cost"])
                row["trips"] += sum(ref["trips"])
                for null in nulls(RL):
                    if hasattr(null, "bind"):
                        null.bind(env, q, SEEDS)
                    J[null.name] += w * np.array(run(env, q, null, SEEDS)[0]["cost"])
                if i in ends:
                    top += w * np.array(
                        ceiling(env, q, name, SEEDS[:5], moves, ref["cost"], 2)["cost"]
                        + [0.0] * 5
                    )
            row["J"] = {k: float(np.median(v)) for k, v in J.items()}
            row["beats_nulls"] = all(
                row["J"]["reference"] < v
                for k, v in row["J"].items()
                if k != "reference"
            )
            if i in ends:
                row["ceiling_J"] = float(np.mean(top[:5]))
                row["reference_J_same_seeds"] = float(np.mean(J["reference"][:5]))
                row["ceiling_beats_reference"] = (
                    row["ceiling_J"] < row["reference_J_same_seeds"]
                )
        row["holds"] = bool(
            best is not None
            and row.get("beats_nulls")
            and row.get("ceiling_beats_reference", True)
        )
        result.append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k != "J"}
                | {"J_ref": row["J"].get("reference")},
                default=float,
            ),
            flush=True,
        )
        (out / "production_sweep.json").write_text(
            json.dumps(result, indent=1, default=float)
        )
    print("sweep holds on every draw:", all(r["holds"] for r in result), flush=True)


if __name__ == "__main__":
    {"build": build, "judge": judge, "measure": measure, "sweep": sweep}[sys.argv[2]](
        pathlib.Path(sys.argv[1])
    )
