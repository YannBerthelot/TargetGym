"""Prototype, not a task: the jacketed CSTR scored on what it makes.

From the cstr review of 2026-11-23 (owner: build its two ends before deciding).
Nothing here is registered, recorded or fingerprinted.

    python scripts/prototypes/cstr_production.py OUT_DIR gains    # bump tests, SIMC
    python scripts/prototypes/cstr_production.py OUT_DIR place    # set points
    python scripts/prototypes/cstr_production.py OUT_DIR measure  # reference, ceiling

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
- The window starts in established production at full cooling: the reactor on
  its setpoint, the water valve at ``valve_rest`` of its span (1: its top) and
  the feed that takes, or the feed at its top and the water that takes.

The reference follows [P1]'s procedure, which is the ladder's recipe (SIMC on
first-order fits of step responses): its three structures for a temperature
whose cooling may saturate while throughput is given up. Split range control
(one temperature controller, the water first and the feed past its top), valve
position control (a second controller holds the water valve at a set point by
the feed) and two temperature controllers with different set points. The fits
and gains come from ``scripts/reference_ladder.py``'s own bump tests on this
plant, with the closed-loop time equal to the dead time (charter J1). Set points
are placed on a fixed grid by the owner's rule of 2026-11-23: the cheapest at
which the normal day's seeds never leave the band. The cheapest candidate that
trips no seed is kept, as the ladder keeps its best rung.

The ceiling is ``pc_gym.foresight``'s, with the band's edge priced at what
leaving it costs (a step's product). Priced at the trip's cost, 960 times
steeper, L-BFGS-B stopped after 19 iterations short of the plan (projected
gradient 4.6e-2) and a closed loop beat it. It starts from the reset's hold and
from the kept reference's moves, restarts from its own end plan, and is never
above any closed loop measured here: their moves are candidates too.

[P1] A. Reyes-Lua, C. Zotica, S. Skogestad, "Optimal Operation with Changing
Active Constraint Regions using Classical Advanced Control", Preprints, 10th
IFAC ADCHEM, Shenyang, 25-27 July 2018, pp. 434-439.
https://skoge.folk.ntnu.no/prost/proceedings/ADCHEM-2018/media/files/0071.pdf
Fetched twice, the two copies byte-identical.
"""

import json
import pathlib
import sys
import time

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
    valve_rest: float = 1.0  # of the water valve's span at the reset: its top
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
            target_T=events(state, p, minute + p.delta_t)[0],
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


def gains(out):
    """SIMC by the ladder's own bump tests (``reference_ladder.build``) on this
    plant, from a reset with both valves in mid-range: the temperature on the
    water, the temperature on the feed, and the water valve on the feed with the
    temperature loop closed (as [P1] tunes its valve position controller)."""
    from target_gym import registry
    from target_gym.identification import fit, simc

    env = registry._ENV_CACHE["cstr"] = ProductionCSTR()  # this process only
    sys.path.insert(0, str(ROOT / "scripts"))
    import reference_ladder as RL

    models, read_loop = {}, RL.read_loop

    def keep(plant, label, *a, **k):
        models[label] = read_loop(plant, label, *a, **k)
        return models[label]

    RL.read_loop = keep
    p = ProdParams()
    lo, hi = (float(x) for x in p.feed_T_range)
    window = int(np.ceil(2 * 0.99 * 34.0 / float(p.delta_t)))
    T = RL.Cv("T on water", 0, lambda o, xp: o[0], lambda o, xp: o[4], "T", 0.1, window)
    top = RL.Limit(
        "T on feed", lambda o, xp: o[0], float(p.T_setpoint + p.T_band), False, 1,
        "T", 0.1, window,
    )  # fmt: skip
    # The jacket as the cstr row tests it: its dead time and five of its lags.
    jacket = RL.Inner(
        0, "T_j on water", lambda o, xp: o[1], None, "T_j", 0.1,
        int(np.ceil((2.0 + 5 * 2.2) / float(p.delta_t))), 0.5, window,
    )  # fmt: skip
    row = RL.Row(
        240, hold, (T,), test={"valve_rest": 0.8}, limits=(top,), cascades=(jacket,),
        points=({"feed_T_range": (lo, lo)}, {"feed_T_range": (hi, hi)}),
        why={"feedforward": "in Loops", "decoupler": "prototype"},
    )  # fmt: skip
    RL.build("cstr", row)
    inst_T, inst_j = (RL.Instrument.of(env, p, tag) for tag in ("T", "T_j"))

    def least(prefix, inst, db):
        """The design asking the smallest gain over the points (``most_sensitive``)."""
        fits = [m for k, m in models.items() if k.startswith(prefix + " at point")]
        return min((RL.design(m, inst, db) for m in fits), key=lambda d: abs(d["Kc"]))

    g = {
        "water": least("T on water", inst_T, RL.RAW_BAND),
        "feed": least("T on feed", RL.Instrument(float(p.delta_t)), None),
        "inner": least("T_j on water", inst_j, RL.RAW_BAND),
    }
    g["outer"] = least(
        "T on water over T_j on water", inst_T, RL.RAW_BAND / abs(g["inner"]["Kc"])
    )
    # The valve on the feed with each temperature rung closed: a feed step of 0.1
    # raw down after an hour, read against the same seeds' unstepped run.
    q = ProdParams(sensor_noise_scale=1.0, valve_rest=0.8)
    n0, du = 240, -0.1
    for tc in TCS:

        def valve(step):
            loop = Loops(q, g, "two", float(q.T_band), None, tc, feed_step=(n0, step))
            return run(env, q, loop, SEEDS)[1][:, :, 0]

        dy = np.median(valve(du) - valve(0.0), axis=0)[n0:]
        m = fit("fopdt", float(q.delta_t), dy, np.full(len(dy), du))
        theta = float(m["theta"]) + float(q.delta_t) / 2
        kc, ti = simc(float(m["k"]), float(m["tau"]), theta, theta)
        g["vpc " + tc] = {
            "Kc": float(kc), "Ti": float(ti), "k": float(m["k"]),
            "tau": float(m["tau"]), "theta": theta,
        }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "production_gains.json").write_text(json.dumps(g, indent=1, default=float))
    print(json.dumps(g, default=float), flush=True)


#: The temperature loop's rungs, up the ladder: the PI, the PI around the water the
#: balances ask for at the measured feed, and the cascade on the jacket outlet
#: around the same, as ``reference_ladder.build`` composes them. The last is the
#: cascade without the balances under it, which the script would build for a row
#: that declares none.
TCS = ("pi", "feedforward", "cascade", "cascade alone")


def _alpha(design, dt):
    return -np.expm1(-dt / design["Tf"]) if design.get("Tf") else 1.0


class Loops:
    """One of [P1]'s structures, vectorised over episodes on the host. The
    temperature controller holds ``b`` under the band's top by the water, started
    from the valve as the reset left it; ``tc`` is its rung (``TCS``). ``kind``:

    - ``"src"``: split range. Past the water's top its output cuts the feed,
      scaled by the two loops' SIMC gains' ratio; one integral for both valves.
    - ``"vpc"``: a PI holds the water valve at ``extra`` of its span by the feed.
    - ``"two"``: a second temperature PI on the feed, ``extra`` kelvin above the
      first; None leaves the feed where the reset put it (the bump tests).

    The feed never asks more than its top, the desired throughput ([P1]'s min
    selector), and an integral holds while its output is limited.
    """

    oracle = False

    def __init__(self, p, g, kind, b, extra, tc="pi", feed_step=None):
        self.p, self.kind, self.b, self.extra, self.tc = p, kind, b, extra, tc
        self.step, self.dt, self.band = feed_step, float(p.delta_t), float(p.T_band)
        self.kw, self.tiw = abs(g["water"]["Kc"]), g["water"]["Ti"]
        self.cascade = tc.startswith("cascade")
        self.a = _alpha(g["outer"] if self.cascade else g["water"], self.dt)
        if self.cascade:
            self.ko, self.tio = abs(g["outer"]["Kc"]), g["outer"]["Ti"]
            self.ki, self.tii = abs(g["inner"]["Kc"]), g["inner"]["Ti"]
            self.aj = _alpha(g["inner"], self.dt)
        # The feed answers like an integrator, so the two valves are matched on
        # their SIMC gains, not on steady process gains.
        self.ratio = abs(g["feed"]["Kc"] / g["water"]["Kc"])
        two = g["feed"] if kind == "two" else g.get("vpc " + tc, g["feed"])
        self.k2, self.ti2 = abs(two["Kc"]), two["Ti"]
        self.name = f"{kind} {tc} b={b} extra={extra}"

    def balance(self, o, sp):
        """The raw water the balances ask for at the measured feed, with the
        reactor at ``sp``, the feed at the middle of its range and full activity:
        the ladder's steady-state feedforward."""
        p = self.p
        q = float(p.q_min) + o[:, 3] / 100.0 * float(p.q_max - p.q_min)
        k = float(p.k0) * np.exp(-float(p.EA_over_R) / sp)
        C_a = float(p.Caf) / (1.0 + k * float(p.V) / q)
        T_f = 0.5 * float(p.feed_T_range[0] + p.feed_T_range[1])
        made = float(-p.deltaHr) * k * C_a * float(p.V)
        duty = q * float(p.rho * p.C) * (T_f - sp) + made
        T_j = sp - duty / float(p.UA)
        F_w = duty / (CP_WATER * np.maximum(T_j - float(p.Tw_in), 1e-3))
        return 2.0 * (F_w - float(p.Fw_min)) / float(p.Fw_max - p.Fw_min) - 1.0

    def reset(self, key, obs, task):
        o = np.asarray(obs, float)
        w, f, zero = o[:, 2] / 50 - 1, o[:, 3] / 50 - 1, np.zeros(len(o))
        if self.kind == "src":  # the output that holds both valves as they stand
            w = np.where(w >= 1 - 1e-6, 1 + (1 - f) / self.ratio, w)
        sp = o[:, 4] + self.band - self.b
        return {
            "i": zero, "j": zero, "ii": zero, "w": w, "f": f, "y": o[:, 0],
            "yj": o[:, 1], "tj0": o[:, 1], "ff0": self.balance(o, sp),
        }  # fmt: skip

    def act(self, m, obs, t):
        o, dt = np.asarray(obs, float), self.dt
        sp = o[:, 4] + self.band - self.b
        y = m["y"] + self.a * (o[:, 0] - m["y"])
        yj, ii = m["yj"], m["ii"]
        e = y - sp  # a hot reactor opens the water
        i = m["i"] + e * dt
        base = m["w"]
        if self.tc in (
            "feedforward",
            "cascade",
        ):  # around the balances, re-based where the reset stood
            base = base + self.balance(o, sp) - m["ff0"]
        if self.cascade:  # the outer loop lowers the jacket's set point
            yj = yj + self.aj * (o[:, 1] - yj)
            e_j = yj - (m["tj0"] - self.ko * (e + i / self.tio))
            ii = m["ii"] + e_j * dt
            v = base + self.ki * (e_j + ii / self.tii)
        else:
            v = base + self.kw * (e + i / self.tiw)
        water, j = np.clip(v, -1.0, 1.0), m["j"]
        if self.kind == "src":
            feed = np.clip(1.0 - self.ratio * np.maximum(v - 1.0, 0.0), -1.0, 1.0)
            i = np.where((v < -1.0) | (feed <= -1.0), m["i"], i)
        else:
            free = v == water
            i, ii = np.where(free, i, m["i"]), np.where(free, ii, m["ii"])
            if self.extra is None:
                feed = m["f"] + (self.step[1] if t >= self.step[0] else 0.0)
            else:
                if self.kind == "vpc":  # the valve too open: less feed
                    e2 = (2.0 * self.extra - 1.0) - water
                else:  # the reactor above the second set point: less feed
                    e2 = (sp + self.extra) - o[:, 0]
                j2 = j + e2 * dt
                asked = m["f"] + self.k2 * (e2 + j2 / self.ti2)
                feed = np.clip(asked, -1.0, 1.0)
                j = np.where(asked == feed, j2, j)
        memory = {**m, "i": i, "j": j, "ii": ii, "y": y, "yj": yj}
        return np.stack([water, feed], axis=1), memory


SCENARIOS = {  # the shipped suite less the setpoint move, with its weights
    "normal": ({}, 0.7),
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
#: The placement grids, fixed before any run: the temperature set point in
#: tenths of a kelvin under the band's top (the score's resolution) down to its
#: centre; the valve's set point from [P1]'s 95 % either way; the second
#: controller's offset from a tenth of a kelvin.
GRID_B = tuple(round(0.1 * k, 1) for k in range(1, 11))
GRID = {
    "src": (None,),
    "vpc": (0.99, 0.98, 0.95, 0.9, 0.85, 0.8),
    "two": (0.1, 0.2, 0.3, 0.5),
}


def run(env, p, policy, seeds):
    """``policy`` on ``seeds``: a summary, its moves and the true traces."""
    task = task_info("cstr", env, p, len(seeds))
    ro = run_policy(policy, env, p, task, seeds=seeds, record=RECORD)
    eps, rec = episodes_of(ro), ro["records"]
    summary = {
        "cost": [float(e.cost.mean()) for e in eps],
        "trips": [bool(np.any(e.failed)) for e in eps],
        "made": rec["made"].mean(1).tolist(),
        "on_spec": (rec["made"] * (1 - rec["off"])).mean(1).tolist(),
        "off_spec": rec["off"].mean(1).tolist(),
        "T_max": rec["T"].max(1).tolist(),
        "valve": ((rec["Fw"].mean(1) / 60 - 40) / 96).tolist(),
        "feed_gps": (rec["q"].mean(1) * 1000 / 60).tolist(),
    }
    return summary, np.asarray(ro["action"], float), rec


def show(name, label, r):
    print(
        f"{name:14s} {label:34s} cost {np.mean(r['cost']):.5f}"
        f" on-spec {np.mean(r['on_spec']):.3f} mol/min off {np.mean(r['off_spec']):.3f}"
        f" valve {np.mean(r['valve']):.3f} feed {np.mean(r['feed_gps']):.1f} g/s"
        f" Tmax {max(r['T_max']) - 273.15:.2f} C trips {sum(r['trips'])}",
        flush=True,
    )


def candidates(g):
    """``(label, kind, extra, tc)``: the two feed-steered structures on the PI,
    and valve position control on every rung of the temperature loop."""
    out = [("split range", "src", None, "pi")]
    out += [(f"two controllers, +{x} K", "two", x, "pi") for x in GRID["two"]]
    for tc in TCS:
        out += [(f"{tc}, valve at {x}", "vpc", x, tc) for x in GRID["vpc"]]
    return out


def place(out):
    """The owner's rule (2026-11-23): on the fixed grid, each candidate keeps the
    cheapest set point at which the normal day's seeds never leave the band."""
    g = json.loads((out / "production_gains.json").read_text())
    env, p, placed = ProductionCSTR(), ProdParams(sensor_noise_scale=1.0), {}
    for label, kind, extra, tc in candidates(g):
        rows = {}
        for b in GRID_B:
            if kind == "two" and extra >= b:  # the second set point inside the band
                continue
            r, _, _ = run(env, p, Loops(p, g, kind, b, extra, tc), SEEDS)
            rows[b] = (
                float(np.mean(r["cost"])),
                float(np.max(r["off_spec"])),
                sum(r["trips"]),
            )
        stays = {
            b: c for b, (c, off, trips) in rows.items() if off == 0.0 and not trips
        }
        b = min(stays, key=stays.get) if stays else None
        placed[label] = {"kind": kind, "extra": extra, "tc": tc, "b": b, "rows": rows}
        print(f"{label:34s} b {b} cost {stays.get(b)} | " + " ".join(
            f"{k}:{c:.4f}{'' if off == 0 and not tr else '*'}" for k, (c, off, tr) in rows.items()
        ), flush=True)  # fmt: skip
    (out / "production_placed.json").write_text(json.dumps(placed, indent=1))


def measure(out, seeds=SEEDS):
    """The cheapest placed setting of each structure and rung, and the ceiling, on
    every scenario, per seed. A third argument names a stored measurement whose
    ceiling is kept instead of solved again (the plant, scenarios and seeds being
    the same); a seed on which a closed loop here costs less takes that cost."""
    stored = json.loads(pathlib.Path(sys.argv[3]).read_text()) if sys.argv[3:] else None
    g = json.loads((out / "production_gains.json").read_text())
    placed = json.loads((out / "production_placed.json").read_text())
    env, base, result, t0 = (
        ProductionCSTR(),
        ProdParams(sensor_noise_scale=1.0),
        {},
        time.time(),
    )
    kept = {k: v for k, v in placed.items() if v["b"] is not None}
    # One candidate per structure: its cheapest placed setting on the normal day.
    best = {}
    for label, v in kept.items():
        key = v["kind"] + v["tc"]
        cost = v["rows"][str(v["b"])][0]
        if key not in best or cost < best[key][1]:
            best[key] = (label, cost)
    for name, (over, weight) in SCENARIOS.items():
        p, rows, moves = base.replace(**over), {}, {}
        for label, _ in best.values():
            v = kept[label]
            rows[label], moves[label], _ = run(
                env, p, Loops(p, g, v["kind"], v["b"], v["extra"], v["tc"]), seeds
            )
        if stored:
            cheapest = np.min([r["cost"] for r in rows.values()], axis=0)
            ceiling = dict(stored[name]["ceiling"])
            ceiling["from_closed_loop"] = int(np.sum(cheapest < ceiling["cost"]))
            ceiling["cost"] = np.minimum(ceiling["cost"], cheapest).tolist()
            rows["ceiling"] = ceiling
        plans, certs = [], []
        for i, seed in enumerate(() if stored else seeds):
            closed = min(moves, key=lambda k: rows[k]["cost"][i])
            first = moves[closed][i]
            options = [(rows[closed]["cost"][i], first, "closed loop " + closed, 0)]
            for start, init in (
                ("hold", np.tile(first[:1], (len(first), 1))),
                ("moves", first),
            ):
                plan = init
                for again in range(SOLVES.get(name, 2)):
                    plan, cert = F.solve(
                        env, p, seed, ceiling_margins, CEILING_TERMS, plan, maxiter=3000
                    )
                if not cert["tripped"]:
                    options.append(
                        (cert["planned_cost"], plan, start, cert["projected_gradient"])
                    )
            solved = sorted(o[0] for o in options[1:])
            cost, plan, start, pg = min(options, key=lambda o: o[0])
            plans.append(plan)
            gap = (
                (solved[-1] - solved[0]) / max(solved[0], 1e-9)
                if len(solved) > 1
                else None
            )
            certs.append(
                {
                    "start": start,
                    "planned": cost,
                    "projected_gradient": pg,
                    "starts_gap": gap,
                }
            )
            print(
                f"  {name} seed {seed}: {cost:.5f} from {start}, gap {gap}, {time.time() - t0:.0f} s",
                flush=True,
            )
        if not stored:
            rows["ceiling"], _, _ = run(env, p, F.PlanPolicy(np.stack(plans)), seeds)
            rows["ceiling"]["certs"] = certs
        result[name] = {"weight": weight, **rows}
        for label, r in rows.items():
            show(name, label, r)
        (out / "production_measure.json").write_text(
            json.dumps(result, indent=1, default=float)
        )
    print("done", flush=True)


if __name__ == "__main__":
    {"gains": gains, "place": place, "measure": measure}[sys.argv[2]](
        pathlib.Path(sys.argv[1])
    )
