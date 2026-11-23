"""Prototype, not a task: the jacketed CSTR scored on what it makes.

From the cstr review of 2026-11-23 (owner: measure it before deciding). Nothing
here is registered, recorded or fingerprinted. It asks one question: if the
operator also moves the feed and is paid for product, where does the reference
lose production against a foresight ceiling?

    python scripts/prototypes/cstr_production.py OUT_DIR [SEEDS]

What changes from ``pc_gym.cstr``, and why:

- A second handle, the feed, 40 to 110 g/s. The top is S1's feed range [S1,
  Table 1]; the floor is ours. Past the cooling's capacity practice cuts the
  feed (cstr/PHYSICS.md D3, D6), which the shipped task cannot do.
- The score is production lost against the nameplate (the top feed fully
  converted), in fractions of it per step. Product made with the reactor more
  than ``T_band`` (ours: 1 K) off its setpoint is off-spec and counts for
  nothing. A trip loses 4 h of nameplate production, as the shipped task prices
  it. One unit throughout, so no price is needed.
- The window starts in established production: the reactor on its setpoint with
  the water valve at ``valve_rest`` of its span (ours: 0.9, a practitioner's
  "maximum desirable valve position") and the feed that takes.

The reference is hand-built, not the ladder's: the shipped SIMC PI on the water,
started bumpless, and a valve position controller on the feed (an integral that
holds the water valve at its rest position) with a proportional feed cut as the
temperature nears the top of its band. Its numbers come from the plant's steady
balances, written where they are set. The ceiling is ``pc_gym.foresight``'s.
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
    valve_rest: float = 0.9  # of the water valve's span (ours)
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
        C_a, T_j, q = rest(T, T_feed, F_w, p)
        lo, hi = p.event_window
        at = jax.random.uniform(event_key, minval=lo, maxval=hi)
        state = ProdState(
            time=0, C_a=C_a, T=T, target_T=T, F_w=F_w, T_j=T_j, T_feed=T_feed,
            feed_phase=phase, event_at=at, event_until=at + p.event_hold,
            event_size=event_sizes(p), q=jnp.clip(q, p.q_min, p.q_max),
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
#: of a kelvin backed off one tenth: the band's two edges and the interlock.
CEILING_TERMS = ("lost_production",)


def ceiling_margins(state, p):
    off = state.T - state.target_T
    edges = [p.T_band - off, p.T_band + off, p.T_trip - state.T]
    return jnp.stack([e / p.e_floor - 1.0 for e in edges]), R.trip_cost(p)


class Reference:
    """The hand-built reference, vectorised over episodes on the host.

    Water: the shipped SIMC PI (Kp 0.105 raw per K, Ti 15.4 min,
    ``data/pid_gains.json``), started from the valve as the reset left it, on
    the setpoint plus ``lift``. Feed: an integral holding the water valve at
    ``valve`` (raw), with closed-loop time ``vpc_min``. The valve answers the
    feed by 1.36 raw per raw at the design point (the steady balances: 90 to
    100 g/s of feed takes 120 to 131.9 g/s of water), so the integral gain is
    1 / (1.36 vpc_min). Past ``T_band - guard`` above the setpoint the feed is
    cut by ``cut`` raw per K.
    """

    oracle, name = False, "reference"

    def __init__(self, p, valve, lift=0.0, vpc_min=75.0, cut=1.0, guard=0.3, gain=1.0):
        self.p, self.valve, self.lift, self.gain = p, valve, lift, gain
        self.ki, self.cut, self.guard = 1.0 / (1.36 * vpc_min), cut, guard

    def reset(self, key, obs, task):
        o = np.asarray(obs, float)
        return {"i": np.zeros(len(o)), "w": o[:, 2] / 50 - 1, "f": o[:, 3] / 50 - 1}

    def act(self, m, obs, t):
        o, dt = np.asarray(obs, float), float(self.p.delta_t)
        e = o[:, 0] - (o[:, 4] + self.lift)  # a hot reactor opens the water
        integral = m["i"] + e * dt
        asked = m["w"] + self.gain * 0.105366 * (e + integral / 15.40763)
        water = np.clip(asked, -1.0, 1.0)
        integral = np.where(asked == water, integral, m["i"])
        feed = np.clip(m["f"] + self.ki * dt * (self.valve - water), -1.0, 1.0)
        over = np.maximum(o[:, 0] - (o[:, 4] + float(self.p.T_band) - self.guard), 0.0)
        action = np.stack([water, np.clip(feed - self.cut * over, -1.0, 1.0)], axis=1)
        return action, {"i": integral, "w": m["w"], "f": feed}


SCENARIOS = {  # the shipped suite less the setpoint move, with its weights
    "normal": ({}, 0.7),
    "catalyst_loss": ({"k0_drop": 0.4, "k0_decay": 90.0}, 0.1),
    "feed_buildup": ({"feed_rise": 0.1}, 0.05),  # with a feed to cut, S3's 10 %
    "cooling_loss": ({"cooling_loss": 5.0}, 0.05),
}
RECORD = {
    "made": lambda s, p, t: made(s, p),
    "T": lambda s, p, t: s.T,
    "off": lambda s, p, t: jnp.abs(s.T - s.target_T) > p.T_band,
    "Fw": lambda s, p, t: s.F_w,
    "q": lambda s, p, t: s.q,
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
        "off_spec": rec["off"].mean(1).tolist(),
        "T_max": rec["T"].max(1).tolist(),
        "valve": ((rec["Fw"].mean(1) / 60 - 40) / 96).tolist(),
        "feed_gps": (rec["q"].mean(1) * 1000 / 60).tolist(),
    }
    return summary, np.asarray(ro["action"], float), rec


def show(name, label, r):
    print(
        f"{name:14s} {label:36s} cost {np.mean(r['cost']):.5f}"
        f" made {np.mean(r['made']):.3f} mol/min off-spec {np.mean(r['off_spec']):.3f}"
        f" valve {np.mean(r['valve']):.3f} feed {np.mean(r['feed_gps']):.1f} g/s"
        f" Tmax {max(r['T_max']) - 273.15:.2f} C trips {sum(r['trips'])}",
        flush=True,
    )


def main():
    out = pathlib.Path(sys.argv[1])
    seeds = tuple(range(100, 100 + (int(sys.argv[2]) if len(sys.argv) > 2 else 10)))
    env, result, t0 = ProductionCSTR(), {}, time.time()
    base = ProdParams(sensor_noise_scale=1.0)
    rest_valve = 2.0 * float(base.valve_rest) - 1.0
    # The charter's zone rule (J1): two standard deviations inside each edge, the
    # deviations read on the normal day under the practitioner's reference, over
    # the second half of each window.
    _, moves, rec = run(env, base, Reference(base, rest_valve), seeds)
    half = moves.shape[1] // 2
    sig_w, sig_T = float(np.std(moves[:, half:, 0])), float(np.std(rec["T"][:, half:]))
    result["sigma"] = {"valve_raw": sig_w, "T_K": sig_T}
    print("sigma", result["sigma"], flush=True)
    for name, (over, weight) in SCENARIOS.items():
        p = base.replace(**over)
        rows, starts = {}, {}
        rows["reference"], starts["reference"], _ = run(
            env, p, Reference(p, rest_valve), seeds
        )
        edge = Reference(
            p, 1.0 - 2 * sig_w, lift=float(p.T_band) - 2 * sig_T, guard=sig_T
        )
        label = "reference, 2 sigma inside the edges"
        rows[label], starts["edge"], _ = run(env, p, edge, seeds)
        plans = []
        for i, seed in enumerate(seeds):
            first = starts["reference"][i]
            inits = {"hold": np.tile(first[:1], (len(first), 1)), "reference": first}
            inits["edge"] = starts["edge"][i]
            best = None
            for start, init in inits.items():
                plan, cert = F.solve(
                    env, p, seed, ceiling_margins, CEILING_TERMS, init, maxiter=3000
                )
                if cert["tripped"] or (best and cert["planned_cost"] >= best[1]):
                    continue
                best = (plan, cert["planned_cost"], start, cert["status"])
            plans.append(best[0])
            print(
                f"  {name} seed {seed}: ceiling {best[1]:.5f} from {best[2]},"
                f" status {best[3]}, {time.time() - t0:.0f} s",
                flush=True,
            )
        rows["ceiling"], _, _ = run(env, p, F.PlanPolicy(np.stack(plans)), seeds)
        result[name] = {"weight": weight, **rows}
        for label, r in rows.items():
            show(name, label, r)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cstr_production.json").write_text(json.dumps(result, indent=1))
    print("done", flush=True)


def grid():
    """How much the hand-built reference's own settings move its production: the
    valve it holds, how far above the setpoint it runs, and its water loop's gain.
    Closed loop only, so it says what a classical loop reaches, not a ceiling."""
    out = pathlib.Path(sys.argv[1])
    seeds, env, result = tuple(range(100, 110)), ProductionCSTR(), {}
    base = ProdParams(sensor_noise_scale=1.0)
    for name, (over, _) in SCENARIOS.items():
        p = base.replace(**over)
        for valve in (0.9, 0.95, 0.98):
            for lift in (0.0, 0.5):
                for gain in (1.0, 8.0):
                    ref = Reference(p, 2 * valve - 1, lift=lift, gain=gain)
                    label = f"valve {valve}, lift {lift}, gain x{gain:g}"
                    result[f"{name}: {label}"], _, _ = run(env, p, ref, seeds)
                    show(name, label, result[f"{name}: {label}"])
    (out / "cstr_production_grid.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    grid() if sys.argv[2:3] == ["grid"] else main()
