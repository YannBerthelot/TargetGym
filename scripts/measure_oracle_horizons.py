"""Measure each oracle's planning horizon by Cooper's DMC recipe.

Run with ``uv run python scripts/measure_oracle_horizons.py [--envs ...]``.
It runs docs/proposals/oracle-horizon-protocol.md, with Amendments 1 and 2, on the
14 process and industrial tasks and writes one file per task,
``src/target_gym/data/oracle_horizons/<task>.json``, a compact summary that
ships in the wheel, and ``evidence/oracle_horizons/<task>.json`` at the
repository root, every test, retest and fit, in git but not in the wheel
(H13, H19). The 10
aircraft tasks' tests fly through ArduPlane 4.0.9's default loops (decision D6),
and tasks sharing a plant run its test once (Amendment 5, H24).

Per training seed 100-109 (``target_gym.identification``):

1. the shipped expert runs one episode from the seed's reset; the moves then
   hold the mean of its last 20 commands (the glass furnace: its last
   reversal cycle), the steady operation of amended H8;
2. each MV is stepped alone by its C2 step, away from the nearer limit, and
   every CV of H6 is read against the unstepped run for one episode, up to
   the first trip of either run (H8, H9);
3. both structures are fitted and judged (H14), failed tests are retested
   (H15, H18), and each pair-seed ends valid, insignificant or invalid under
   its CV's class (H16, H17, H19).

Only valid fits enter T_c. P also reads a fit that failed V1 alone at a fit of
at least 95 % (``target_gym.oracles.horizon``, Amendment 4). A task with no
valid fit is non-compliant: no protocol horizon is written, and it keeps its
current oracle (Amendment 2, N3). unstable_cstr is measured
on its stabilised plant (section 4.4): the cascade's outer loop in manual,
the inner PD closed, its setpoint stepped 0.5 K down as in the expert test.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import jax.numpy as jnp
import numpy as np

from target_gym import records
from target_gym.identification import (
    TRAIN_SEEDS,
    Plant,
    evidence,
    identify,
    quiet_fits,
)
from target_gym.oracles.horizon import (
    episode_samples,
    sample_time,
    task_horizon,
    task_ramp,
)
from target_gym.registry import REGISTRY

EVIDENCE = pathlib.Path(__file__).resolve().parent.parent / "evidence"
#: What the package keeps of a task's horizon (H13); the rest is evidence.
SUMMARY = ("protocol", "compliant", "T_c_steps", "N", "M", "ramp", "P")
SUMMARY += ("T_c_set_by", "P_set_by", "ramp_set_by", "counts", "cpu", "generated")
#: H21: a task's tracked CV with a target range, and each rate-limited MV's
#: slower rate, in raw action units per second as the steps and gains are.
RATE_LIMITS = {
    "compressor_surge": (
        "dp",
        lambda p: p.p_ref_range[1] - p.p_ref_range[0],
        lambda p: {
            "speed": 2 * p.drive_rate / (p.N_max - p.N_min),
            "recycle": 2 * min(p.valve_open_rate, p.valve_close_rate),
        },
    ),
    "wind_turbine": (
        "P",
        lambda p: (p.target_power_range[1] - p.target_power_range[0]) / 1.0e6,
        lambda p: {
            "pitch": 2 * p.pitch_rate_max / (p.pitch_max - p.pitch_min),
            "torque": 2 * p.torque_rate_max / p.torque_max,
        },
    ),
    "glass_furnace": (
        "T_crown",
        lambda p: p.target_T_crown_range[1] - p.target_T_crown_range[0],
        lambda p: {"fuel": 2 * p.actuator_rate},
    ),
    "boiler_drum": (
        "pressure",
        lambda p: p.target_pressure_range[1] - p.target_pressure_range[0],
        lambda p: {"fuel": 2 * p.actuator_rate[0], "feedwater": 2 * p.actuator_rate[1]},
    ),
    "reactor": (
        "T_avg",
        lambda p: _tref_span(p),
        lambda p: {"rod": 2 * p.rod_speed / (p.rho_ext_max - p.rho_ext_min)},
    ),
}


def _tref_span(p):
    """What the 100-50-100 day moves Tref by."""
    from target_gym.reactor.env import tavg_program

    return tavg_program(1.0, p) - tavg_program(p.load_min, p)


def write(name: str, row: dict) -> None:
    """The compact summary in the package and the full evidence beside the
    repository's sources (H13)."""
    evidence = EVIDENCE / "oracle_horizons" / f"{name}.json"
    summary = {k: row[k] for k in SUMMARY if k in row}
    # The evidence keeps which test each class came from.
    summary["classes"] = {
        c.split(":")[-1]: v["class"] for c, v in row["classes"].items()
    }
    summary["evidence"] = evidence.relative_to(EVIDENCE.parent).as_posix()
    records.write("oracle_horizons", name, summary, compact=True)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


def _s(field, lo=None, hi=None):
    """A CV read from the state, with its trip limits as params fields."""
    return (lambda obs, state, p: getattr(state, field)), lo, hi


def _kiln_lime(obs, state, p):
    from target_gym.cement_kiln.env import discharge_lime

    return discharge_lime(state)


def _omega(factor):
    from target_gym.energy.wind_turbine.env import omega_rated

    return lambda p: getattr(p, factor) * omega_rated(p)


_H = {f"h{i}": _s(f"h{i}", "h_min", "h_max") for i in range(1, 5)}
_GLASS = {c: _s(c, "T_glass_min", "T_glass_max") for c in ("T_melt", "T_work")}


def _air_speed(s, p):
    """The airspeed the trips read, in the wind the state is left in."""
    from target_gym.plane.dynamics import total_wind_2d, total_wind_3d

    if hasattr(s, "y_dot"):
        wind = total_wind_3d(s.z, s.gust_x, s.gust_y, s.gust_z, p)
        v = jnp.stack([s.x_dot - wind[0], s.y_dot - wind[1], s.z_dot - wind[2]])
    else:
        wind = total_wind_2d(s.z, s.gust_x, s.gust_z, p)
        v = jnp.stack([s.x_dot - wind[0], s.z_dot - wind[1]])
    return jnp.sqrt(jnp.sum(v**2))


def _eas(o, s, p):
    from target_gym.plane.dynamics import compute_air_density_from_altitude as rho

    return _air_speed(s, p) * jnp.sqrt(rho(s.z) / p.air_density_at_sea_level)


def _mach(o, s, p):
    from target_gym.plane.dynamics import compute_speed_of_sound_from_altitude as a

    return _air_speed(s, p) / a(s.z)


_ALTITUDE = _s("z", "min_alt", "max_alt")
#: H6: every variable with a trip limit in ``check_is_terminal``.
_ENVELOPE = {
    "eas": (_eas, None, "vmo_eas"),
    "mach": (_mach, None, "mmo"),
    "load": _s("load_factor", "load_factor_min", "load_factor_max"),
}
_LIMITS_2D = {"altitude": _ALTITUDE, **_ENVELOPE}
_LIMITS_3D = {**_LIMITS_2D, "bank": _s("phi", lambda p: -p.bank_limit, "bank_limit")}
_LONGITUDINAL = ({"throttle": 0.1, "pitch": 0.1}, _LIMITS_2D, {})
#: Section 4.5: a path task's bank demand comes from a course hold, stepped 2 deg.
_COURSE_STEP = math.radians(2.0)


def _path(distance):
    """A path task's own row: the course step, read on its distance. The limit CVs
    are read only so that a trip names its CV (H9)."""
    steps = {"throttle": 0.1, "pitch": 0.1, "course": _COURSE_STEP}
    cvs = {"path": ((lambda o, s, p: distance(s, p)), None, None), **_LIMITS_3D}
    return steps, cvs, {}


def _follower(cv):
    """A CV of an aircraft, read on a patrol's follower."""
    read, lo, hi = cv
    return (lambda o, s, p: read(o, s.follower, p)), lo, hi


def _slot_axis(axis):
    def read(o, s, p):
        from target_gym.patrol.env import slot_error_vector

        return slot_error_vector(s)[axis]

    return read


def _separation(o, s, p):
    from target_gym.patrol.env import separation

    return separation(s)


def _slot_error(o, s, p):
    from target_gym.patrol.env import slot_error

    return slot_error(s)


_PATROL_LIMITS = {
    **{k: _follower(v) for k, v in _LIMITS_3D.items()},
    "separation": (_separation, "min_separation", None),
    "slot": (_slot_error, None, "max_slot_error"),
}


def _lemniscate_distance(s, p):
    from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

    return nearest_point_on_twisted_lemniscate(s, p)[3]


def _circle_distance(s, p):
    from target_gym.plane3d.env import distance_to_circle

    return distance_to_circle(s)


def _racetrack_distance(s, p):
    from target_gym.plane3d.env import racetrack_guidance

    return racetrack_guidance(s)[0]


# Per task: the MVs with their C2 steps in raw units (existing steps stay, a
# new one is 0.1, 5 % of [-1, 1]); the CVs of H6 (tracked outputs, then every
# variable check_is_terminal trips on) with their trip limits; and, on the
# multi-input tasks, the MV the shipped expert holds each CV with (H18).
TASKS = {
    "first_order": ({"u": 0.1}, {"x": _s("x", "x_min", "x_max")}, {}),
    "cstr": ({"coolant": 0.1}, {"C_a": _s("C_a")}, {}),
    "four_tank": ({"pump1": 0.1, "pump2": 0.1}, _H, {"h1": "pump1", "h2": "pump2"}),
    # +0.1 mL/s on the 10 to 22 mL/s base flow.
    "ph_neutralization": ({"base": 2 * 0.1 / 12.0}, {"pH": _s("pH")}, {}),
    "distillation": (
        {"L": 0.1, "V": 0.1},
        {
            "yD": ((lambda o, s, p: s.x[-1]), "yD_floor", None),
            "xB": ((lambda o, s, p: s.x[0]), None, "xB_ceiling"),
        },
        {"yD": "L", "xB": "V"},
    ),
    "glass_furnace": (
        {"fuel": 0.05},
        {"T_crown": _s("T_crown", "T_crown_min", "T_crown_max"), **_GLASS},
        {},
    ),
    "reactor": (
        {"rod": 0.1},
        {
            "T_avg": (
                (lambda o, s, p: 0.5 * (s.T_coolant + s.T_cold)),
                "T_avg_min",
                None,
            ),
            "n": _s("n", "n_min", "n_max"),
            "T_fuel": _s("T_fuel", "T_fuel_min", "T_fuel_max"),
            "T_hot": _s("T_coolant", None, "T_coolant_max"),
        },
        {},
    ),
    "hvac": ({"heating": 0.1}, {"T_air": _s("T_air", "T_air_min", "T_air_max")}, {}),
    "cement_kiln": (
        {"fuel": 0.1, "speed": 0.1},
        {
            "lime": (_kiln_lime, None, "lime_max"),
            "T_bz": ((lambda o, s, p: jnp.max(s.T_solid)), "T_bz_min", "T_bz_max"),
        },
        {"lime": "fuel", "T_bz": "fuel"},
    ),
    # Firing +0.02 raw; feedwater +2.0 kg/s on 0 to 200 kg/s.
    "boiler_drum": (
        {"fuel": 0.02, "feedwater": 2 * 2.0 / 200.0},
        {
            "level": ((lambda o, s, p: s.level), lambda p: -p.level_trip, "level_trip"),
            "pressure": _s("pressure", "pressure_min", "pressure_max"),
        },
        {"level": "feedwater", "pressure": "fuel"},
    ),
    "wind_turbine": (
        {"pitch": 0.1, "torque": 0.1},
        {
            "P": ((lambda o, s, p: o[3]), None, None),
            "omega": _s(
                "omega", _omega("underspeed_factor"), _omega("overspeed_factor")
            ),
        },
        {"P": "torque", "omega": "pitch"},
    ),
    "battery": (
        {"power": 0.1},
        {
            "P": ((lambda o, s, p: o[3]), None, None),
            "soc": _s("soc", "soc_min", "soc_max"),
            "V_cell": ((lambda o, s, p: o[1]), "V_cell_min", "V_cell_max"),
            "T_cell": _s("T_cell", None, "T_max"),
        },
        {},
    ),
    # Speed +1.75 % of rated on 70 to 105 %; recycle +0.05 on 0 to 1.
    "compressor_surge": (
        {"speed": 2 * 0.0175 / 0.35, "recycle": 2 * 0.05 / 1.0},
        {"dp": _s("dp"), "phi_min": _s("phi_min", lambda p: 2 * p.W, None)},
        {"dp": "speed", "phi_min": "recycle"},
    ),
    "unstable_cstr": (
        {"T_sp": -0.5},
        {"C_a": _s("C_a", lambda p: 0.0, None), "T": _s("T", "T_valid_min", "T_trip")},
        {},
    ),
    # Section 4.5: the plant's inner loop holds the pitch demand, 5 % of its
    # range, or the throttle moves 5 %.
    "plane": _LONGITUDINAL,
    # Bank demand, 5 % of +-45 deg, read on the track.
    "plane3d_heading": (
        {"throttle": 0.1, "pitch": 0.1, "bank": 0.1},
        {"heading": _s("psi"), **_LIMITS_3D},
        {},
    ),
    "plane3d_circle": _path(_circle_distance),
    "plane3d_racetrack": _path(_racetrack_distance),
    "plane3d_figure8": _path(_lemniscate_distance),
    # Section 4.5's altitude rows on a follower, whose air the lead's wake moves.
    "patrol": ({"throttle": 0.1, "pitch": 0.1, "bank": 0.1}, _PATROL_LIMITS, {}),
    # Section 4.5's slot rows: speed +5 m/s and course 2 deg, through TECS and a
    # course hold. The height demand holds.
    "patrol_slots": (
        {"speed": 5.0, "height": 0.0, "course": _COURSE_STEP},
        {
            "along": (_slot_axis(0), None, None),
            "across": (_slot_axis(1), None, None),
            **_PATROL_LIMITS,
        },
        {},
    ),
}

#: Amendment 5, H24: tasks sharing a plant run its test once, on the shortest
#: episode's window. Each task's fits come from these tests.
PLANT_TESTS = {
    "plane": ("plane",),
    "plane_energy": ("plane",),
    "plane_sine": ("plane",),
    "plane3d_heading": ("plane3d_heading",),
    "plane3d_circle": ("plane3d_heading", "plane3d_circle"),
    "plane3d_racetrack": ("plane3d_heading", "plane3d_racetrack"),
    "plane3d_figure8": ("plane3d_heading", "plane3d_figure8"),
    "patrol": ("patrol", "patrol_slots"),
    "patrol_range_bearing": ("patrol", "patrol_slots"),
    # Section 4.5: its lead's rows as plane3d_heading, its wingmen's as patrol.
    "formation": ("plane3d_heading", "patrol", "patrol_slots"),
}
#: The window of a family's test, the shortest episode among its tasks, and what a
#: test's own row reads: ``(MVs, CVs)``, every one if not listed.
TEST_WINDOW = {
    "plane3d_circle": 300, "plane3d_racetrack": 300, "plane3d_figure8": 300,
    "patrol": 200, "patrol_slots": 200,
}  # fmt: skip
TEST_ROWS = {
    "plane3d_circle": (("course",), ("path",)),
    "plane3d_racetrack": (("course",), ("path",)),
    "plane3d_figure8": (("course",), ("path",)),
    "patrol_slots": (("speed", "course"), ("along", "across")),
}
#: Why a task does not read its own window (Amendment 5, H24).
OWN_WINDOW = {
    "plane_energy": (
        "a test on this task's own 1600-step window gave 0 valid of 20: a fit on the "
        "first 280 steps tracks the same run within about 2 % at 420 s, and drifts by "
        "8 to 13 % at 800 s and 31 to 45 % at 1600 s, as the climb moves the air"
    ),
    "plane_sine": "takes plane's test: the same dynamics, on the shortest window",
}
#: The registered task a test runs on.
TEST_TASK = {"patrol_slots": "patrol"}


def _stabilised(p):
    """The unstable_cstr base loop: the expert's inner PD on T, its setpoint
    held at the mean of the cascade's last 20 outputs and stepped as given."""
    from target_gym.experts.pid import filtered_derivative
    from target_gym.pc_gym.unstable_cstr import experts as X

    g, dt = X.load_gains(), float(p.delta_t)

    def actuate(u, y, T_prev):
        T = y[1]  # the CV "T"
        T_prev = T if T_prev is None else T_prev
        rate, T_filtered = filtered_derivative(T, T_prev, g["Kp_T"], g["Kd_T"], dt)
        T_c = g["Tc_bias"] + g["Kp_T"] * (u[0] - T) - g["Kd_T"] * rate
        raw = float(X.raw_from_coolant(np.clip(T_c, p.T_c_min, p.T_c_max), p))
        return [raw], T_filtered

    return {
        "actuate": actuate,
        "start": lambda acts, mems: np.asarray(mems.T_sp_prev)[:, -20:].mean(1),
        "mv_range": (-math.inf, math.inf),
        "direction": lambda u, size: size,
    }


#: NAVL1_PERIOD and NAVL1_DAMPING at ArduPlane 4.0.9's defaults.
_L1_PERIOD, _L1_DAMPING = 17.0, 0.75


def _default_loops(spec):
    """What settles an aircraft (decision D6): TECS and L1 at ArduPlane 4.0.9's
    defaults on this aircraft's measured numbers, not the expert's gains, so the
    horizon stays fixed when the expert is retuned. A course hold's time constant
    is the period over 8.9 (aircraft protocol, 3.4)."""
    ctrl = spec.make_pid_controller()
    p = ctrl.params
    tecs = p.tecs.replace(
        time_const=5.0, ptch_damp=0.0, integ_gain=0.1, thr_damp=0.5,
        spd_weight=1.0, roll_comp=10.0,
    )  # fmt: skip
    nav = {}
    if hasattr(p, "period"):
        nav = dict(period=_L1_PERIOD, damping=_L1_DAMPING, tau_hdg=_L1_PERIOD / 8.9)
    return ctrl.replace(params=p.replace(tecs=tecs, **nav))


def _course_hold():
    """Section 4.5's course demand through ``update_heading_hold``: lateral
    acceleration 2 V w sin(Nu), w = sqrt(2) pi / NAVL1_PERIOD, Nu clipped to +-90
    deg, as a coordinated turn's bank. The MV starts on the settled track."""
    from target_gym.plane.dynamics import BANK_DEMAND_MAX

    w = math.sqrt(2.0) * math.pi / _L1_PERIOD

    def track(obs, state, p):
        return state.psi

    def speed(obs, state, p):
        return jnp.sqrt(state.x_dot**2 + state.y_dot**2 + state.z_dot**2)

    def actuate(u, y, memory):
        psi, v = y[-2], y[-1]
        nu = math.atan2(math.sin(u[2] - psi), math.cos(u[2] - psi))
        nu = float(np.clip(nu, -math.pi / 2.0, math.pi / 2.0))
        bank = math.atan(2.0 * v * w * math.sin(nu) / 9.81)
        raw = np.clip([u[0], u[1], bank / BANK_DEMAND_MAX], -1.0, 1.0)
        return raw, memory

    def settled(u0, y):
        u0 = np.array(u0)
        u0[:, 2] = y[:, -2]
        return u0

    return {
        "actuate": actuate, "mv_range": (-math.inf, math.inf), "settled": settled,
        "reads": (track, speed),
        "direction": lambda u, size: size,
    }  # fmt: skip


def _slot_hold(spec):
    """Section 4.5's slot rows: the speed and height demands through TECS at its
    defaults, the course through ``update_heading_hold``, on the follower."""
    from target_gym.plane.dynamics import BANK_DEMAND_MAX, PITCH_DEMAND_MAX
    from target_gym.plane.experts_tecs import TECSState, tecs_step

    tecs = _default_loops(spec).params.tecs
    w = math.sqrt(2.0) * math.pi / _L1_PERIOD

    def follower(field):
        return lambda o, s, p: getattr(s.follower, field)

    def speed(o, s, p):
        a = s.follower
        return jnp.sqrt(a.x_dot**2 + a.y_dot**2 + a.z_dot**2)

    def actuate(u, y, memory):
        psi, v, z, z_dot, phi = y[-5:]
        memory = TECSState() if memory is None else memory
        power, pitch, memory = tecs_step(tecs, memory, z, z_dot, v, u[1], u[0], phi, np)
        nu = math.atan2(math.sin(u[2] - psi), math.cos(u[2] - psi))
        nu = float(np.clip(nu, -math.pi / 2.0, math.pi / 2.0))
        bank = math.atan(2.0 * v * w * math.sin(nu) / 9.81)
        raw = [power, pitch / PITCH_DEMAND_MAX, bank / BANK_DEMAND_MAX]
        return np.clip(raw, -1.0, 1.0), memory

    def settled(u0, y):
        return np.stack([y[:, -4], y[:, -3], y[:, -5]], axis=1)

    return {
        "actuate": actuate, "mv_range": (-math.inf, math.inf), "settled": settled,
        "reads": (
            follower("psi"), speed, follower("z"), follower("z_dot"), follower("phi"),
        ),
        "direction": lambda u, size: size,
    }  # fmt: skip


def _clean(x, dt):
    """JSON-safe, to 6 significant digits (every decision is taken before):
    non-finite numbers as strings; a delay or lag at its lower bound written
    "< dt" (10.3)."""
    if isinstance(x, dict):
        low = {"theta", "tau", "tau2"}
        return {
            k: (
                "< dt"
                if k in low and isinstance(v, float) and v <= 1e-6 * dt
                else _clean(v, dt)
            )
            for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [_clean(v, dt) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.6g}") if math.isfinite(x) else str(float(x))
    return x


def _test_record(run, a, mvs, cvs):
    loops = [
        {"cv": cvs[lp.cv], "mv": mvs[lp.mv], "kc": lp.kc, "n_pre": lp.n_pre}
        for lp in run.loops
    ]
    trips = {
        k: {"index": i, "cv": cvs[c] if c is not None else None}
        for k, (i, c) in run.trips.items()
    }
    head = {"step": run.step, "window": run.window, "loops": loops, "trips": trips}
    return {**head, "samples": a["samples"], **a["fits"]}


_RUNS: dict = {}


def run_test(test: str) -> dict:
    """One bump test, once a process: its pairs under the protocol's keys, its
    classes, and its plant's step and episode."""
    if test in _RUNS:
        return _RUNS[test]
    spec = REGISTRY[TEST_TASK.get(test, test)]
    window = TEST_WINDOW.get(test)
    p = spec.make_test_params(**({"max_steps_in_episode": window} if window else {}))
    steps, cv_defs, expert = TASKS[test]
    mvs, cvs = list(steps), list(cv_defs)

    def limit(b):
        return None if b is None else float(b(p) if callable(b) else getattr(p, b))

    extra = _stabilised(p) if test == "unstable_cstr" else {}
    if test == "glass_furnace":
        extra["window"] = int(round(float(p.reversal_period) / float(p.delta_t)))
    if spec.group == "aircraft":
        extra["controller"] = _default_loops(spec)
    if "speed" in steps:
        extra.update(_slot_hold(spec))
    elif "course" in steps:
        extra.update(_course_hold())
    if "heading" in cvs:
        extra["angles"] = (cvs.index("heading"),)
    plant = Plant(
        spec,
        p,
        [f for f, _, _ in cv_defs.values()],
        [(limit(lo), limit(hi)) for _, lo, hi in cv_defs.values()],
        expert_mv={cvs.index(c): mvs.index(m) for c, m in expert.items()},
        **extra,
    )
    row_mvs, row_cvs = TEST_ROWS.get(test, (mvs, cvs))
    tested = [mvs.index(m) for m in row_mvs]
    sizes = list(steps.values())
    pairs, classes = identify(plant, sizes, tested)
    read = {cvs.index(c) for c in row_cvs}
    pairs = {k: v for k, v in pairs.items() if k[1] in read}
    for r in pairs.values():
        r["names"] = (mvs, cvs)
    run = dict(
        pairs=pairs, classes=classes, mvs=mvs, cvs=cvs, p=p, dt=plant.dt,
        n_ep=plant.n_ep, steps=steps, extra=extra,
    )  # fmt: skip
    _RUNS[test] = run
    return run


def measure(name: str) -> dict:
    tests = PLANT_TESTS.get(name, (name,))
    runs = [run_test(t) for t in tests]
    p = REGISTRY[name].make_test_params()
    first = runs[0]
    dt, n_ep = first["dt"], int(p.max_steps_in_episode)
    steps, extra = first["steps"], first["extra"]
    key, pairs, classes, mvs, cvs, test_of = {}, {}, {}, [], [], {}
    for t, run in zip(tests, runs):
        n_m, n_c = len(mvs), len(cvs)
        mvs += [f"{t}:{m}" if len(tests) > 1 else m for m in run["mvs"]]
        cvs += [f"{t}:{c}" if len(tests) > 1 else c for c in run["cvs"]]
        for (j, i, s), r in run["pairs"].items():
            at = (j + n_m, i + n_c, s)
            pairs[at] = r
            key[at] = (mvs[j + n_m], cvs[i + n_c], TRAIN_SEEDS[s])
            test_of[at] = t
        classes.update({i + n_c: c for i, c in run["classes"].items()})
    valid = {key[k]: r["fit"] for k, r in pairs.items() if r["label"] == "valid"}
    quiet = {(*key[k[:3]], k[3]): f for k, f in quiet_fits(pairs, classes).items()}
    tc, rows, top = task_horizon(valid, dt, quiet)
    at = ("mv", "cv", "seed", "test")
    horizon = {}
    if top is not None:  # else non-compliant: no protocol horizon (N3)
        (N, M, P), n_samples = rows[top], episode_samples(n_ep, tc)
        fastest = min(valid, key=lambda k: sample_time(valid[k]))
        ramp, ramp_key = 0, None
        without = max(v[2] for k, v in rows.items() if len(k) == 3)
        if name in RATE_LIMITS:
            # A gain needs a settled fit, not a quiet one: V1, the noise test
            # alone failing, leaves it usable (H21).
            usable = {
                key[k]: r["fit"]
                for k, r in pairs.items()
                if r.get("fit") and r["label"] != "insignificant"
                and set(r.get("failed", [])) <= {"V1"}
            }  # fmt: skip
            cv, span, rates = RATE_LIMITS[name]
            ramp, ramp_key = task_ramp(usable, cv, float(span(p)), rates(p), tc * dt)
        P += ramp
        horizon = {
            "episode_samples": n_samples,
            "T_c_steps": tc,
            "T_c_seconds": tc * dt,
            "T_c_set_by": dict(zip(at, fastest)),
            "N": N,
            "M": M,
            "ramp": ramp,
            "P": P,
            "P_without_quiet_fits": without + ramp,
            "P_steps": P * tc,
            "P_set_by": dict(zip(at, top)),
            **({"ramp_set_by": dict(zip(at, ramp_key))} if ramp_key else {}),
            "plans_to_episode_end": P >= n_samples,
        }
    labels = [r["label"] for r in pairs.values()]
    read = {key[k][:2] for k in pairs}
    zero = sorted(
        f"{m} -> {c}"
        for m, c in read
        if all(
            r["label"] == "insignificant"
            for k, r in pairs.items()
            if key[k][:2] == (m, c)
        )
    )
    first_cv = {j: min(i for jj, i, _ in pairs if jj == j) for j, _, _ in pairs}
    row = {
        "protocol": "docs/proposals/oracle-horizon-protocol.md, Amendments 1 to 5",
        "dt": dt,
        "episode_steps": n_ep,
        "compliant": bool(horizon),
        "status": (
            "compliant"
            if horizon
            else "non-compliant: no valid pair, so no protocol horizon; the task "
            "keeps its current oracle, and C1 and C2 leave it as it is (N3)"
        ),
        **horizon,
        "counts": {
            **{k: labels.count(k) for k in ("valid", "invalid", "insignificant")},
            "quiet": len(quiet),
        },
        "quiet_fits": [
            {**dict(zip(at, k)), "fit": f, "P": rows.get(k, (None,) * 3)[2]}
            for k, f in quiet.items()
        ],
        "retests": sum(
            len(r["tests"]) - 1 for (j, i, s), r in pairs.items() if i == first_cv[j]
        ),
        "zero_models": zero,
        "cvs_without_valid_pair": sorted({c for _, c in read} - {k[1] for k in valid}),
        "classes": {cvs[i]: c for i, c in classes.items()},
        "tests": list(tests),
        "test": {
            "seeds": list(TRAIN_SEEDS),
            "settle_steps": first["n_ep"],
            "window_steps": first["n_ep"],
            "steps": steps,
            "hold_window": extra.get("window", 20),
            "noise": "residual of each fitted model over its fit window (N1)",
        },
        "pairs": [
            {
                **dict(zip(at, key[k])),
                "label": r["label"],
                "failed": r.get("failed", []),
                "kept": r["kept"],
                "evidence": evidence(r["tests"][r["kept"]][1]),
                "fit": r.get("fit"),
                **dict(zip(("N", "M", "P"), rows.get(key[k], (None,) * 3))),
                "tests": [_test_record(run, a, *r["names"]) for run, a in r["tests"]],
            }
            for k, r in pairs.items()
            if test_of[k] == name
        ],
    }
    if name in OWN_WINDOW:
        row["own_window"] = OWN_WINDOW[name]
    return _clean(row, dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--envs", nargs="*", default=list(TASKS) + list(PLANT_TESTS))
    args = ap.parse_args()
    known = [n for n in TASKS if n in REGISTRY and n not in PLANT_TESTS]
    known += list(PLANT_TESTS)
    for name in (n for n in args.envs if n in known):
        row = measure(name)
        write(name, row := records.stamped(row))
        if row["compliant"]:
            end = ", to the end" if row["plans_to_episode_end"] else ""
            what = (
                f"T_c {row['T_c_steps']}, P {row['P']} ({row['P_steps']} of "
                f"{row['episode_steps']} steps{end}), set by {row['P_set_by']}"
            )
        else:
            what = "NON-COMPLIANT, keeps its current oracle"
        classes = {c: v["class"] for c, v in row["classes"].items()}
        print(
            f"{name:18s} {what}; {row['counts']}, retests {row['retests']}; "
            f"classes {classes}"
        )


if __name__ == "__main__":
    main()
