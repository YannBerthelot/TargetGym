"""Central registry of every single-agent TargetGym environment.

Motivation
----------
Before this module existed, each environment was wired into the library by
hand in several independent places -- ``target_gym.__init__``, the runner
tables in ``target_gym.runners.runners``, the PID/MPC factory exports in
``target_gym.experts.__init__``, and the test suite.  Nothing checked that
those lists agreed, so an environment could silently fall out of one of them:
``glass_furnace`` was fully implemented, complete with its own runner module,
yet absent from every runner table, so ``make figures`` and ``make videos``
never touched it.

The registry is the single source of truth.  Anything that wants to iterate
over "all environments" -- the conformance test suite, the runner CLI, the
docs table -- reads it from here, so adding an environment to the library is
one edit and a missing baseline is a test failure rather than silence.

Multi-agent environments (``PlanePatrolMARL``) are deliberately *not*
registered: they expose a dict-based JaxMARL-style API rather than the
single-agent gymnax one, so the shared conformance contract does not apply.
They are covered by their own tests in ``tests/patrol/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, cast

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

#: Human-readable name for each group, in the order they should be presented.
GROUPS: dict[str, str] = {
    "aircraft": "Aircraft",
    "process": "Process Control",
    "industrial": "Industrial / Energy",
    "energy": "Renewable Energy",
}


# How a registry name is written for a human. Only the names that
# ``name.replace("_", " ").title()`` gets wrong need an entry -- that fallback
# turns acronyms into words ("Cstr", "Hvac") and lowercases the chemistry
# ("Ph Neutralization"). Kept here rather than in the documentation scripts
# because more than one of them needs it and two copies would drift.
DISPLAY_NAMES: dict[str, str] = {
    "cstr": "CSTR",
    "hvac": "Building HVAC",
    "ph_neutralization": "pH neutralisation",
    "four_tank": "Four-tank",
    "first_order": "First order",
    "plane3d_heading": "3D heading",
    "plane3d_circle": "3D circle",
    "plane3d_racetrack": "3D holding pattern",
    "plane3d_figure8": "3D figure-8",
    "plane_sine": "Altitude - sinusoid",
    "plane_energy": "Altitude and airspeed",
    "plane": "Altitude hold",
    "patrol": "Patrol - MARL formation",
    "patrol_bearing_only": "Patrol - MARL, bearing-only",
}


def display_name(name: str) -> str:
    """The human-facing name for a registry key."""
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# Keyed by spec name: EnvSpec is frozen, so the cache lives beside it.
_ENV_CACHE: dict[str, Any] = {}


def control_step_seconds(env, params) -> float:
    """Seconds of simulated time one ``step_env`` call advances.

    ``delta_t`` alone is not it. Two plants carry their models' native minutes
    (the PC-gym CSTR and distillation column declare ``time_unit_seconds``),
    and the reactor holds one action across ``control_period`` physics
    sub-steps. Without both factors the generated facts table labelled a
    200-minute distillation episode "3 min" and a 25-minute CSTR one "25 s",
    and the reactor's 2.4 h would read as 864 s.

    Lives here rather than in ``utils.py`` because that module is part of every
    environment's provenance fingerprint, and a bookkeeping helper must not
    move twenty-one fingerprints.
    """
    dt = float(getattr(params, "delta_t", 1.0))
    unit = float(getattr(params, "time_unit_seconds", 1.0))
    period = int(getattr(env, "control_period", 1))
    return dt * unit * period


@dataclass(frozen=True)
class EnvSpec:
    """Everything the library needs to know about one environment.

    Attributes
    ----------
    name:
        Registry key.  Matches the runner module prefix and the key used in
        ``src/target_gym/data/pid_gains.json`` where gains are tuned.
    group:
        One of :data:`GROUPS`.
    env_factory:
        Zero-argument callable returning a fresh environment instance.
    params_cls:
        The ``EnvParams`` subclass for this environment.
    make_pid:
        Zero-argument callable returning a *stateful* PID controller -- an
        object with ``reset()`` and ``__call__(obs) -> action``.  ``None``
        means no PID baseline exists for this environment yet.
    make_mpc:
        Callable ``(env, params) -> controller`` with ``reset()`` and
        ``step(obs, state) -> action``.  ``None`` means no MPC baseline
        exists for this environment yet.
    test_params:
        Parameter overrides producing an episode short enough for the test
        suite while still exercising the interesting dynamics.  Applied via
        ``params_cls(**test_params)``.
    tuned_gains_key:
        Key under which this environment's PID gains live in
        ``src/target_gym/data/pid_gains.json``.  ``None`` means the controller is not a
        single flat SISO loop and the gains are stored per sub-loop.
    baselines_note:
        Set when ``make_pid``/``make_mpc`` are ``None``: a short explanation
        of why, surfaced by the baseline-coverage test so a missing expert is
        a documented gap rather than a silent one.
    disturbance_fields:
        State fields holding a *zero-mean stochastic disturbance* (gusts, load
        noise). The conformance suite asserts these behave like disturbances --
        in particular that they do not ratchet monotonically when the
        environment is stepped with a constant PRNG key, which is how every
        rollout helper in this repo drives ``step_env``. Deliberately excludes
        deliberately-drifting processes such as the reactor's OU demand.
    disturbance_overrides:
        Parameter overrides that switch the disturbance on, when it is off by
        default (e.g. the aircraft's ``turbulence_sigma``).
    effectiveness_overrides:
        Parameter overrides for the controller-effectiveness contract, when the
        default ``test_params`` episode is too short to tell a working
        controller from a constant. The reactor needs this: its xenon transient
        runs for hours, so a 20-minute episode separates nothing.
    expert_degraded:
        Set when a baseline *exists and is well-formed* but does not yet meet
        the effectiveness contract -- it loses to a constant action. Distinct
        from ``baselines_note``, which marks a baseline that is absent
        entirely. Surfaced by the conformance suite so a weak expert is a
        recorded, explained gap rather than a silently bad benchmark number.
    mpc_degraded:
        The same, for the MPC: set when the MPC exists and runs but returns
        materially less than the PID, so it is not the upper bound the benchmark
        presents it as. The reason should carry the measured numbers.
    noise_fields:
        Names of parameters that scale process noise. A planner that rolls the
        real environment forward gets a copy of the params with these set to
        zero, so it plans on the mean disturbance instead of one invented
        realisation of it.
    """

    name: str
    group: str
    env_factory: Callable[[], Any]
    params_cls: Callable[..., Any]
    make_pid: Callable[[], Any] | None
    make_mpc: Callable[[Any, Any], Any] | None
    # Behavioural version of the environment, part of its public identity as
    # ``<name>-v<version>``. Every environment ships as v1 in the 0.6 release,
    # which is where versioning starts; nothing before that is versioned,
    # because the package had no users to preserve results for. Bump it
    # whenever the dynamics, the reward, the parameters or the observation
    # layout change, so that a published number keeps meaning what it meant.
    version: int = 1
    test_params: dict[str, Any] = field(default_factory=dict)
    tuned_gains_key: str | None = None
    baselines_note: str | None = None
    expert_degraded: str | None = None
    mpc_degraded: str | None = None
    effectiveness_overrides: dict[str, Any] = field(default_factory=dict)
    disturbance_fields: tuple[str, ...] = ()
    #: Parameters that scale *process* noise, zeroed in the copy of the params
    #: a planner uses for its internal model. See ``experts.mpc.plan_params``.
    noise_fields: tuple[str, ...] = ()
    disturbance_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def versioned_name(self) -> str:
        """The name a published result should cite, e.g. ``plane-v1``.

        Registry keys stay unversioned because they are an internal handle,
        used for gains, recorded baselines and file paths. This is the public
        identity: it changes when the environment's behaviour changes, so a
        number quoted against it stays meaningful. Bumping it is enforced by
        ``tests/test_env_versions.py``.
        """
        return f"{self.name}-v{self.version}"

    @property
    def has_pid(self) -> bool:
        return self.make_pid is not None

    @property
    def has_mpc(self) -> bool:
        return self.make_mpc is not None

    def make_env(self):
        """The environment for this spec, built once and shared.

        Deliberately not a fresh instance per call. ``env.step_env`` is a *bound
        method*, so every new instance is a new callable to JAX, which keys its
        compilation cache on the callable and holds the resulting executable for
        the life of the process. Building environments in a loop therefore leaks:
        measured on the 2D aircraft, forty fresh instances retained 104 MB, about
        2.6 MB apiece, and each one also paid its compile again. Re-wrapping the
        *same* instance in ``jax.jit`` repeatedly costs nothing, so the instance
        is what matters.

        Sharing is safe because these environments carry no per-episode state:
        every field is configuration (``obs_shape``, ``integration_method``,
        ``observe_wind``), and ``positions_history`` is a render buffer the
        stepping code never writes. All state lives in the ``EnvState`` passed
        through each call.

        Anyone constructing an environment directly rather than through the
        registry should hold onto the instance for the same reason.
        """
        cached = _ENV_CACHE.get(self.name)
        if cached is None:
            cached = _ENV_CACHE[self.name] = self.env_factory()
        return cached

    def make_test_params(self, **overrides):
        """Parameters for a short test episode, plus any extra overrides."""
        return self.params_cls(**{**self.test_params, **overrides})


# ---------------------------------------------------------------------------
# Lazy factories
#
# Importing the environments eagerly would make ``import target_gym.registry``
# pull in matplotlib, pygame, casadi and do-mpc.  Each factory therefore
# imports inside the call, so the registry itself stays cheap to import.
# ---------------------------------------------------------------------------


def _plane():
    from target_gym.plane.env_jax import Airplane2D

    return Airplane2D()


def _plane3d_heading():
    from target_gym.plane3d.env_jax import Plane3DHeading

    return Plane3DHeading()


def _plane3d_circle():
    from target_gym.plane3d.env_jax import Plane3DCircle

    return Plane3DCircle()


def _plane3d_racetrack():
    from target_gym.plane3d.env_jax import Plane3DRacetrack

    return Plane3DRacetrack()


def _plane3d_figure8():
    from target_gym.plane3d.env_jax import Plane3DFigureEight

    return Plane3DFigureEight()


def _patrol():
    from target_gym.patrol.env_jax import PlanePatrol

    return PlanePatrol()


def _patrol_bearing_only():
    from target_gym.patrol.env_jax import PlanePatrolBearingOnly

    return PlanePatrolBearingOnly()


def _cstr():
    from target_gym.pc_gym.cstr.env_jax import CSTR

    return CSTR()


def _first_order():
    from target_gym.pc_gym.first_order.env_jax import FirstOrderSystem

    return FirstOrderSystem()


def _distillation():
    from target_gym.pc_gym.distillation.env_jax import DistillationColumn

    return DistillationColumn()


def _ph_neutralization():
    from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization

    return PHNeutralization()


def _four_tank():
    from target_gym.pc_gym.four_tank.env_jax import FourTank

    return FourTank()


def _glass_furnace():
    from target_gym.glass_furnace.env_jax import GlassFurnace

    return GlassFurnace()


def _reactor():
    from target_gym.reactor.env_jax import Reactor

    return Reactor()


def _battery():
    from target_gym.energy.battery.env_jax import GridBattery

    return GridBattery()


def _wind_turbine():
    from target_gym.energy.wind_turbine.env_jax import WindTurbine

    return WindTurbine()


def _cement_kiln():
    from target_gym.cement_kiln.env_jax import CementKiln

    return CementKiln()


def _boiler_drum():
    from target_gym.boiler_drum.env_jax import BoilerDrum

    return BoilerDrum()


def _hvac():
    from target_gym.hvac.env_jax import BuildingHVAC

    return BuildingHVAC()


# -- params classes (imported lazily through the same mechanism) -------------


def _params_cls(module: str, name: str) -> type:
    from importlib import import_module

    # getattr is Any-typed; the cast records the contract callers rely on.
    return cast(type, getattr(import_module(module), name))


class _LazyParams:
    """Stand-in that resolves to the real params class on first use.

    ``EnvSpec`` is a frozen dataclass, so it stores this proxy rather than the
    class itself; attribute access and instantiation forward to the real one.
    """

    def __init__(self, module: str, name: str):
        self._module = module
        self._name = name
        self._cls: type | None = None

    def _resolve(self) -> type:
        if self._cls is None:
            self._cls = _params_cls(self._module, self._name)
        return self._cls

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self):
        return f"<LazyParams {self._module}.{self._name}>"


# -- PID factories ----------------------------------------------------------


def _pid(factory_name: str) -> Callable[[], Any]:
    def make():
        from importlib import import_module

        return getattr(import_module("target_gym.experts.pid"), factory_name)()

    return make


def _mpc(factory_name: str) -> Callable[[Any, Any], Any]:
    def make(env, params, **kwargs):
        from importlib import import_module

        return getattr(import_module("target_gym.experts.mpc"), factory_name)(
            env, params, **kwargs
        )

    return make


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_SPECS: tuple[EnvSpec, ...] = (
    # -- Aircraft -----------------------------------------------------------
    EnvSpec(
        name="plane",
        group="aircraft",
        env_factory=_plane,
        params_cls=_LazyParams("target_gym.plane.env", "PlaneParams"),
        make_pid=_pid("make_plane_cascaded_pid"),
        make_mpc=_mpc("make_plane_mpc"),
        test_params={"max_steps_in_episode": 280},
        tuned_gains_key="plane",
        disturbance_fields=("gust_x", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
    ),
    # Two moving-setpoint variants of the same aircraft. They are the same plant
    # and the same controllers; only the commanded altitude differs, which is
    # the point -- holding one altitude is a task a tuned PID finishes with
    # 0.0 m of settled error, so it can no longer distinguish controllers.
    #
    # Measured with the shipped PID: hold scores 1595 at 0.0 m, steps 1472 at
    # 57.6 m, sinusoid 870 at 120.3 m. Monotone in difficulty, and each exposes
    # something a constant setpoint cannot -- steps the transient response,
    # repeatedly, and the sinusoid the closed loop's bandwidth, since amplitude
    # ratio and phase lag against frequency *are* its frequency response.
    EnvSpec(
        name="plane_energy",
        group="aircraft",
        env_factory=_plane,
        params_cls=_LazyParams("target_gym.plane.env", "PlaneParams"),
        make_pid=_pid("make_plane_cascaded_pid"),
        make_mpc=_mpc("make_plane_mpc"),
        # Altitude *and* airspeed. The aircraft has always carried two
        # actuators, thrust and elevator, against one scored objective, so a
        # controller had a spare degree of freedom and could trade airspeed for
        # altitude for free. Scoring both removes it: thrust is finite, climbing
        # and accelerating compete, and the exchange between them is the
        # phugoid. Steps on the altitude so the trade is forced repeatedly
        # rather than settled once.
        #
        # This absorbed ``plane_steps``, which was this environment with
        # ``speed_weight`` at zero and nothing else different: same plant, same
        # schedule, same disturbances, same episode. Two registered environments
        # for one reward coefficient is not two tasks, and the pair cost 9 h of
        # the 12.9 h it took to record the aircraft. The pure altitude staircase
        # is still available, as ``PlaneParams(speed_weight=0.0)`` -- the term is
        # a parameter, not a fork.
        #
        # Gains come from ``tuned_gains_key="plane"``, tuned on the plain
        # altitude-hold task and deliberately not re-tuned here. The PID is the
        # baseline a learner has to beat, and a PID that has been fitted to the
        # airspeed trade is no longer the honest reference for whether that
        # trade is worth making.
        test_params={
            "max_steps_in_episode": 1200,
            "target_pattern": 1,
            "target_amplitude": 900.0,
            "target_steps": 8,
            "speed_weight": 0.5,
        },
        tuned_gains_key="plane",
        disturbance_fields=("gust_x", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
    ),
    EnvSpec(
        name="plane_sine",
        group="aircraft",
        env_factory=_plane,
        params_cls=_LazyParams("target_gym.plane.env", "PlaneParams"),
        make_pid=_pid("make_plane_cascaded_pid"),
        make_mpc=_mpc("make_plane_mpc"),
        # 800 steps is 3.3 periods of the 240 s sinusoid, satisfying the three
        # periods the episode-length criterion asks of a periodic task.
        #
        # The amplitude is set by the aircraft's climb rate, not by taste. A
        # sinusoid of amplitude A and period T commands a peak vertical rate of
        # 2*pi*A/T, and the aircraft's best sustained climb at these altitudes
        # measures 14.6 m/s. At the 800 m this ran at, the command peaked at
        # 20.9 m/s -- half again what the aircraft can produce -- so the target
        # was unreachable by construction and no controller could score near
        # the top of the range. 300 m peaks at 7.9 m/s, comfortably inside the
        # envelope, which is what a bandwidth probe needs: the response has to
        # be limited by the closed loop, not by the actuator saturating.
        test_params={
            "max_steps_in_episode": 480,
            "target_pattern": 3,
            "target_amplitude": 300.0,
            "target_period": 240.0,
        },
        tuned_gains_key="plane",
        disturbance_fields=("gust_x", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
    ),
    EnvSpec(
        name="plane3d_heading",
        group="aircraft",
        env_factory=_plane3d_heading,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid("make_plane3d_heading_cascaded_pid"),
        make_mpc=_mpc("make_plane3d_mpc"),
        test_params={"max_steps_in_episode": 200},
        tuned_gains_key="plane3d_heading",
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
    ),
    EnvSpec(
        name="plane3d_circle",
        group="aircraft",
        env_factory=_plane3d_circle,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid("make_plane3d_circle_cascaded_pid"),
        make_mpc=_mpc("make_plane3d_mpc"),
        test_params={"max_steps_in_episode": 300},
        tuned_gains_key="plane3d_circle",
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
    ),
    EnvSpec(
        name="plane3d_racetrack",
        group="aircraft",
        env_factory=_plane3d_racetrack,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid("make_plane3d_racetrack_cascaded_pid"),
        make_mpc=_mpc("make_plane3d_mpc"),
        # A lap is two legs plus two half-circles: 2 * (2 * 2r) + 2 * pi * r,
        # about 8.3 r of path. At an 8.4 km radius and 230 m/s that is ~300 s,
        # so 650 steps is the three laps the episode-length criterion asks of a
        # periodic task.
        test_params={"max_steps_in_episode": 650},
        tuned_gains_key="plane3d_racetrack",
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        # No ``expert_degraded``: this expert used to carry one, and what it
        # said was that its gains had never been searched. They have been now.
        # Coordinate descent on the cross-track gain alone, over five seeds and
        # full 900-step episodes, takes the settled cross-track error from
        # 3.02 km to 0.31 km against an 8.4 km turn radius, and the return from
        # 269.2 to 319.0. It holds the pattern rather than merely flying its
        # shape.
        #
        # Getting there needed one fix outside the gains: the class declared
        # ``obs_value_index`` and no ``obs_target_index``, so ``rollout``
        # raised on it, the search scored every candidate as -inf inside its
        # own ``try`` and reported success having changed nothing, and no
        # baseline could be recorded for this environment at all.
        # ``tests/test_env_conformance.py`` now checks both indices across the
        # whole registry.
        #
        # Three earlier defects, kept because two of them are mistakes this
        # repository has made before. The cross-track error was taken straight
        # from the geometry as side * (|v| - r), which reverses meaning between
        # the outbound and return legs, so the correction steered *away* from
        # the path on one of them; it is now signed in the tangent's own frame.
        # There was no coordinated-turn feedforward, so the law chased the
        # tangent through the turns and lagged by construction -- the same
        # failure the circle expert had, and the geometry supplies the exact
        # curvature, so it is now fed forward. And the roll-damping term had
        # the sign that makes it positive feedback, which let bank reach 52
        # degrees against a 30 degree command limit. Together those took the
        # settled error from 48.7 km to 4.4 km.
    ),
    EnvSpec(
        name="plane3d_figure8",
        group="aircraft",
        env_factory=_plane3d_figure8,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid("make_plane3d_figure8_stateful_pid"),
        make_mpc=_mpc("make_plane3d_mpc"),
        test_params={"max_steps_in_episode": 400},
        tuned_gains_key="plane3d_figure8",
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
    ),
    EnvSpec(
        name="patrol",
        group="aircraft",
        env_factory=_patrol,
        params_cls=_LazyParams("target_gym.patrol.env", "PatrolParams"),
        make_pid=_pid("make_patrol_stateful_pid"),
        make_mpc=_mpc("make_patrol_mpc"),
        test_params={"max_steps_in_episode": 200},
        tuned_gains_key="patrol",
    ),
    EnvSpec(
        name="patrol_bearing_only",
        group="aircraft",
        env_factory=_patrol_bearing_only,
        params_cls=_LazyParams("target_gym.patrol.env", "PatrolParams"),
        make_pid=_pid("make_patrol_bearing_only_stateful_pid"),
        make_mpc=None,
        test_params={"max_steps_in_episode": 200},
        tuned_gains_key="patrol",
        baselines_note=(
            "PID present -- a lead-state estimator feeding the same pursuit "
            "law the full-observation variant uses. Range with azimuth and "
            "elevation is a complete relative-position measurement, so the "
            "only genuinely unobservable quantity is the lead's HEADING, "
            "which the commanded slot needs because the slot is expressed in "
            "the lead's frame; it is recovered by differencing the estimated "
            "relative position and filtering. Measured performance matches "
            "the full-observation expert (4 of 8 seeds complete, ~229 m "
            "settled slot error vs ~260 m), so the partial observation costs "
            "essentially nothing here. "
            "No MPC, and the reason is the withheld observation rather than "
            "the manoeuvring lead. This note used to blame the lead, on the "
            "grounds that an MPC would need its future trajectory as a "
            "time-varying parameter. That holds for a CasADi model and not "
            "for a gradient planner: `patrol` now ships a GradientMPC that "
            "differentiates step_env, and because the lead is scripted and "
            "deterministic the plan propagates it for free. What blocks one "
            "here is that the planner reads the slot error out of the state, "
            "which is precisely what this variant withholds. Handing it the "
            "true state anyway would make it an oracle on a task defined by "
            "what is hidden, so it needs a planner built on the estimator."
        ),
    ),
    # -- Process control ----------------------------------------------------
    EnvSpec(
        name="cstr",
        group="process",
        env_factory=_cstr,
        params_cls=_LazyParams("target_gym.pc_gym.cstr.env", "CSTRParams"),
        make_pid=_pid("make_cstr_stateful_pid"),
        make_mpc=_mpc("make_cstr_mpc"),
        tuned_gains_key="cstr",
    ),
    EnvSpec(
        name="first_order",
        group="process",
        env_factory=_first_order,
        params_cls=_LazyParams("target_gym.pc_gym.first_order.env", "FirstOrderParams"),
        make_pid=_pid("make_first_order_stateful_pid"),
        make_mpc=_mpc("make_first_order_mpc"),
        tuned_gains_key="first_order",
    ),
    EnvSpec(
        name="four_tank",
        group="process",
        env_factory=_four_tank,
        params_cls=_LazyParams("target_gym.pc_gym.four_tank.env", "FourTankParams"),
        make_pid=_pid("make_four_tank_stateful_pid"),
        make_mpc=_mpc("make_four_tank_mpc"),
        # 500 steps. The lower tanks have a ~58 s time constant at these
        # levels, so the previous 100-step horizon was under two of them --
        # every controller was still mid-transient and they all scored alike,
        # which is why the effectiveness contract could not separate them here.
        tuned_gains_key="four_tank",
    ),
    EnvSpec(
        name="ph_neutralization",
        group="process",
        env_factory=_ph_neutralization,
        params_cls=_LazyParams("target_gym.pc_gym.ph_neutralization.env", "PHParams"),
        make_pid=_pid("make_ph_stateful_pid"),
        make_mpc=_mpc("make_ph_mpc"),
        # 300 steps = 25 min ~ 17 residence times, enough for the buffer
        # disturbance to move the operating point.
        tuned_gains_key="ph_neutralization",
        disturbance_fields=("q2",),
    ),
    EnvSpec(
        name="distillation",
        group="process",
        env_factory=_distillation,
        params_cls=_LazyParams(
            "target_gym.pc_gym.distillation.env", "DistillationParams"
        ),
        make_pid=_pid("make_distillation_stateful_pid"),
        make_mpc=_mpc("make_distillation_mpc"),
        # 200 min ~ one dominant time constant. The column is the slowest
        # environment per step (41 states, 16 substeps for stability), so the
        # test episode is kept short.
        tuned_gains_key="distillation",
        disturbance_fields=("zF",),
    ),
    # -- Industrial / energy ------------------------------------------------
    EnvSpec(
        name="glass_furnace",
        group="industrial",
        env_factory=_glass_furnace,
        params_cls=_LazyParams("target_gym.glass_furnace.env", "GlassFurnaceParams"),
        make_pid=_pid("make_glass_furnace_stateful_pid"),
        make_mpc=_mpc("make_glass_furnace_mpc"),  # 13.3 h at dt=30 s, 12.1 tau
        tuned_gains_key="glass_furnace",
        disturbance_fields=("m_pull_disturbance",),
    ),
    EnvSpec(
        name="reactor",
        group="industrial",
        env_factory=_reactor,
        params_cls=_LazyParams("target_gym.reactor.env", "ReactorParams"),
        make_pid=_pid("make_reactor_stateful_pid"),
        make_mpc=_mpc("make_reactor_mpc"),
        # Env steps of 10 s (``control_period`` physics sub-steps each): 2.4 h.
        #
        # This used to be written as 8640 *physics* steps while the shipped
        # rollout counted env steps, so the recorded baselines ran 8640 env
        # steps of which the last 7776 scored a frozen plant at a tenth of the
        # reward (see ReactorParams.max_steps_in_episode). Same task, now
        # counted in the unit every other plant uses.
        #
        # Before that it was 1200 physics steps, twenty minutes, with 8640
        # applied only through ``effectiveness_overrides`` on the stated
        # grounds that the shorter one was not long enough "for the
        # xenon/demand dynamics to actually distinguish a controller": over
        # twenty minutes the xenon state moves 2.5%, against a 13.2 h time
        # constant. 2.4 h is 0.18 xenon time constants, enough that the poison
        # moves materially while the controller works. Full pit dynamics would
        # need about ten hours, which alone would cost more to record than the
        # rest of the suite together.
        test_params={"max_steps_in_episode": 864},
        tuned_gains_key="reactor",
        # v2: ``max_steps_in_episode`` and ``state.time`` count env steps, not
        # physics sub-steps. The physics within an episode is unchanged, but a
        # return published against v1 was taken over 8640 env steps of which
        # 7776 scored a frozen plant, so its numbers do not carry over.
        version=2,
    ),
    EnvSpec(
        name="hvac",
        group="industrial",
        env_factory=_hvac,
        params_cls=_LazyParams("target_gym.hvac.env", "HVACParams"),
        make_pid=_pid("make_hvac_stateful_pid"),
        make_mpc=_mpc("make_hvac_mpc"),
        # 720 steps at dt = 900 s is 7.5 days, so fifteen setback recoveries
        # and as many solar cycles -- which is what distinguishes controllers
        # here. (This comment read "2 days" long after the episode-length audit
        # raised it from the 192 steps that actually was two days.)
        tuned_gains_key="hvac",
        disturbance_fields=("weather_dev",),
    ),
    EnvSpec(
        name="cement_kiln",
        group="industrial",
        env_factory=_cement_kiln,
        params_cls=_LazyParams("target_gym.cement_kiln.env", "CementKilnParams"),
        make_pid=_pid("make_cement_kiln_stateful_pid"),
        make_mpc=_mpc("make_cement_kiln_mpc"),
        # 700 steps = 5.8 hours at dt = 30 s, about fourteen transport delays,
        # so a controller lives with the consequences of its own fuel changes
        # many times over. (This comment read "240 steps = 2 hours ... about
        # five transport delays" long after the episode-length audit raised it
        # to 700.)
        tuned_gains_key="cement_kiln",
        disturbance_fields=("raw_meal",),
    ),
    EnvSpec(
        name="boiler_drum",
        group="industrial",
        env_factory=_boiler_drum,
        params_cls=_LazyParams("target_gym.boiler_drum.env", "BoilerDrumParams"),
        make_pid=_pid("make_boiler_drum_stateful_pid"),
        make_mpc=_mpc("make_boiler_drum_mpc"),
        # 400 steps = 800 s at dt = 2 s, about 20 times the ~35 s swell peak,
        # so a controller has to survive many inverse-response transients.
        tuned_gains_key="boiler_drum",
        disturbance_fields=("q_steam",),
    ),
    EnvSpec(
        name="wind_turbine",
        group="energy",
        env_factory=_wind_turbine,
        params_cls=_LazyParams(
            "target_gym.energy.wind_turbine.env", "WindTurbineParams"
        ),
        make_pid=_pid("make_wind_turbine_stateful_pid"),
        make_mpc=_mpc("make_wind_turbine_mpc"),
        # 400 steps = 100 s ~ 7 rotor time constants.
        tuned_gains_key="wind_turbine",
        noise_fields=("turbulence_std",),
        disturbance_fields=("v_wind",),
    ),
    EnvSpec(
        name="battery",
        group="energy",
        env_factory=_battery,
        params_cls=_LazyParams("target_gym.energy.battery.env", "BatteryParams"),
        make_pid=_pid("make_battery_stateful_pid"),
        make_mpc=_mpc("make_battery_mpc"),
        # 360 steps = 30 min, a real fraction of the ~96 min it takes to
        # traverse the usable state-of-charge range at full power.
        tuned_gains_key="battery",
        noise_fields=("dispatch_noise_std",),
        disturbance_fields=("target_power",),
    ),
)

REGISTRY: dict[str, EnvSpec] = {spec.name: spec for spec in _SPECS}


def all_specs() -> Iterator[EnvSpec]:
    """Iterate over every registered environment spec."""
    return iter(_SPECS)


def specs_in_group(group: str) -> Iterator[EnvSpec]:
    """Iterate over the specs belonging to ``group``."""
    if group not in GROUPS:
        raise KeyError(f"Unknown group {group!r}; expected one of {sorted(GROUPS)}")
    return (spec for spec in _SPECS if spec.group == group)


def get(name: str) -> EnvSpec:
    """Look up one spec by registry name."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown environment {name!r}; registered: {sorted(REGISTRY)}"
        ) from None


def env_names() -> list[str]:
    """Names of every registered environment, in registration order."""
    return [spec.name for spec in _SPECS]
