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

The multi-agent formation is registered in its joint form: ``formation`` takes
one action for every aircraft and returns their observations concatenated, so
the shared contract, the baselines and the protocol apply to it as to any
task. ``target_gym.formation.marl.FormationMARL`` is the same step with one
observation and one action per aircraft, in the JaxMARL style.
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
    "patrol": "Patrol - wingman",
    "patrol_bearing_only": "Patrol - wingman, bearing-only",
    "unstable_cstr": "Unstable CSTR",
}


def display_name(name: str) -> str:
    """The human-facing name for a registry key."""
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# Keyed by spec name: EnvSpec is frozen, so the cache lives beside it.
_ENV_CACHE: dict[str, Any] = {}


def control_step_seconds(env, params) -> float:
    """Seconds of simulated time one ``step_env`` call advances.

    ``delta_t`` times ``time_unit_seconds`` (the PC-gym CSTR and distillation
    column run in minutes) times ``control_period`` (the reactor holds one
    action across several physics sub-steps). Kept out of ``utils.py``, which
    is in every fingerprint.
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
        Registry key, and the key of its gains in
        ``src/target_gym/data/pid_gains.json`` where they are tuned.
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
    tuned_gains_key:
        Key under which this environment's PID gains live in
        ``src/target_gym/data/pid_gains.json``.  ``None`` means the controller
        reads no single entry there: its gains are stored per sub-loop, or kept
        in its package's ``experts.py`` (the 3D aircraft's guidance laws).
        ``provenance.baseline_fingerprint`` hashes this key's entry.
    gains_keys:
        Other keys of ``pid_gains.json`` this task's baseline and protocol
        records depend on, beyond those starting with its name and
        ``tuned_gains_key``; without them a retune of that key would leave the
        records looking fresh (``plane_sine`` declares ``plane_cascaded``).
        Feeds the baseline fingerprint only, not the version stamp.
    baselines_note:
        Set when ``make_pid``/``make_mpc`` are ``None``: a short explanation
        of why, surfaced by the baseline-coverage test so a missing expert is
        a documented gap rather than a silent one. Also set where both ship
        but one needs saying what it is (``patrol_bearing_only``'s MPC is a
        full-state bound); the environment pages print it beside the numbers.
    disturbance_fields:
        State fields holding a *zero-mean stochastic disturbance* (gusts, load
        noise). The conformance suite asserts these behave like disturbances --
        in particular that they do not ratchet under a constant PRNG key,
        which a caller may pass at every step. Excludes deliberately drifting
        processes such as the reactor's demand.
    disturbance_overrides:
        Parameter overrides that switch the disturbance on, when it is off by
        default (e.g. the aircraft's ``turbulence_sigma``).
    effectiveness_overrides:
        Parameter overrides for the controller-effectiveness contract, when the
        scored episode is too short to tell a working
        controller from a constant (the reactor, whose xenon transient outlasts
        the scored episode).
    expert_degraded:
        Set when a baseline exists but loses to a constant action, so the gap
        is recorded and explained rather than a silently bad number.
    mpc_degraded:
        The same, for the MPC: set when the MPC exists and runs but returns
        materially less than the PID, so it is not the upper bound the benchmark
        presents it as. The reason should carry the measured numbers.
    noise_fields:
        Parameters that scale process noise. A planner's copy of the params
        has them zeroed, so it plans on the mean disturbance
        (``experts.mpc.plan_params``).
    fingerprint_sources:
        Source files outside the task's own package that its physics imports,
        relative to ``src/target_gym`` (e.g. ``"pc_gym/cstr/env.py"``). Both
        fingerprints hash only the task's own package, so an imported file has
        to be declared here for an edit to it to make this task's records
        stale.
    """

    name: str
    group: str
    env_factory: Callable[[], Any]
    params_cls: Callable[..., Any]
    make_pid: Callable[[], Any] | None
    make_mpc: Callable[[Any, Any], Any] | None
    # Behavioural version of the environment, part of its public identity as
    # ``<name>-v<version>``. Bump it whenever the dynamics, the reward, the
    # parameters or the observation layout change, so that a published number
    # keeps meaning what it meant.
    #
    # Versioning starts at v1 in the 0.6 release. v2 is the floor-normalised
    # reward (docs/reward-shaping.md), and some tasks have reached v3 since.
    # The v1 reward is still constructible with ``reward_version=1`` on any
    # params, and the v1 baselines are kept in
    # ``src/target_gym/data/baseline_returns_v1.json``.
    version: int = 2
    tuned_gains_key: str | None = None
    #: Further ``pid_gains.json`` keys the task reads, hashed into its
    #: baseline fingerprint (``provenance.gains_entries``).
    gains_keys: tuple[str, ...] = ()
    baselines_note: str | None = None
    expert_degraded: str | None = None
    mpc_degraded: str | None = None
    effectiveness_overrides: dict[str, Any] = field(default_factory=dict)
    disturbance_fields: tuple[str, ...] = ()
    #: Parameters that scale *process* noise, zeroed in the copy of the params
    #: a planner uses for its internal model. See ``experts.mpc.plan_params``.
    noise_fields: tuple[str, ...] = ()
    disturbance_overrides: dict[str, Any] = field(default_factory=dict)
    fingerprint_sources: tuple[str, ...] = ()

    @property
    def versioned_name(self) -> str:
        """The name a published result should cite, e.g. ``plane-v2``.

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

    def make_pid_controller(self):
        """The scored PID as a pure ``Controller``: ``.act`` traces under
        ``jit``/``vmap``, ``.host()`` is what ``make_pid()`` returns."""
        if not self.has_pid:
            raise NotImplementedError(self.baselines_note or "no PID ships")
        return self.make_pid.controller()

    @property
    def has_mpc(self) -> bool:
        return self.make_mpc is not None

    def make_env(self):
        """The environment for this spec, built once and shared.

        Not a fresh instance per call: ``env.step_env`` is a bound method, so
        every new instance is a new callable to JAX, compiled again and held for
        the life of the process. Sharing is safe because environments carry no
        per-episode state; it all lives in the ``EnvState`` passed through each
        call. Hold onto an instance you construct directly for the same reason.
        """
        cached = _ENV_CACHE.get(self.name)
        if cached is None:
            cached = _ENV_CACHE[self.name] = self.env_factory()
        return cached

    def make_test_params(self, **overrides):
        """The scored configuration, the environment's ``default_params``, plus
        any extra overrides. The baselines and the protocol are measured on it."""
        return self.make_env().default_params.replace(**overrides)


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


def _plane_energy():
    from target_gym.plane.env_jax import PlaneEnergy

    return PlaneEnergy()


def _plane_sine():
    from target_gym.plane.env_jax import PlaneSine

    return PlaneSine()


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


def _unstable_cstr():
    from target_gym.pc_gym.unstable_cstr.env_jax import UnstableCSTR

    return UnstableCSTR()


def _compressor_surge():
    from target_gym.compressor_surge.env_jax import CompressorSurge

    return CompressorSurge()


def _formation():
    from target_gym.formation.env_jax import Formation

    return Formation()


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


def _pid(factory_name: str, module: str) -> Callable[[], Any]:
    # Every task's PID lives in its own package's ``experts`` module, as a
    # factory returning a pure ``Controller``: experts/pid.py holds only the
    # shared core and is in every task's baseline fingerprint, so a controller
    # kept there would stale all of them. ``make()`` is the host form the
    # baselines are scored with; ``make.controller()`` the traceable one.
    def controller():
        from importlib import import_module

        return getattr(import_module(module), factory_name)()

    def make():
        return controller().host()

    # An attribute on the function rather than a small class, because the
    # pinned-spec test reads the factory name from this closure's cells.
    make.controller = controller  # type: ignore[attr-defined]
    return make


def _mpc(factory_name: str, module: str) -> Callable[[Any, Any], Any]:
    # Every task's oracle lives in its own package's ``experts`` module, so
    # changing one re-records only that package's tasks. experts/mpc.py holds
    # the shared machinery and is in every task's baseline fingerprint.
    def make(env, params, **kwargs):
        from importlib import import_module

        return getattr(import_module(module), factory_name)(env, params, **kwargs)

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
        make_pid=_pid("make_plane_pid_controller", module="target_gym.plane.experts"),
        make_mpc=_mpc("make_plane_mpc", module="target_gym.plane.experts"),
        tuned_gains_key="plane",
        disturbance_fields=("gust_x", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
    ),
    # Two moving-setpoint variants of the same aircraft: same plant, same
    # controllers, a different commanded altitude. Steps expose the transient
    # response repeatedly, and the sinusoid the closed loop's bandwidth.
    EnvSpec(
        name="plane_energy",
        group="aircraft",
        env_factory=_plane_energy,
        params_cls=_LazyParams("target_gym.plane.env", "PlaneParams"),
        make_pid=_pid("make_plane_pid_controller", module="target_gym.plane.experts"),
        make_mpc=_mpc("make_plane_mpc", module="target_gym.plane.experts"),
        # Gains are the altitude-hold task's, deliberately not re-tuned here,
        # so the PID stays an honest reference for whether the trade pays.
        tuned_gains_key="plane",
        disturbance_fields=("gust_x", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
        version=4,
    ),
    EnvSpec(
        name="plane_sine",
        group="aircraft",
        env_factory=_plane_sine,
        params_cls=_LazyParams("target_gym.plane.env", "PlaneParams"),
        make_pid=_pid("make_plane_pid_controller", module="target_gym.plane.experts"),
        make_mpc=_mpc("make_plane_mpc", module="target_gym.plane.experts"),
        tuned_gains_key="plane",
        disturbance_fields=("gust_x", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
    ),
    EnvSpec(
        name="plane3d_heading",
        group="aircraft",
        env_factory=_plane3d_heading,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_heading_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=_mpc("make_plane3d_mpc", module="target_gym.plane3d.experts"),
        tuned_gains_key="plane3d_heading",
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
        # The 2D aircraft's dynamics, which the 3D model extends, and the
        # TECS its experts fly (an experts file, so in the baseline
        # fingerprint only).
        fingerprint_sources=("plane/dynamics.py", "plane/experts_tecs.py"),
        version=4,
    ),
    EnvSpec(
        name="plane3d_circle",
        group="aircraft",
        env_factory=_plane3d_circle,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_circle_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=_mpc("make_plane3d_mpc", module="target_gym.plane3d.experts"),
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
        # The 2D aircraft's dynamics, which the 3D model extends, and the
        # TECS its experts fly (an experts file, so in the baseline
        # fingerprint only).
        fingerprint_sources=("plane/dynamics.py", "plane/experts_tecs.py"),
        version=3,
    ),
    EnvSpec(
        name="plane3d_racetrack",
        group="aircraft",
        env_factory=_plane3d_racetrack,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_racetrack_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=_mpc("make_plane3d_mpc", module="target_gym.plane3d.experts"),
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
        # The 2D aircraft's dynamics, which the 3D model extends, and the
        # TECS its experts fly (an experts file, so in the baseline
        # fingerprint only).
        fingerprint_sources=("plane/dynamics.py", "plane/experts_tecs.py"),
        version=4,
    ),
    EnvSpec(
        name="plane3d_figure8",
        group="aircraft",
        env_factory=_plane3d_figure8,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_figure8_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=_mpc("make_plane3d_mpc", module="target_gym.plane3d.experts"),
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
        # The 2D aircraft's dynamics, which the 3D model extends, and the
        # TECS its experts fly (an experts file, so in the baseline
        # fingerprint only).
        fingerprint_sources=("plane/dynamics.py", "plane/experts_tecs.py"),
        # v4: the distance to the curve observed as the tracked output.
        version=4,
    ),
    EnvSpec(
        name="patrol",
        group="aircraft",
        env_factory=_patrol,
        params_cls=_LazyParams("target_gym.patrol.env", "PatrolParams"),
        make_pid=_pid("make_patrol_pid_controller", module="target_gym.patrol.experts"),
        make_mpc=_mpc("make_patrol_mpc", module="target_gym.patrol.experts"),
        # The oracle and the lead fly the 3D heading autopilot. The expert's
        # gains are in its experts.py, as the 3D aircraft's are.
        gains_keys=("plane3d_heading",),
        noise_fields=("turbulence_sigma",),
        # The 3D aircraft it flies, and the heading PID its lead flies on
        # (physics here, so declared for the version stamp too); the
        # autopilot and TECS its expert flies under the formation hold (experts
        # files, so in the baseline fingerprint only).
        fingerprint_sources=(
            "plane/dynamics.py",
            "plane3d/dynamics.py",
            "plane3d/env.py",
            "experts/pid.py",
            "plane3d/experts.py",
            "plane/experts_tecs.py",
        ),
        version=3,
    ),
    EnvSpec(
        name="patrol_bearing_only",
        group="aircraft",
        env_factory=_patrol_bearing_only,
        params_cls=_LazyParams("target_gym.patrol.env", "PatrolParams"),
        make_pid=_pid(
            "make_patrol_bearing_only_pid_controller",
            module="target_gym.patrol.experts",
        ),
        make_mpc=_mpc(
            "make_patrol_bearing_only_mpc", module="target_gym.patrol.experts"
        ),
        gains_keys=("plane3d_heading",),
        baselines_note=(
            "The PID is a lead-state estimator feeding the same formation "
            "hold the full-observation variant uses. Range with azimuth and "
            "elevation is a complete relative-position measurement, so the "
            "only unobservable quantity is the lead's heading, which the slot "
            "needs because it is expressed in the lead's frame; the estimator "
            "takes the lead's velocity from the follower's own and the change "
            "in the relative position, and its heading from that track less "
            "the crab the follower flies in the same air. The MPC slot holds patrol's oracle reading the true "
            "state, so it is a full-state bound, not a bearing-only "
            "controller. On a task defined by what the observation withholds, "
            "its score measures information and control together: what the "
            "hidden lead state is worth plus what a policy leaves on the "
            "table. It never reads the observation, so its per-step costs "
            "equal patrol's seed for seed."
        ),
        noise_fields=("turbulence_sigma",),
        # As patrol's.
        fingerprint_sources=(
            "plane/dynamics.py",
            "plane3d/dynamics.py",
            "plane3d/env.py",
            "experts/pid.py",
            "plane3d/experts.py",
            "plane/experts_tecs.py",
        ),
        version=3,
    ),
    # -- Process control ----------------------------------------------------
    EnvSpec(
        name="cstr",
        group="process",
        env_factory=_cstr,
        params_cls=_LazyParams("target_gym.pc_gym.cstr.env", "CSTRParams"),
        make_pid=_pid(
            "make_cstr_pid_controller", module="target_gym.pc_gym.cstr.experts"
        ),
        make_mpc=_mpc("make_cstr_mpc", module="target_gym.pc_gym.cstr.experts"),
        # 150 steps, over twice the 67 steps the SIMC cascade takes to come
        # within twice the level it holds. At 100 the scored hold was the last
        # third.
        tuned_gains_key="cstr",
        version=3,
    ),
    EnvSpec(
        name="first_order",
        group="process",
        env_factory=_first_order,
        params_cls=_LazyParams("target_gym.pc_gym.first_order.env", "FirstOrderParams"),
        make_pid=_pid(
            "make_first_order_pid_controller",
            module="target_gym.pc_gym.first_order.experts",
        ),
        make_mpc=_mpc(
            "make_first_order_mpc", module="target_gym.pc_gym.first_order.experts"
        ),
        # 150 steps, twice the 63 steps the SIMC PI takes to come within twice
        # the level it holds. At 100 the scored hold was the last 37 steps.
        tuned_gains_key="first_order",
        version=3,
    ),
    EnvSpec(
        name="four_tank",
        group="process",
        env_factory=_four_tank,
        params_cls=_LazyParams("target_gym.pc_gym.four_tank.env", "FourTankParams"),
        make_pid=_pid(
            "make_four_tank_pid_controller",
            module="target_gym.pc_gym.four_tank.experts",
        ),
        make_mpc=_mpc(
            "make_four_tank_mpc", module="target_gym.pc_gym.four_tank.experts"
        ),
        # 1200 steps, twice the 570 s the decoupled SIMC expert takes to bring
        # both levels within the 1 mm transmitter resolution: the plant is
        # non-minimum phase (gamma1 + gamma2 = 0.4), which caps any feedback
        # loop near a 64 s time constant. At 500 steps it was still settling.
        tuned_gains_key="four_tank",
        version=3,
    ),
    EnvSpec(
        name="ph_neutralization",
        group="process",
        env_factory=_ph_neutralization,
        params_cls=_LazyParams("target_gym.pc_gym.ph_neutralization.env", "PHParams"),
        make_pid=_pid(
            "make_ph_pid_controller",
            module="target_gym.pc_gym.ph_neutralization.experts",
        ),
        make_mpc=_mpc(
            "make_ph_mpc", module="target_gym.pc_gym.ph_neutralization.experts"
        ),
        # 300 steps = 25 min ~ 17 residence times, enough for the buffer
        # disturbance to move the operating point.
        tuned_gains_key="ph_neutralization",
        disturbance_fields=("q2",),
        # The buffer flow's OU innovation. The CasADi oracle never rolls the
        # simulator, so it could not see it anyway; declared so no planner can.
        noise_fields=("q2_noise_std",),
    ),
    EnvSpec(
        name="distillation",
        group="process",
        env_factory=_distillation,
        params_cls=_LazyParams(
            "target_gym.pc_gym.distillation.env", "DistillationParams"
        ),
        make_pid=_pid(
            "make_distillation_pid_controller",
            module="target_gym.pc_gym.distillation.experts",
        ),
        make_mpc=_mpc(
            "make_distillation_mpc", module="target_gym.pc_gym.distillation.experts"
        ),
        # 200 min ~ one dominant time constant. The column is the slowest
        # environment per step (41 states, 16 substeps for stability), so the
        # test episode is kept short.
        tuned_gains_key="distillation",
        # v3 observes the stage-15 temperature, the thermocouple the base
        # layer's temperature loop holds.
        version=3,
        disturbance_fields=("zF",),
        # The feed composition's OU innovation. Declared so the planner plans on the
        # mean (``plan_params``) rather than on one fixed noise path.
        noise_fields=("zF_noise_std",),
    ),
    # -- Industrial / energy ------------------------------------------------
    EnvSpec(
        name="glass_furnace",
        group="industrial",
        env_factory=_glass_furnace,
        params_cls=_LazyParams("target_gym.glass_furnace.env", "GlassFurnaceParams"),
        make_pid=_pid(
            "make_glass_furnace_pid_controller",
            module="target_gym.glass_furnace.experts",
        ),
        # 13.3 h at dt=30 s, 12.1 tau
        make_mpc=_mpc(
            "make_glass_furnace_mpc", module="target_gym.glass_furnace.experts"
        ),
        tuned_gains_key="glass_furnace",
        disturbance_fields=("m_pull_disturbance",),
        # The pull disturbance's AR(1) innovation. The CasADi oracle never rolls
        # the simulator and plans on the pull's conditional mean, read from
        # the state, so it could not see the innovation anyway; declared so no
        # planner can.
        noise_fields=("m_pull_noise_std",),
    ),
    EnvSpec(
        name="reactor",
        group="industrial",
        env_factory=_reactor,
        params_cls=_LazyParams("target_gym.reactor.env", "ReactorParams"),
        make_pid=_pid(
            "make_reactor_pid_controller", module="target_gym.reactor.experts"
        ),
        make_mpc=_mpc("make_reactor_mpc", module="target_gym.reactor.experts"),
        tuned_gains_key="reactor",
        version=3,
    ),
    EnvSpec(
        name="hvac",
        group="industrial",
        env_factory=_hvac,
        params_cls=_LazyParams("target_gym.hvac.env", "HVACParams"),
        make_pid=_pid("make_hvac_pid_controller", module="target_gym.hvac.experts"),
        make_mpc=_mpc("make_hvac_mpc", module="target_gym.hvac.experts"),
        # 720 steps at dt = 900 s is 7.5 days, so fifteen setback recoveries
        # and as many solar cycles, which is what distinguishes controllers
        # here.
        tuned_gains_key="hvac",
        disturbance_fields=("weather_dev",),
    ),
    EnvSpec(
        name="cement_kiln",
        group="industrial",
        version=3,
        env_factory=_cement_kiln,
        params_cls=_LazyParams("target_gym.cement_kiln.env", "CementKilnParams"),
        make_pid=_pid(
            "make_cement_kiln_pid_controller", module="target_gym.cement_kiln.experts"
        ),
        make_mpc=_mpc("make_cement_kiln_mpc", module="target_gym.cement_kiln.experts"),
        # 700 steps = 5.8 hours at dt = 30 s, about fourteen transport delays,
        # so a controller lives with the consequences of its own fuel changes
        # many times over.
        tuned_gains_key="cement_kiln",
        disturbance_fields=("raw_meal",),
        # The raw-meal feed's OU innovation. Declared so the planner plans on the
        # mean (``plan_params``) rather than on one fixed noise path.
        noise_fields=("feed_noise_std",),
    ),
    EnvSpec(
        name="boiler_drum",
        group="industrial",
        env_factory=_boiler_drum,
        params_cls=_LazyParams("target_gym.boiler_drum.env", "BoilerDrumParams"),
        make_pid=_pid(
            "make_boiler_drum_pid_controller", module="target_gym.boiler_drum.experts"
        ),
        make_mpc=_mpc("make_boiler_drum_mpc", module="target_gym.boiler_drum.experts"),
        # 400 steps = 800 s at dt = 2 s, about 20 times the ~35 s swell peak,
        # so a controller has to survive many inverse-response transients.
        tuned_gains_key="boiler_drum",
        disturbance_fields=("q_steam",),
        # The steam demand's OU innovation. Declared so the planner plans on the
        # mean (``plan_params``) rather than on one fixed noise path.
        noise_fields=("q_steam_noise_std",),
    ),
    EnvSpec(
        name="wind_turbine",
        group="energy",
        env_factory=_wind_turbine,
        params_cls=_LazyParams(
            "target_gym.energy.wind_turbine.env", "WindTurbineParams"
        ),
        make_pid=_pid(
            "make_wind_turbine_pid_controller",
            module="target_gym.energy.wind_turbine.experts",
        ),
        make_mpc=_mpc(
            "make_wind_turbine_mpc", module="target_gym.energy.wind_turbine.experts"
        ),
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
        make_pid=_pid(
            "make_battery_pid_controller", module="target_gym.energy.battery.experts"
        ),
        make_mpc=_mpc("make_battery_mpc", module="target_gym.energy.battery.experts"),
        # 360 steps = 30 min, a real fraction of the ~96 min it takes to
        # traverse the usable state-of-charge range at full power.
        tuned_gains_key="battery",
        noise_fields=("dispatch_noise_std",),
        disturbance_fields=("target_power",),
    ),
    # Append new tasks here, never insert: a task's position in REGISTRY is its
    # benchmark seed index, and TargetFoundation derives seeds the same way.
    # A task added after the 21 keeps its controllers in its own package's
    # experts.py and declares any physics it imports (tests/test_registry_rules.py).
    EnvSpec(
        name="unstable_cstr",
        group="process",
        env_factory=_unstable_cstr,
        params_cls=_LazyParams(
            "target_gym.pc_gym.unstable_cstr.env", "UnstableCSTRParams"
        ),
        make_pid=_pid(
            "make_unstable_cstr_pid_controller",
            module="target_gym.pc_gym.unstable_cstr.experts",
        ),
        make_mpc=_mpc(
            "make_unstable_cstr_mpc", module="target_gym.pc_gym.unstable_cstr.experts"
        ),
        # 1200 steps = 60 min at 3 s: six 10-minute blocks, with no
        # effectiveness_overrides.
        tuned_gains_key="unstable_cstr",
        disturbance_fields=("Ti_dev",),
        noise_fields=("Ti_sigma",),
        # The balances and the ten parameter values are imported from cstr, so an
        # edit there moves this task's stamp and baseline too.
        fingerprint_sources=("pc_gym/cstr/env.py",),
    ),
    EnvSpec(
        name="compressor_surge",
        group="industrial",
        env_factory=_compressor_surge,
        params_cls=_LazyParams(
            "target_gym.compressor_surge.env", "CompressorSurgeParams"
        ),
        make_pid=_pid(
            "make_compressor_surge_pid_controller",
            module="target_gym.compressor_surge.experts",
        ),
        make_mpc=_mpc(
            "make_compressor_surge_mpc", module="target_gym.compressor_surge.experts"
        ),
        # 1200 steps = 120 s at 0.1 s: four 30-second setpoint blocks and six
        # 20-second demand blocks, with no effectiveness_overrides.
        tuned_gains_key="compressor_surge",
        # The OU deviation of the consumers' opening, zero-mean;
        # demand_sigma is its stationary sd.
        disturbance_fields=("demand_dev",),
        noise_fields=("demand_sigma",),
        # No fingerprint_sources: the package imports only itself and the
        # shared plumbing.
    ),
    EnvSpec(
        name="formation",
        group="aircraft",
        env_factory=_formation,
        params_cls=_LazyParams("target_gym.formation.env", "FormationParams"),
        make_pid=_pid(
            "make_formation_pid_controller", module="target_gym.formation.experts"
        ),
        make_mpc=_mpc("make_formation_mpc", module="target_gym.formation.experts"),
        # The oracle's lead autopilot reads plane3d_heading's gains. The
        # expert's are in the experts.py files, as the 3D aircraft's are.
        gains_keys=("plane3d_heading",),
        baselines_note=(
            "The PID flies the lead seat as a formation lead: the 3D heading "
            "task's course hold and pitch on height, aimed at the heading the "
            "route commands, at a set power, so its wingmen have a steady "
            "reference. Each wingman seat flies the patrol's formation hold. "
            "The MPC slot is the patrol oracle's design applied to the "
            "whole formation: the patrol's lead autopilot in the lead seat and "
            "its twin-autopilot law in each wingman seat, with one residual "
            "planner over every seat on the team cost."
        ),
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        noise_fields=("turbulence_sigma",),
        # The aircraft and the patrol's geometry and observation, and the
        # controllers the PID team and the oracle reuse: the patrol's, the 3D
        # autopilot and TECS (experts files, so in the baseline fingerprint
        # only).
        fingerprint_sources=(
            "patrol/env.py",
            "patrol/experts.py",
            "plane3d/env.py",
            "plane3d/dynamics.py",
            "plane/dynamics.py",
            "plane3d/experts.py",
            "plane/experts_tecs.py",
        ),
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
