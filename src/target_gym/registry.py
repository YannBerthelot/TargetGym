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
    "patrol_range_bearing": "Patrol - wingman, range and bearing",
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
        ``provenance.expert_fingerprint`` hashes this key's entry.
    gains_keys:
        Other keys of ``pid_gains.json`` this task's expert rows depend on,
        beyond those starting with its name and ``tuned_gains_key``; without
        them a retune of that key would leave the records looking fresh.
        Feeds the expert fingerprint only, not the version stamp.
    oracle_reads, oracle_gains_keys:
        The expert code (files relative to ``src/target_gym``) and the
        ``pid_gains.json`` keys the task's oracle still reads, hashed into its
        oracle fingerprint, so an expert retune stales those oracle rows too.
        Empty once an oracle stands on its own.
    oracle_data:
        The data files (relative to ``src/target_gym``) an oracle on the
        shared solver (``target_gym.oracles``) reads: its horizon summary and
        its smoothing widths. Their values enter its oracle fingerprint, which
        then covers the shared solver's modules in place of experts/mpc.py.
    baselines_note:
        Set when ``make_pid``/``make_mpc`` are ``None``: a short explanation
        of why, surfaced by the baseline-coverage test so a missing expert is
        a documented gap rather than a silent one. Also set where both ship
        but one needs saying what it is (``patrol_range_bearing``'s MPC is a
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
    scenarios:
        Zero-argument callable returning the task's scenario suite
        (``target_gym.scenarios``), kept in the task's own package so the
        fingerprints cover it. ``None`` is one ``normal`` scenario.
    extra:
        The reason a task is in Extra (charter rule 5): kept in the
        repository and listed with this reason, never scored. ``None`` is a
        benchmark candidate (``benchmark_specs``).
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
    #: Further ``pid_gains.json`` keys the task's expert reads, hashed into its
    #: expert fingerprint (``provenance.gains_entries``).
    gains_keys: tuple[str, ...] = ()
    oracle_reads: tuple[str, ...] = ()
    oracle_gains_keys: tuple[str, ...] = ()
    oracle_data: tuple[str, ...] = ()
    baselines_note: str | None = None
    expert_degraded: str | None = None
    mpc_degraded: str | None = None
    effectiveness_overrides: dict[str, Any] = field(default_factory=dict)
    disturbance_fields: tuple[str, ...] = ()
    #: Parameters that scale *process* noise, zeroed in the copy of the params
    #: a planner uses for its internal model. See ``experts.mpc.plan_params``.
    noise_fields: tuple[str, ...] = ()
    scenarios: Callable[[], tuple] | None = None
    extra: str | None = None
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

    def make_pid_controller(self, sensor_noise_scale: float | None = None):
        """The scored PID as a pure ``Controller``: ``.act`` traces under
        ``jit``/``vmap``, ``.host()`` is what ``make_pid()`` returns.

        ``sensor_noise_scale`` takes the gains tuned at that noise level. None
        and 1 mean the shipped gains, and a level with no stored gains raises
        (``provenance.level_gains``)."""
        if not self.has_pid:
            raise NotImplementedError(self.baselines_note or "no PID ships")
        from target_gym.provenance import level_gains

        make: Any = self.make_pid  # ``_pid`` hangs ``controller`` on the function
        stored = level_gains(self, sensor_noise_scale)
        if stored is None:
            return make.controller()
        from target_gym.experts.pid import gains_from

        with gains_from(stored):
            return make.controller()

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
# pull in matplotlib and pygame.  Each factory therefore
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


def _patrol_range_bearing():
    from target_gym.patrol.env_jax import PlanePatrolRangeBearing

    return PlanePatrolRangeBearing()


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


def _glass_furnace_scenarios():
    from target_gym.glass_furnace.physical_scenarios import suite

    return suite()


def _reactor():
    from target_gym.reactor.env_jax import Reactor

    return Reactor()


def _battery_scenarios():
    # The package lists its suite as plain rows, so it imports nothing outside
    # its fingerprint; the rows become Scenarios here.
    from target_gym.energy.battery.scenarios import battery_scenarios
    from target_gym.scenarios import Scenario

    return tuple(Scenario(**row) for row in battery_scenarios())


def _battery():
    from target_gym.energy.battery.env_jax import GridBattery

    return GridBattery()


def _wind_turbine():
    from target_gym.energy.wind_turbine.env_jax import WindTurbine

    return WindTurbine()


def _cement_kiln():
    from target_gym.cement_kiln.env_jax import CementKiln

    return CementKiln()


def _boiler_drum_scenarios():
    from target_gym.boiler_drum.scenarios import boiler_drum_scenarios
    from target_gym.scenarios import Scenario

    return tuple(Scenario(**row) for row in boiler_drum_scenarios())


def _boiler_drum():
    from target_gym.boiler_drum.env_jax import BoilerDrum

    return BoilerDrum()


def _hvac_scenarios():
    from target_gym.hvac.scenarios import hvac_scenarios
    from target_gym.scenarios import Scenario

    return tuple(Scenario(**row) for row in hvac_scenarios())


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
    # shared core and is in every task's expert fingerprint, so a controller
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
    # Every task's oracle lives in its own package's ``oracle`` module, so
    # changing one re-records only the oracle rows of that package's tasks.
    # experts/mpc.py holds the shared machinery and is in every task's
    # oracle fingerprint.
    def make(env, params, **kwargs):
        from importlib import import_module

        return getattr(import_module(module), factory_name)(env, params, **kwargs)

    return make


def _scenarios(factory_name: str, module: str) -> Callable[[], tuple]:
    """A task's scenario suite, imported when asked for: ``module.factory_name()``
    returns ``Scenario`` records or plain dicts of their fields. The suite lives in
    the task's own package, so its fingerprint covers it."""

    def suite():
        from importlib import import_module

        from target_gym.scenarios import Scenario

        return tuple(
            s if isinstance(s, Scenario) else Scenario(**s)
            for s in getattr(import_module(module), factory_name)()
        )

    return suite


def _shared_oracle(name: str) -> tuple[str, ...]:
    """The data an oracle on the shared solver reads: its horizon summary
    and its smoothing widths (``EnvSpec.oracle_data``)."""
    return (f"data/oracle_horizons/{name}.json", f"data/oracle_widths/{name}.json")


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
        make_mpc=None,
        baselines_note=(
            "No oracle ships until the 2D oracle is rebuilt for the Amendment 2 "
            "score (+-65 ft and +-10 kt bands, plane/PHYSICS.md): the shipped "
            "model plans the earlier quadratic cost, so plane's room is not "
            "measured (charter rule 9)."
        ),
        tuned_gains_key="plane",
        oracle_data=_shared_oracle("plane"),
        disturbance_fields=("gust_x", "gust_z"),
        # Stressed: heavier gusts, in a zone from the first step (the scored
        # episode puts its zone anywhere, plane/PHYSICS.md).
        disturbance_overrides={
            "turbulence_sigma": 3.0,
            "zone_onset_range": (0.0, 0.0),
        },
        noise_fields=("turbulence_sigma",),
        # v4: Amendment 2, the job (plane/PHYSICS.md): protected limits, air
        # data read, established start, a turbulence zone, band costs.
        fingerprint_sources=("scenarios.py",),
        scenarios=_scenarios("suite", module="target_gym.plane.scenarios"),
        # v5: Amendment 3 (plane/PHYSICS.md): cruise at FL290 to 370, the light
        # zone's sigma re-derived there, and a scenario suite.
        version=5,
    ),
    # The same aircraft on ATC's level and speed clearances (plane_energy, Extra
    # under rule 18: thrust never runs out, plane/EXTRA.md), and on a sinusoid
    # (plane_sine, Extra: no real duty, plane/EXTRA.md).
    EnvSpec(
        name="plane_energy",
        group="aircraft",
        env_factory=_plane_energy,
        params_cls=_LazyParams("target_gym.plane.env", "PlaneParams"),
        make_pid=_pid("make_plane_pid_controller", module="target_gym.plane.experts"),
        make_mpc=None,
        baselines_note=(
            "No oracle ships until the 2D oracle is rebuilt for the Amendment 2 "
            "score (clearances on the AIM and RVSM bands, plane/PHYSICS.md): "
            "the shipped model plans the earlier cost, so plane_energy's room "
            "is not measured (charter rule 9)."
        ),
        # Gains are the altitude-hold task's, deliberately not re-tuned here,
        # so the PID stays an honest reference for whether the trade pays.
        tuned_gains_key="plane",
        oracle_data=_shared_oracle("plane_energy"),
        disturbance_fields=("gust_x", "gust_z"),
        # Stressed: heavier gusts, in a zone from the first step (the scored
        # episode puts its zone anywhere, plane/PHYSICS.md).
        disturbance_overrides={
            "turbulence_sigma": 3.0,
            "zone_onset_range": (0.0, 0.0),
        },
        noise_fields=("turbulence_sigma",),
        # v6: the 2D plant's Amendment 2 changes (plane/PHYSICS.md).
        # v7: Amendment 2's job: ATC clearances, scored on the AIM and RVSM
        # bands (plane/PHYSICS.md). v8: the altitude band travels with the
        # 500 fpm profile instead of starting where it ends.
        version=8,
    ),
    EnvSpec(
        name="plane_sine",
        group="aircraft",
        env_factory=_plane_sine,
        params_cls=_LazyParams("target_gym.plane.env", "PlaneParams"),
        make_pid=_pid("make_plane_pid_controller", module="target_gym.plane.experts"),
        make_mpc=_mpc("make_plane_mpc", module="target_gym.plane.oracle"),
        tuned_gains_key="plane",
        oracle_data=_shared_oracle("plane_sine"),
        disturbance_fields=("gust_x", "gust_z"),
        # Stressed: heavier gusts, in a zone from the first step (the scored
        # episode puts its zone anywhere, plane/PHYSICS.md).
        disturbance_overrides={
            "turbulence_sigma": 3.0,
            "zone_onset_range": (0.0, 0.0),
        },
        noise_fields=("turbulence_sigma",),
        # v4: the 2D plant's Amendment 2 changes (plane/PHYSICS.md)
        version=4,
    ),
    EnvSpec(
        name="plane3d_heading",
        group="aircraft",
        env_factory=_plane3d_heading,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_heading_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=None,
        baselines_note=(
            "No oracle ships until the 3D oracle is rebuilt for the Amendment 2 "
            "score (heading select scored on the RVSM, AIM and ACS bands, plane3d/PHYSICS.md): the shipped model plans the "
            "earlier cost, so the room is not measured (charter rule 9)."
        ),
        tuned_gains_key="plane3d_heading",
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        # Stressed: heavier gusts, in a zone from the first step.
        disturbance_overrides={
            "turbulence_sigma": 3.0,
            "zone_onset_range": (0.0, 0.0),
        },
        noise_fields=("turbulence_sigma",),
        # The 2D aircraft's dynamics, which the 3D model extends, and the
        # TECS its experts fly (an experts file, so in the baseline
        # fingerprint only).
        fingerprint_sources=(
            "plane/dynamics.py",
            "plane/experts_tecs.py",
            "scenarios.py",
        ),
        # v6: Amendment 2 (plane3d/PHYSICS.md): protected limits, air data,
        # established start, a turbulence zone, heading select scored on the RVSM, AIM and ACS bands.
        scenarios=_scenarios("heading_suite", module="target_gym.plane3d.scenarios"),
        # v7: Amendment 3 (plane3d/PHYSICS.md): the protected 30 deg bank,
        # vectors every 150 s, and a scenario suite.
        version=7,
    ),
    EnvSpec(
        name="plane3d_circle",
        group="aircraft",
        env_factory=_plane3d_circle,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_circle_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=_mpc("make_plane3d_mpc", module="target_gym.plane3d.oracle"),
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
        name="plane3d_racetrack",
        group="aircraft",
        env_factory=_plane3d_racetrack,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_racetrack_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=None,
        baselines_note=(
            "No oracle ships until the 3D oracle is rebuilt for the Amendment 2 "
            "score (the holding pattern's own legs, turns and speed, scored on the RVSM, AIM and ACS bands, plane3d/PHYSICS.md): the shipped model plans the "
            "earlier cost, so the room is not measured (charter rule 9)."
        ),
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        # Stressed: heavier gusts, in a zone from the first step.
        disturbance_overrides={
            "turbulence_sigma": 3.0,
            "zone_onset_range": (0.0, 0.0),
        },
        noise_fields=("turbulence_sigma",),
        # The 2D aircraft's dynamics, which the 3D model extends, and the
        # TECS its experts fly (an experts file, so in the baseline
        # fingerprint only).
        fingerprint_sources=(
            "plane/dynamics.py",
            "plane/experts_tecs.py",
            "scenarios.py",
        ),
        # v6: Amendment 2 (plane3d/PHYSICS.md): protected limits, air data,
        # established start, a turbulence zone, the holding pattern's own legs, turns and speed, scored on the RVSM, AIM and ACS bands.
        scenarios=_scenarios("racetrack_suite", module="target_gym.plane3d.scenarios"),
        # v7: Amendment 3 (plane3d/PHYSICS.md): the protected 30 deg bank and
        # a scenario suite with steady winds.
        # v8: the altitude band is the ACS holding tolerance, +-100 ft, at the
        # hold's FL100 to FL260, below RVSM airspace (plane3d/PHYSICS.md).
        version=8,
    ),
    EnvSpec(
        name="plane3d_figure8",
        group="aircraft",
        env_factory=_plane3d_figure8,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid(
            "make_plane3d_figure8_pid_controller", module="target_gym.plane3d.experts"
        ),
        make_mpc=_mpc("make_plane3d_mpc", module="target_gym.plane3d.oracle"),
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
        noise_fields=("turbulence_sigma",),
        # The 2D aircraft's dynamics, which the 3D model extends, and the
        # TECS its experts fly (an experts file, so in the baseline
        # fingerprint only).
        fingerprint_sources=("plane/dynamics.py", "plane/experts_tecs.py"),
        # v4: the distance to the curve observed as the tracked output.
        version=5,
    ),
    EnvSpec(
        name="patrol",
        group="aircraft",
        env_factory=_patrol,
        params_cls=_LazyParams("target_gym.patrol.env", "PatrolParams"),
        make_pid=_pid("make_patrol_pid_controller", module="target_gym.patrol.experts"),
        make_mpc=_mpc("make_patrol_mpc", module="target_gym.patrol.oracle"),
        # The oracle's twin law flies the 3D heading PID, on the old lead's gains
        # (task constants, patrol/oracle.py). The expert's gains are in its
        # experts.py, as the 3D aircraft's are.
        oracle_reads=("experts/pid.py",),
        noise_fields=("turbulence_sigma",),
        # The 3D aircraft it flies; the PID its oracle's twin law flies (physics
        # here, so declared for the version stamp too); the autopilot and TECS
        # its lead and its expert fly (experts files, so in the expert
        # fingerprint only: the lead's values are constants in patrol/env.py).
        fingerprint_sources=(
            "plane/dynamics.py",
            "plane3d/dynamics.py",
            "plane3d/env.py",
            "experts/pid.py",
            "plane3d/experts.py",
            "plane/experts_tecs.py",
        ),
        version=4,
    ),
    EnvSpec(
        name="patrol_range_bearing",
        group="aircraft",
        env_factory=_patrol_range_bearing,
        params_cls=_LazyParams("target_gym.patrol.env", "PatrolParams"),
        make_pid=_pid(
            "make_patrol_range_bearing_pid_controller",
            module="target_gym.patrol.experts",
        ),
        make_mpc=_mpc(
            "make_patrol_range_bearing_mpc", module="target_gym.patrol.oracle"
        ),
        oracle_reads=("experts/pid.py",),
        baselines_note=(
            "The PID is a lead-state estimator feeding the same formation "
            "hold the full-observation variant uses. Range with azimuth and "
            "elevation is a complete relative-position measurement, so the "
            "only unobservable quantity is the lead's heading, which the slot "
            "needs because it is expressed in the lead's frame; the estimator "
            "takes the lead's velocity as the filtered rate of its position, "
            "with the follower's own motion added back, and its heading from "
            "that track less "
            "the crab the follower flies in the same air. The MPC slot holds patrol's oracle reading the true "
            "state, so it is a full-state bound, not a range-and-bearing "
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
        version=4,
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
        make_mpc=_mpc("make_cstr_mpc", module="target_gym.pc_gym.cstr.oracle"),
        tuned_gains_key="cstr",
        oracle_data=_shared_oracle("cstr"),
        # v5 (charter Amendment 3): a PARSEX-like jacketed reactor with the
        # jacket in code, cooling water as the move, a high-temperature
        # interlock, temperature as the job, scenarios.
        # v6: cooling water enters at 29.9 C and runs to 136 g/s, so the design
        # point takes S1's 120 g/s [S1, p. 6, Fig. 2]; catalyst_loss is S1's
        # 40 % as a 90 min decay, not a step [S1, p. 6]; the feed follows the
        # room through the day across S1's 23.1 to 24.9 C [S1, Tables 3, 4].
        version=6,
        scenarios=_scenarios(
            "cstr_scenarios", module="target_gym.pc_gym.cstr.scenarios"
        ),
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
            "make_first_order_mpc", module="target_gym.pc_gym.first_order.oracle"
        ),
        # 150 steps, twice the 63 steps the SIMC PI takes to come within twice
        # the level it holds. At 100 the scored hold was the last 37 steps.
        tuned_gains_key="first_order",
        oracle_data=_shared_oracle("first_order"),
        version=4,
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
            "make_four_tank_mpc", module="target_gym.pc_gym.four_tank.oracle"
        ),
        tuned_gains_key="four_tank",
        oracle_data=_shared_oracle("four_tank"),
        extra=(
            "a lab rig with no industrial job: Johansson's teaching process "
            "tracks levels to 1 mm, and no plant family stands behind its "
            "flow-split zero (charter Amendment 3)"
        ),
        # 4: the levels are read through instruments (PHYSICS.md, Readings).
        # 5 (charter Amendment 2): the Lund rig's P+ constants, no trip, an
        # overflowing rim, and an episode of setpoint steps from the operating
        # point with an extra upper outlet and a cup. 6: failure_cost 0, as
        # for every plant with no trip (charter B5).
        version=6,
        # One scenario, the window as scored, declared for the stream's event
        # rate (long-strata draft 4, S1).
        scenarios=_scenarios(
            "four_tank_scenarios", module="target_gym.pc_gym.four_tank.scenarios"
        ),
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
            "make_ph_mpc", module="target_gym.pc_gym.ph_neutralization.oracle"
        ),
        tuned_gains_key="ph_neutralization",
        disturbance_fields=("q2",),
        # The buffer flow's OU innovation, zeroed for the oracle, which rolls
        # the simulator and would otherwise plan on one seed's draws.
        noise_fields=("q2_noise_std",),
        # v4 (charter Amendment 2): the permit zone with its 60 min timed
        # limit, diversion below pH 5, acid strength upsets, a 2 h episode from
        # established operation, and a job score in reagent litres. v5
        # (Amendment 3): two 10 min stages with a valve each, a finite
        # diversion basin pumped back, influent flow swings, scenarios.
        version=5,
        scenarios=_scenarios(
            "ph_neutralization_scenarios",
            module="target_gym.pc_gym.ph_neutralization.scenarios",
        ),
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
            "make_distillation_mpc", module="target_gym.pc_gym.distillation.oracle"
        ),
        # 400 min ~ two dominant time constants, the length the episode rule asks
        # of the analyser-fed expert. The column is the slowest environment per
        # step (80 states, 16 substeps for stability).
        tuned_gains_key="distillation",
        # v3 observes the stage-15 temperature, the thermocouple the base
        # layer's temperature loop holds. v5 (charter Amendment 2): product
        # zones instead of trips, feed-rate and composition steps [D1, Fig.
        # 18], the feed flow observed, and a job score in kmol of boilup.
        # The feed steps are drawn at reset, so no step key moves them: no
        # ``disturbance_fields``. v6 (charter Amendment 3): D2's realistic
        # drums, specs at the family's give-away margin, scenario suite. v7:
        # failure_cost 0, as for every plant with no trip (charter B5).
        # v8: light key lost to the bottoms is priced at 100 kmol of boilup a
        # kmol, Column A's distillate over bottoms price and energy [D8].
        version=8,
        scenarios=_scenarios(
            "distillation_scenarios", module="target_gym.pc_gym.distillation.scenarios"
        ),
    ),
    # -- Industrial / energy ------------------------------------------------
    EnvSpec(
        name="glass_furnace",
        group="industrial",
        env_factory=_glass_furnace,
        params_cls=_LazyParams(
            "target_gym.glass_furnace.env_jax", "GlassFurnaceParams"
        ),
        make_pid=_pid(
            "make_glass_furnace_pid_controller",
            module="target_gym.glass_furnace.experts",
        ),
        # Attempt 3 (2026-11-08) replaced the plant. Its oracle plans the reward on
        # the env's own step with the episode's foresight (glass_furnace/oracle.py).
        make_mpc=_mpc(
            "make_glass_furnace_mpc", module="target_gym.glass_furnace.oracle"
        ),
        baselines_note=(
            "The plant was rebuilt (attempt 3: a float melter in two dimensions, "
            "scored in money on fuel and lost glass). The PID is the reference "
            "ladder's first rung, a PI on the cold-end fining reading two sigma "
            "above its pass mark. The oracle plans the valve on the env's own step "
            "and reward, with the episode's walks foreseen when given its seed, by "
            "SLSQP, which does not converge on this plant's ratchet: its plans "
            "bound the ceiling from above (glass_furnace/oracle.py)."
        ),
        expert_degraded=(
            "The fining PI costs 335,604 EUR a 7-day episode, mean of the conformance "
            "test's two seeds, against 242,995 for the best constant valve (raw 0.0, "
            "0.5 kg/s). Its setpoint sits two sigma above a pass mark the reward "
            "prices as a step, so it loses glass where a wide constant margin does "
            "not: full back-off is what this reward pays for (PHYSICS.md, cold "
            "review item 4). A finding for the reference ladder; not tuned."
        ),
        tuned_gains_key="glass_furnace",
        disturbance_fields=("walk",),
        scenarios=_glass_furnace_scenarios,
        version=4,
    ),
    EnvSpec(
        name="reactor",
        group="industrial",
        env_factory=_reactor,
        params_cls=_LazyParams("target_gym.reactor.env", "ReactorParams"),
        make_pid=_pid(
            "make_reactor_pid_controller", module="target_gym.reactor.experts"
        ),
        make_mpc=_mpc("make_reactor_mpc", module="target_gym.reactor.oracle"),
        tuned_gains_key="reactor",
        oracle_data=_shared_oracle("reactor"),
        version=6,
        noise_fields=("demand_sigma",),
        scenarios=_scenarios("suite", module="target_gym.reactor.scenarios"),
    ),
    EnvSpec(
        name="hvac",
        group="industrial",
        env_factory=_hvac,
        params_cls=_LazyParams("target_gym.hvac.env", "HVACParams"),
        make_pid=_pid("make_hvac_pid_controller", module="target_gym.hvac.experts"),
        make_mpc=_mpc("make_hvac_mpc", module="target_gym.hvac.oracle"),
        # v4 (charter Amendments 1 and 2): a TABS concrete slab between the
        # supply water and the room ([9]'s first-order model), supply water
        # as the action, the 21 to 26 C band at all hours with no setback,
        # two-week episodes from established operation, a score in euros of
        # heat and kelvin-hours of discomfort (hvac/PHYSICS.md).
        tuned_gains_key="hvac",
        oracle_data=_shared_oracle("hvac"),
        disturbance_fields=("weather_dev",),
        noise_fields=("T_out_noise_std",),
        # v5 (Amendment 3): supply capped at 49 C [14], a brine heat pump whose
        # COP falls with supply temperature, a three-period tariff, comfort
        # priced as lost office work, a day-ahead forecast in the observation,
        # and a scenario suite.
        scenarios=_hvac_scenarios,
        version=5,
    ),
    EnvSpec(
        name="cement_kiln",
        group="industrial",
        version=4,
        env_factory=_cement_kiln,
        params_cls=_LazyParams("target_gym.cement_kiln.env", "CementKilnParams"),
        make_pid=_pid(
            "make_cement_kiln_pid_controller", module="target_gym.cement_kiln.experts"
        ),
        make_mpc=_mpc("make_cement_kiln_mpc", module="target_gym.cement_kiln.oracle"),
        # 700 steps = 5.8 hours at dt = 30 s, about fourteen transport delays,
        # so a controller lives with the consequences of its own fuel changes
        # many times over.
        tuned_gains_key="cement_kiln",
        disturbance_fields=("raw_meal",),
        # The raw-meal feed's OU innovation. Declared so the planner plans on the
        # mean (``plan_params``) rather than on one fixed noise path.
        oracle_data=_shared_oracle("cement_kiln"),
        noise_fields=("feed_noise_std",),
        # One scenario, the normal kiln as scored, declared for the stream's
        # target-change rate (long-strata draft 4, S1).
        scenarios=_scenarios(
            "cement_kiln_scenarios", module="target_gym.cement_kiln.scenarios"
        ),
    ),
    EnvSpec(
        name="boiler_drum",
        group="industrial",
        env_factory=_boiler_drum,
        params_cls=_LazyParams("target_gym.boiler_drum.env", "BoilerDrumParams"),
        make_pid=_pid(
            "make_boiler_drum_pid_controller", module="target_gym.boiler_drum.experts"
        ),
        make_mpc=_mpc("make_boiler_drum_mpc", module="target_gym.boiler_drum.oracle"),
        # v4 (charter Amendments 1 and 2): Astrom and Bell's published model,
        # the sourced timed level trips, a load plan ramping at 1 to 2 %/min
        # over an hour, and a score in tonnes of steam off the plan
        # (boiler_drum/PHYSICS.md).
        tuned_gains_key="boiler_drum",
        # The load plan is drawn at reset and read by the agent: no hidden
        # disturbance moves with the step's key.
        oracle_data=_shared_oracle("boiler_drum"),
        # v5 (Amendment 3): scenarios with an ID fan runback and a feed pump
        # trip, ramps over the family's 1 to 6 %/min, a score in GBP at the
        # imbalance price past the steam meter's resolution.
        scenarios=_boiler_drum_scenarios,
        version=5,
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
            "make_wind_turbine_mpc", module="target_gym.energy.wind_turbine.oracle"
        ),
        # v5 (charter Amendment 3): the tower's fore-aft mode on the rotor's
        # thrust, the overspeed feather as the one trip, 10 min windows of
        # above-rated operation in six scenarios, and a score in dollars: lost
        # production against the available power, tower fatigue and pitch wear
        # at their shares of the turbine's cost (energy/wind_turbine/PHYSICS.md).
        tuned_gains_key="wind_turbine",
        oracle_data=_shared_oracle("wind_turbine"),
        noise_fields=("turbulence_iref",),
        disturbance_fields=("v_wind",),
        scenarios=_scenarios(
            "wind_turbine_scenarios", module="target_gym.energy.wind_turbine.scenarios"
        ),
        version=5,
    ),
    EnvSpec(
        name="battery",
        group="energy",
        env_factory=_battery,
        params_cls=_LazyParams("target_gym.energy.battery.env", "BatteryParams"),
        make_pid=_pid(
            "make_battery_pid_controller", module="target_gym.energy.battery.experts"
        ),
        make_mpc=_mpc("make_battery_mpc", module="target_gym.energy.battery.oracle"),
        # v4 (charter Amendments 1 and 2): an LFP Megapack whose BMS reports a
        # drifting state of charge, selling Dynamic Regulation for one 4 h EFA
        # block, steered by its baseline past gate closure, and scored in MW h
        # of availability lost (energy/battery/PHYSICS.md).
        tuned_gains_key="battery",
        # The regulation signal's innovation, and the BMS error drawn at reset.
        noise_fields=("signal_std", "soc_error_std"),
        disturbance_fields=("regulation",),
        # v5 (Amendment 3): the named property is the 60 to 90 min gate-closure
        # delay; energy priced at the GB wholesale price; six scenarios.
        scenarios=_battery_scenarios,
        version=5,
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
            "make_unstable_cstr_mpc", module="target_gym.pc_gym.unstable_cstr.oracle"
        ),
        # v5 (charter Amendment 3): an LDPE-like adiabatic autoclave at its
        # documented unstable steady state, initiator flow as the move, a
        # high-temperature trip, temperature as the job, scenarios.
        # v6: the cold monomer feed wanders 1 K around its 353 K over 10 min
        # (size and period ours), so the normal day moves the unstable point.
        version=6,
        tuned_gains_key="unstable_cstr",
        oracle_data=_shared_oracle("unstable_cstr"),
        scenarios=_scenarios(
            "unstable_cstr_scenarios",
            module="target_gym.pc_gym.unstable_cstr.scenarios",
        ),
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
            "make_compressor_surge_mpc", module="target_gym.compressor_surge.oracle"
        ),
        # 1200 steps = 120 s at 0.1 s: four 30-second setpoint blocks and six
        # 20-second demand blocks, with no effectiveness_overrides.
        tuned_gains_key="compressor_surge",
        oracle_data=_shared_oracle("compressor_surge"),
        # The OU deviation of the consumers' opening, zero-mean;
        # demand_sigma is its stationary sd.
        disturbance_fields=("demand_dev",),
        noise_fields=("demand_sigma",),
        # No fingerprint_sources: the package imports only itself and the
        # shared plumbing. v4 (charter Amendment 3): the counted-surge trip,
        # the valve's positioner lag, a constant header setpoint, a recycle
        # energy score inside a header zone, scenarios.
        version=4,
        scenarios=_scenarios(
            "compressor_surge_scenarios", module="target_gym.compressor_surge.scenarios"
        ),
    ),
    EnvSpec(
        name="formation",
        group="aircraft",
        env_factory=_formation,
        params_cls=_LazyParams("target_gym.formation.env", "FormationParams"),
        make_pid=_pid(
            "make_formation_pid_controller", module="target_gym.formation.experts"
        ),
        make_mpc=_mpc("make_formation_mpc", module="target_gym.formation.oracle"),
        # The oracle's lead autopilot reads plane3d_heading's gains. The
        # expert's are in the experts.py files, as the 3D aircraft's are.
        oracle_reads=("experts/pid.py",),
        oracle_gains_keys=("plane3d_heading",),
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
        # autopilot and TECS (controller files, so in the expert or oracle
        # fingerprint only).
        fingerprint_sources=(
            "patrol/env.py",
            "patrol/experts.py",
            "patrol/oracle.py",
            "plane3d/env.py",
            "plane3d/dynamics.py",
            "plane/dynamics.py",
            "plane3d/experts.py",
            "plane/experts_tecs.py",
        ),
        version=3,
    ),
)

REGISTRY: dict[str, EnvSpec] = {spec.name: spec for spec in _SPECS}


def all_specs() -> Iterator[EnvSpec]:
    """Iterate over every registered environment spec."""
    return iter(_SPECS)


def benchmark_specs() -> Iterator[EnvSpec]:
    """The benchmark candidates: every spec not in Extra (charter rule 5)."""
    return (spec for spec in _SPECS if spec.extra is None)


def specs_in_group(group: str) -> Iterator[EnvSpec]:
    """Iterate over the specs belonging to ``group``."""
    if group not in GROUPS:
        raise KeyError(f"Unknown group {group!r}; expected one of {sorted(GROUPS)}")
    return (spec for spec in _SPECS if spec.group == group)


#: Names that changed, so an old script is told where its task went.
RENAMED = {"patrol_bearing_only": "patrol_range_bearing"}


def get(name: str) -> EnvSpec:
    """Look up one spec by registry name."""
    try:
        return REGISTRY[name]
    except KeyError:
        if name in RENAMED:
            raise KeyError(f"{name!r} is now {RENAMED[name]!r}") from None
        raise KeyError(
            f"Unknown environment {name!r}; registered: {sorted(REGISTRY)}"
        ) from None


def env_names() -> list[str]:
    """Names of every registered environment, in registration order."""
    return [spec.name for spec in _SPECS]
