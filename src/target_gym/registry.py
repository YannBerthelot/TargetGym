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
from typing import Any, Callable, Iterator

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


@dataclass(frozen=True)
class EnvSpec:
    """Everything the library needs to know about one environment.

    Attributes
    ----------
    name:
        Registry key.  Matches the runner module prefix and the key used in
        ``data/pid_gains.json`` where gains are tuned.
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
        ``data/pid_gains.json``.  ``None`` means the controller is not a
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
    """

    name: str
    group: str
    env_factory: Callable[[], Any]
    params_cls: type
    make_pid: Callable[[], Any] | None
    make_mpc: Callable[[Any, Any], Any] | None
    test_params: dict[str, Any] = field(default_factory=dict)
    tuned_gains_key: str | None = None
    baselines_note: str | None = None
    expert_degraded: str | None = None
    effectiveness_overrides: dict[str, Any] = field(default_factory=dict)
    disturbance_fields: tuple[str, ...] = ()
    disturbance_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def has_pid(self) -> bool:
        return self.make_pid is not None

    @property
    def has_mpc(self) -> bool:
        return self.make_mpc is not None

    def make_env(self):
        """Instantiate the environment."""
        return self.env_factory()

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

    return getattr(import_module(module), name)


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
        test_params={"max_steps_in_episode": 200},
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
        test_params={"max_steps_in_episode": 200},
        tuned_gains_key="plane3d_circle",
        disturbance_fields=("gust_x", "gust_y", "gust_z"),
        disturbance_overrides={"turbulence_sigma": 3.0},
    ),
    EnvSpec(
        name="plane3d_figure8",
        group="aircraft",
        env_factory=_plane3d_figure8,
        params_cls=_LazyParams("target_gym.plane3d.env", "PlaneParams3D"),
        make_pid=_pid("make_plane3d_figure8_stateful_pid"),
        make_mpc=_mpc("make_plane3d_mpc"),
        test_params={"max_steps_in_episode": 200},
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
        make_mpc=None,
        test_params={"max_steps_in_episode": 200},
        tuned_gains_key="patrol",
        baselines_note=(
            "PID present -- a stateful wrapper around the functional pursuit "
            "expert, which already held formation; what was missing was the "
            "adapter, not the controller. No MPC yet: the follower's plant is "
            "the full 3D aircraft and the reference is a *manoeuvring lead*, "
            "so an MPC needs the lead's future trajectory as a time-varying "
            "parameter, which is not yet wired."
        ),
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
            "essentially nothing here. No MPC: the follower's plant is the "
            "full 3D aircraft and the reference is a manoeuvring lead."
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
        test_params={"max_steps_in_episode": 100},
        tuned_gains_key="cstr",
    ),
    EnvSpec(
        name="first_order",
        group="process",
        env_factory=_first_order,
        params_cls=_LazyParams("target_gym.pc_gym.first_order.env", "FirstOrderParams"),
        make_pid=_pid("make_first_order_stateful_pid"),
        make_mpc=_mpc("make_first_order_mpc"),
        test_params={"max_steps_in_episode": 100},
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
        test_params={"max_steps_in_episode": 500},
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
        test_params={"max_steps_in_episode": 300},
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
        test_params={"max_steps_in_episode": 200},
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
        make_mpc=_mpc("make_glass_furnace_mpc"),
        test_params={"max_steps_in_episode": 240},  # 2 h at dt=30 s
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
        # max_steps_in_episode is in *physics* steps; the reactor runs
        # ``control_period`` (10) of them per env step, so this is 120 env steps.
        test_params={"max_steps_in_episode": 1200},
        # 8640 physics steps = 864 control steps = 2.4 h, long enough for the
        # xenon/demand dynamics to actually distinguish a controller.
        effectiveness_overrides={"max_steps_in_episode": 8640},
        tuned_gains_key="reactor",
    ),
    EnvSpec(
        name="hvac",
        group="industrial",
        env_factory=_hvac,
        params_cls=_LazyParams("target_gym.hvac.env", "HVACParams"),
        make_pid=_pid("make_hvac_stateful_pid"),
        make_mpc=_mpc("make_hvac_mpc"),
        # 2 days at dt=900 s. Long enough to cover two setback recoveries and
        # two solar cycles, which is what distinguishes controllers here.
        test_params={"max_steps_in_episode": 192},
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
        # 240 steps = 2 hours at dt = 30 s, about five transport delays -- long
        # enough that a controller has to live with the consequences of its
        # own earlier fuel changes.
        test_params={"max_steps_in_episode": 240},
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
        test_params={"max_steps_in_episode": 400},
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
        test_params={"max_steps_in_episode": 400},
        tuned_gains_key="wind_turbine",
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
        test_params={"max_steps_in_episode": 360},
        tuned_gains_key="battery",
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
