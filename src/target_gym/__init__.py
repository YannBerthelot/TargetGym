from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from target_gym.api import (
    list_envs,
    make,
    make_expert,
    make_gymnasium,
    make_oracle,
)
from target_gym.boiler_drum.env import BoilerDrumParams
from target_gym.boiler_drum.env_jax import BoilerDrum
from target_gym.cement_kiln.env import CementKilnParams
from target_gym.cement_kiln.env_jax import CementKiln
from target_gym.compressor_surge.env import CompressorSurgeParams
from target_gym.compressor_surge.env_jax import CompressorSurge
from target_gym.energy.battery.env import BatteryParams
from target_gym.energy.battery.env_jax import GridBattery
from target_gym.energy.wind_turbine.env import WindTurbineParams
from target_gym.energy.wind_turbine.env_jax import WindTurbine
from target_gym.formation.env import FormationParams
from target_gym.formation.env_jax import Formation
from target_gym.formation.marl import FormationMARL
from target_gym.glass_furnace.env import GlassFurnaceParams
from target_gym.glass_furnace.env_jax import GlassFurnace
from target_gym.hvac.env import HVACParams
from target_gym.hvac.env_jax import BuildingHVAC
from target_gym.patrol.env import PatrolParams
from target_gym.patrol.env_jax import PlanePatrol, PlanePatrolBearingOnly
from target_gym.pc_gym.cstr.env_jax import CSTR, CSTRParams
from target_gym.pc_gym.distillation.env import DistillationParams
from target_gym.pc_gym.distillation.env_jax import DistillationColumn
from target_gym.pc_gym.first_order.env_jax import FirstOrderParams, FirstOrderSystem
from target_gym.pc_gym.four_tank.env_jax import FourTank, FourTankParams
from target_gym.pc_gym.ph_neutralization.env import PHParams
from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization
from target_gym.pc_gym.unstable_cstr.env import UnstableCSTRParams
from target_gym.pc_gym.unstable_cstr.env_jax import UnstableCSTR
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D, PlaneEnergy, PlaneSine
from target_gym.plane3d.env import PlaneParams3D
from target_gym.plane3d.env_jax import (
    Plane3DCircle,
    Plane3DFigureEight,
    Plane3DHeading,
    Plane3DRacetrack,
)
from target_gym.reactor.env import ReactorParams
from target_gym.reactor.env_jax import Reactor
from target_gym.scoring import score_policy

try:
    __version__ = _version("target-gym")
except _PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for dev environments

# Friendly aliases. The class's own ``__name__`` is exported too, because
# ``type(env).__name__`` is the first thing anyone reads off an object and it
# has to be importable -- this package's own README got that wrong, reaching for
# ``Airplane2D`` because that is what the object calls itself.
Plane = Airplane2D
Plane3D = Plane3DHeading


def __getattr__(name):
    # Deprecated names, kept working for a release. Served here rather than
    # bound, so using them warns and the import of the package does not.
    import warnings
    from importlib import import_module

    if name not in ("gym_wrapper_factory", "GymnasiumPlane"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # import_module, not ``from target_gym import wrapper``: that form looks
    # the name up on this module first, which lands back here.
    wrapper = import_module("target_gym.wrapper")
    if name == "gym_wrapper_factory":
        return wrapper.gym_wrapper_factory
    warnings.warn(
        "GymnasiumPlane is deprecated; use target_gym.make_gymnasium('plane')",
        DeprecationWarning,
        stacklevel=2,
    )
    return wrapper._class_wrapper(Plane)


__all__ = (
    # The front door: every task by its registry name
    "make",
    "make_expert",
    "make_oracle",
    "make_gymnasium",
    "list_envs",
    "score_policy",
    # Environment and parameter classes
    "Plane",
    "Airplane2D",  # the class's own name; Plane is the friendly alias
    "PlaneEnergy",
    "PlaneSine",
    "PlaneParams",
    "Plane3D",
    "Plane3DHeading",
    "Plane3DCircle",
    "Plane3DRacetrack",
    "Plane3DFigureEight",
    "PlaneParams3D",
    # Close patrol (formation-keeping)
    "PlanePatrol",
    "PlanePatrolBearingOnly",
    "PatrolParams",
    # Formation: a lead and its wingmen, every aircraft a learner
    "Formation",
    "FormationMARL",
    "FormationParams",
    # PC-Gym environments
    "CSTR",
    "CSTRParams",
    "FirstOrderSystem",
    "FirstOrderParams",
    "FourTank",
    "FourTankParams",
    # Unstable CSTR
    "UnstableCSTR",
    "UnstableCSTRParams",
    # Distillation
    "DistillationColumn",
    "DistillationParams",
    # pH neutralisation
    "PHNeutralization",
    "PHParams",
    # Glass furnace
    "GlassFurnace",
    "GlassFurnaceParams",
    # Grid battery
    "GridBattery",
    "BatteryParams",
    # Boiler drum
    "BoilerDrum",
    "BoilerDrumParams",
    # Cement kiln
    "CementKiln",
    "CementKilnParams",
    # Compressor surge
    "CompressorSurge",
    "CompressorSurgeParams",
    # Wind turbine
    "WindTurbine",
    "WindTurbineParams",
    # Building HVAC
    "BuildingHVAC",
    "HVACParams",
    # Nuclear reactor
    "Reactor",
    "ReactorParams",
)
