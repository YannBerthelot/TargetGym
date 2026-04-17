from importlib.metadata import PackageNotFoundError, version

from target_gym.glass_furnace.env import GlassFurnaceParams
from target_gym.glass_furnace.env_jax import GlassFurnace
from target_gym.pc_gym.cstr.env_jax import CSTR, CSTRParams
from target_gym.pc_gym.first_order.env_jax import FirstOrderParams, FirstOrderSystem
from target_gym.pc_gym.four_tank.env_jax import FourTank, FourTankParams
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D as Plane
from target_gym.plane3d.env import PlaneParams3D
from target_gym.plane3d.env_jax import Plane3DCircle, Plane3DFigureEight, Plane3DHeading
from target_gym.plane3d.env_jax import Plane3DHeading as Plane3D
from target_gym.reactor.env import ReactorParams
from target_gym.reactor.env_jax import Reactor
from target_gym.wrapper import gym_wrapper_factory

try:
    __version__ = version("target-gym")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for dev environments

GymnasiumPlane = gym_wrapper_factory(Plane)


__all__ = (
    "Plane",
    "PlaneParams",
    "Plane3D",
    "Plane3DHeading",
    "Plane3DCircle",
    "Plane3DFigureEight",
    "PlaneParams3D",
    "GymnasiumPlane",
    # PC-Gym environments
    "CSTR",
    "CSTRParams",
    "FirstOrderSystem",
    "FirstOrderParams",
    "FourTank",
    "FourTankParams",
    # Glass furnace
    "GlassFurnace",
    "GlassFurnaceParams",
    # Nuclear reactor
    "Reactor",
    "ReactorParams",
)
