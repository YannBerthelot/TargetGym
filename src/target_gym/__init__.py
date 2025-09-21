from importlib.metadata import PackageNotFoundError, version

from target_gym.bicycle.env import EnvParams as BikeParams
from target_gym.bicycle.env_jax import RandlovBicycle as Bike
from target_gym.car.env import EnvParams as CarParams
from target_gym.car.env_jax import Car2D as Car
from target_gym.plane.env import EnvParams as PlaneParams
from target_gym.plane.env_gymnasium import Airplane2D as PlaneGymnasium
from target_gym.plane.env_jax import Airplane2D as Plane

try:
    __version__ = version("target-gym")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for dev environments

__all__ = (
    "Car",
    "PlaneGymnasium",
    "Plane",
    "Bike",
    "PlaneParams",
    "CarParams",
    "BikeParams",
)  # Make Flake8 Happy
