from importlib.metadata import PackageNotFoundError, version

from target_gym.bicycle.env import BikeParams
from target_gym.bicycle.env_jax import RandlovBicycle as Bike
from target_gym.car.env import CarParams
from target_gym.car.env_jax import Car2D as Car
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D as Plane
from target_gym.wrapper import gym_wrapper_factory

try:
    __version__ = version("target-gym")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for dev environments

GymnasiumPlane = gym_wrapper_factory(Plane)
GymnasiumCar = gym_wrapper_factory(Car)
GymnasiumBike = gym_wrapper_factory(Bike)


__all__ = (
    "Car",
    "PlaneGymnasium",
    "Plane",
    "Bike",
    "PlaneParams",
    "CarParams",
    "BikeParams",
    "GymnasiumPlane",
    "GymnasiumCar",
    "GymnasiumBike",
)  # Make Flake8 Happy
