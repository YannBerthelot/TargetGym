from flax import struct


@struct.dataclass
class EnvParams:
    delta_t: float = 1.0
    max_steps_in_episode: int = 1_000


@struct.dataclass
class EnvState:
    time: int
