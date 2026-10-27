"""Shared controller machinery.

Every task's controllers live in its own package's ``experts`` module and are
reached by registry name, which is stable where the per-task factories are
not:

    expert = target_gym.make_expert("cstr")  # pure form: you carry its memory
    pid = expert.host()                      # host form: keeps its own memory
    oracle = target_gym.make_oracle("cstr")  # full-state MPC, scored params
"""

from target_gym.experts.episodes import reset_where_done
from target_gym.experts.mpc import CasadiMPC, GradientMPC, SamplingMPC
from target_gym.experts.pid import (
    Controller,
    FunctionalExpertPolicy,
    HostController,
    mimo_pid,
    mimo_pid_step,
    pid_step,
    register_learnable_gains,
    siso_pid,
)

__all__ = [
    "Controller",
    "HostController",
    "FunctionalExpertPolicy",
    "siso_pid",
    "mimo_pid",
    "pid_step",
    "mimo_pid_step",
    "register_learnable_gains",
    "reset_where_done",
    "GradientMPC",
    "CasadiMPC",
    "SamplingMPC",
]
