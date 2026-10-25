"""Shared controller machinery.

Every task's controllers live in its own package's ``experts`` module and are
reached through the registry, which is stable where the per-task factories
are not:

    spec = target_gym.registry.REGISTRY["cstr"]
    pid = spec.make_pid()                    # host form: keeps its own memory
    ctrl = spec.make_pid_controller()        # pure form: you carry its memory
    mpc = spec.make_mpc(spec.make_env(), spec.make_test_params())

``env.make_pid()``, ``env.make_mpc()`` and ``env.expert_policy`` resolve the
same way.
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
