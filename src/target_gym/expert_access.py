"""Env-level access to the shipped experts, resolved through the registry.

Kept out of ``base.py`` and the environment packages on purpose: those files
are hashed into every provenance fingerprint, and how a controller is looked
up is not part of what an environment is.

``make_pid`` and ``expert_policy`` are deprecated in favour of
``target_gym.make_expert(name)``, which makes the caller pick the expert's
form, and ``make_mpc`` in favour of ``target_gym.make_oracle(name)``.
"""

import warnings


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"env.{old} is deprecated; use target_gym.{new}",
        DeprecationWarning,
        stacklevel=3,
    )


class RegistryExperts:
    """``make_pid`` and ``make_mpc`` resolved through the registry.

    An environment class sets ``registry_name`` and gets the same controllers
    ``REGISTRY[registry_name].make_pid`` / ``.make_mpc`` build, so the env
    method and the registry cannot name different factories.
    """

    registry_name: str

    def make_pid(self):
        """Deprecated: ``target_gym.make_expert(name).host()``."""
        _deprecated("make_pid()", "make_expert(name).host()")
        from target_gym.registry import REGISTRY

        spec = REGISTRY[self.registry_name]
        if not spec.has_pid:
            raise NotImplementedError(spec.baselines_note or "no PID ships")
        return spec.make_pid()

    @property
    def expert_policy(self):
        """Deprecated: ``target_gym.make_expert(name).batched()``."""
        _deprecated("expert_policy", "make_expert(name).batched()")
        from target_gym.registry import REGISTRY

        return REGISTRY[self.registry_name].make_pid_controller().batched()

    def make_mpc(self, params=None, **kwargs):
        """Deprecated: ``target_gym.make_oracle(name)``, which plans on the mean
        disturbance as the recorded baselines do.

        Returns the shipped MPC oracle for ``params`` (default params if None),
        planning on ``params`` as given."""
        _deprecated("make_mpc()", "make_oracle(name)")
        from target_gym.registry import REGISTRY

        spec = REGISTRY[self.registry_name]
        if not spec.has_mpc:
            raise NotImplementedError(spec.baselines_note or "no MPC ships")
        if params is None:
            params = self.default_params
        return spec.make_mpc(self, params, **kwargs)
