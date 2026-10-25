"""Env-level access to the shipped experts, resolved through the registry.

Kept out of ``base.py`` and the environment packages on purpose: those files
are hashed into every provenance fingerprint, and how a controller is looked
up is not part of what an environment is.
"""


class RegistryExperts:
    """``make_pid`` and ``make_mpc`` resolved through the registry.

    An environment class sets ``registry_name`` and gets the same controllers
    ``REGISTRY[registry_name].make_pid`` / ``.make_mpc`` build, so the env
    method and the registry cannot name different factories.
    """

    registry_name: str

    def make_pid(self):
        """Return the shipped PID baseline, ready to call on observations."""
        from target_gym.registry import REGISTRY

        spec = REGISTRY[self.registry_name]
        if not spec.has_pid:
            raise NotImplementedError(spec.baselines_note or "no PID ships")
        return spec.make_pid()

    @property
    def expert_policy(self):
        """The scored PID in batched JAX form, for ``jit``/``vmap``/GPU: the
        pure controller batched, so the caller carries its memory.
        ``policy.init_state(num_envs)``, then
        ``actions, state = policy(state, obs)``; never ``policy(obs)``, which
        has no memory. Built from the same step
        function as ``make_pid()``, so the two act alike."""
        from target_gym.registry import REGISTRY

        return REGISTRY[self.registry_name].make_pid_controller().batched()

    def make_mpc(self, params=None, **kwargs):
        """Return the shipped MPC oracle for ``params`` (default params if None)."""
        from target_gym.registry import REGISTRY

        spec = REGISTRY[self.registry_name]
        if not spec.has_mpc:
            raise NotImplementedError(spec.baselines_note or "no MPC ships")
        if params is None:
            params = self.default_params
        return spec.make_mpc(self, params, **kwargs)
