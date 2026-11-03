"""Reference controller (the MPC slot) for the first-order task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

from target_gym.experts.mpc import NewtonMPC


def make_first_order_mpc(env, params, horizon: int = 5, n_iter: int = 10):
    """The oracle: ``NewtonMPC`` on the plant's own step and reward, over
    PC-gym's five-step horizon (0.25 s)."""
    return NewtonMPC(env, params, horizon=horizon, n_iter=n_iter)


def make_first_order_pid_controller():
    """The scored PI as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. SIMC gains from a step test,
    stored in src/target_gym/data/pid_gains.json by
    ``scripts/tune_process_experts.py`` (PHYSICS.md section 6)."""
    from target_gym.experts.pid import gain, siso_pid

    return siso_pid(
        Kp=gain("first_order", "Kp", 5.0),
        Ki=gain("first_order", "Ki", 25.0),
        Kd=gain("first_order", "Kd", 0.0),
        dt=0.05,
        state_index=0,
        setpoint_index=1,
    )
