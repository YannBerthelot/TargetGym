"""The expert (the PID slot) for the first-order task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""


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
