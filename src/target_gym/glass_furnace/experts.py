"""The expert (the PID slot) for the glass furnace task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""


def make_glass_furnace_pid_controller():
    """The scored PID as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains from
    src/target_gym/data/pid_gains.json."""
    from target_gym.experts.pid import gain, siso_pid

    return siso_pid(
        Kp=gain("glass_furnace", "Kp", 0.01),
        Ki=gain("glass_furnace", "Ki", 0.001),
        Kd=gain("glass_furnace", "Kd", 0.0),
        dt=30.0,
        state_index=0,
        setpoint_index=4,
    )
