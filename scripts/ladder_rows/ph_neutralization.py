"""The reference ladder's row for ``ph_neutralization`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score
(PHYSICS.md section 10). Two stages in series, each with its own valve and probe:
stage 1's base valve on its probe, stage 2's trim valve on the discharge probe. The
discharge is a zone, pH 6 to 8 [P7], cheap side low (less base holds a lower pH), so
its loop holds 2 sigma above 6; stage 1 has no zone and follows the same setpoint.
"""

from __future__ import annotations

import math

import numpy as np
from reference_ladder import (  # noqa: E402
    Cv,
    Limit,
    Row,
)

from target_gym.registry import REGISTRY

DISCHARGE = "pH on trim"


def row() -> Row:
    from target_gym.pc_gym.ph_neutralization.env import steady_base_flow

    spec = REGISTRY["ph_neutralization"]
    p = spec.make_test_params()
    # Bump tests at the reset steady state with the upsets and the buffer's drift at
    # 0 (PHYSICS.md section 10).
    test = {"acid_upset_size": 0.0, "flow_swing": 0.0, "q2_noise_std": 0.0}
    low, high = float(p.permit_low), float(p.permit_high)
    centre = 0.5 * (low + high)
    q3_min, span = float(p.q3_min), float(p.q3_max - p.q3_min)
    q2_nominal = float(p.q2_nominal)

    # The build fills ``placed`` with the discharge loop's setpoint. Stage 1 and the
    # feedforward read it when their rungs are traced; before that, while the build
    # measures sigma with the loops at the zone's centre, they read the centre.
    placed: dict = {}

    def setpoint() -> float:
        return float(placed[DISCHARGE][1]) if DISCHARGE in placed else centre

    def raw(q3, xp=np):
        return 2.0 * (q3 - q3_min) / span - 1.0

    def rest_at(q):
        """Stage 1's valve on the steady balance at the point's setpoint, buffer and
        streams (``steady_base_flow``, as the reset puts it), the trim shut."""
        q3 = float(steady_base_flow(float(q.setpoint_pH), float(q.q2_nominal), q))
        u = np.array([raw(q3), -1.0], np.float32)
        return lambda o, xp: xp.asarray(u)

    def balance(o, xp):
        """Feedforward from the metered influent (obs 6): the base the steady
        balance needs at the setpoint, linear in the influent, with the hidden
        buffer at its nominal flow (the lab titration's). The trim rests shut."""
        sp = setpoint()
        per_influent = float(steady_base_flow(sp, 0.0, p.replace(q1=1.0)))
        per_buffer = float(steady_base_flow(sp, 1.0, p.replace(q1=0.0)))
        q3 = per_influent * o[6] + per_buffer * q2_nominal
        return xp.stack([xp.clip(raw(q3, xp), -1.0, 1.0), xp.asarray(-1.0)])

    # Each stage is one stirred tank: its residence V / Q is 10.2 min at the
    # nominal 32.8 mL/s (PHYSICS.md section 2), 122 steps of 5 s. A window of five
    # residence times reads the whole response, past the probe's 10 s lag and 10 s
    # dead time. Each bump is 1 % of its valve's span (0.02 raw); the noise band
    # sizes it up where that moves the probe too little (N3).
    residence = float(p.V_s1) / 32.8
    window = math.ceil(5 * residence / float(p.delta_t))
    du = 0.02

    stage_1 = Cv(
        "pH_s1 on base",
        0,
        lambda o, xp: o[4],
        lambda o, xp: xp.asarray(setpoint(), dtype=xp.float32),
        "pH_s1",
        du,
        window,
    )
    discharge = Cv(
        DISCHARGE,
        1,
        lambda o, xp: o[0],
        lambda o, xp: o[2],
        "pH",
        du,
        window,
        zone=(low, high, "low"),
    )
    # The field case's "kicker" steps the second stage's valves open on low pH
    # [P3, pp. 4-5]: an override on the trim, holding the discharge above the
    # permit's low edge.
    kicker = Limit("kicker on trim", lambda o, xp: o[0], low, True, 1, "pH", du, window)

    # Every load range the loops visit, one end at a time from the base point: the
    # influent's acid strength and flow at +-20 % (scenarios.py), and the buffer
    # from none (``buffer_loss``) to its nominal plus two of its standard
    # deviations. Their corners are left out: 20 % more acid in 20 % more flow
    # needs more base than stage 1's valve passes, so no rest holds it. The loops
    # hold one setpoint, so there is no target range; the zone's edges are no rest
    # either, since a plant held on the permit's edge is outside it on the
    # slightest error, and the 60 min excursion clock trips it during the hold.
    buffer_high = q2_nominal + 2.0 * float(p.q2_noise_std)
    points = (
        {"Wa1": float(p.Wa1) * (1.0 - float(p.acid_upset_size))},
        {"Wa1": float(p.Wa1) * (1.0 + float(p.acid_upset_size))},
        {"q1": float(p.q1) * (1.0 - float(p.flow_swing))},
        {"q1": float(p.q1) * (1.0 + float(p.flow_swing))},
        {"q2_nominal": float(p.q2_min)},
        {"q2_nominal": buffer_high},
    )
    base = spec.make_test_params(**test)
    return Row(
        720,
        rest_at(base),
        (stage_1, discharge),
        test=test,
        points=points,
        rest_at=rest_at,
        balance=balance,
        limits=(kicker,),
        placed=placed,
        why={
            "cascade": "no inner measurement: each valve sets its reagent flow "
            "directly, and the field case's cascade [P3] is a set point optimizer "
            "over the pH loops, not a loop under them",
            "decoupler": "the coupling runs one way, stage 1 into stage 2, and the "
            "trim rests shut and only adds base, so it cannot cancel stage 1's moves",
        },
    )
