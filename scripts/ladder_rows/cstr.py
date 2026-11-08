"""The reference ladder's row for ``cstr`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Row,
    corners,
)

from target_gym.registry import REGISTRY


def _cstr_rest(p, C_a, xp):
    """T and raw coolant at which ``C_a`` is steady, from the two balances."""
    qV = p.q / p.V
    rate = qV * (p.Caf - C_a)
    T = -p.EA_over_R / xp.log(rate / (p.k0 * C_a))
    heat = qV * (p.Ti - T) + (-p.deltaHr) * rate / (p.rho * p.C)
    T_c = T - heat * p.rho * p.C * p.V / p.UA
    return T, 2 * (T_c - p.T_c_min) / (p.T_c_max - p.T_c_min) - 1


def row() -> Row:
    p = REGISTRY["cstr"].make_test_params()

    def balance(o, xp):
        return xp.reshape(_cstr_rest(p, o[2], xp)[1], (1,))

    C_a = Cv(
        "C_a on coolant", 0, lambda o, xp: o[0], lambda o, xp: o[2], "C_a", 0.1, 80
    )
    T = Inner(
        0,
        "T on coolant",
        lambda o, xp: o[1],
        lambda o, xp: _cstr_rest(p, o[2], xp)[0],
        "T",
        0.1,
        60,
        0.5,
        80,
    )
    return Row(
        150,
        balance,
        (C_a,),
        test={"load_scale": 0.0},
        # The feed's drawn loads, held: Ti and Caf at each end of their steps.
        points=tuple(
            {**q, "Ti": float(p.Ti) + dT, "Caf": float(p.Caf) + dC}
            for q in corners(target_CA_range=tuple(p.target_CA_range))
            for dT in (-float(p.Ti_load), float(p.Ti_load))
            for dC in (-float(p.Caf_load), float(p.Caf_load))
        ),
        rest_at=lambda q: lambda o, xp: xp.reshape(_cstr_rest(q, o[2], xp)[1], (1,)),
        balance=balance,
        cascades=(T,),
        why={"decoupler": "one loop", "override": "no reachable limit"},
    )
