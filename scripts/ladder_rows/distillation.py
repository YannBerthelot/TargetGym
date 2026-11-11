"""The reference ladder's row for ``distillation`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score
(PHYSICS.md section 11). Column A in the LV configuration: distillate purity on
reflux, bottoms impurity on boilup, each read by its analyser (5 min cycle, 5 min
dead time). Both products are zones, held 2 sigma inside their cheap edges.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Row,
)

from target_gym.registry import REGISTRY

#: Column A's design products [D2], at which the rest and the balance hold it.
DESIGN_YD, DESIGN_XB = 0.99, 0.01
#: Column A's design flows, kmol/min [D2] (``colamod.m``).
L0, V0 = 2.70629, 3.20629
#: Eq. 31's steady gains, d[yD, xB] / d[L, V] in kmol/min [D1].
EQ_31 = ((0.8754, -0.8618), (1.0846, -1.0982))


@functools.lru_cache(maxsize=None)
def _rest_flows(F: float, zF: float, yD: float, xB: float) -> tuple:
    """``(L, V)`` holding the products at ``yD`` and ``xB`` with feed ``F`` and
    ``zF``: the tray balances at steady state, solved by Newton in float64."""
    from jax.experimental import enable_x64

    from target_gym.pc_gym.distillation.env import compute_velocity, steady_liquid
    from target_gym.pc_gym.distillation.env_jax import _NOMINAL_PROFILE

    p = REGISTRY["distillation"].make_test_params(F=F)
    n = len(_NOMINAL_PROFILE)
    with enable_x64():

        def residual(z):
            x, L, V = z[:n], z[n], z[n + 1]
            here = jnp.concatenate([x, steady_liquid(L, p)])
            dx = compute_velocity(here, (L, V), zF, p)[0][:n]
            products = jnp.stack([x[-1] - yD, x[0] - xB])
            return jnp.concatenate([dx, products])

        @jax.jit
        def solve(z):
            def newton(_, z):
                return z - jnp.linalg.solve(jax.jacfwd(residual)(z), residual(z))

            z = jax.lax.fori_loop(0, 20, newton, z)
            return z, jnp.max(jnp.abs(residual(z)))

        start = np.concatenate([np.asarray(_NOMINAL_PROFILE, float), [L0, V0]])
        z, left = solve(jnp.asarray(start, jnp.float64))
        z = np.asarray(z)
    assert float(left) < 1e-9, (F, zF, yD, xB, float(left))
    return float(z[n]), float(z[n + 1])


def row() -> Row:
    spec = REGISTRY["distillation"]
    p = spec.make_test_params()
    L_min, V_min = float(p.L_min), float(p.V_min)
    per_L, per_V = 2 / float(p.L_max - p.L_min), 2 / float(p.V_max - p.V_min)
    F0, zF0 = float(p.F), float(p.zF_nominal)
    specs = float(p.yD_spec), float(p.xB_spec)

    def raw(L, V, xp=np):
        return xp.stack([per_L * (L - L_min) - 1.0, per_V * (V - V_min) - 1.0])

    def rest_at(q):
        """The flows the tray balances need at the point's feed: for the design
        products at the design feed, the reset's point; for the specs at the feed's
        ends. The specs are the zones' cheap edges, where the loops run, and the
        design products at 20 % more feed need more reflux than the box allows."""
        F, zF = float(q.F), float(q.zF_nominal)
        products = (DESIGN_YD, DESIGN_XB) if (F, zF) == (F0, zF0) else specs
        u = np.asarray(raw(*_rest_flows(F, zF, *products)), np.float32)
        return lambda o, xp: xp.asarray(u)

    design = np.asarray(raw(L0, V0), np.float32)

    # The build fills ``placed`` with each zone loop's setpoint; the feedforward
    # reads them when its rung is traced, after the build has placed them.
    placed: dict = {}

    def held() -> tuple:
        if "yD on L" in placed and "xB on V" in placed:
            return float(placed["yD on L"][1]), float(placed["xB on V"][1])
        return DESIGN_YD, DESIGN_XB

    # Feedforward from the metered feed (obs 7, FC 1 [D4, p. 5]): the flows the
    # tray balances need for the held products at the nominal feed, kept in their
    # ratio to the feed. That is the steady state of a feed-rate change at fixed zF
    # (the balances are homogeneous in the flows), and it closes the overall
    # balance D = F (zF - xB) / (yD - xB) with the unmeasured zF at nominal.
    def balance(o, xp):
        # Traced inside the rung's jit: the solve runs once, eagerly, on constants.
        with jax.ensure_compile_time_eval():
            L, V = _rest_flows(F0, zF0, *held())
        F = o[7] / F0
        return xp.clip(raw(L * F, V * F, xp), -1.0, 1.0)

    # Each analyser's noise band is 6 of its 4e-4 sigmas; a bump moves the product
    # five bands (N3) by eq. 31's larger gain on that MV, so the gain matrix reads
    # the same step. The window is three of the column's slowest time constant with
    # D2's drums, 219.5 min (PHYSICS.md section 9).
    move = 5 * 6 * 4e-4
    du_L = move / abs(EQ_31[0][0]) * per_L
    du_V = move / max(abs(EQ_31[0][1]), abs(EQ_31[1][1])) * per_V
    window = math.ceil(3 * 219.5 / float(p.delta_t))
    yD = Cv(
        "yD on L",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[4],
        "x",
        du_L,
        window,
        zone=(float(p.yD_spec), 1.0, "low"),
    )
    xB = Cv(
        "xB on V",
        1,
        lambda o, xp: o[1],
        lambda o, xp: o[5],
        "x",
        du_V,
        window,
        zone=(0.0, float(p.xB_spec), "high"),
    )
    # The stage-15 tray temperature, on line [D5], under the bottoms loop. Its
    # first response to boilup is a ramp on the column's slow mode, so it is read
    # as integrating over 20 min, under a tenth of that mode; the outer test steps
    # the tray setpoint by about one RTD band (6 x 0.04 K) and reads xB over the
    # outer window.
    tray = Inner(
        1,
        "T_15 on V",
        lambda o, xp: o[6],
        None,
        "",
        du_V,
        20,
        0.24,
        window,
        model="integrating_lag",
    )
    # Bump tests at the design point with the feed steps at 0 (PHYSICS.md section
    # 11), and at the corners of the feed's two steps [D1, Fig. 18] with the
    # products at their specs. From the reset profile a point's column settles over
    # the hold, two 600 min episodes, five of its slowest time constants.
    return Row(
        600,
        lambda o, xp: xp.asarray(design),
        (yD, xB),
        test={"feed_step": 0.0, "zF_step": 0.0},
        points=tuple(
            {"F": F0 * (1 + dF), "zF_nominal": zF0 + dz}
            for dF in (-float(p.feed_step), float(p.feed_step))
            for dz in (-float(p.zF_step), float(p.zF_step))
        ),
        rest_at=rest_at,
        balance=balance,
        cascades=(tray,),
        decouple=True,
        placed=placed,
        why={
            "override": "no trip and no limit a second loop could take an MV for: "
            "the products are zones, and flooding has no sourced figure [D6]",
        },
    )
