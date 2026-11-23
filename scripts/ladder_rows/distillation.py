"""The reference ladder's row for ``distillation`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score
(PHYSICS.md section 11). Column A in the LV configuration: distillate purity on
reflux, bottoms impurity on boilup, each read by its analyser (5 min cycle, 5 min
dead time). The distillate is a zone, held 2 sigma inside its cheap edge. The bottoms'
zone has no cheap edge: purer bottoms cost boilup, less pure bottoms lose light key at
the distillate's price [D8], so the loop rests where the two balance (``_bottoms_rest``),
as far as the flow box can hold it at the metered feed (``_bottoms_schedule``).
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
from scipy.optimize import brentq, minimize_scalar

from target_gym.experts.ladder import K_BACKOFF
from target_gym.oracles.controller import enable_x64
from target_gym.registry import REGISTRY

#: Column A's design products [D2], at which the rest and the balance hold it.
DESIGN_YD, DESIGN_XB = 0.99, 0.01
#: Column A's design flows, kmol/min [D2] (``colamod.m``).
L0, V0 = 2.70629, 3.20629
#: Eq. 31's steady gains, d[yD, xB] / d[L, V] in kmol/min [D1].
EQ_31 = ((0.8754, -0.8618), (1.0846, -1.0982))
#: The flow box is 5 % over the flows the specs need (env.py, ``L_max``): the loops'
#: room to act. A rest point leaves the reflux and boilup that much.
ROOM = 1.05


@functools.lru_cache(maxsize=None)
def _newton() -> tuple:
    """The tray balances at steady state by Newton in float64, compiled once:
    ``solve(start, [F, zF, yD, xB])`` gives the solution (profile, L, V) and its
    largest residual. Also the start, the design point, and the profile's length."""
    from target_gym.pc_gym.distillation.env import compute_velocity, steady_liquid
    from target_gym.pc_gym.distillation.env_jax import _NOMINAL_PROFILE

    base = REGISTRY["distillation"].make_test_params()
    n = len(_NOMINAL_PROFILE)

    def residual(z, at):
        F, zF, yD, xB = at
        p = base.replace(F=F)
        x, L, V = z[:n], z[n], z[n + 1]
        here = jnp.concatenate([x, steady_liquid(L, p)])
        dx = compute_velocity(here, (L, V), zF, p)[0][:n]
        return jnp.concatenate([dx, jnp.stack([x[-1] - yD, x[0] - xB])])

    @jax.jit
    def solve(z, at):
        def newton(_, z):
            return z - jnp.linalg.solve(jax.jacfwd(residual)(z, at), residual(z, at))

        z = jax.lax.fori_loop(0, 20, newton, z)
        return z, jnp.max(jnp.abs(residual(z, at)))

    start = np.concatenate([np.asarray(_NOMINAL_PROFILE, float), [L0, V0]])
    return solve, start, n


@functools.lru_cache(maxsize=None)
def _rest(F: float, zF: float, yD: float, xB: float) -> tuple:
    """The column at rest with the products at ``yD`` and ``xB`` on feed ``F`` and
    ``zF``, as ``(profile, L, V)``."""
    solve, start, n = _newton()
    with enable_x64():
        at = jnp.asarray([F, zF, yD, xB], jnp.float64)
        z, left = solve(jnp.asarray(start, jnp.float64), at)
        z = np.asarray(z)
    assert float(left) < 1e-9, (F, zF, yD, xB, float(left))
    return z[:n], float(z[n]), float(z[n + 1])


def _rest_flows(F: float, zF: float, yD: float, xB: float) -> tuple:
    """``(L, V)`` of ``_rest``."""
    return _rest(F, zF, yD, xB)[1:]


def _rest_cost(yD: float, xB: float) -> float:
    """The score per step of the column at rest on its design feed with the products
    at ``yD`` and ``xB``: the env's own reward terms on that steady state."""
    from target_gym.pc_gym.distillation.env import (
        DistillationState,
        compute_reward_terms,
    )

    p = REGISTRY["distillation"].make_test_params()
    x, L, V = _rest(float(p.F), float(p.zF_nominal), yD, xB)
    with enable_x64():
        state = DistillationState(
            time=0, x=jnp.asarray(x), zF=p.zF_nominal, L=L, V=V, F=p.F,
            target_yD=p.yD_spec, target_xB=p.xB_spec,
        )  # fmt: skip
        return float(sum(compute_reward_terms(state, p).values()))


def _bottoms_rest(p) -> float:
    """The bottoms impurity at which the column at rest costs least, with the
    distillate two analyser sigmas inside its spec. The score charges boilup and the
    light key the bottoms carry away (PHYSICS.md section 7), and with energy this
    cheap their balance lies inside the zone, the bottoms over-purified [D8, p. 2966]:
    the operating target is the price's, not the spec's. Never nearer the spec than
    rule 22's two sigmas."""
    yD = float(p.yD_spec) + K_BACKOFF * float(p.e_floor_top)
    edge = float(p.xB_spec) - K_BACKOFF * float(p.e_floor_bottom)
    best = minimize_scalar(
        lambda xB: _rest_cost(yD, float(xB)),
        bounds=(float(p.e_floor_bottom), edge),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(best.x)


def _bottoms_schedule(p, n: int = 9) -> tuple:
    """``(feeds, setpoints)``: the bottoms' rest point at each metered feed rate over
    the sourced step [D1, Fig. 18], on the design feed composition. The cheapest
    (``_bottoms_rest``) where its reflux and boilup leave the loops the flow box's
    5 %; less pure where they do not, since the flows fall with the purity; never
    nearer the spec than rule 22's two sigmas. The balances are homogeneous in the
    flows, so a feed rate scales the design feed's."""
    yD = float(p.yD_spec) + K_BACKOFF * float(p.e_floor_top)
    edge = float(p.xB_spec) - K_BACKOFF * float(p.e_floor_bottom)
    F0, zF0 = float(p.F), float(p.zF_nominal)
    best = _bottoms_rest(p)

    def short(xB, scale):
        L, V = _rest_flows(F0, zF0, yD, float(xB))
        return max(ROOM * scale * L - float(p.L_max), ROOM * scale * V - float(p.V_max))

    feeds = F0 * (1.0 + float(p.feed_step) * np.linspace(-1.0, 1.0, n))
    points = []
    for scale in feeds / F0:
        if short(best, scale) <= 0.0:
            points.append(best)
        elif short(edge, scale) >= 0.0:
            points.append(edge)
        else:
            points.append(float(brentq(short, best, edge, args=(scale,), xtol=1e-7)))
    return feeds, np.array(points)


def row() -> Row:
    spec = REGISTRY["distillation"]
    p = spec.make_test_params()
    L_min, V_min = float(p.L_min), float(p.V_min)
    per_L, per_V = 2 / float(p.L_max - p.L_min), 2 / float(p.V_max - p.V_min)
    F0, zF0 = float(p.F), float(p.zF_nominal)
    specs = float(p.yD_spec), float(p.xB_spec)
    feeds, bottoms = _bottoms_schedule(p)

    def scheduled(o, xp, values=bottoms):
        """``values`` at the metered feed (obs 7, FC 1 [D4, p. 5])."""
        return xp.interp(
            o[7], xp.asarray(feeds, xp.float32), xp.asarray(values, xp.float32)
        )

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

    # The build fills ``placed`` with the distillate loop's setpoint; the feedforward
    # reads it when its rung is traced, after the build has placed it.
    placed: dict = {}

    def held() -> float:
        return float(placed["yD on L"][1]) if "yD on L" in placed else DESIGN_YD

    # Feedforward from the metered feed (obs 7, FC 1 [D4, p. 5]): the flows the
    # tray balances need for the held products at the nominal feed composition,
    # in their ratio to the feed. That is the steady state of a feed-rate change at
    # fixed zF (the balances are homogeneous in the flows), and it closes the
    # overall balance D = F (zF - xB) / (yD - xB) with the unmeasured zF at nominal.
    def balance(o, xp):
        # Traced inside the rung's jit: the solves run once, eagerly, on constants.
        with jax.ensure_compile_time_eval():
            unit = [_rest_flows(F0, zF0, held(), float(xB)) for xB in bottoms]
            L, V = (np.array(unit) * (feeds / F0)[:, None]).T
        return xp.clip(raw(scheduled(o, xp, L), scheduled(o, xp, V), xp), -1.0, 1.0)

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
        scheduled,
        "x",
        du_V,
        window,
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
