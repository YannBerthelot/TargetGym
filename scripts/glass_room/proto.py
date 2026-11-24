"""Glass room study, prototype: what a temperature move costs.

Written against the plant at 4be0ab9, before these defects went into it; on the plant
as it is now, use ``plant_runs.py``. Kept as the record of the prototype's runs.

Two defect sources from REALITY.md section 10, carried with the glass as the fining share
is, on top of the registered plant, which is not touched:

- **contact**: the tank blocks give off bubbles, knots and stones when their temperature
  rises [TR4; S7]. Each cell's block follows its glass over ``TAU_BLOCK`` (ours: the 2 h
  dwell of Dunkl's cycles), and releases in proportion to its rise and to the wall and
  bottom area the cell touches. Nothing removes these on the way out (knots and stones do
  not fine out; ours for the bubbles).
- **reboil**: the fining reaction still running in glass past the last port [S45; TR3],
  by the plant's own fining kinetics.

Rates have no source. They enter only when a cost is read, as two numbers swept over
decades: ``q10``, the share of the tank's glass a 10 K rise of every block would reject,
and ``p1``, the chance that glass doing 1 % of its fining past the last port is rejected.
"""

import os
import pathlib
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

P, E = L.P, L.E
H = L.HOUR
DT = L.DT
TAU_BLOCK = 2.0 * 3600.0
#: A check of the reading: only blocks above this temperature release (K). Dunkl's tests
#: ran at 1450 to 1550 C, and most of the tank's contact area is its floor, far colder.
#: 0 lets every block release; set GLASS_PROTO_HOT=1 for 1400 C.
HOT_ONLY = 1673.15 if os.environ.get("GLASS_PROTO_HOT") else 0.0
BASE = L.scenario_params("normal")
PULL0 = float(BASE.pull)
A = P.layout(BASE.grid)
ND, NC, NM = int(BASE.grid.n_depth), int(A.nc), int(A.nm)
RHO = float(BASE.plant.rho_glass)
DZ = A.dz[:, None]
PLAN = A.dx * A.wc
CELL_MASS = RHO * DZ * PLAN  # kg, (layers, columns)
# Block area a cell touches, over its mass: two side walls for every layer, the floor
# for the bottom layer. m^2 per kg.
_bottom = jnp.zeros((ND, 1)).at[-1].set(1.0)
CONTACT = 2.0 / (RHO * A.wc)[None, :] + _bottom / (RHO * DZ)
CONTACT_PER_KG = float(
    jnp.sum(CONTACT * CELL_MASS) / jnp.sum(CELL_MASS)
)  # m^2/kg, whole tank
PAST_PORTS = jnp.concatenate([jnp.zeros(NM), jnp.ones(NC - NM)])[None, :]
ROW = L.ladder_row()
G_TABLE, CROWN0 = 185.0, 1820.75  # K per unit pull; K at the design pull (probes 3, 7)


class Aux(NamedTuple):
    block: jax.Array  # the blocks' temperature, K, (layers, columns)
    D: jax.Array  # (2, layers, columns): contact, in K m^2 per kg; reboil, a share


def transport(D, books, pp):
    """One step of a passive scalar on the plant's own flow, as ``plant_step`` carries
    the fining share: flux limited along the tank, implicit in depth, fresh melt
    arriving with none."""
    u, melt = books["u"], books["melt"]
    source = jnp.concatenate([melt * A.dx[:NM], jnp.zeros(NC - NM)])
    Q = jnp.cumsum(source * A.wc) / pp.rho_glass
    out = (Q[-1] * A.dz / A.depth)[:, None]
    F = jnp.concatenate([jnp.zeros((ND, 1)), u * DZ * A.wf[1:-1], out], 1)
    W = jnp.concatenate(
        [(source / A.dx / pp.rho_glass)[None], (F[:, :-1] - F[:, 1:]) / PLAN], 0
    )
    W = jnp.cumsum(W, 0)
    Wp, Wm = jnp.maximum(W, 0.0), jnp.maximum(-W, 0.0)
    zero, pad = jnp.zeros((ND, 1)), jnp.zeros((1, NC))
    carry = jnp.concatenate(
        [zero, F[:, 1:-1] * P.carried(D, u, DT, A.dx), out * D[:, -1:]], 1
    )
    aa = jnp.concatenate([pad, -Wp[1:-1]], 0)
    cc = jnp.concatenate([-Wm[1:-1], pad], 0)
    bb = DZ / DT + Wm[:-1] + Wp[1:]
    rhs = DZ / DT * D + (carry[:, :-1] - carry[:, 1:]) / PLAN
    left = jnp.sum(out[:, 0] * D[:, -1]) * pp.rho_glass * DT  # what leaves this step
    return P.thomas(aa, bb, cc, rhs), left


def step(key, state, aux, action, params):
    """The env's own step, then the tracers. ``info`` holds what a cost needs."""
    obs, new, reward, _, _ = L.ENV.step_env(key, state, action, params)
    walk = new.walk
    pp = params.plant.replace(
        delta_t=params.delta_t, sulfate_batch=params.sulfate * (1.0 + walk[4])
    )
    lhv = E.gas_factor(walk, new.time, state, params)
    again, books = P.plant_step(
        state.plant, new.fuel, new.pull, new.cullet, params.thickness * (1.0 + walk[3]),
        state.cooling, params.grid, pp, lhv=lhv,
    )  # fmt: skip
    T = new.plant.T
    block = aux.block + (T - aux.block) * (1.0 - jnp.exp(-DT / TAU_BLOCK))
    rise = jnp.maximum(block - aux.block, 0.0)
    contact = rise * CONTACT * (block > HOT_ONLY)
    X = new.plant.fining
    released = (1.0 - X) * (1.0 - jnp.exp(-DT * P.fining_done(T, pp) / pp.fining_time))
    reboil = released * PAST_PORTS
    carried, left = jax.vmap(lambda d: transport(d, books, pp))(aux.D)
    D = carried + jnp.stack([contact, reboil])
    read = P.sensors(new.plant, params.grid, params.plant)
    terms = E.compute_reward_terms(new, params)
    info = dict(
        fuel=terms["running"],
        cliff=terms["tracking"] > 0.0,
        pull=new.pull,
        D_out=jnp.sum(D[:, :, -1] * A.dz, -1) / jnp.sum(A.dz),
        X=read["fining"],
        hot=read["hot"],
        rise=jnp.sum(rise * CELL_MASS) / jnp.sum(CELL_MASS),  # K, mass mean, this step
        rise_end=jnp.sum((rise * CELL_MASS)[:, -4:]) / jnp.sum(CELL_MASS[:, -4:]),
        mismatch=jnp.max(jnp.abs(again.T - T)),
        made=jnp.sum(jnp.stack([contact, reboil]) * CELL_MASS, (1, 2)),
        left=left,
        held=jnp.sum(D * CELL_MASS, (1, 2)),
    )
    return obs, new, Aux(block, D), info


def controller(ch):
    """The ladder's cascade with its crown setpoint following the pull in channel ``ch``
    (8 charged, 9 planned), as a function of ``(sp, outer, rate, offset)``: the fining
    setpoint, a factor on the outer loop's gain, the fastest the table may move the
    crown setpoint, K per hour, and a constant added to it, K."""
    ko, tio, ao = L.OUTER["kc"], L.OUTER["ti"], L._alpha(L.OUTER["tf"])
    ki, tii, ai = L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"])

    def start(obs):
        return (
            obs[4],
            jnp.zeros(()),
            obs[0],
            jnp.zeros(()),
            CROWN0 + G_TABLE * (obs[ch] / PULL0 - 1.0),
        )

    def act(par, m, obs):
        sp, outer, rate, offset = par
        y, Io, f, Ii, ff = m
        y = y + ao * (obs[4] - y)
        e = sp - y
        Io2 = Io + e * DT
        trim = outer * ko * (e + Io2 / tio)
        target = CROWN0 + G_TABLE * (obs[ch] / PULL0 - 1.0)
        lim = rate * DT / 3600.0
        ff = ff + jnp.clip(target - ff, -lim, lim)
        f = f + ai * (obs[0] - f)
        ei = ff + offset + trim - f
        Ii2 = Ii + ei * DT
        asked = ROW.balance(obs, jnp)[0] + ki * (ei + Ii2 / tii)
        u = jnp.clip(asked, -1.0, 1.0)
        moved = u != asked
        return jnp.reshape(u, (1,)), (
            y,
            jnp.where(moved, Io, Io2),
            f,
            jnp.where(moved, Ii, Ii2),
            ff,
        )

    return start, act


def settle(ch, par, p0, days=25):
    """The plant and the controller's memory settled at ``p0``, walks off."""
    quiet = p0.replace(walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
    start, act = controller(ch)
    obs, st = L.ENV.reset_env(jax.random.PRNGKey(0), quiet)

    def f(c, k):
        s, m, o = c
        a_, m = act(par, m, o)
        o1, s1, _, _, _ = L.ENV.step_env(k, s, a_, quiet)
        return (s1, m, o1), None

    (s, m, _), _ = jax.lax.scan(
        f,
        (st, start(obs), obs),
        jax.random.split(jax.random.PRNGKey(31), days * 24 * H),
    )
    return s, m


def episode(ch, par, p0, p, seed, n, settled=None):
    """``n`` steps of the scored params ``p`` on seed ``seed`` from the state settled
    at ``p0``: the per-step info."""
    _, act = controller(ch)
    s, m = settle(ch, par, p0) if settled is None else settled
    k_reset, keys = L.episode_keys(seed, n)
    _, st = L.ENV.reset_env(k_reset, p)
    st = st.replace(
        plant=s.plant,
        cooling=s.cooling,
        integral=s.integral,
        fuel=s.fuel,
        limit=s.limit,
    )
    aux = Aux(st.plant.T, jnp.zeros((2, ND, NC)))

    def f(c, k):
        s0, x, mem, o = c
        a_, mem = act(par, mem, o)
        o1, s1, x1, info = step(k, s0, x, a_, p)
        return (s1, x1, mem, o1), info

    return jax.lax.scan(f, (st, aux, m, L.ENV.get_obs(st, p)), keys)[1]


def d_contact(q10):
    """The contact tracer's density that rejects a plate, for ``q10``."""
    return 10.0 * CONTACT_PER_KG / -np.log1p(-q10)


def costs(info, q10, p1, start=0, price=600.0):
    """EUR per step from ``start`` on: fuel, glass lost to the fining and canal marks,
    glass lost to contact defects and to reboil, and the total, the lost shares
    combined and never over the whole pull."""
    t = {k: np.asarray(v)[start:] for k, v in info.items()}
    worth = t["pull"] * DT / 1000.0 * price
    l1 = (
        1.0 - np.exp(-t["D_out"][:, 0] / d_contact(q10))
        if q10 > 0
        else np.zeros_like(worth)
    )
    l2 = (
        1.0 - np.exp(t["D_out"][:, 1] * np.log1p(-p1) / 0.01)
        if p1 > 0
        else np.zeros_like(worth)
    )
    lost = np.maximum(t["cliff"].astype(float), 1.0 - (1.0 - l1) * (1.0 - l2))
    return dict(
        fuel=float(t["fuel"].mean()),
        marks=float((worth * t["cliff"]).mean()),
        contact=float((worth * l1).mean()),
        reboil=float((worth * l2).mean()),
        total=float((t["fuel"] + worth * lost).mean()),
        cliff_h=float(t["cliff"].sum() / H),
        contact_share=float(l1.mean()),
        reboil_share=float(l2.mean()),
    )
