"""Attempt 2: the glass furnace from its equations, unregistered.

Every term is a conservation law or a sourced relation. Citations are in
``PHYSICS.md`` under "Physical plant". Temperatures are in K here unless a name says
otherwise; the shipped clocks and regenerator constants are reused through
``GlassFurnaceParams``.

Layers count down from the surface, columns along the tank from the charger.
"""

import functools
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.glass_furnace.env import (
    KELVIN_OFFSET,
    N_REGEN_NODES,
    GlassFurnaceParams,
    reversal_phase,
)

SIGMA = 5.670374e-8
C2 = 14387.77  # um K, Planck's second constant

# ---- Radiation properties of the melt ------------------------------------------


def band_fraction(lam_T):
    """Share of blackbody emission below ``lam_T`` (um K), Planck's series."""
    x = C2 / np.asarray(lam_T, dtype=float)
    n = np.arange(1, 40)[(None,) * x.ndim]
    x_ = x[..., None]
    terms = np.exp(-n * x_) / n * (x_**3 + 3 * x_**2 / n + 6 * x_ / n**2 + 6 / n**3)
    return 15.0 / np.pi**4 * terms.sum(-1)


def _rosseland_share(lam_T):
    """Share of dB/dT below ``lam_T``."""
    x = np.geomspace(C2 / lam_T, 60.0, 4000)
    full = np.geomspace(1e-3, 60.0, 8000)
    w = lambda v: v**4 * np.exp(v) / np.expm1(v) ** 2  # noqa: E731
    return np.trapezoid(w(x), x) / np.trapezoid(w(full), full)


def radiative_conductivity(T, kappa1, kappa2, n):
    """Liu 2016 eqs 6 and 7: bands 0 to 2.8 and 2.8 to 5 um, opaque past 5 um."""
    below_1 = _rosseland_share(2.8 * T)
    below_2 = _rosseland_share(5.0 * T)
    over_kappa = below_1 / kappa1 + (below_2 - below_1) / kappa2
    return 16.0 * n**2 * SIGMA * T**3 * over_kappa / 3.0


def vft_viscosity(T):
    """DGG X, Pa s, ``T`` in K."""
    C = T - KELVIN_OFFSET
    return 0.1 * 10.0 ** (-1.56151 + 4289.18 / (C - 250.737))


#: Clear float glass, 0.084 wt % iron oxide, Liu 2016 Table III; n ours, about 1.5.
KAPPA1, KAPPA2, N_GLASS = 42.8, 459.2, 1.5
_T0, _DT, _NT = 1000.0, 5.0, 281  # the tables' grid, K


def _table(f):
    return jnp.asarray(np.array([f(_T0 + i * _DT) for i in range(_NT)]))


_K_TABLE = _table(
    lambda T: 1.31 + 5.90e-4 * (T - KELVIN_OFFSET)
    + radiative_conductivity(T, KAPPA1, KAPPA2, N_GLASS)
)  # fmt: skip
_LOG_MU_TABLE = _table(lambda T: np.log(float(vft_viscosity(T))))


def _lookup(table, T):
    s = jnp.clip((T - _T0) / _DT, 0.0, _NT - 1.001)
    i = s.astype(jnp.int32)
    a = table[i]
    return a + (s - i) * (table[i + 1] - a)


def melt_conductivity(T):
    """k = kc + kr, Liu 2016 eqs 5 to 7, W/(m K)."""
    return _lookup(_K_TABLE, T)


def viscosity(T):
    return jnp.exp(_lookup(_LOG_MU_TABLE, T))


#: Pa s, about 1700 C: past every trip. A numerical guard on the explicit current, so
#: a run-away stays finite until it trips.
MU_GUARD = 3.0


# ---- Zone method ---------------------------------------------------------------


def _surface_patches(n_bays, length, width, height, walls, n=6):
    """(points, normals, weights, node) of each surface sample. Crown nodes first."""
    d = length / n_bays
    pts, nor, wts, node = [], [], [], []

    def rect(origin, u, v, normal, index):
        a = (np.arange(n) + 0.5) / n
        s, t = np.meshgrid(a, a, indexing="ij")
        p = np.asarray(origin) + s[..., None] * np.asarray(u) + t[..., None] * np.asarray(v)
        pts.append(p.reshape(-1, 3))
        nor.append(np.tile(normal, (n * n, 1)))
        wts.append(np.full(n * n, np.linalg.norm(np.cross(u, v)) / n**2))
        node.append(np.full(n * n, index))

    for i in range(n_bays):
        x0 = i * d
        rect((x0, 0, height), (d, 0, 0), (0, width, 0), (0, 0, -1.0), i)
        rect((x0, 0, 0), (d, 0, 0), (0, width, 0), (0, 0, 1.0), n_bays + i)
        if walls:
            rect((x0, 0, 0), (d, 0, 0), (0, 0, height), (0, 1.0, 0), i)
            rect((x0, width, 0), (d, 0, 0), (0, 0, height), (0, -1.0, 0), i)
    if walls:
        rect((0, 0, 0), (0, width, 0), (0, 0, height), (1.0, 0, 0), 0)
        rect((length, 0, 0), (0, width, 0), (0, 0, height), (-1.0, 0, 0), n_bays - 1)
    return tuple(np.concatenate(v) for v in (pts, nor, wts, node))


def _gas_points(n_bays, length, width, height, n=(5, 8, 5)):
    d = length / n_bays
    grids = [(np.arange(k) + 0.5) / k for k in n]
    a, b, c = np.meshgrid(*grids, indexing="ij")
    pts, node = [], []
    for i in range(n_bays):
        pts.append(np.stack([(i + a) * d, b * width, c * height], -1).reshape(-1, 3))
        node.append(np.full(a.size, i))
    return np.concatenate(pts), np.concatenate(node), d * width * height / a.size


def direct_areas(n_bays, length, width, height, kappa, walls=True):
    """Feischl eqs 8 to 11 by midpoint rule, then scaled to close exactly: each node
    leaves as much as it emits. ``walls`` puts side and end walls in the crown nodes."""
    P, N, W, node = _surface_patches(n_bays, length, width, height, walls)
    r = P[:, None] - P[None]
    r2 = (r**2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.sqrt(r2)
        cos_i = -(r * N[:, None]).sum(-1) / dist
        cos_j = (r * N[None]).sum(-1) / dist
        kernel = np.nan_to_num(
            np.clip(cos_i, 0, None) * np.clip(cos_j, 0, None) * np.exp(-kappa * dist)
            / (np.pi * r2)
        )  # fmt: skip
    m = 2 * n_bays
    ss = np.zeros((m, m))
    np.add.at(ss, (node[:, None], node[None]), kernel * W[:, None] * W[None])
    area = np.bincount(node, W, minlength=m)
    out = {"ss": 0.5 * (ss + ss.T), "area": area, "sg": np.zeros((m, n_bays))}
    out["gg"] = np.zeros((n_bays, n_bays))
    out["emission"] = np.zeros(n_bays)
    if kappa > 0:
        G, g_node, dV = _gas_points(n_bays, length, width, height)
        r = G[None] - P[:, None]
        r2 = (r**2).sum(-1)
        dist = np.sqrt(r2)
        cos_s = np.clip((r * N[:, None]).sum(-1) / dist, 0, None)
        sg_k = kappa * cos_s * np.exp(-kappa * dist) / (np.pi * r2)
        sg = np.zeros((m, n_bays))
        np.add.at(sg, (node[:, None], g_node[None]), sg_k * W[:, None] * dV)
        out["sg"] = sg
        r = G[:, None] - G[None]
        r2 = (r**2).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gg_k = kappa**2 * np.exp(-kappa * np.sqrt(r2)) / (np.pi * r2)
        gg_k[~np.isfinite(gg_k)] = 0.0
        gg = np.zeros((n_bays, n_bays))
        np.add.at(gg, (g_node[:, None], g_node[None]), gg_k * dV**2)
        emission = 4 * kappa * dV * np.bincount(g_node, minlength=n_bays)
        np.fill_diagonal(gg, 0.0)
        np.fill_diagonal(gg, emission - gg.sum(1) - sg.sum(0))
        out["gg"], out["emission"] = 0.5 * (gg + gg.T), emission
    if walls:  # an open pair of plates does not close, by design
        zero = out["emission"] == 0
        Z = np.block([[out["ss"], out["sg"]], [out["sg"].T, out["gg"]]])
        target = np.concatenate([area, out["emission"]])
        keep = np.concatenate([np.zeros(m, bool), zero])
        Z, tgt = Z[~keep][:, ~keep], target[~keep]
        for _ in range(200):
            s = np.sqrt(tgt / Z.sum(1))
            Z = s[:, None] * Z * s[None]
        out["ss"], gs = Z[:m, :m], Z[:m, m:]
        if kappa > 0:
            out["sg"], out["gg"] = gs, Z[m:, m:]
    return out


def exchange(n_bays, length, width, height, kappa, eps):
    """Linear maps from emissive powers: incident flux ``H = Hs E + Hg E_g`` and the
    gas's net emission ``Ugg E_g - Ugs E``, reflections included (Feischl eqs 5 to 7).
    ``eps`` is (crown, load)."""
    ex = direct_areas(n_bays, length, width, height, kappa)
    e = np.concatenate([np.full(n_bays, eps[0]), np.full(n_bays, eps[1])])
    A, rho = ex["area"], np.diag(1.0 - e)
    D, G, gg = ex["ss"], ex["sg"], ex["gg"]
    inv = np.linalg.inv(np.diag(A) - rho @ D)
    Pj, Qj = inv @ np.diag(A * e), inv @ rho @ G
    return {
        "Hs": (D @ Pj) / A[:, None],
        "Hg": (D @ Qj + G) / A[:, None],
        "Ugg": np.diag(ex["emission"]) - gg - G.T @ Qj,
        "Ugs": G.T @ Pj,
        "area": A,
    }


# ---- Flow, foam, blanket -------------------------------------------------------


def column_flow(grad, mu, dz, Q, cover, v_top, g_rho_beta):
    """Long-tank momentum at each face: hydrostatic in depth, d/dz (mu du/dz) = dp/dx,
    floor no-slip, top free, or following the blanket at ``v_top`` where ``cover`` is 1.
    The pressure gradient is whatever carries the net flux ``Q`` (m^2/s). All shapes
    (layers, faces); returns the velocity at layer centres."""
    dz_ = dz[:, None]
    head = jnp.cumsum(grad * dz_, 0) - 0.5 * grad * dz_
    b = -g_rho_beta * head
    S_b = jnp.cumsum(b * dz_, 0) - 0.5 * b * dz_
    z = jnp.cumsum(dz) - 0.5 * dz

    def solution(shear):
        I = shear / mu * dz_  # noqa: E741
        rev = jnp.cumsum(I[::-1], 0)[::-1]
        u = -(rev - 0.5 * I)
        return u, jnp.sum(u * dz_, 0), -rev[0]

    u_b, F_b, top_b = solution(S_b)
    u_g, F_g, top_g = solution(z[:, None] * jnp.ones_like(mu))
    u_t, F_t, top_t = solution(jnp.ones_like(mu))
    det = F_g * top_t - F_t * top_g
    tau = cover * (F_g * (v_top - top_b) - top_g * (Q - F_b)) / det
    G = (Q - F_b - tau * F_t) / F_g
    return u_b + G * u_g + tau * u_t


def foam_thickness(j, mu, sigma, r0, rho, g):
    """Pilon, Fedorov, Viskanta 2001 eq 19, m, with no onset velocity (none found)."""
    return (
        2905.0 * sigma * (mu * jnp.maximum(j, 0.0)) ** 0.8
        / (r0**2.6 * (rho * g) ** 1.8)
    )  # fmt: skip


#: Foam's extinction, 1/m: two-flux transmittance 1 / (1 + 0.75 beta L) = 0.2 at 2 cm,
#: the bound in Fedorov and Pilon 2002. Its gas fraction: 0.82, the same paper's chapter.
FOAM_BETA = 4.0 / (0.75 * 0.02)
FOAM_POROSITY = 0.82


def foam_conductivity(T):
    return 16.0 * SIGMA * T**3 / (3.0 * FOAM_BETA) + (2.0 / 3.0) * (
        1.0 - FOAM_POROSITY
    ) * (1.31 + 5.90e-4 * (T - KELVIN_OFFSET))


def bare_flux(H, T, eps, h=0.0, T_gas=0.0):
    return eps * (H - SIGMA * T**4) + h * (T_gas - T)


def foam_flux(H, T, thickness, eps, h=0.0, T_gas=0.0):
    """What passes a foam into glass at ``T``, and the foam's top temperature."""
    top = T + 200.0
    thick = jnp.maximum(thickness, 1e-4)
    for _ in range(4):
        k = foam_conductivity(0.5 * (top + T)) / thick
        f = eps * (H - SIGMA * top**4) + h * (T_gas - top) - k * (top - T)
        top = top + f / (4.0 * eps * SIGMA * top**3 + h + k)
    return foam_conductivity(0.5 * (top + T)) / thick * (top - T), top


def cover(m):
    """Share of a cell under blanket: a linear wedge's tip, from the mass next to it."""
    behind = jnp.concatenate([m[:1], m[:-1]])
    return jnp.cumprod(jnp.clip(2.0 * m / jnp.maximum(behind, 1e-9), 0.0, 1.0))


def advance_blanket(m, c, melt, m_in, c_in, v, dx, dt):
    """Upwind transport at the charging speed, UV1986; ``melt`` kg/(m^2 s) leaves."""
    behind_m = jnp.concatenate([jnp.atleast_1d(m_in), m[:-1]])
    behind_c = jnp.concatenate([jnp.atleast_1d(c_in), c[:-1]])
    flow = v * dt / dx
    return (
        jnp.maximum(m + flow * (behind_m - m) - dt * melt, 0.0),
        c + flow * (behind_c - c),
    )


def thomas(a, b, c, d):
    """Tridiagonal solve along axis 0: ``a`` below, ``b`` on, ``c`` above the diagonal."""
    n = b.shape[0]
    cp, dp = [c[0] / b[0]], [d[0] / b[0]]
    for i in range(1, n):
        w = b[i] - a[i] * cp[-1]
        cp.append(c[i] / w)
        dp.append((d[i] - a[i] * dp[-1]) / w)
    x = [dp[-1]]
    for i in range(n - 2, -1, -1):
        x.append(dp[i] - cp[i] * x[-1])
    return jnp.stack(x[::-1])


# ---- Plant ---------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    n_bays: int = 6  # six ports, W2000b
    cols_per_bay: int = 2
    n_refiner: int = 4
    n_depth: int = 10
    skin: float = 0.006  # m, the top layer
    n_roof: int = 8
    roof_skin: float = 0.003  # m, the hot-face cell


def layout(g):
    """Arrays of a grid, safe to build while tracing."""
    with jax.ensure_compile_time_eval():
        return _layout(g)


@functools.lru_cache(maxsize=8)
def _layout(g, length=21.5, refiner=20.5, width=11.0, height=2.4, depth=1.2, crown=0.45):

    def graded(total, first, n):
        lo, hi = 1.0001, 4.0
        for _ in range(60):
            r = 0.5 * (lo + hi)
            lo, hi = (r, hi) if first * (r**n - 1) / (r - 1) < total else (lo, r)
        return first * r ** np.arange(n)

    dz = graded(depth, g.skin, g.n_depth)
    dr = graded(crown, g.roof_skin, g.n_roof)
    nm = g.n_bays * g.cols_per_bay
    dx = np.concatenate([np.full(nm, length / nm), np.full(g.n_refiner, refiner / g.n_refiner)])
    z = np.concatenate([[0.0], np.cumsum(dz)])
    F1 = float(band_fraction(2.8 * 1900.0))  # the crown and flame's own spectrum
    F2 = float(band_fraction(5.0 * 1900.0)) - F1
    F3 = 1.0 - F1 - F2
    e1, e2 = np.exp(-KAPPA1 * z), np.exp(-KAPPA2 * z)
    D = F1 * (e1[:-1] - e1[1:]) + F2 * (e2[:-1] - e2[1:])
    D[0] += F3
    D = D / D.sum()
    ex = exchange(g.n_bays, length, width, height, 0.08, (0.8, 0.8))
    return SimpleNamespace(
        nc=nm + g.n_refiner, nm=nm, dz=jnp.asarray(dz), dx=jnp.asarray(dx),
        dzc=jnp.asarray(0.5 * (dz[1:] + dz[:-1])),
        dxc=jnp.asarray(0.5 * (dx[1:] + dx[:-1])),
        dr=jnp.asarray(dr), drc=jnp.asarray(0.5 * (dr[1:] + dr[:-1])),
        width=width, depth=depth, deposit=jnp.asarray(D),
        bay=jnp.repeat(jnp.arange(g.n_bays), g.cols_per_bay),
        Hs=jnp.asarray(ex["Hs"]), Hg=jnp.asarray(ex["Hg"]),
        Ugg=jnp.asarray(ex["Ugg"]), Ugs=jnp.asarray(ex["Ugs"]),
        A_crown=jnp.asarray(ex["area"][: g.n_bays]),
        A_load=jnp.asarray(ex["area"][g.n_bays :]),
        A_roof=length / g.n_bays * width,
    )  # fmt: skip


@struct.dataclass
class PhysicalParams(GlassFurnaceParams):
    rho_glass: float = 2319.0  # kg/m^3 at 1400 C
    beta_glass: float = 5.95e-5  # 1/K, cubical
    c_glass: float = 1300.0  # J/(kg K), the shipped form at 1200 C
    g_accel: float = 9.81
    eps_surface: float = 0.8  # ours: no surface emissivity was found
    h_gas: float = 50.0  # W/(m^2 K), VW1983 "typical"
    U_bottom: float = 1.0  # W/(m^2 K), of those judged usable
    U_side: float = 2.9  # W/(m^2 K), 4 kW/m^2 at 1400 C, B2011
    k_brick: float = 2.48  # W/(m K)
    rho_brick: float = 1870.0  # kg/m^3
    cp_brick: float = 1245.0  # J/(kg K), 3R per atom on SiO2
    U_crown: float = 1.70  # W/(m^2 K), 2 kW/m^2 at 1200 C
    U_wall: float = 3.6  # W/(m^2 K), 5 kW/m^2 at 1400 C, W2000b
    U_refiner: float = 6.8  # W/(m^2 K), uninsulated: four times U_crown, Ford float
    U_regen: float = 1.4  # W/(m^2 K), 1726 W/m^2 at 1250 C, ICG slides 12 to 14
    T_melt_K: float = 1450.0  # K, clear melt, Pilon II
    rho_batch: float = 1400.0  # kg/m^3
    H_melt: float = 2.2e6  # J/kg, 320 K to 1450 K, Pilon II
    cullet_gain: float = 0.352  # 2.5 % less fuel per 10 points
    sulfate_batch: float = 0.01  # Na2SO4 per kg batch, Laimbock
    so3_glass: float = 0.0027  # float, Rudolph
    glass_per_batch: float = 0.834  # 297 of 356 t/d, Pilon II
    T_fine_low: float = 1693.15  # K, 1420 C, Beerkens review
    T_fine_high: float = 1773.15  # K, 1500 C
    sigma_glass: float = 0.3  # N/m, Table 1 of JCIS2001
    bubble_radius: float = 8.3e-3  # m, top of the lab range in Ch16
    max_steps_in_episode: int = 8640


@struct.dataclass
class Fields:
    T: jnp.ndarray  # melt, (layers, columns), K
    T_top: jnp.ndarray  # radiating top of each melter column, K
    T_roof: jnp.ndarray  # (cells, bays), K
    T_gas: jnp.ndarray  # bays, K
    T_rA: jnp.ndarray  # regenerators, C, as shipped
    T_rB: jnp.ndarray
    m: jnp.ndarray  # blanket, kg/m^2, melter columns
    c: jnp.ndarray  # cullet ratio in the blanket
    q_under: jnp.ndarray  # W/m^2 already taken from the melt into the blanket


def initial_fields(g, p):
    a = layout(g)
    z = np.cumsum(np.asarray(a.dz)) - 0.5 * np.asarray(a.dz)
    x = np.cumsum(np.asarray(a.dx)) - 0.5 * np.asarray(a.dx)
    surface = np.interp(x, [0, 5, 14, 21.5, 42], [1500, 1700, 1820, 1750, 1500])
    deep = np.interp(x, [0, 21.5, 42], [1600, 1650, 1450])
    w = np.clip(z[:, None] / 0.4, 0, 1)
    T = (1 - w) * surface + w * deep
    dr = np.asarray(a.dr)
    r = np.cumsum(dr) - 0.5 * dr
    roof = 1850.0 + (1480.0 - 1850.0) * r / 0.45
    xm = np.asarray(jnp.cumsum(a.dx)[: a.nm])
    m = np.clip(280.0 * (1.0 - xm / 6.0), 0.0, None)
    return Fields(
        T=jnp.asarray(T), T_top=jnp.asarray(T[0, : a.nm]),
        T_roof=jnp.tile(jnp.asarray(roof)[:, None], (1, g.n_bays)),
        T_gas=jnp.full((g.n_bays,), 2000.0),
        T_rA=jnp.linspace(1300.0, 500.0, N_REGEN_NODES),
        T_rB=jnp.linspace(1300.0, 500.0, N_REGEN_NODES),
        m=jnp.asarray(m), c=jnp.full((a.nm,), 0.25), q_under=jnp.zeros((a.nc,)),
    )  # fmt: skip


def _stack(nodes, T_in, m_air, m_gas, is_air, p):
    """A checker stack, as ``_regenerator_flows``, written out: (Q, air out, flue out)."""
    e = p.eps_regen_node
    T_e, T_a = T_in, jnp.asarray(p.T_ambient, nodes.dtype)
    Qe, Qa = [], []
    for i in range(N_REGEN_NODES):
        out = T_e - e * (T_e - nodes[i])
        Qe.append(m_gas * p.c_p_gas * (T_e - out))
        T_e = out
    for i in range(N_REGEN_NODES - 1, -1, -1):
        out = T_a + e * (nodes[i] - T_a)
        Qa.append(-m_air * p.c_p_air * (out - T_a))
        T_a = out
    return jnp.where(is_air, jnp.stack(Qa[::-1]), jnp.stack(Qe)), T_a, T_e


def _gas_balance(s, E_load, T_load, fuel_heat, m_gas, H_air, a, g, p):
    """Each bay's gas, algebraically: firing in, flue out, radiation, convection."""
    share = 1.0 / g.n_bays
    E = jnp.concatenate([SIGMA * s.T_roof[0] ** 4, E_load])
    drive = a.Ugs @ E
    mc = share * m_gas * p.c_p_gas
    inflow = share * (fuel_heat + H_air) + mc * (p.T_ambient + KELVIN_OFFSET)
    ha_c, ha_l = p.h_gas * a.A_crown, p.h_gas * a.A_load
    sink = ha_c * s.T_roof[0] + ha_l * T_load
    T = s.T_gas
    diag = jnp.diagonal(a.Ugg)
    for _ in range(5):
        Eg = SIGMA * T**4
        f = inflow + sink + drive - a.Ugg @ Eg - (mc + ha_c + ha_l) * T
        T = T + f / (4.0 * diag * SIGMA * T**3 + mc + ha_c + ha_l)
    return T, E


def _melt_column_fluxes(s, H, T_g, f, c_mean, charge_melt, a, g, p):
    """Surface fluxes of the melter's columns: (into melt, into blanket, top, foam)."""
    nm = a.nm
    Ts = s.T[0, :nm]
    w = jnp.clip((Ts - p.T_fine_low) / (p.T_fine_high - p.T_fine_low), 0.0, 1.0) * (1 - f)
    dA = a.dx[:nm] * a.width
    mol = (
        (p.sulfate_batch * 80.06 / 142.04 / p.glass_per_batch - p.so3_glass)
        / 0.08006 * 1.5
    )  # fmt: skip
    gas = charge_melt * (1.0 - c_mean) * mol * 8.314 * p.T_fine_high / 101325.0
    j = gas * w / jnp.maximum(jnp.sum(w * dA), 1.0)
    foam = foam_thickness(
        j, viscosity(Ts), p.sigma_glass, p.bubble_radius, p.rho_glass, p.g_accel
    )
    q_foam, T_foam = foam_flux(H, Ts, foam, p.eps_surface, p.h_gas, T_g)
    q_bare = bare_flux(H, Ts, p.eps_surface, p.h_gas, T_g)
    has = foam > 1e-3
    q = jnp.where(has, q_foam, q_bare)
    top = jnp.where(has, T_foam, Ts)
    q_batch = bare_flux(H, p.T_melt_K, p.eps_surface, p.h_gas, T_g)
    return q, q_batch, top, foam


def _refiner_loss(T, p):
    """Surface to its own roof by radiation, the roof to outside: W/m^2, positive out."""
    e = 1.0 / (2.0 / p.eps_surface - 1.0)
    Ta = p.T_ambient + KELVIN_OFFSET
    R = T - 300.0
    for _ in range(4):
        f = e * SIGMA * (T**4 - R**4) - p.U_refiner * (R - Ta)
        R = R + f / (4.0 * e * SIGMA * R**3 + p.U_refiner)
    return p.U_refiner * (R - Ta)


def plant_step(s, fuel, charge, cullet, thickness, time, g, p, lhv=0.0):
    """One step of the physics: ``(fields, books)``. ``fuel`` kg/s actually fired,
    ``charge`` kg/s, ``cullet`` and ``thickness`` as the charger lays them."""
    a = layout(g)
    dt = p.delta_t
    Ta = p.T_ambient + KELVIN_OFFSET
    rc = p.rho_glass * p.c_glass
    A_is_air = reversal_phase(time, p) < 0.5
    m_air = p.AFR * (1.0 + p.excess_air) * fuel
    m_gas = fuel + m_air
    fuel_heat = fuel * p.LHV * (1.0 + lhv)

    # ---- Combustion space
    _, air_A, _ = _stack(s.T_rA, 0.0, m_air, m_gas, True, p)
    _, air_B, _ = _stack(s.T_rB, 0.0, m_air, m_gas, True, p)
    T_air = jnp.where(A_is_air, air_A, air_B)
    H_air = m_air * p.c_p_air * (T_air - p.T_ambient)
    f = cover(s.m)
    bay = a.bay
    weights = a.dx[: a.nm] * a.width / a.A_load[bay]
    surf = f * p.T_melt_K + (1.0 - f) * s.T_top
    emit = f * SIGMA * p.T_melt_K**4 + (1.0 - f) * SIGMA * s.T_top**4
    T_load = jax.ops.segment_sum(weights * surf, bay, g.n_bays)
    E_load = jax.ops.segment_sum(weights * emit, bay, g.n_bays)
    T_g, E = _gas_balance(s, E_load, T_load, fuel_heat, m_gas, H_air, a, g, p)
    H = a.Hs @ E + a.Hg @ (SIGMA * T_g**4)
    H_crown, H_load = H[: g.n_bays], H[g.n_bays :]

    # ---- Surfaces and the blanket
    dA = a.dx[: a.nm] * a.width
    H_melt = p.H_melt * (1.0 - p.cullet_gain * s.c)
    q, q_batch, top, foam = _melt_column_fluxes(
        s, H_load[bay], T_g[bay], f, jnp.mean(s.c[:2]), charge, a, g, p
    )
    taken = f * q_batch + s.q_under[: a.nm]
    melt = jnp.minimum(jnp.maximum(taken, 0.0) / H_melt, s.m / dt)
    spare = taken - melt * H_melt  # a tip with no mass left hands its heat back
    h0 = thickness
    v = charge / (p.rho_batch * h0 * a.width)
    m_new, c_new = advance_blanket(
        s.m, s.c, melt, charge / (v * a.width), cullet, v, a.dx[: a.nm], dt
    )
    in_blanket = jnp.sum(f * q_batch * dA) + jnp.sum(s.q_under[: a.nm] * dA)
    used = jnp.sum(melt * H_melt * dA)

    # ---- Melt: flow
    T = s.T
    Tf = 0.5 * (T[:, 1:] + T[:, :-1])
    grad = (T[:, 1:] - T[:, :-1]) / a.dxc
    source = jnp.concatenate([melt * a.dx[: a.nm], jnp.zeros(g.n_refiner)])
    Q = jnp.cumsum(source) / p.rho_glass
    cover_f = jnp.concatenate([f, jnp.zeros(g.n_refiner)])
    cover_face = 0.5 * (cover_f[1:] + cover_f[:-1])
    u = column_flow(
        grad, jnp.maximum(viscosity(Tf), MU_GUARD), a.dz, Q[:-1], cover_face, v,
        p.g_accel * p.rho_glass * p.beta_glass,
    )  # fmt: skip
    dz_ = a.dz[:, None]
    out = (Q[-1] * a.dz / a.depth)[:, None]
    F = jnp.concatenate([jnp.zeros((g.n_depth, 1)), u * dz_, out], 1)
    W = jnp.concatenate(
        [(source / a.dx / p.rho_glass)[None], (F[:, :-1] - F[:, 1:]) / a.dx], 0
    )
    W = jnp.cumsum(W, 0)
    Wp, Wm = jnp.maximum(W, 0.0), jnp.maximum(-W, 0.0)

    # ---- Melt: energy, explicit along the tank and implicit in depth
    Fp, Fm = jnp.maximum(F[:, 1:-1], 0.0), jnp.minimum(F[:, 1:-1], 0.0)
    carry = Fp * T[:, :-1] + Fm * T[:, 1:]
    zero = jnp.zeros((g.n_depth, 1))
    carry = jnp.concatenate([zero, carry, out * T[:, -1:]], 1)
    kf = melt_conductivity(Tf)
    along = kf * grad / rc * dz_
    along = jnp.concatenate([zero, along, zero], 1)
    R = (carry[:, :-1] - carry[:, 1:] + along[:, 1:] - along[:, :-1]) / a.dx
    R = R - 2.0 * p.U_side / a.width * (T - Ta) / rc * dz_
    R = R.at[-1].add(-p.U_bottom * (T[-1] - Ta) / rc)

    loss_ref = _refiner_loss(T[0, a.nm :], p)
    q_all = jnp.concatenate([(1.0 - f) * q + spare, -loss_ref])
    R = R + a.deposit[:, None] * q_all / rc
    Tt0 = T[0]
    ht = 4.0 * p.eps_surface * SIGMA * Tt0**3 + p.h_gas
    hl = a.deposit[0] * jnp.concatenate(
        [(1.0 - f) * jnp.where(foam > 1e-3, 0.0, ht[: a.nm]), jnp.zeros(g.n_refiner)]
    )
    k0 = melt_conductivity(Tt0)
    under = cover_f * k0 / (0.5 * a.dz[0])

    kz = melt_conductivity(0.5 * (T[1:] + T[:-1]))
    d = kz / (rc * a.dzc[:, None])
    lower = -(d + Wp[1:-1])
    upper = -(d + Wm[1:-1])
    pad = jnp.zeros((1, a.nc))
    aa = jnp.concatenate([pad, lower], 0)
    cc = jnp.concatenate([upper, pad], 0)
    bb = dz_ / dt + jnp.concatenate([pad, d], 0) + jnp.concatenate([d, pad], 0)
    bb = bb + Wm[:-1] + Wp[1:]
    bb = bb.at[0].add((hl + under) / rc)
    rhs = dz_ / dt * T + R
    rhs = rhs.at[0].add(W[0] * p.T_melt_K + (hl * Tt0 + under * p.T_melt_K) / rc)
    Tn = thomas(aa, bb, cc, rhs)
    q_under = under * (Tn[0] - p.T_melt_K)

    # ---- Crown: conduction, the face linearised
    ep = p.eps_surface
    Tr = s.T_roof
    U = (a.A_roof * p.U_crown + (a.A_crown - a.A_roof) * p.U_wall) / a.A_crown
    rb = p.rho_brick * p.cp_brick
    gd = jnp.tile((p.k_brick / (rb * a.drc))[:, None], (1, g.n_bays))
    gpad = jnp.zeros((1, g.n_bays))
    dr_ = a.dr[:, None]
    bb_r = dr_ / dt + jnp.concatenate([gpad, gd], 0) + jnp.concatenate([gd, gpad], 0)
    aa_r = jnp.concatenate([gpad, -gd], 0)
    cc_r = jnp.concatenate([-gd, gpad], 0)
    h_face = 4.0 * ep * SIGMA * Tr[0] ** 3 + p.h_gas
    q_face = ep * (H_crown - SIGMA * Tr[0] ** 4) + p.h_gas * (T_g - Tr[0])
    back_U = 1.0 / (1.0 / U + 0.5 * a.dr[-1] / p.k_brick)
    bb_r = bb_r.at[0].add(h_face / rb).at[-1].add(back_U / rb)
    rhs_r = dr_ / dt * Tr
    rhs_r = rhs_r.at[0].add((q_face + h_face * Tr[0]) / rb).at[-1].add(back_U * Ta / rb)
    Trn = thomas(aa_r, bb_r, cc_r, rhs_r)

    # ---- Regenerators
    T_flue_K = jnp.mean(T_g)
    T_flue = T_flue_K - KELVIN_OFFSET
    QA, _, st_A = _stack(s.T_rA, T_flue, m_air, m_gas, A_is_air, p)
    QB, _, st_B = _stack(s.T_rB, T_flue, m_air, m_gas, ~A_is_air, p)
    T_stack = jnp.where(A_is_air, st_B, st_A)
    UA = p.U_regen * p.A_regen / N_REGEN_NODES
    loss_A = UA * (s.T_rA - p.T_ambient)
    loss_B = UA * (s.T_rB - p.T_ambient)
    rate = dt / p.C_regen_node
    new = Fields(
        T=Tn, T_top=top, T_roof=Trn, T_gas=T_g,
        T_rA=s.T_rA + rate * (QA - loss_A), T_rB=s.T_rB + rate * (QB - loss_B),
        m=m_new, c=c_new, q_under=q_under,
    )  # fmt: skip

    w_area = a.dx * a.width
    books = dict(
        **{"in": fuel_heat + rc * jnp.sum(W[0] * w_area) * p.T_melt_K},
        out=m_gas * p.c_p_gas * (T_stack - p.T_ambient)
        + jnp.sum(loss_A + loss_B)
        + jnp.sum(a.A_crown * back_U * (Trn[-1] - Ta))
        + jnp.sum(2.0 * p.U_side * (T - Ta) * dz_ * a.dx)
        + jnp.sum(p.U_bottom * (T[-1] - Ta) * w_area)
        + jnp.sum(loss_ref * w_area[a.nm :])
        + rc * jnp.sum(out[:, 0] * T[:, -1]) * a.width
        + used,
        T_stack=T_stack, T_air=T_air, foam=foam, melt=melt, q=q_all, v=v,
        in_blanket=in_blanket, u=u, H_load=H_load, H_crown=H_crown,
        to_crown=jnp.sum(a.A_crown * q_face),
        to_load=jnp.sum(dA * ((1.0 - f) * q + f * q_batch)),
        from_gas=fuel_heat + H_air - m_gas * p.c_p_gas * (T_flue - p.T_ambient),
    )  # fmt: skip
    return new, books


def stored_energy(s, g, p):
    a = layout(g)
    rc = p.rho_glass * p.c_glass
    melt = rc * jnp.sum(s.T * a.dz[:, None] * a.dx * a.width)
    roof = p.rho_brick * p.cp_brick * jnp.sum(s.T_roof * a.dr[:, None] * a.A_crown)
    regen = p.C_regen_node * (jnp.sum(s.T_rA) + jnp.sum(s.T_rB))
    held = p.delta_t * jnp.sum(s.q_under * a.dx * a.width)
    return melt + roof + regen + held


def reading_cell(depth, g):
    """(cell, weight) to read the crown ``depth`` under the hot face by interpolation."""
    a = layout(g)
    r = np.cumsum(np.asarray(a.dr)) - 0.5 * np.asarray(a.dr)
    i = int(np.searchsorted(r, depth)) - 1
    return i, (depth - r[i]) / (r[i + 1] - r[i])


def sensors(s, depth, bay, g):
    """What instruments could see: the crown's hot face and its couple at ``depth`` in
    ``bay``, the glass bottom under that bay, the refiner's end surface. K, unfiltered."""
    i, w = reading_cell(depth, g)
    col = bay * g.cols_per_bay + g.cols_per_bay // 2
    return (
        s.T_roof[0, bay],
        (1.0 - w) * s.T_roof[i, bay] + w * s.T_roof[i + 1, bay],
        s.T[-1, col],
        s.T[0, -1],
    )
