"""The column against its plant record (REALITY.md, COLD_REVIEW.md tests 1 to 10).

Contract, from Skogestad's Column A [D1, D2] and the real splitter [D4], not
from the code (tolerances are the audit's, fixed before any run):
1. at L 2.706, V 3.206, zF 0.5 the column settles at yD 0.989 to 0.991, xB
   0.009 to 0.011, D within 0.5 % of 0.5 [D2];
2. a 0.1 % reflux step raises yD by 0.0015 to 0.0025 and xB by 0.0025 to
   0.0045 [D1, p. 15, Fig. 5]; its 194 min is test 6's mode;
3. the same boilup step moves both the other way, by Fig. 5's sizes at 500 min
   within 25 % (the audit's 0.8 to 1.25 of the reflux step reads "almost the
   same" more tightly than Fig. 5 draws it);
4. L and V up 0.27 together make both products purer, with a gain 30 to 300
   times smaller, and faster [D1, p. 15, Fig. 6] (the audit's 7 to 30 min is
   D1's case without liquid flow dynamics);
5. and 7. are in test_distillation_hydraulics.py, with tighter tolerances (the
   reflux reaches the reboiler after 39 lags of 0.063 min; the gains are eq. 31
   to 1e-4 and the RGA 35.94 to 0.01);
6. the three slowest modes are 193.9, 12.0 and 3.5 min within 5 % [D1, 4.4];
8. under D1's PI loops (eq. 82) with a 1 min delay, a 20 % feed step dips yD by
   3.0e-3 to 5.6e-3, lifts xB by 2.5e-3 to 4.5e-3, and both are back within 20
   to 60 min [D1, Fig. 18];
9. a zF step 0.5 to 0.6 moves xB 0.5e-3 to 2e-3 and xD less, each under half
   of 8 (D1: zF moves xD less than xB);
10. the analyser reports a change 3 to 30 min after it happens [D5].
Two more, from the job [D4, p. 5]: the product limits are zones, not trips,
and inside them the run on less boilup scores better.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from target_gym.pc_gym.distillation.env import (
    DistillationParams,
    compute_velocity,
    steady_liquid,
)
from target_gym.pc_gym.distillation.env_jax import _NOMINAL_PROFILE, DistillationColumn

L0, V0 = 2.70629, 3.20629  # colamod.m [D2]
#: [D1, p. 14]: "increased the reflux L by 0.0027 (about 0.1%)", the boilup "by
#: the same amount", and both together "by 0.27 (about 10%)".
DL = 0.0027
ENV = DistillationColumn()


def quiet(**kw):
    """No feed moves, no drift, no instrument noise."""
    return DistillationParams(sensor_noise_scale=0.0, feed_step=0.0, zF_step=0.0, **kw)


def raw(L, V, p):
    return jnp.array(
        [
            2.0 * (L - p.L_min) / (p.L_max - p.L_min) - 1.0,
            2.0 * (V - p.V_min) / (p.V_max - p.V_min) - 1.0,
        ]
    )


def start(p):
    _, s = ENV.reset_env(jax.random.PRNGKey(0), p)
    return s.replace(
        x=jnp.asarray(_NOMINAL_PROFILE), liquid=steady_liquid(L0, p), L=L0, V=V0
    )


def hold(p, s, L, V, minutes):
    """Products per minute at constant flows, from ``s``."""
    action = raw(L, V, p)

    def body(s, _):
        _, s, *_ = ENV.step_env(jax.random.PRNGKey(0), s, action, p)
        return s, jnp.stack([s.x[-1], s.x[0]])

    return jax.lax.scan(body, s, None, length=minutes)


def t63(trace):
    y = np.asarray(trace)
    frac = (y - y[0]) / (y[-1] - y[0])
    return float(np.argmax(frac >= 1.0 - np.exp(-1.0)))


@pytest.fixture(scope="module")
def rest():
    """Column A at rest at its nominal flows, in float64 so the 194 min mode
    reaches rest."""
    with enable_x64():
        p = quiet()
        s = jax.tree.map(
            lambda a: (
                jnp.asarray(a).astype(jnp.float64)
                if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating)
                else a
            ),
            start(p),
        )
        s, _ = hold(p, s, L0, V0, 3000)
    return p, s


def test_the_nominal_flows_give_the_published_products(rest):
    p, s = rest
    assert 0.989 <= float(s.x[-1]) <= 0.991
    assert 0.009 <= float(s.x[0]) <= 0.011
    assert abs((V0 - L0) - 0.5) <= 0.5 * 0.005


def _step(rest, L, V, minutes=2000):
    p, s = rest
    with enable_x64():
        _, tr = hold(p, s, L, V, minutes)
    tr = np.concatenate([[[float(s.x[-1]), float(s.x[0])]], np.asarray(tr)])
    return tr - tr[0], tr


def test_a_reflux_step_moves_both_products_as_published(rest):
    """[D1, p. 14]: xD 0.99 to about 0.992, xB 0.01 to about 0.0135. Its "time
    constant of about 194 minutes" is the linearised slowest mode, which D1
    section 4.4 computes and says corresponds to it: that is
    ``test_the_slowest_modes_are_the_published_ones``. The nonlinear response
    to this step is faster, 160 min to 63 % and a 170 min tail, as D1's own
    Fig. 5 shows (PHYSICS.md, section 7)."""
    d, _ = _step(rest, L0 + DL, V0)
    assert 0.0015 <= d[-1, 0] <= 0.0025
    assert 0.0025 <= d[-1, 1] <= 0.0045


#: [D1, Fig. 5], read off the plot at 500 min, x 1e-3: L step xD +1.8, xB
#: +3.15; V step xD -2.55, xB -2.3. 25 % for a plot reading, the audit's.
FIG5_AT_500 = {"L": (1.8e-3, 3.15e-3), "V": (-2.55e-3, -2.3e-3)}


def test_a_boilup_step_moves_both_products_the_other_way_as_figure_5(rest):
    """[D1, p. 14]: the boilup step by the same amount moves the products
    "almost the same, but in opposite direction"; Fig. 5 gives the sizes."""
    for name, (L, V) in (("L", (L0 + DL, V0)), ("V", (L0, V0 + DL))):
        d, _ = _step(rest, L, V, minutes=500)
        for j in (0, 1):
            want = FIG5_AT_500[name][j]
            assert np.sign(d[-1, j]) == np.sign(want)
            assert abs(d[-1, j] - want) <= 0.25 * abs(want), (name, j, d[-1])


def test_internal_flows_purify_both_products_with_a_smaller_faster_gain(rest):
    """[D1, p. 15, Fig. 6]: L and V up 0.27 together, D constant. Both products
    purer, the effect of a given change in L "about 100 times smaller" (30 to
    300, the audit's), and "the dynamics are much faster". The audit's 7 to
    30 min is D1's case with no liquid flow dynamics (the dotted lines); with
    them, Fig. 6 rises fast and then creeps, so "much faster" is tested as a
    63 % time under a quarter of the external step's (ours)."""
    dL, ext = _step(rest, L0 + DL, V0)
    d, tr = _step(rest, L0 + 100 * DL, V0 + 100 * DL, minutes=2000)
    assert d[-1, 0] > 0 and d[-1, 1] < 0
    gain_step = abs(dL[-1, 0]) / DL
    gain_internal = abs(d[-1, 0]) / (100 * DL)
    assert 30 <= gain_step / gain_internal <= 300
    assert t63(tr[:500, 0]) < 0.25 * t63(ext[:, 0])


def test_the_slowest_modes_are_the_published_ones():
    p = quiet()
    here = jnp.concatenate([jnp.asarray(_NOMINAL_PROFILE), steady_liquid(L0, p)])
    with enable_x64():
        here = here.astype(jnp.float64)
        A = jax.jacfwd(lambda z: compute_velocity(z, (L0, V0), 0.5, p)[0])(here)
    taus = np.sort(-1.0 / np.linalg.eigvals(np.asarray(A)).real)[::-1][:3]
    np.testing.assert_allclose(taus, [193.9, 12.0, 3.5], rtol=0.05)


def _closed_loop(p, s, minutes=300):
    """D1's two PI loops (eq. 82) on a 1 min old measurement: k1 = 26.1 (1 +
    3.76 s) / 3.76 s on xD with L, k2 = -37.5 (1 + 3.31 s) / 3.31 s on xB with
    V, around the nominal flows."""
    step = jax.jit(ENV.step_env)
    key = jax.random.PRNGKey(0)
    seen = (float(s.x[-1]), float(s.x[0]))
    i1 = i2 = 0.0
    out = []
    for _ in range(minutes):
        e1, e2 = 0.99 - seen[0], 0.01 - seen[1]
        i1, i2 = i1 + e1, i2 + e2
        L = L0 + 26.1 * (e1 + i1 / 3.76)
        V = V0 - 37.5 * (e2 + i2 / 3.31)
        seen = (float(s.x[-1]), float(s.x[0]))  # read before the move: 1 min old
        _, s, *_ = step(key, s, raw(L, V, p), p)
        out.append(seen)
    return np.asarray(out) - np.array([0.99, 0.01])


def _feed_run(feed, zF):
    """A step at minute 10 of the feed rate by ``feed`` (fraction) or of its
    composition by ``zF``, from rest at the nominal point."""
    p = quiet().replace(feed_step=abs(feed), zF_step=abs(zF))
    s = start(p).replace(
        feed_step_at=jnp.asarray(10 if feed else 10**6),
        feed_sign=jnp.sign(jnp.asarray(feed or 1.0)),
        zF_step_at=jnp.asarray(10 if zF else 10**6),
        zF_sign=jnp.sign(jnp.asarray(zF or 1.0)),
    )
    return _closed_loop(p, s)


def _back(dev):
    """Minutes from the step until the deviation last leaves 10 % of its
    extreme: the figure's "back"."""
    big = np.nonzero(np.abs(dev) > 0.1 * np.max(np.abs(dev)))[0]
    return float(big[-1] - 10)


def test_a_feed_rate_step_upsets_both_products_as_figure_18():
    dev = _feed_run(0.2, 0.0)
    assert 3.0e-3 <= -dev[:, 0].min() <= 5.6e-3
    assert 2.5e-3 <= dev[:, 1].max() <= 4.5e-3
    for j in (0, 1):
        assert 20 <= _back(dev[:, j]) <= 60


def test_a_feed_composition_step_is_smaller_than_a_feed_rate_step():
    """[D1, Fig. 18]: about 1e-3 (0.5e-3 to 2e-3, the audit's), on xB: D1
    adds that "zF has a smaller effect on xD then on xB", so xD is tested as
    smaller. Each under half the feed-rate step's, the figure's main point."""
    rate = _feed_run(0.2, 0.0)
    comp = _feed_run(0.0, 0.1)
    peak = np.max(np.abs(comp), axis=0)
    assert 0.5e-3 <= peak[1] <= 2e-3
    assert peak[0] < peak[1]
    for j in (0, 1):
        assert peak[j] < 0.5 * np.max(np.abs(rate[:, j]))


def test_the_analyser_reports_a_change_3_to_30_min_late():
    p = quiet()
    s = start(p)
    step = jax.jit(ENV.step_env)
    key = jax.random.PRNGKey(0)
    obs = ENV.get_obs(s, p)
    first = float(obs[0])
    for t in range(1, 40):
        obs, s, *_ = step(key, s, raw(L0 * 1.05, V0, p), p)
        if float(obs[0]) != first:
            break
    assert 3 <= t <= 30


def test_off_spec_product_is_priced_not_tripped():
    """[D4, p. 5, Table 1]: product limits are control zones."""
    p = quiet()
    s = start(p)
    s = s.replace(x=s.x.at[-1].set(0.85).at[0].set(0.15))
    assert not bool(ENV.is_terminated(s, p))


def test_inside_spec_less_boilup_scores_better():
    """[D4, p. 5]: hold both impurities in their zones at least reflux and
    boilup. Two columns inside spec, one purer on more boilup."""
    p = quiet()
    s = start(p)
    pure = s.replace(x=s.x.at[-1].set(0.995).at[0].set(0.005), V=V0 * 1.05)
    lean = s.replace(x=s.x.at[-1].set(0.991).at[0].set(0.009), V=V0)
    assert float(ENV.compute_reward(lean, p)) > float(ENV.compute_reward(pure, p))
