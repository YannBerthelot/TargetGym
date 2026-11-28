"""The reagent capacity pH's tanks hold, and what its evaluation charges for it
(v1 task pages, item 2; owner, 2026-11-28: "charge, at the episode's end, any
reagent capacity the tanks have lost").

Contract:

- The env reports the capacity its two stages hold above a pH, in litres of the
  base stream (``stored_capacity``): the base a policy can spend before the tanks
  fall to that pH, negative where they sit below it. Above the permit's low edge it
  is the start's inventory the ceiling counts.
- Every run records, after each step, the capacity above the pH an episode starts
  at (``setpoint_pH``) at the reagent's price, as ``stored_value``. pH's evaluation
  names it as its edge charge: the first step only marks the start, every later one
  is scored, and the change in the capacity across the window is charged once at
  that price, a fuller tank credited at it (``Evaluation.edge_credit``). Near the
  start's pH the tanks' content leaves in the outflow holding no base above that
  pH, so restoring a litre of capacity there takes a litre of reagent: the charge is
  what restoring costs, and the credit what filling cost, with nothing between them
  to trade. The reward carries no such charge.
- Draining the tanks at the episode's end never pays (the medium test below, the v1
  pages' end probe).

Where the expected values come from, none of them the function under test:

- capacity by counting base, as ``test_ph_ceiling`` counts the ceiling's
  inventory: a tank at rest at a pH is the steady mix of the three streams there, a
  fraction ``f = q3 / (q1 + q2 + q3)`` of it base, and the water in it needs
  ``q_ref / (q1 + q2)`` of base per unit to sit at the reference pH; the base flows
  are roots of the charge balance in ``env.py``'s docstring, found here by Brent's
  method in float64;
- a tank whose content is ``V - x`` mL of water at a pH and ``x`` mL of the base
  stream holds ``x`` mL above that pH, by the definition: removing that base leaves
  the water at the pH;
- the charge: the window's mean cost plus the capacity's change over its steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from target_gym import evaluation
from target_gym.eval import Episode
from target_gym.pc_gym.ph_neutralization.env import PHParams
from target_gym.registry import REGISTRY

SPEC = REGISTRY["ph_neutralization"]
P = PHParams()
#: Steps of the scored episode: the first marks the start, 1439 are scored.
STEPS = int(P.max_steps_in_episode)
#: float32 capacities near 1 L, a difference of terms near 1e-3 M: 1e-4.
CAPACITY_RTOL = 1e-4


def _balance(pH, Wa, Wb, p):
    """The charge balance of env.py's docstring, in float64."""
    carbonate = (1 + 2 * 10 ** (pH - p.pK2)) / (
        1 + 10 ** (p.pK1 - pH) + 10 ** (pH - p.pK2)
    )
    return Wa + 10 ** (pH - 14) - 10**-pH + Wb * carbonate


def _pH(Wa, Wb, p):
    return brentq(lambda x: _balance(x, Wa, Wb, p), 0.0, 14.0, xtol=1e-14)


def _mix(q1, q2, q3, p):
    """The invariants of a steady mix of the three streams at these flows."""
    total = q1 + q2 + q3
    Wa = (q1 * float(p.Wa1) + q2 * float(p.Wa2) + q3 * float(p.Wa3)) / total
    Wb = (q1 * float(p.Wb1) + q2 * float(p.Wb2) + q3 * float(p.Wb3)) / total
    return Wa, Wb


def _base_flow(pH, q1, q2, p):
    return brentq(lambda q3: _pH(*_mix(q1, q2, q3, p), p) - pH, 0.0, 60.0, xtol=1e-12)


def _plant(pH_s1, pH_s2, q2=0.55, p=P):
    """A reset plant with each stage at rest at its own pH on influent and ``q2``."""
    from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization

    _, s = PHNeutralization().reset_env(jax.random.PRNGKey(0), p)
    (Wa1, Wb1), (Wa2, Wb2) = (
        _mix(16.6, q2, _base_flow(x, 16.6, q2, p), p) for x in (pH_s1, pH_s2)
    )
    f32 = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
    return s.replace(Wa_s1=f32(Wa1), Wb_s1=f32(Wb1), Wa=f32(Wa2), Wb=f32(Wb2))


def _counted(pH, ref, q2=0.55, p=P):
    """Litres of base stream one 20 L tank at rest at ``pH`` holds above the pH
    ``ref``, by counting base."""
    water = 16.6 + q2
    at = _base_flow(pH, 16.6, q2, p)
    needed = _base_flow(ref, 16.6, q2, p)
    base = at / (water + at)
    return 20.0 * (base - (1.0 - base) * needed / water)


@pytest.mark.parametrize("ref", [6.0, 6.45])
@pytest.mark.parametrize("pH_s1, pH_s2", [(6.45, 6.45), (7.5, 6.2), (6.0, 6.0)])
def test_the_tanks_hold_the_base_above_a_pH_that_counting_gives(pH_s1, pH_s2, ref):
    from target_gym.pc_gym.ph_neutralization.env import stored_capacity

    held = float(stored_capacity(_plant(pH_s1, pH_s2), P, ref))
    want = _counted(pH_s1, ref) + _counted(pH_s2, ref)
    assert held == pytest.approx(want, rel=CAPACITY_RTOL, abs=2e-5)


def test_the_capacity_defaults_to_the_permits_low_edge_and_is_negative_below_it():
    from target_gym.pc_gym.ph_neutralization.env import stored_capacity

    on_start = _plant(6.45, 6.45)
    assert float(stored_capacity(on_start, P)) == pytest.approx(
        float(stored_capacity(on_start, P, float(P.permit_low))), rel=1e-6
    )
    assert float(stored_capacity(_plant(5.5, 5.8), P)) < 0.0


def test_base_mixed_into_water_at_a_pH_is_held_litre_for_litre():
    from target_gym.pc_gym.ph_neutralization.env import stored_capacity

    s = _plant(6.3, 6.3)
    x = 300.0  # mL of the base stream in each 20 L stage
    V = float(P.V)
    Wa = (float(s.Wa) * (V - x) + float(P.Wa3) * x) / V
    Wb = (float(s.Wb) * (V - x) + float(P.Wb3) * x) / V
    f32 = lambda v: jnp.asarray(v, jnp.float32)  # noqa: E731
    mixed = s.replace(Wa_s1=f32(Wa), Wb_s1=f32(Wb), Wa=f32(Wa), Wb=f32(Wb))
    assert float(stored_capacity(mixed, P, 6.3)) == pytest.approx(
        2 * x / 1000.0, rel=CAPACITY_RTOL
    )


def test_the_value_is_the_capacity_above_the_start_at_the_reagents_price():
    from target_gym.pc_gym.ph_neutralization.env import stored_capacity
    from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization

    env = PHNeutralization()
    s = _plant(6.8, 6.6)
    dear = P.replace(reagent_price=2.5)
    want = 2.5 * float(stored_capacity(s, P, float(P.setpoint_pH)))
    assert float(env.stored_value(s, dear)) == pytest.approx(want, rel=1e-6)
    # A plant reset on its start holds nothing above it.
    _, fresh = env.reset_env(jax.random.PRNGKey(3), P)
    assert abs(float(env.stored_value(fresh, P))) < 1e-3


def test_pHs_evaluation_marks_the_start_and_charges_the_capacitys_change():
    ev = evaluation.of(SPEC)
    assert (ev.unscored_steps, ev.scored_steps) == (1, None)
    assert (ev.edge_charge, ev.edge_credit) == ("stored_value", True)


def _episode(before, after, cost=0.07):
    """A scored episode at ``cost`` a step, the store worth ``before`` after the
    first step and ``after`` after the last, and more in between."""
    store = np.full(STEPS, 2.0 * max(before, after, 1.0))
    store[0], store[-1] = before, after
    return Episode(
        cost=np.full(STEPS, cost),
        in_band=np.ones(STEPS, bool),
        target_change=np.zeros(STEPS, bool),
        terms={"stored_value": store},
    )


def test_ending_at_the_start_capacity_is_neutral():
    assert evaluation.episode_cost(SPEC, _episode(0.2, 0.2)) == pytest.approx(0.07)


def test_capacity_lost_is_charged_and_capacity_gained_credited_once():
    steps = STEPS - 1
    assert evaluation.episode_cost(SPEC, _episode(0.2, -1.3)) == pytest.approx(
        0.07 + 1.5 / steps
    )
    assert evaluation.episode_cost(SPEC, _episode(0.2, 1.7)) == pytest.approx(
        0.07 - 1.5 / steps
    )


def test_the_reward_carries_no_capacity_charge():
    from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization

    terms = PHNeutralization().reward_terms(_plant(6.45, 6.45), P)
    assert set(terms) == {"reagent", "excursion", "diversion", "failure"}


# --- The v1 pages' end probe, on pH's own evaluation (medium) -----------------
#
# The frozen reference, and the same with both valves shut over the episode's last
# 5, 10, 20 and 40 minutes, each scored as ``tg.score`` scores a policy: every
# scenario at its weight on the scoring seeds, read by pH's evaluation. Before the
# charge the 5 min cut saved 4.1 % on the normal day and 0.2 % on the score. With
# it every cut loses on the score. Shorter cuts, not tested: 0.5, 1 and 2 min
# change the score by -0.0001 %, +0.00005 % and +0.0025 %, and the 5 min cut gains
# 0.045 % on the normal day alone, where the discharge falls below the start pH in
# the last minutes: base the outflow does not carry is saved, as it is by holding
# a lower pH anywhere, and the excursion the fall would bring comes after the
# window (scratchpad probes of the pH v1 session, 2026-11-28).

#: Steps of each cut: 5, 10, 20 and 40 minutes of 5 s steps.
CUTS = (60, 120, 240, 480)


class _Cut:
    """The reference with both valves shut from step ``start`` on."""

    def __init__(self, ref, start):
        self.ref, self.start = ref, int(start)

    def reset(self, n_envs=None):
        t = jnp.zeros(() if n_envs is None else (n_envs,), jnp.int32)
        return (self.ref.reset(n_envs), t)

    def step(self, memory, obs):
        m, t = memory
        a, m = self.ref.step(m, obs)
        shut = jnp.where(t >= self.start, -1.0, jnp.asarray(a, jnp.float32))
        return shut, (m, t + 1)


@pytest.fixture(scope="module")
def end_cuts() -> dict:
    """Each cut's saving on the score over the reference, as ``tg.score`` reads it."""
    import target_gym as tg

    ref = tg.make_reference("ph_neutralization")
    return {
        n: tg.score("ph_neutralization", _Cut(ref, STEPS - n))["saving"] for n in CUTS
    }


@pytest.mark.medium
@pytest.mark.records
@pytest.mark.parametrize("steps", CUTS)
def test_draining_the_tanks_at_the_episodes_end_never_pays(end_cuts, steps):
    assert end_cuts[steps] <= 0.0
