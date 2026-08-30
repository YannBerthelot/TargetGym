# CSTR — physics model, provenance and validation

Reference process: a **non-isothermal continuous stirred-tank reactor** running
an irreversible first-order exothermic reaction A → B, cooled through a jacket.
The coolant temperature is the manipulated variable.

Contract for `target_gym.pc_gym.cstr`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

Two states — reactant concentration and reactor temperature — under perfect
mixing:

```
dCa/dt = (q/V)(Caf − Ca) − r
dT/dt  = (q/V)(Ti − T) + (−ΔHr)·r / (ρ c_p) + UA(Tc − T) / (ρ c_p V)

with    r = k0 · exp(−E/RT) · Ca
```

**Provenance: adapted from [PC-gym](https://github.com/MaximilianB2/pc-gym).**
The parameters and both ODEs were compared against PC-gym's `model_classes.py`
term for term and match exactly. PC-gym itself carries no citation for the
model; the parameter set is the standard non-isothermal CSTR used throughout
process-control teaching.

The interesting structure is the feedback between the two states: heat released
raises the temperature, the Arrhenius term raises the rate exponentially, and
that releases more heat. The jacket is the only thing holding it.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Jacket dynamics | Coolant temperature is commanded directly, with no lag or flow limit. |
| Feed disturbances | `Caf` and `Ti` are constant; the plant has no stochastic disturbance at all. |
| Reverse reaction, side reactions, catalyst decay | One irreversible first-order step. |
| Level / volume dynamics | Constant holdup, perfectly mixed. |
| Mixing and measurement lag | Both instantaneous. |

**Regime of validity.** The low-conversion (extinguished) branch, roughly
315–330 K, with coolant between 295 and 302 K. The model is written for the
whole state space but the environment only operates on that branch — see §2.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Parameters and ODEs vs PC-gym | exact match | **verbatim** | ✅ |
| Residence time V/q | — | **1.00 min** | ✅ |
| Steady-state multiplicity | 3 solutions over a coolant window | **3 for Tc ≈ 299–304 K** | ✅ |
| Operating branch is stable | required | eigenvalues < 0 throughout | ✅ |
| Operating temperature | 315–330 K | 317.7 → 328.7 K | ✅ |
| Conversion on that branch | low | 7.3 % → 16.5 % | ✅ |
| Every target concentration reachable | required | 0.835 ≤ Ca ≤ 0.927 covers (0.84, 0.91) | ⚠️ D1 |
| Reaction is exothermic | ΔHr < 0 | −50 kJ/mol | ✅ |
| Runaway trip sits above the branch | required | 350 K trip, branch ≤ 329 K | ✅ |

**Multiplicity is the load-bearing check.** It is not something the parameters
were tuned to produce — it falls out of the Arrhenius feedback, and its
presence is what makes this a genuine reactor rather than a first-order lag
with a nonlinear gain:

| coolant Tc | steady states (T) | reading |
|---|---|---|
| 295 K | 317.7 | extinguished only |
| 300 K | 324.5 · 350.0 · 369.7 | **three** — the middle one is unstable |
| 302 K | 328.7 · 343.5 · 373.6 | three |
| 305 K | 378.1 | ignited only |

The environment operates on the low branch and trips at 350 K, which is where
the unstable middle solution sits at Tc = 300. So the termination is not an
arbitrary safety number: it fires exactly when the reactor leaves the
extinguished branch and begins to ignite.

**The plant also slows as it is pushed.** The eigenvalues run from −1.53 /min
at Tc = 295 to −0.75 /min at Tc = 302, so the closed loop that is comfortable
at the cold end is half as fast at the hot end.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `q`, `V` | 100, 100 | L/min, L | PC-gym | ✅ |
| `rho`, `C` | 1000, 0.239 | g/L, J/(g K) | PC-gym | ✅ |
| `deltaHr` | −5.0e4 | J/mol | PC-gym | ✅ |
| `EA_over_R` | 8750 | K | PC-gym | ✅ |
| `k0` | 7.2e10 | 1/min | PC-gym | ✅ |
| `UA` | 5.0e4 | J/(min K) | PC-gym | ✅ |
| `Ti`, `Caf` | 350 K, 1.0 mol/L | – | PC-gym | ✅ |
| `T_c_min`, `T_c_max` | 295, 302 | K | THIS REPO — not in PC-gym, which has no bounds | ⚠️ |
| `T_max` | 350 | K | THIS REPO — runaway trip, at the unstable branch | ✅ |
| `target_CA_range` | (0.84, 0.91) | mol/L | THIS REPO — see D1 | ⚠️ |
| `delta_t` | 0.25 | min | 4 steps per residence time | ✅ |

PC-gym's model class carries **no bounds, targets or termination**. Everything
in the lower half of this table is this repository's own choice, and is
therefore where to look first when something is unreachable — as it was for the
four-tank environment.

---

## 4. Task design

**Why it is interesting.** Concentration is controlled indirectly: the coolant
sets the temperature, and the temperature sets the reaction rate through an
exponential. The gain is roughly −0.011 mol/L per K, so the entire 7 K input
range buys 0.078 mol/L of concentration — a narrow, nonlinear authority.

**Observation** `[Ca, T, target_CA]` — a real reactor has a thermocouple and a
composition analyser. Nothing is hidden here, which makes this one of the two
fully-observed environments in the suite.

**Reward** — a squared normalised tracking band on `Ca`.

**No disturbance.** Unusually for this suite, the CSTR is deterministic apart
from the sampled setpoint. That is inherited from PC-gym and is why the shared
conformance suite skips its PRNG-hygiene checks.

---

## 5. Known deviations

**⚠️ D1 — the bottom of the target range is nearly unreachable.** Targets are
sampled from (0.84, 0.91) mol/L. At the coolant limit `T_c_max = 302 K` the
steady concentration is 0.835 mol/L, so a target of 0.84 needs the coolant
within about 0.2 K of its stop — roughly 3 % of the input range as margin. It
is *reachable*, unlike the four-tank case, but a controller asked for the
bottom of the band has essentially no authority left for disturbance rejection.
Asserted by a test so it cannot silently become unreachable.

**⚠️ D2 — the jacket is ideal.** Coolant temperature is commanded directly.
A real jacket has a flow actuator, a transport lag and a finite cooling duty,
all of which matter for a reactor whose failure mode is thermal runaway.

**⚠️ D3 — no feed disturbance.** Feed concentration and temperature are fixed,
so the only variation across episodes is the sampled setpoint and initial
condition. The environment is effectively deterministic.

**⚠️ D4 — the ignited branch is unreachable by construction.** The 350 K
termination fires before the reactor can settle on its high-conversion steady
state, so the multiplicity is present in the model but only one branch is ever
visited. The trip is the *point* — runaway is the irrecoverable state — but it
means the environment does not exercise the reactor's full behaviour.
