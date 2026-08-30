# Quadruple tank — physics model, provenance and validation

Reference process: **Johansson's quadruple-tank process**, the standard
laboratory MIMO benchmark with an adjustable multivariable zero. Two pumps feed
four tanks; each pump splits its flow between the lower tank on its own side
and the *diagonal* upper tank, which then drains into the other lower tank.

Contract for `target_gym.pc_gym.four_tank`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

Mass balances with Torricelli outflow:

```
A1 dh1/dt = −a1 √(2g h1) + a3 √(2g h3) + γ1 k1 v1
A2 dh2/dt = −a2 √(2g h2) + a4 √(2g h4) + γ2 k2 v2
A3 dh3/dt = −a3 √(2g h3) + (1 − γ2) k2 v2
A4 dh4/dt = −a4 √(2g h4) + (1 − γ1) k1 v1
```

**Provenance: adapted from [PC-gym](https://github.com/MaximilianB2/pc-gym).**
Every parameter and all four ODEs were compared against PC-gym's
`model_classes.py` term for term and match exactly. The process itself is
Johansson (2000), *The Quadruple-Tank Process: A Multivariable Laboratory
Process with an Adjustable Zero*, IEEE TCST 8(3).

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Pump dynamics and deadband | Voltage maps to flow instantaneously and linearly. |
| Level-sensor noise and lag | Both tanks are read exactly. |
| Valve-position drift | γ1 and γ2 are fixed, so the zero cannot be moved at run time — which is the one thing the real rig is famous for being able to do. |
| Inflow disturbances | No stochastic disturbance at all. |

**Regime of validity.** Levels between roughly 0.05 and 0.45 m, pump voltages
0–10 V. Torricelli outflow assumes free discharge and no orifice submergence.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Parameters and ODEs vs PC-gym | exact match | **verbatim** | ✅ |
| Configuration | γ1 + γ2 < 1 → non-minimum phase | **0.40** | ✅ |
| RGA element λ11 = γ1γ2/(γ1+γ2−1) | negative in this regime | **−0.067** | ✅ |
| Off-diagonal gains dominate | required for the cross pairing | ~4× the diagonal | ✅ |
| Maximum reachable levels | — | h1 0.360 m, h2 0.429 m | ✅ |
| Every target individually reachable | required | yes | ✅ |
| Every target pair **jointly** reachable | required | 49/49 | ✅ |
| Voltage headroom at the top target | < 90 % | yes | ✅ |
| Upper-tank margin above the trip | ≥ 30 mm | **≥ 50 mm** | ✅ |
| Episode covers settling | ≥ 4 τ | 500 steps ≈ 8 τ | ✅ |

**The negative RGA is the load-bearing result**, because it dictates how the
plant must be controlled. Johansson's formula gives λ11 = −0.067 and a
numerically differentiated gain matrix at the operating point agrees:

```
G = [[0.0092, 0.0412],      dh1/dv1 is four times SMALLER than dh1/dv2
     [0.0469, 0.0131]]
```

A negative RGA element means closing one diagonal loop reverses the sign of the
other, so integral action on the obvious pairing is unstable. The shipped
controller therefore drives **v1 from h2 and v2 from h1**. This is not a tuning
preference; the diagonal pairing tripped the low-level limit within 40 steps.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `gamma1`, `gamma2` | 0.2, 0.2 | – | PC-gym; sum < 1 → non-minimum phase | ✅ |
| `k1`, `k2` | 8.5e-4, 9.5e-4 | m³/(V s) | PC-gym | ✅ |
| `a1`–`a4` | 3.5, 3.0, 2.0, 2.5 (×10⁻³) | m² | PC-gym | ✅ |
| `A1`–`A4` | 1.0 | m² | PC-gym | ✅ |
| `v_min`, `v_max` | 0, 10 | V | THIS REPO — PC-gym has no bounds | ✅ |
| `h_min`, `h_max` | 0.05, 1.5 | m | THIS REPO | ⚠️ D2 |
| `tracking_band` | 0.05 | m | THIS REPO — see D1 | ✅ |
| `target_h1_range` | (0.11, 0.19) | m | THIS REPO — sized against the envelope | ✅ |
| `target_h2_range` | (0.14, 0.28) | m | THIS REPO | ✅ |
| `delta_t` | 1.0 | s | ~58 s tank time constant | ✅ |

As with the other PC-gym adaptations, the plant is upstream and everything
below `v_min` is this repository's. That distinction matters: every defect
found in this environment was in the lower half of the table, never the model.

---

## 4. Task design

**Why it is hard.** Two coupled loops whose obvious pairing is unstable, with
each pump sending 80 % of its flow to the tank on the *other* side. The useful
control move is a coordinated one, and a controller that treats the loops as
independent and diagonal will destabilise itself.

**Observation** `[h1, h2, h3, h4, target_h1, target_h2]` — all four levels are
measured, so unlike most of this suite the plant is fully observed. The
difficulty is structural, not informational.

**Reward** — mean of two squared normalised tracking bands.

---

## 5. Known deviations

**⚠️ D1 — the reward band was three times the operating range.** It was
originally normalised by the full tank span (`h_max − h_min` = 1.45 m), so a
half-metre miss still scored 0.43 and a saturated controller looked much like a
working one. Now an explicit 0.05 m band. Recorded because it is *why* a much
worse defect went unnoticed: the target range once sat entirely above the
reachable envelope — no sampled setpoint was attainable and every episode was
unwinnable — and neither the reward nor the shared effectiveness contract
registered it.

**⚠️ D2 — the level limits are not physical.** `h_max = 1.5 m` cannot be
reached by any input (the plant tops out at 0.43 m), so the high-level trip is
dead code. `h_min = 0.05 m` is live and binds on the *upper* tanks, which is
the subtle constraint: holding a high h1 with a low h2 requires a low v1, and
v1 is what feeds tank 4. The target box is sized against that.

**⚠️ D3 — the zero is fixed.** The real apparatus is celebrated for letting you
move the multivariable zero across the imaginary axis by turning two valves.
Here γ1 and γ2 are constants, so only the non-minimum-phase configuration is
available. Making them episode parameters would recover the benchmark's most
distinctive property.

**⚠️ D4 — no disturbance and no measurement noise.** Deterministic given the
sampled setpoint and initial condition, which is why the conformance suite
skips its PRNG-hygiene checks.
