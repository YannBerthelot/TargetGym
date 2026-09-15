# Nuclear reactor — physics model, provenance and validation

Reference plant: a **3 GW thermal pressurised-water reactor**, modelled with
point kinetics, six delayed-neutron groups, xenon-135 poisoning, two-node
thermal feedback and rate-limited control rods.

Contract for `target_gym.reactor`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

**Point kinetics** with the standard six-group U-235 thermal delayed-neutron
data of Keepin (1965):

```
dn/dt   = ((rho − beta) / Lambda) n + sum_i lambda_i C_i
dC_i/dt = (beta_i / Lambda) n − lambda_i C_i          i = 1..6
```

Reactivity is the sum of the rod, the thermal feedbacks and the xenon term:

```
rho = rho_ext
    + alpha_fuel    (T_fuel    − T_fuel_ref)
    + alpha_coolant (T_coolant − T_coolant_ref)
    − rho_Xe_full   (Xe_hat − 1)
```

Both temperature coefficients are negative, which is what makes a PWR
passively self-regulating: a power excursion heats the fuel, Doppler broadening
absorbs more neutrons, and the excursion damps itself without the rods moving.

**Xenon-135** is tracked through its iodine parent in normalised units, so
equilibrium at full power is `I_hat = Xe_hat = 1` and contributes no reactivity
by construction:

```
dI_hat/dt  = lambda_I (n − I_hat)
dXe_hat/dt = a n + b I_hat − (lambda_Xe + sigma_phi0 n) Xe_hat
```

Xe-135 has the largest thermal absorption cross-section of any known nuclide.
After a power reduction, iodine keeps decaying into xenon while the burn-up
that destroys it collapses — so the poison *builds* for hours after the power
has already come down. That is the xenon pit, and it is the environment's
central mechanic.

Integration is **TR-BDF2 on the fast block** (neutrons plus precursors) with
the thermal and fission-product states advanced explicitly. The stiffness ratio
is enormous — prompt neutrons at ~1e-4 s against xenon at ~1e5 s, nine orders
of magnitude — and an explicit scheme would need a step small enough to make a
24-hour episode unaffordable.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Spatial neutronics (flux tilt, axial offset, xenon oscillations) | Point kinetics is a zero-dimensional reduction. Real PWRs can develop *spatial* xenon oscillations that this cannot represent. |
| Samarium-149 and other poisons | Xenon dominates the short-term reactivity balance. |
| Burn-up and fuel depletion | Episodes are 24 h; depletion acts over months. |
| Boron / chemical shim | Real PWRs trim reactivity with dissolved boron over a cycle; here the rods do everything. |
| Primary-loop transport delay, steam generator, turbine | The coolant node is lumped and the secondary side is absent. |
| Decay heat | Power follows the neutron flux, so a scram removes all heat instantly. Real decay heat is ~7 % immediately after shutdown. |

**Regime of validity.** Power manoeuvring between roughly 20 % and 100 % of
rated, with reactivity well below prompt critical. Not valid for startup,
shutdown transients, or any accident scenario.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Total delayed fraction β | 0.0065 (U-235 thermal) | **0.00650** | ✅ |
| Neutron generation time Λ | 1e-5 – 1e-4 s (PWR) | 1e-4 | ✅ |
| Doppler coefficient | −2 to −4 pcm/K | **−3.0** | ✅ |
| Moderator coefficient | −10 to −50 pcm/K | −5.0 | ⚠️ D1 |
| I-135 half-life | 6.57 h | **6.57 h** | ✅ |
| Xe-135 half-life | 9.14 h | **9.13 h** | ✅ |
| Stable period at +100 pcm (inhour) | tens of seconds | **55 s** | ✅ |
| Maximum rod worth vs prompt critical | must stay < β | **0.77 β** | ✅ |
| Rod speeds, insert / withdraw | ~40 / 20 pcm/s | **40 / 20** | ✅ |
| Coolant rise across the core | 30–40 K | **33.3 K** | ✅ |
| Fuel temperature at full power | below the 1473 K trip | **1043 K** | ✅ |
| Xenon peak after a scram | 9–11 h | 8.3 h | ⚠️ D2 |
| Peak xenon worth after a scram | ~2000–3000 pcm | **−2213 pcm** | ✅ |
| Reactivity budget spans the target range | rods must cover it | −671 to +470 pcm | ✅ |

The **inhour check** is the load-bearing one: it ties Λ and all twelve
delayed-neutron constants together through the dispersion relation, and a
reactor period is the quantity an operator would actually recognise.

The **reactivity budget** is the second. Holding a power level requires the
rods to cancel the thermal feedback at that level, and the required worth is an
*output* of the model rather than something tuned:

| power | T_fuel | T_coolant | feedback | rod worth needed |
|---|---|---|---|---|
| 0.30 | 701 K | 565 K | +671 pcm | −671 |
| 0.50 | 799 K | 572 K | +345 pcm | −345 |
| 1.00 | 1043 K | 588 K | −470 pcm | **+470** |

Rod range is −1000 to +500 pcm, so full power is reachable with **30 pcm of
margin**. That is deliberate, and it is what gives the xenon pit its teeth:
after a power reduction the xenon overshoot removes far more than 30 pcm, so
returning to *full* power becomes impossible for hours even though intermediate
levels stay available.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `beta_i`, `lambda_i` | 6-group | – | Keepin (1965), U-235 thermal | ✅ |
| `Lambda_gen` | 1e-4 | s | PWR range | ✅ |
| `alpha_fuel` | −3.0e-5 | 1/K | Doppler, PWR range | ✅ |
| `alpha_coolant` | −5.0e-5 | 1/K | Weaker than a real PWR — see D1 | ⚠️ |
| `rho_ext_min/max` | −0.010 / +0.005 | – | Sized by the reactivity budget above | ✅ |
| `rod_speed_insert/withdraw` | 4e-4 / 2e-4 | 1/s | Insertion is the safety direction | ✅ |
| `P_thermal_ref` | 3.0 | GW | Large PWR | ✅ |
| `C_fuel`, `C_coolant`, `UA` | 3.3e7, 7.0e7 J/K, 6.6e6 W/K | – | ~100 t UO₂; gives a 5 s fuel time constant | ⚠️ |
| `m_dot_cp` | 9.0e7 | W/K | Gives the 33 K core rise | ✅ |
| `sigma_phi0` | 7.95e-5 | 1/s | σ_a 2.65e-18 cm² × Φ ≈ 3e13 n/cm²/s | ✅ |
| `gamma_ratio` | 0.049 | – | Direct Xe yield / I yield (0.003 / 0.061) | ✅ |
| `rho_Xe_full` | 0.025 | – | Equilibrium xenon worth, 2500 pcm | ✅ |
| `delta_t` | 1.0 | s | Physics sub-step; 10 per env step, so 8640 env steps = 24 h | ✅ |

---

## 4. Task design

**Why it is hard.**

1. **Xenon is a multi-hour memory.** A power change today constrains what is
   reachable six hours from now. Reacting when the poison arrives is far too
   late; the controller has to anticipate it.
2. **Rod authority is deliberately tight.** +500 pcm against an equilibrium
   xenon worth of 2500 pcm, and only 30 pcm spare at full power.
3. **Rod motion is asymmetric.** Insertion runs at twice the withdrawal rate —
   the reactor can always be shut down faster than it can be brought up.
4. **Four timescales at once**: prompt neutrons (ms), precursors (s to min),
   thermal (s to min), xenon (hours).
5. **Most of the state is hidden.** The six precursors, fuel temperature,
   iodine and xenon are all unobserved.

**Observation** `[n, T_coolant, rho_ext_norm, target_n]` — a plant measures
neutron flux, coolant temperature and rod position. Fuel temperature and the
fission-product inventory are not directly instrumented.

**Reward** = power tracking − rod-motion penalty.

---

## 5. Known deviations

**⚠️ D1 — the moderator coefficient is weak.** −5 pcm/K against a −10 to
−50 pcm/K range for a real PWR. The consequence is that this reactor is less
self-regulating against coolant-temperature swings than the real machine, so
the rods carry more of the load than they would in practice. The Doppler
coefficient, which dominates the fast feedback, is in range.

**⚠️ D2 — the xenon peak is early and shallow.** 8.3 h after a full scram at
1.89× equilibrium, against a documented 9–11 h. The peak *worth* (−2213 pcm)
lands in the right band, so the reactivity consequence is realistic even though
the timing is roughly an hour early.

**⚠️ D3 — reset is not thermally lined out.** Temperatures start at fixed
values (900 K fuel, 580 K coolant) regardless of the initial power, so the
first minute of an episode carries a thermal transient of up to ~70 K on the
fuel and a corresponding reactivity swing. Precursors and xenon *are* started
at their steady values, so this is inconsistent within the same reset.

**⚠️ D4 — the coolant node is nearly algebraic.** Its time constant works out
at 0.72 s, so the coolant follows the fuel almost instantly. A real primary
loop has a transport delay of order ten seconds.

**❌ D5 — no decay heat.** Power follows the neutron flux exactly, so a scram
removes all heat immediately. A real reactor still produces ~7 % of rated power
seconds after shutdown, decaying over hours. This makes shutdown transients in
this model markedly more benign than the real thing.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `reactor` | 864 | 10 | 2.4 h | 1 in [-1, 1] | 4 | 14 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per 10 s control step

`compute_reward = -(tracking + running + failure)`; each physics sub-step charges one second and `step_env` sums the ten, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.00451 of rated | certified hold floor: `scripts/floor_reactor_hold.py`, a one-state dynamic programme on the error against the demand's within-period random walk (sd 7.9e-3 over 10 s) with the rod-limited power slews; the slew never binds and the floor is the walk. The shipped MPC holds 0.0069 at period boundaries against the 0.0063 boundary floor (`scripts/measure_hold.py`, 2000 hold steps after a 2000-step burn-in); the PID 0.08 over the same 11 h (3% over the 2.4 h test episode). |
| `e_tol` | 0 | none |
| `tracking_exponent` | 1 | imbalance energy is settled linearly |
| `imbalance_multiple` | 3 x `spot_price_per_MWh` on `P_electric_GW` | 240 000 $/h per unit of \|error\|; tracking at the floor costs $3.0 per step |
| `rod_wear_weight` | 1 | **provisional.** Reactivity asked beyond what the rods can deliver, as a fraction of the rod range, charged per unit at what tracking at the floor costs; the audit found the optimum insensitive to it below ten times the floor. Sweep 0.5 / 1 / 2. |
| `failure_cost` | 2000 $ per step | twice the 1.49 envelope's imbalance |
| `restart_steps` | 17280 (48 h) | restart time priced into a trip, `restart_steps x failure_cost` (a SCRAM's xenon-limited restart; provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the tracking cost per step at the floor in the reward's
units (the NEA floor for tracking) and `rho_floor` the full floor including
consumption charged in full; `floor_is_documented_minimum` records whether
`e_floor` is a measured/certified floor or a resolution used as a scale, and
where it is a resolution (a deterministic plant) both references are 0, since
exact hold is achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce, and `restart_steps` the time a restart would take,
so a trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`): the step that leaves the envelope is
charged the trip cost, with tracking and running cost zeroed, and the plant
restarts at once as `reset_env` would, on the same clock. `terminated` is never
raised; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
