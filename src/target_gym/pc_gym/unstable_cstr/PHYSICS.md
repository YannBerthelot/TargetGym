# Unstable CSTR: physics model, provenance and validation

Reference process: the shipped `cstr` reactor, a jacketed continuous
stirred-tank reactor running an irreversible first-order exothermic reaction
A → B, held on its **open-loop unstable middle steady state**. The coolant
command reaches the reactor through a valve and a first-order jacket lag, the
reactor temperature reads through a thermowell, the targets move on a ramp, and
the feed temperature drifts. **The jacket is synthetic** (section 6).

Contract for `target_gym.pc_gym.unstable_cstr`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

Every number carries a label: **read** (from a source or from shipped code),
**derived** (computed from read values by a script or test named beside it),
**ours** (a design choice), **synthetic** (ours and load-bearing, and not what
hardware gives), **measured** (from a repo script run on the built env).
"The numbers script" is `scripts/unstable_cstr_numbers.py`; a flag such as
`--section pnr` or `--sensitivity` names the part of it that prints the
figure. Its tables score a hold on a fixed window of their own, minutes 6 to
8.4 of each block (120 <= block_clock % 200 < 168), which ends before the
32-step horizon of the MPC before 0.8 reaches the next switch. The hold that sets the floor
comes from `scripts/measure_hold.py` instead (the reward section).

**What the instruments changed.** Figures derived from the plant alone (sections
2 and 4, and the reset and integrator figures) include the thermowell, the valve
and the ramp. Figures measured on the expert or the oracle (sections 5 to 8 and the
reward table) were taken without them. PR 9 of `docs/proposals/industrial-fixes.md`
measures those again when it retunes the expert, which trips on this plant
(section 7).

---

## 1. Model and provenance

Three states under perfect mixing, and the feed drift:

```
dC_a/dt = (q/V)(Caf − C_a) − r
dT/dt   = (q/V)(Ti + dTi − T) + (−deltaHr) r / (rho C) + UA (T_j − T) / (rho C V)
dT_j/dt = (T_c − T_j) / tau_j

T_c[k] = T_c[k−1] + clip((1 − exp(−delta_t / tau_v)) (u[k] − T_c[k−1]), −r_v delta_t, r_v delta_t)

with    r = k0 · exp(−EA_over_R / T) · C_a

dTi[k+1] = a dTi[k] + Ti_sigma sqrt(1 − a²) xi[k],   a = exp(−delta_t / Ti_tau),   xi[k] ~ N(0, 1)
```

| symbol | quantity | unit |
|---|---|---|
| C_a | reactant concentration | mol/L |
| T | reactor temperature | K |
| T_j | jacket temperature, seen by the energy balance | K |
| u | coolant command, the action | K |
| T_c | coolant the valve delivers, `actuate`'s output | K |
| tau_v, r_v | the valve's lag and slew limit, `actuator_lag` and `actuator_rate` | min, K/min |
| dTi (`Ti_dev`) | feed-temperature drift, added to `Ti` | K |
| t | time; one step is `delta_t` = 0.05 min (3 s) | min |

**The balances and the ten values are imported.** `compute_velocity` calls
cstr's own `compute_velocity` (`import target_gym.pc_gym.cstr.env as
cstr_env`) with `T_j` in the coolant slot and `Ti + Ti_dev` as the feed, and
the ten physical defaults are read from `CSTRParams()` at import. The spec
declares `pc_gym/cstr/env.py` in `fingerprint_sources`, so an edit there moves
this task's version stamp and recorded baselines too.
`test_the_ten_parameters_are_cstrs` checks both, bit for bit.

**Provenance chain** (all read):

| source | what it gives | |
|---|---|---|
| PC-gym, `src/pcgym/model_classes.py`, https://github.com/MaximilianB2/pc-gym | the two ODEs and the ten values, verbatim in cstr (verified locally, code) | ✅ |
| Decardi-Nelson and Liu (2022), https://arxiv.org/html/2109.09810v1 | Table 1: the ten values; Eq. 36a and 36b: the same two balances; Section 5.3: two published middle-branch states | ✅ |
| Seborg, Edgar, Mellichamp and Doyle, *Process Dynamics and Control*, Table 2.3 (3rd ed.; the 4th ed. numbering is assumed) | the ten values and the textbook point (0.5 mol/L, 350 K at a 300 K coolant), read through Sandrock's notebook (https://github.com/alchemyst/Dynamics-and-Control, "Nonlinear CSTR") and Kantor's (https://jckantor.github.io/cbe30338-book/notebooks/02.07-Exothermic-CSTR.html) | ✅ |
| Bequette (1998), *Process Dynamics: Modeling, Analysis, and Simulation* (book snippet) | rho c_p = 239 J/(L K), the same heat capacity (deviation D1) | ✅ |
| APMonitor, https://apmonitor.com/pdc/index.php/Main/StirredReactor | the low steady state at a 300 K coolant, to 15 digits | ✅ |

**Added by this task.** The jacket lag (`tau_j` 6 s, synthetic; section 6) and the
feed drift (`Ti_sigma` 2 K, `Ti_tau` 10 min, ours), neither of them sourced, and the
valve (section 3), the thermowell (section Readings) and the ramp (section 3).

**Integration.** One RK4 step per env step (`rk4_1`) over `[C_a, T, T_j]`,
with the valve's output and the drift held over the step. The drift is then advanced
by its exact Ornstein-Uhlenbeck update, with the innovation drawn from the
step key folded with the pre-increment `time` (as `glass_furnace` and
`battery` do), so a constant rollout key still gives a zero-mean process.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Aqueous heat capacity (D1) | C = 0.239 J/(g K) is what every source above uses; section 9. |
| Jacket energy balance (D2) | The jacket follows the command through a fixed lag, whatever heat it removes. |
| Analyser dead time (D3) | C_a is observed at once, with noise (section Readings). |
| Feed-concentration disturbance (D4) | Only the feed temperature drifts. |
| Side reactions, catalyst decay | One irreversible first-order step, as in cstr. |
| Volume, boiling, pressure | Constant holdup, liquid phase throughout. |
| The ignited branch above the trip | The 365 K interlock trips first (section 2). |

**Regime of validity.** RK4 at `delta_t` is stable for |lambda| dt < 2.785
on the negative real axis (read). The fastest dynamics are the runaway just
below the trip. Under full heating until the trip, the largest |lambda| dt is
0.720 from the 135 corners of the reset box and 0.936 from 33 extinguished
starts, and anywhere with 0 <= C_a <= Caf and T <= 365 K it is 1.616, at
C_a = Caf and T = 365 K (derived, `--section integrator`;
`test_rk4_is_stable_where_the_plant_can_go`). One step per 3 s is converged.
Over a 1200-step episode of the shipped cascade, 16 RK4 substeps move T by at
most 9.16e-4 K and C_a by at most 2.74e-6 mol/L, and under full heating from
the low state both integrations trip at step 25, at most 7.0e-4 K apart
before it (derived, `--section integrator`; `test_rk4_1_matches_rk4_16`
bounds them at 1e-3 K, 1e-5 mol/L and 5e-3 K). Steps to trip under full
heating from the 135 corners are the same in float32 and float64 (derived,
same section).

The trip keeps every continuing state inside T < 365 K. The validity guard
(T <= 250 K, C_a < 0, or a non-finite state) is unreachable. With random
coolant and the drift within 3 standard deviations the lowest T is 310.0 K
and C_a stays within 0.33 to 0.96 mol/L (measured,
`test_states_stay_physical_on_the_reachable_set`). It exists for the
off-envelope sweep of conformance check 5, which moves each state field 75 %
either side of a reset state. Where one RK4 step from such a start overshoots
to a negative or non-finite state, the guard trips the proposal instead of
integrating it.

---

## 2. Validation targets

Computed with the env's own functions in float64 (the closed forms and
`jax.jacfwd` of `compute_velocity`) by the tests named, and printed by the
numbers script (`--section steady`, `--section targets`, `--section pnr`).
All derived.

| Quantity | Target | Model | Test | |
|---|---|---|---|---|
| Balances and ten values | cstr's | imported, bit for bit | `test_the_ten_parameters_are_cstrs` | ✅ |
| Low state at 300 K (APMonitor, read) | 0.87725294608097 mol/L, 324.475443431599 K | 0.877252946 mol/L, 324.4754434 K | `test_published_low_branch_state` | ✅ |
| Middle state at 299.709 K (Decardi-Nelson and Liu, read) | 0.483 mol/L, 350.970 K | 0.48277 mol/L, 350.969 K, a saddle | `test_published_middle_branch_states` | ✅ |
| Middle state at 299.413 K (same) | 0.465 mol/L, 352.000 K | 0.46460 mol/L, 351.998 K, a saddle | `test_published_middle_branch_states` | ✅ |
| Nominal middle state at 300 K | textbook 0.5 mol/L, 350 K | 0.499918 mol/L, 350.0055 K; eigenvalues −0.4542, +2.8344 /min | `test_nominal_middle_state` | ✅ |
| Steady-state multiplicity | 1, 3, 3, 1 states at 295, 300, 302, 305 K | the same, within 0.05 K of cstr's table | `test_steady_state_counts_and_roots` | ✅ |
| Folds | three states only between them | Tc 298.080 K (T 360.511 K) and 303.229 K (T 335.654 K) | `test_multiplicity_window_and_folds` | ✅ |
| Every target is a saddle | lambda+ > 0 | lambda+ 3.158 /min at 0.45 to 1.391 at 0.65 | `test_every_target_is_a_saddle`, `test_unforced_error_grows_at_the_saddle_rate` | ✅ |
| No P or PI loop on C_a alone stabilises | a closed-loop coefficient negative whatever the gain | det − tr/tau_j, the s coefficient under P and the s² coefficient under PI, is −28.95 /min² at 0.45 to −9.35 /min² at 0.65 (`--section targets`) | `test_no_p_or_pi_loop_on_c_a_stabilises` | ✅ |
| Ignited branch | cstr's table: "ignited only" at 305 K | unstable foci at 300 and 305 K; Hopf at Tc 306.22 K, T 379.61 K; at 310 K a stable focus at 383.89 K, above the trip | `test_upper_branch_is_an_unstable_focus` | ✅ |
| Energy invariant | T + A C_a constant in an adiabatic batch (q 0, UA 0) | moves by at most 1.7e-13 K over 64 steps to 99 % conversion from 0.1 mol/L at 340 K (`--section deviations`) | `test_energy_balance_closes_in_an_adiabatic_batch` | ✅ |
| Integrator convergence | `rk4_1` against 16 substeps | 9.16e-4 K and 2.74e-6 mol/L at most over an episode of the shipped cascade (`--section integrator`) | `test_rk4_1_matches_rk4_16` | ✅ |

**The textbook point is not an exact steady state.** At (0.5 mol/L, 350 K) and
a 300 K coolant the derivatives are +3.40e-5 mol/(L min) and −7.12e-3 K/min,
as Sandrock's notebook also prints; the exact middle root is (0.499918,
350.0055 K).

**Steady states** with the jacket held at the coolant (derived,
`--section steady`). Each entry is C_a (mol/L), T (K), the type, and the
trace of the two-state Jacobian (/min).

| Tc (K) | extinguished branch | middle branch | ignited branch |
|---|---|---|---|
| 295 | 0.926772, 317.742, stable node, −2.843 | none | none |
| 300 | 0.877253, 324.475, stable focus, −2.098 | 0.499918, 350.006, saddle, +2.380 | 0.208761, 369.705, unstable focus, +2.715 |
| 302 | 0.834785, 328.702, stable focus, −1.491 | 0.615767, 343.521, saddle, +1.244 | 0.170497, 373.647, unstable focus, +1.919 |
| 305 | none | none | 0.135196, 378.065, unstable focus, +0.587 |
| 310 | none | none | 0.099141, 383.888, stable focus, −1.989 |

The extinction fold is at Tc 298.080 K (T 360.511 K, C_a 0.3255) and the
ignition fold at Tc 303.229 K (T 335.654 K, C_a 0.7443). With the lag the
nominal middle state gains a third eigenvalue, −10.000 /min, and keeps the
other two.

**Correction to cstr's multiplicity table.** Its 300 and 305 K ignited states
are unstable foci (trace +2.715 and +0.587 /min). The branch turns stable only
past a Hopf point at a 306.22 K coolant (T 379.611 K, C_a 0.12455), and at
305 K with the trip disabled the plant settles on a limit cycle between
362.5 and 405.4 K (derived, `--section steady`, 300 min of
`compute_next_state`). cstr never goes there (its coolant stops at 302 K);
here it is why full heating always runs away to the trip.

**Targets** (derived, `--section targets`). T* and Tc* are the equilibrium
on target L; the eigenvalues are those of the two balances, and the lag adds
−1/tau_j = −10.000 /min without moving them (`test_every_target_is_a_saddle`).

| L (mol/L) | T* (K) | Tc* at dTi −6 / 0 / +6 K (K) | eigenvalues (/min) | 1/lambda+ (s) | lambda+ tau_j |
|---|---|---|---|---|---|
| 0.45 | 352.833 | 302.055 / 299.187 / 296.319 | −0.385, +3.158 | 19.0 | 0.316 |
| 0.50 | 350.001 | 302.869 / 300.001 / 297.133 | −0.454, +2.834 | 21.2 | 0.283 |
| 0.55 | 347.214 | 303.750 / 300.882 / 298.014 | −0.500, +2.422 | 24.8 | 0.242 |
| 0.60 | 344.415 | 304.613 / 301.745 / 298.877 | −0.526, +1.940 | 30.9 | 0.194 |
| 0.65 | 341.544 | 305.370 / 302.502 / 299.634 | −0.529, +1.391 | 43.1 | 0.139 |

**Point of no return** (derived, `--section pnr`, float64). The point of no
return is the lowest reactor temperature from which a full cooling command,
through the valve from the jacket's temperature, still reaches the trip
(`point_of_no_return`, bisection on the env's own step;
`test_the_point_of_no_return_is_reached_through_the_valve`). The
margin "in pure T" raises T alone, with C_a = L and the jacket at
Tc*(L, dTi). The margin "along the eigenvector" moves along the unstable
direction, where C_a falls as T rises (dC_a/dT −0.0072 to −0.0090
mol/(L K)). The reset corner is (L + 0.01 mol/L, T* + 1.5 K), the box's
hottest.

| L (mol/L) | PNR − T* in pure T, dTi −6 / 0 / +6 K (K) | along the eigenvector, dTi 0 (K) | reset corner margin, dTi 0 / +6 K (K) |
|---|---|---|---|
| 0.45 | 3.437 / 2.973 / 2.356 | 4.576 | 1.009 / 0.390 |
| 0.50 | 3.865 / 3.414 / 2.782 | 5.037 | 1.501 / 0.867 |
| 0.55 | 4.449 / 4.023 / 3.374 | 5.780 | 2.149 / 1.498 |
| 0.60 | 5.230 / 4.792 / 4.160 | 6.811 | 2.951 / 2.315 |
| 0.65 | 6.259 / 5.804 / 5.186 | 8.323 | 3.989 / 3.368 |

The margin grows with the target, so 0.45 mol/L is the tightest at every
drift (`test_point_of_no_return_at_the_hottest_target`).

**The line the controllers read.** `PNR_LINE` is (354.9, −40.7), meaning
T = 354.9 − 40.7 (C_a − 0.45) K. It was the supporting line under the exact
point of no return at dTi +6 K over C_a 0.40 to 0.70. **The valve has moved that
curve, and the frozen line now sits above it** by up to 0.254 K at +6 K, near
C_a 0.57, and stays 0.390 K below it at dTi 0 and 0.818 K at −6 K. The refit is
354.808 − 42.195 (C_a − 0.45) (derived, `--section pnr`). The line belongs to the
tuning, so PR 9 redid it (`test_pnr_line_is_conservative`, a strict xfail until
then). The PID's setpoint ceiling at 0.45 is the line less 0.3 K, 354.600 K or
T* + 1.767 K, and the line is the binding ceiling for targets up to 0.464 mol/L.

**Boundary to peak.** Started 0.01 K inside the point of no return, full cooling
turns the excursion at its peak 16 to 20 steps later at dTi 0 (0.80 to 1.00 min;
the highest peak is 363.75 K, at 0.45) and 20 to 21 steps later at dTi +6 K (1.00
to 1.05 min) (derived, `--section pnr`).

---

## 3. Parameter table

The reward's parameters (`e_floor`, `e_tol`, `tracking_exponent`,
`failure_cost`, `restart_steps`, `rho_floor_tracking`, `rho_floor`) are in the
reward table at the end, with their sources.

| Symbol | Value | Unit | Label | Source | |
|---|---|---|---|---|---|
| `q`, `V` | 100, 100 | L/min, L | read | cstr's `CSTRParams` (PC-gym, Decardi-Nelson and Liu Table 1, Seborg Table 2.3) | ✅ |
| `rho`, `C` | 1000, 0.239 | g/L, J/(g K) | read | same; `C` is deviation D1 | ⚠️ |
| `deltaHr` | −5.0e4 | J/mol | read | same | ✅ |
| `EA_over_R` | 8750 | K | read | same | ✅ |
| `k0` | 7.2e10 | 1/min | read | same | ✅ |
| `UA` | 5.0e4 | J/(min K) | read | same | ✅ |
| `Ti`, `Caf` | 350 K, 1.0 mol/L | | read | same | ✅ |
| `precision_floor` | 5e-4 | mol/L | read | cstr's online-analyser scatter, the ceiling's resolution | ✅ |
| `time_unit_seconds` | 60 | s per min | read | cstr's convention: `delta_t` in minutes | ✅ |
| `T_c_min` | 290 | K | read | PC-gym's CSTR action-space lower bound | ✅ |
| `T_c_max` | 310 | K | ours | spans both folds, with at least 4.63 K headroom at \|dTi\| 6 K (derived, `--section targets`) | ⚠️ |
| `tau_j` | 0.1 (6 s) | min | synthetic | not hardware: the longest lag the cascade survives, and it needs about 240 L/min through a 25 L jacket (section 6) | ⚠️ |
| `actuator_lag` | 2/60 (2 s) | min | read, for the owner to accept | the "Nominal" class of [EnTech's Control Valve Dynamic Specification 3.0](https://www.academia.edu/21406639/Control_Valve_Dynamic_Specification) (11/98): a 2 s time constant, T86 4 s, for a loop of about 10 s, this loop's scale, since lambda+ is 19 s at its fastest. The 4-inch assemblies B and C in the [Emerson Control Valve Handbook, 6th ed.](https://www.emerson.com/documents/automation/control-valve-handbook-en-3661206.pdf) (2023), Valve Response Time table, reach 63 % in 1.14 to 1.63 s on 10 % steps and 2.0 to 7.1 s on 5 % steps. The valve standing in front of a coolant temperature is ours | ⚠️ |
| `actuator_rate` | 4.8 (8 %/s, 12.5 s full stroke) | span per min | derived from read, for the owner to accept | same table, assemblies B and C on 10 % steps, opening and closing: 6.3 % of span over T63 − T(d) of 0.72 to 0.94 s is 6.7 to 8.8 %/s, rounded to 8. Assembly A, a specified valve, does 27 to 45 %/s. This is the mean speed on a 10 % step, a lower bound on the top speed, and full-stroke tests, which that handbook says do not show control performance, are not used. EnTech's own limit, T86 4 s up to a 10 % step, only implies at least 2.2 %/s, which would leave the worst reset 0.17 K past the point of no return at dTi +6 K, so a spec's floor is not used as a typical figure | ⚠️ |
| `T_trip` | 365 | K | ours | high-high interlock, 4.49 K above the extinction fold and 12.17 K above the hottest target (derived) | ⚠️ |
| `T_valid_min` | 250 | K | ours | validity guard, unreachable (section 1) | ✅ |
| `Ti_sigma` | 2.0 | K | ours | stationary standard deviation of the feed drift | ⚠️ |
| `Ti_tau` | 10 | min | ours | feed-drift correlation time | ⚠️ |
| `Ti_dev_reset_clip` | 3.0 | sd | ours | clip on the initial drift draw (section 5) | ⚠️ |
| `target_CA_range` | (0.45, 0.65) | mol/L | ours | the middle branch, 5.89 K from the ignition fold and 7.68 K from the extinction fold (derived) | ⚠️ |
| `switch_sizes` | (0.05, 0.05, 0.10, 0.10, 0.10) | mol/L | ours | a fixed multiset, permuted per episode | ⚠️ |
| `target_ramp` | 0.1/3 (1/600 mol/L per step) | mol/L per min | ours, set before any result | the largest switch takes three residence times, V/q = 1 min, when a mixed tank has gone 1 − exp(−3) = 95 % of the way to a new steady state. T* then moves at 1.9 K/min, under the expert's 3 K/min `sp_rate`, which no longer does the ramping | ⚠️ |
| `block_steps` | 200 (10 min) | steps | ours | 3 x the settle of the cascade on `DEFAULT_GAINS` after its worst switch (62 steps or 3.10 min, 0.55 to 0.65 mol/L), rounded up to whole minutes, with a 6 min floor (derived, `--section settle`). The shipped gains settle in 60 steps, which would give 9 min. | ⚠️ |
| `initial_CA_offset` | 0.01 | mol/L | ours | reset box half-width in C_a | ⚠️ |
| `initial_T_offset` | 1.5 | K | ours | reset box half-width in T | ⚠️ |
| `delta_t` | 0.05 (3 s) | min | ours | lambda+ dt = 0.16 at the fastest target (derived) | ✅ |
| `max_steps_in_episode` | 1200 (60 min) | steps | ours | six blocks of `block_steps` | ✅ |
| `CA_error_max` | 0.55 | mol/L | derived | max(Caf − 0.45, 0.65 − C_a at 365 K on the steady-state curve) = max(0.55, 0.386), with C_a at 365 K 0.2636 mol/L (`--section params`, `test_the_error_envelope_is_certified`); the version-1 envelope and the base of `failure_cost` | ✅ |
| `reward_version` | 2 | | convention | | ✅ |
| `floor_is_documented_minimum` | False | | convention | the plant is disturbed | ✅ |

The ten reactor values are cstr's and carry its sources. Everything below them
is this task's own choice, and is therefore where to look first when something
is unreachable, as cstr's contract says of its own lower half.

---

## 4. Figures of merit

All derived by the numbers script (`--section params`, `--section targets`,
`--section reach`) and, where a test is named, asserted by it.

| figure | value | source |
|---|---|---|
| Adiabatic temperature rise A Caf, A = −deltaHr / (rho C) | 209.205 K (A = 209.205 K L/mol) | `--section params` |
| Jacket coupling beta = UA / (rho C V) | 2.09205 /min. At a fixed target Tc* moves by −(q/V) dTi / beta, which is −dTi / 2.092 since q/V = 1 /min | `test_steady_coolant_is_an_equilibrium` |
| Residence time V/q | 1.00 min | `--section params` |
| Uppal-Ray-Poore groups, all dimensionless: gamma = EA_over_R / Ti, B = gamma A Caf / Ti, the heat-transfer group H = UA / (q rho C), Da = k0 exp(−gamma) V/q | gamma 25.000, B 14.943, H 2.0921, Da 0.99993 | `--section params` |
| Open-loop growth rate lambda+ over the band | 1.391 to 3.158 /min (43 to 19 s), a 2.3x swing | `test_every_target_is_a_saddle` |
| Steady coolant Tc* over the band, dTi 0 | 299.187 to 302.502 K | `test_steady_coolant_is_an_equilibrium` |
| Coolant headroom over the band | 7.50 K at dTi 0, 4.63 K at dTi −6 K and 6.32 K at dTi +6 K | same test |
| Full heating from the 135 reset corners (five targets, 3 x 3 corners, dTi −6, 0 and +6 K) | trips in 5 to 20 steps, median 10 (15 to 60 s); from 5 to 13 steps at 0.45 up to 9 to 20 at 0.65 | `--section reach`, `test_full_heating_trips_from_every_corner` |
| Full cooling for 60 min from the same corners | no trip; the hottest point is 355.63 K | `--section reach`, `test_full_cooling_never_trips` |
| Re-ignition from the low state at a 300 K jacket (324.48 K) under full heating | 340 K after 19 steps, trip after 26, so 21 s between them | `--section reach`, `test_extinction_is_recoverable` |
| Zero action (a 300 K jacket) from T* ± 0.5 K | leaves ±2 K in 6 to 28 steps; the starts 0.5 K above T* at 0.45 and 0.50 trip at steps 13 and 18, the other eight extinguish to 324.5 K | `--section reach`, `test_zero_action_leaves_the_middle_state` |
| Restart | cstr's 240 steps of 0.25 min are 1200 steps at 3 s | `--section params`, `test_restart_time_is_cstrs_hour` |
| Drift per step | a = exp(−delta_t / Ti_tau) = 0.99501; innovation sd 0.1995 K | `--section params` |

---

## 5. Task design

### Task: `unstable_cstr`

Hold the concentration of A in a jacketed exothermic reactor on its open-loop
unstable middle steady state, commanding the coolant through a lagging jacket
while the feed temperature drifts unmeasured. Left alone the reactor runs away
within tens of seconds, and the target steps five times per episode, so every
move crosses the unstable region under control. A trip costs an hour of
restart.

| obs | meaning |
|---|---|
| 0 | `C_a`, concentration of A (mol/L) |
| 1 | `T`, reactor temperature (K) |
| 2 | `T_j`, jacket temperature (K) |
| 3 | target `C_a`, on its ramp to the current block's level (mol/L) |

**Observation.** Hidden: the drift `Ti_dev`, the
later levels and the block clock. C_a is read instantly, with the analyser's
noise (D3, section Readings), and T through its thermowell, 10 s behind (section
Readings). T is observed because no P or PI loop on C_a alone can stabilise
the plant (`test_no_p_or_pi_loop_on_c_a_stabilises`). On the linearisation
with the lag, a loop on C_a alone leaves one coefficient of the closed-loop
polynomial at det − tr/tau_j, which is −28.95 at 0.45 and −9.35 at 0.65
(derived, `--section targets`).

**Action.** One entry, raw in [−1, 1] mapped linearly to a coolant command in
[290, 310] K, raw 0 = 300 K, delivered by a valve with a 2 s lag and a slew limit of
8 % of its span per second (`actuate`, section 3). A reset or a restart leaves the
valve where the jacket is (`initial_action`), so holding moves nothing. The two signs disagree
(`test_short_term_and_static_signs_are_opposite`). Raising the coolant by
1 K lowers C_a within a step or two, by 1.0e-5 to 1.5e-5 mol/L after one step
and 7.4e-5 to 1.2e-4 after two, since the reactor heats and burns faster.
Holding a higher C_a needs a warmer jacket, with Tc* rising by 13.4 to
17.8 K per mol/L across the band (derived, `--section targets`).

**Schedule.** Six 10-minute blocks. The target ramps from one level to the next
at 1/30 mol/L per minute, so a 0.10 switch takes 3 minutes and a 0.05 switch 1.5,
and holds for the rest of the block
(`test_the_target_ramps_from_level_to_level`). The level is reached by the plant's
own schedule, so every policy gets the ramp, not only one with an `sp_rate`. The
first level is uniform in (0.45, 0.65) mol/L. The five switches take the sizes 0.05, 0.05, 0.10, 0.10 and 0.10
mol/L in a random order, each with a fair-coin sign that is flipped when the
move would leave the band (`test_reset_draws_a_stratified_schedule`). The
sizes are fixed so that every episode has the same total squared move,
0.035 (mol/L)² (derived), which keeps the spread of returns across seeds
small. Where each target is drawn on its own, as in cstr, the return is set
mostly by how far the target sits from the start: compare the spread of the
recorded returns over the 10 seeds of
[cstr](../../data/baseline_returns/cstr.json) and of
[this task](../../data/baseline_returns/unstable_cstr.json).

**Scoring at a switch.** The reward and the observation of one state use the
same live target (`scheduled_level`). The step that enters a new block is scored
against the target's first ramp step, which that step's observation already shows
(`test_live_target_follows_the_block_clock`). The last level holds past step 1200.
`benchmark.target_changes` counts a ramp once, where it starts, so a cycle's fixed
settle of 99 steps now runs from the start of the ramp, 39 steps past the end of the
longest one. PR 8 sets the hold window again.

**Reset.** A warm start. C_a is within 0.01 mol/L and T within 1.5 K of the
first level's equilibrium, the jacket is already at Tc*(L0, dTi0), and the
initial drift is drawn from its stationary law clipped at 3 standard
deviations (6 K) (`test_reset_is_warm_with_a_clipped_drift`). Every reset is
recoverable. The worst corner of the box (C_a 0.46 mol/L, T*(0.45) + 1.5 K)
is 1.009 K inside the point of no return at dTi 0 and 0.390 K inside at
dTi +6 K, and would cross it only at dTi +8.69 K (4.34 sd), which the clip
excludes (derived, `--section pnr`;
`test_every_reset_is_inside_the_point_of_no_return`).

**Trip and restart.** The trip is one-sided. T >= 365 K trips the plant, as
does the validity guard, and there is no low trip. The extinguished branch,
below the ignition fold at 335.654 K, is charged as tracking error and is
always recoverable, since at a 310 K coolant the only steady state (383.89 K)
lies above the trip and full heating re-ignites the plant from the low state
in 19 steps (section 4). A trip is part of the kernel (`failure_kernel`). The
step is charged the trip cost, and the plant restarts from a fresh draw of
`reset_env` (with `continuing = False` the episode ends there instead), with a new schedule and drift and
the block clock at 0, while `time` runs on
(`test_a_trip_restarts_fresh_on_the_same_clock`). The draw is keyed on the
step (`base.restart_key`), so even under a constant rollout key, as every
shipped rollout helper passes, a run of trips does not replay the episode's
initial state.

**The drift.** A zero-mean Ornstein-Uhlenbeck process in the feed temperature,
sd 2 K, correlation time 10 min (`test_feed_drift_is_a_zero_mean_ou_process`).
Over 64 seeds of 1200 steps its mean is −0.028 K (3 standard errors 0.37 K),
its RMS 2.005 K and its lag-1 autocorrelation 0.99518 against the 0.99501 of
the law (measured, `--section disturbance`). It moves the steady coolant by
−(q/V) dTi / beta = −dTi / 2.092 K, so a feedforward from the target alone
leaves an offset. It also narrows the point-of-no-return margin on its hot
side, from 3.304 K to 2.476 K at 0.45 mol/L and dTi +6 K. One innovation
(0.200 K) with the coolant held moves C_a by 1.0e-5 mol/L after one step and
3.1e-4 after five at 0.45 (derived), and under the shipped cascade its
largest effect on C_a is 9.8e-5 mol/L (measured), both from
`--section disturbance`.

**What the task adds to the suite.** A process plant held on an open-loop
unstable operating point while its reference moves, 2.4 to 6.3 K below a
point of no return at the lag and drifts it runs at (section 2), where a
failure is one-sided and costs an hour of downtime. On its validation
episodes the shipped cascade comes no closer than 1.336 K to the exact point
of no return, and the MPC no closer than 1.519 K (measured, section 7).

---

## 6. Jacket lag sensitivity

**Written exception under the credibility check on sourced figures, accepted by the
owner on 2026-11-06.** `tau_j` of 6 s is synthetic: no published jacket gives it. It
stays because this section shows what it changes.

**The jacket is synthetic.** `tau_j` is not a hardware figure. 6 s is the longest
lag at which the cascade held, chosen from the tables below. Real hardware does
not give it: a mixed jacket has 1/tau_j = F/V_j + UA/(rho_w c_w V_j), so with the
25 L this reactor's size suggests (a [100 L glass reactor](https://www.torontech.com/pilot-scale-jacketed-glass-reactor/)
lists 20 L) and water at 4184 J/(L K), 6 s takes F = 25 (10 − 0.48) = 238 L/min.
That is 2.4 times the reactor's own throughput, pumped through its jacket. At
60 L/min, for comparison, it would lag about 21 s, where this table has resets past
the point of no return. So `tau_j` stays at 6 s because the task needs it, and no
result on this plant says what a real jacket would do. It needs the owner's written
exception under check 2 of `docs/proposals/industrial-fixes.md`.

These tables predate the thermowell and the valve (`--sensitivity`). lambda+ is 3.158 /min at 0.45 at
every lag, since the lag adds its own eigenvalue and moves neither of the
plant's.

The plant at each lag (derived, float64; the full-heating counts are float32
steps from rest):

| lag | lambda+ tau_j at 0.45 | PNR − T* at 0.45, dTi 0 / +6 K (K) | worst reset corner margin, dTi 0 / +6 K (K) | resets past the PNR, of 20000 draws | full heating to trip from rest at 0.45 / 0.65 (steps) | stabilising P-only inner gain at 0.45 (K per K) |
|---|---|---|---|---|---|---|
| 0 s | 0 | 4.58 / 3.38 | +2.61 / +1.42 | 0 % | 5 / 10 | 1.41 to 19.18 |
| 3 s | 0.158 | 3.81 / 2.85 | +1.85 / +0.88 | 0 % | 6 / 11 | 1.65 to 15.58 |
| 6 s | 0.316 | 3.30 / 2.48 | +1.34 / +0.51 | 0 % | 7 / 12 | 2.25 to 8.77 |
| 12 s | 0.632 | 2.65 / 1.99 | +0.69 / +0.02 | 0 % | 8 / 14 | none |
| 20 s | 1.053 | 2.13 / 1.59 | +0.17 / −0.37 | 0.005 % | 9 / 15 | none |
| 30 s | 1.579 | 1.74 / 1.29 | −0.23 / −0.68 | 0.110 % | 10 / 17 | none |

The cascade at each lag (measured, float32 episodes: 256 scheduled episodes,
256 one-hour holds per target and 256 episodes of the 0.55/0.45 alternation at
0 to 6 s, 1024 of each at 12 to 30 s; the closed-loop rates are derived from
the linearisation, and a positive rate grows). "Re-tuned" is a coordinate
search on episode return (33 evaluations of 24 seeds), run under the guard
from `DEFAULT_GAINS` at that lag. The best rates are over a grid of 18816 gain
sets, and the best P-only inner loop has Kd_T = 0.

| lag | `DEFAULT_GAINS`: episodes with a trip (rate at 0.45, /min) | re-tuned Kp_T, Kd_T, Kc_Ca, Ti_Ca | re-tuned: episodes with a trip | 1 h holds and 0.55/0.45 alternations that trip | mean abs error, minutes 6 to 8.4 of a block (mol/L) | v2 cost per step | rate at 0.45 (/min): re-tuned / best / best P-only | smallest PNR margin (K) |
|---|---|---|---|---|---|---|---|---|
| 0 s | 0 % (+4.68) | 7, 0.1, 98, 0.35 | 0 % | none | 1.78e-4 to 1.91e-4 | 5.582e4 | −2.78 / −8.16 / −8.16 | 2.23 |
| 3 s | 0 % (−1.71) | 14, 0.2, 100, 0.5 | 0 % | none | 9.8e-5 to 1.09e-4 | 5.647e4 | −0.84 / −6.50 / −2.67 | 1.90 |
| 6 s | 0 % (−1.73) | 9.8, 0.4, 140, 0.35 | 0 % | none | 1.13e-4 to 1.71e-4 | 5.752e4 | −0.59 / −4.34 / −0.42 | 1.59 |
| 12 s | 72.2 %, 5.5 trips each (+1.13) | 7, 0.8, 70, 0.49 | 0 % | 0.3 % at 0.45 | 3.11e-4 to 3.54e-4 | 6.086e4 | −1.69 / −3.63 / +0.98 | 0.56 |
| 20 s | 100 %, 24.7 trips each (+2.09) | 2.45, 1.6, 70, 1 | 0.1 % | 1.8 % at 0.45, 0.3 % at 0.50 | 1.30e-3 to 1.38e-3 | 5.991e6 | −0.76 / −3.50 / +1.40 | 0.35 |
| 30 s | 100 %, 29.8 trips each (+2.45) | 1.75, 1.6, 50, 1 | 0.5 % | 5.7 % at 0.45, 1.5 % at 0.50, 0.1 % at 0.55; 0.1 % of the 0.55/0.45 alternation | 2.50e-3 to 2.80e-3 | 2.946e7 | −0.75 / −3.35 / +1.51 | 0.24 |

**Why 6 s.** It is the longest lag on this grid at which the cascade holds
every target for an hour without a trip. At 6 s both `DEFAULT_GAINS` and the
re-tuned gains run every scheduled episode trip-free, and the re-tuned gains
are the shipped ones. At 12 s the re-tuned cascade still runs 1024 scheduled
episodes without a trip, but 0.3 % of one-hour holds at 0.45 mol/L trip and
its closest approach to the point of no return falls to 0.56 K. A slower
1.5 K/min setpoint ramp at 12 s leaves the hold at 0.45 tripping on 0.3 % of
hours as well, with a smallest margin of 0.58 K, and lowers version 1 from
0.7336 to 0.6790 per step. A 12 s jacket would therefore need the band to
start above 0.45 mol/L. From 20 s on, some resets start past the point of no
return.

6 s is also where derivative action on T becomes necessary at the hot end. A
P-only inner loop still stabilises 0.45 mol/L at 6 s, but its best rate
there is −0.42 /min, slower than the guard's 0.5 /min, and from 12 s no
P-only gain stabilises it at all.

---

## 7. Experts

**The recipe on this plant** (section 9 of
`docs/proposals/expert-protocols-process.md`, run by
`scripts/tune_process_experts.py --envs unstable_cstr`). Figures from here to the
next heading are measured on the plant with its instruments. Older figures below
this block predate them.

- **Pairing.** One coolant, so there is no pairing to choose, and the RGA has
  nothing to say. The check is which reading closes the inner loop: the unstable
  mode, 2.42 /min at 0.55 mol/L, reads 4.2 noise bands per unit on T and 2.6 on C_a,
  so T closes it, as the cascade does.
- **Inner PD.** SIMC on the datasheet: slope UA / (rho C V) = 2.09 K/min per K, lag
  0.217 min (the thermowell and half the jacket), delay 0.108 min (the other half of
  the jacket, the valve and half a step). Kp_T = 2.21, Kd_T = 0.478. A closed test
  over 36 s read 12 K/min/K at a fit of 60 %, since a gain and a lag draw one curve
  there, so the datasheet stands. No P gain tried holds this plant: it needs the D.
- **Outer PI.** C_a on the setpoint, inner loop closed: k = -0.0195 mol/L per K,
  tau1 = 0.234 min, theta = 0.30 min. Kc_Ca = 26.6, Ti_Ca = 0.345 min, Tf_Ca = 0.033
  min. The spread of 1.47 stays under C5's limit, so the median stands.
- **Limits.** Tc_bias 300.9 K, window 4.73 K each side, line margin 2.07 K, ramp 9.7
  K/min. `PNR_LINE` is refit below the exact point of no return with the valve in.
- **Held-out check** (seeds 200 to 209). It holds 1.3e4 per step (worst seed 4.7e4),
  with 8 trips on 4 seeds, from 4.1e5 and 407 trips on the gains that predate the
  instruments. The best null is constant 0, ratio 0.008, 4 of 10 wins, so it passes
  the gate.
- **Findings.** `check_cascade_gains`, which now sees the thermowell and the valve,
  reports the closed loop growing at +0.36 /min at 0.45 mol/L and three equilibrium
  setpoints outside the window. D2 makes that a report. Trips restart cycles, so the
  cycle and doubling rules also fail (`tests/test_episode_length.py`).

**Cascade PID** (`make_unstable_cstr_pid`). An outer PI on C_a sets a
rate-limited temperature setpoint around the feedforward T*(L), clipped
between T* − 8 K and min(T* + 2 K, the point-of-no-return line at L − 0.3 K);
an inner PD on T sets the coolant. The setpoint starts at the measured T
(`cascade_init`). On the 50 catches from the extinguished branch that
`--pid-validation` runs, a setpoint started at T* instead trips 98 times in
42 of the 50 episodes (measured). Four gains are tuned. The setpoint limits
are held, since their right value is set by the distance to the point of no
return, which the return sees only once a trip happens.

A stability guard (`check_cascade_gains`) reports gains whose linearised
closed loop decays slower than `GUARD_MIN_DECAY` (0.5 /min, ours) at any of
the five targets, or whose equilibrium setpoint falls outside the setpoint
window at a drift of −6, 0 or +6 K
(`test_guard_refuses_unstable_and_marginal_gains`,
`test_shipped_gains_pass_the_guard`). Without it, the tuner's search at 6 s
ends at Kp_T 14, Kd_T 0.28, Kc_Ca 70, Ti_Ca 0.35, whose linearised loop grows
at +0.44 /min at 0.45. Those gains run 256 episodes without a trip, but they
put the coolant on a bound on 4.7 % of steps and hold 0.45 mol/L only to
1.95e-3 mol/L over minutes 6 to 8.4 of a block, about 11 times the guarded
gains' 1.71e-4 (measured, `--sensitivity`, 6 s row).

The shipped gains are SIMC settings (Skogestad and Grimholt, "The SIMC
method for smooth PID controller tuning", 2012) from step tests on the
simulator at the levels of the training seeds 100 to 109, derived and stored
in `src/target_gym/data/pid_gains.json` under `unstable_cstr` by
`scripts/tune_process_experts.py --envs unstable_cstr`. They replaced a
coordinate search on episode return over seeds 0 to 23, which included the
scored ones.

- **Inner loop.** A 1 K coolant step from the equilibrium at each of the 37
  training levels, read over its first 12 steps against an unstepped run on
  the same drift. The fit `k' / ((s - a)(tau2 s + 1))` gives, at the median,
  a slope k' = 2.34 K/min per K, an unstable pole a = 3.38 /min, the jacket
  lag tau2 = 0.111 min and, with half the 3 s sample, theta = 0.025 min
  (measured). 1/a is 11.8 theta, above SIMC's 8 theta, so the loop is
  lag-dominant and takes the integrating-with-lag rule with tau_c = theta:
  Kp_T = 1 / (k' (tau_c + theta)) = 8.49 and Kd_T = Kp_T tau2 = 0.946 min.
  The integral is left to the outer loop, as the cascade had it.
- **Outer loop.** A 0.5 K step of the temperature setpoint at every third
  training level, with the inner PD closed. The median first-order fit is
  k = -0.0185 mol/L per K, tau1 = 0.471 min and theta = 0.139 min. SIMC with
  tau_c = theta gives Kc_Ca = 92.0 K per mol/L and Ti_Ca = 0.471 min.
  2 theta and 3 theta cost the same on the training seeds to within 0.03 %,
  so the tight default stays.

The closed-loop rates at 0.45, 0.50, 0.55, 0.60 and 0.65 mol/L are −1.92,
−2.05, −2.22, −2.46 and −2.52 /min, and the drift does not change them
(derived, `closed_loop_rates`, `--pid-validation` and `--section
invariance`). The smallest margin of an equilibrium setpoint inside its
window is 1.37 K. On the held-out seeds 200 to 209 the cascade holds 6.0 per
step (worst seed 9.0) against a ceiling of 1 and never trips
(`tests/pc_gym/test_expert_specs.py`); the searched gains held 3.2 there.

**PID validation** (`--pid-validation`, 300 episodes, float32 episodes and
float64 margins, measured). The margin is to the exact point of no return;
the line margin is to `pnr_line`.

| episodes | trips | re-extinctions after capture | smallest PNR margin (K) | smallest line margin (K) | mean abs error, minutes 6 to 8.4 of a block (mol/L) | v2 cost per step |
|---|---|---|---|---|---|---|
| 200 random schedules, drift on | 0 | 0 | 1.320 (p1 1.534, median 2.384) | 0.322 | 1.95e-4 | 6.124e4 |
| 50 of the 0.55/0.45 alternation, dTi +6 K held | 0 | 0 | 1.422 | 0.743 | 2.23e-7 | 8.853e4 |
| 50 catches from extinction, dTi −6, 0 or +6 K held | 0 | 0 | 1.396 (median 3.491) | 0.714 | 3.27e-2 | 1.505e6 |

No step goes above the line. The smallest margin, 1.320 K, is above the
0.5 K the numbers script requires before it keeps `GUARD_MIN_DECAY` at
0.5 /min. `test_pid_holds_every_target` builds and runs its own 130 episodes
the same way: the first 64 random schedules, the first 16 alternating ones
and the same 50 catches. It asserts that no episode trips, that each brings
C_a within 0.01 mol/L of its target, that T stays above the ignition fold
(335.654 K) from then on, that T is at most 0.1 K above `pnr_line(C_a)` on
every step, and that at the eight steps of each episode closest to that line
the exact point of no return is at least 0.25 K above T.

**MPC** (`make_unstable_cstr_mpc`). The MPC slot is a problem definition,
not a tuned controller (docs/proposals/oracle-audit.md): the shared solver's
oracle (`target_gym.oracles`). It reads no expert code and no gains. Once
every step (Cooper's sample time from the horizon protocol,
`data/oracle_horizons/unstable_cstr.json`, measured on the stabilised plant
as its section 4.4 sets out), it plans P = 76 moves of the raw coolant from
the true state, fewer near the episode's end, and applies the first. The plan
maximises the episode reward over the horizon on the environment's own
dynamics, with the feed drift on its noise-free mean path, a^k dTi
(`plan_params`). The command's clip is replaced by a bound on the move, and
the tracking term against the schedule's live level is written as
(e / e_floor)^2, the reward itself since e_tol is 0 and p is 2. The trip,
T_valid_min < T < T_trip and C_a >= 0, is a hard limit on every planned
step, elastic at the trip cost per unit, planned inside its bounds by one
innovation of the feed drift at its 1 % quantile (K11: 0.055 K on T,
`data/oracle_widths/unstable_cstr.json`). The plan's last state must be an
equilibrium under its last move (the horizon protocol's terminal line-out,
D4): the change of each state over one more step is held at zero, elastic
at the trip cost per unit. Nothing is smoothed. IPOPT's algorithm solves it
with exact Hessians to a scaled KKT error of 1e-8, or stops at 150
iterations. A cold start puts every planned state at the measured one and
holds the jacket where the reset put it (simulating that held jacket ahead
would run the open-loop unstable reactor away), a later
solve starts from the last plan shifted one step, and a warm solve that fails or hits the cap is solved again cold,
and one that still fails applies the last plan's next move. Every solve's certificate (KKT error,
iterations, violation, and the plan replayed on `step_env`) is summed into
the record.

Until 0.8 the oracle was an iterative LQR over 32 steps (1.52 times the
longest boundary-to-peak time), guided by the cascade PID, with a barrier on
T past `pnr_line(C_a)` that was added after one trip on the recorded seeds.
The rule removes both: the horizon comes from the protocol, and the only
limit is the task's own trip.

**Recorded results.** Both controllers' returns over 10 seeds
(`scripts/record_baselines.py`) are on the
[environment page](../../../../docs/environments/unstable_cstr.md#baselines), and their
protocol row over 3 seeds (`scripts/evaluate_baselines.py`: gain, hold, reach
and transient cost per change, failure rate) is on the
[results page](https://yannberthelot.github.io/TargetGym/baseline-results/). Both are generated from the records in
`src/target_gym/data/`, so neither is copied here to go stale. The MPC's solver
statistics are in its row of
`src/target_gym/data/baseline_returns/unstable_cstr.json`. The gain is the mean
cost over every step.
The hold leaves out the
steps in which the MPC already moves toward the next level, which
`target_gym.eval` finds from the cost and counts in the reach and transient
cost of the change they prepare. The reach cost is a controller's cost above
its own hold level from a change until it settles. The transient cost sums
the cost over the plant's burn-in on each side of a change, and with no
burn-in here that is the first step of each block and, where the MPC already
moves toward the new level, the step before it.

---

## 8. Version-1 reward and its exploit

Version 1 is ours. It is `log_scaled_reward` of |target − C_a| between
`precision_floor` and `CA_error_max`, 1 at zero error and 0 at 0.55 mol/L, and
**exactly −1200 (`-restart_steps`) on a tripped step**
(`test_v1_is_log_scaled_and_charges_downtime`). It is therefore not bounded
in [0, 1] on this task.

With 0 on the tripped step, a policy that trips and restarts warm near the
target scores above a trip-free constant coolant, since each restart puts the
plant back beside its level for free. Charging the downtime at version 1's best
per-step score, as version 2 prices it, restores the order.
TargetFoundation should report trips beside version 1.

The policies, on 128 seeds (measured, `--v1`, float32). "v1, 0 on a trip" is
the unfixed score; "v1 as shipped" charges −1200.

| policy | v1 per step, 0 on a trip | trips per episode | v2 cost per step | v1 per step as shipped |
|---|---|---|---|---|
| (a) full heating, 310 K | 0.4117 | 134.27 | 8.123e9 | −133.85 |
| (b) full cooling, 290 K | 0.0432 | 0 | 1.6e7 | 0.0432 |
| (c) cascade on `DEFAULT_GAINS` | 0.7623 | 0 | 5.794e4 | 0.7623 |
| (c') shipped cascade | 0.7933 | 0 | 5.75e4 | 0.7933 |
| (d) constant Tc*(0.45) = 299.19 K | 0.1185 | 13.88 | 8.488e8 | −13.76 |
| (d) constant Tc*(0.50) = 300.00 K | 0.1541 | 21.75 | 1.324e9 | −21.60 |
| (d) constant Tc*(0.55) = 300.88 K | 0.2202 | 34.94 | 2.119e9 | −34.72 |
| (d) constant Tc*(0.60) = 301.75 K | 0.2669 | 46.30 | 2.805e9 | −46.04 |
| (d) constant Tc*(0.65) = 302.50 K | 0.3381 | 60.95 | 3.689e9 | −60.61 |
| (e) zero action, 300 K | 0.1540 | 21.75 | 1.324e9 | −21.60 |
| best constant under v1 with 0 on a trip, 304.75 K (0.25 K scan) | 0.4216 | 94.34 | 5.707e9 | −93.92 (derived: the unfixed score less the trips per episode) |
| best trip-free constant, 294.00 K | 0.0496 | 0 | 1.451e7 | 0.0496 |

With 0 on a trip, each of the eight tripping policies outscores both
trip-free constants (full cooling and 294.00 K), 16 pairs in all, and the
Spearman rank correlation with version 2 over the 12 policies is
−0.133. A tripped step must score below −6.5 to reverse every pair. The
shipped −1200 reverses them all, and the rank correlation becomes 1.000 (zero
action and Tc*(0.50) are one policy). `test_v1_ranks_tripping_below_safe`
checks the order on 32 seeds with full heating, a 305.75 K constant, a
295.25 K constant and the shipped PID.

---

## 9. Known deviations

Each carries a strict xfail that states what a real plant does and fails the
suite if the model starts doing it. D1 and D4 are in
`tests/pc_gym/unstable_cstr/test_unstable_cstr_physics.py`, and D2 and D3 in
`tests/pc_gym/unstable_cstr/test_unstable_cstr_env.py`. The numbers below
are what the model gives in each test.

**⚠️ D1. The heat capacity is far below an aqueous feed's.** C = 0.239
J/(g K), about a seventeenth of water's 4.184 (read), gives an adiabatic rise
A Caf of 209.2 K, where an aqueous feed with this heat of reaction would rise
12.0 K (derived, `--section deviations`).
`test_adiabatic_rise_matches_an_aqueous_feed` measures the rise by stepping
an adiabatic batch, gets 209.2 K, and asks for at most 20 K. The value is in
Seborg, PC-gym, Decardi-Nelson and Liu and Bequette alike, and it is what
gives the model its multiplicity at these temperatures, so it is kept.

**⚠️ D2. No jacket energy balance.** The jacket temperature follows the command
through a fixed lag, whatever heat it removes. A real jacket warms during a
runaway at a held command. In `test_jacket_heats_in_a_runaway` the command is
held at the jacket's temperature from 2 K above T* at 0.45 mol/L. T climbs
from 354.83 to 364.82 K in 7 steps while T_j moves by 0.0 K, where the test
asks for more than 0.1 K (measured).

**⚠️ D3. No analyser dead time.** C_a is observed at once. An online analyser
reports a sample a minute or more after it was drawn. In
`test_analyser_has_dead_time` two plants 0.01 mol/L apart in C_a give
observations 9.06e-3 mol/L apart after the first step, where a 1 min dead
time would keep them identical for 19 steps (measured).

**⚠️ D4. No feed-concentration disturbance.** `Caf` is constant; only the feed
temperature drifts. `test_feed_composition_moves_the_equilibrium` turns the
temperature drift off and steps each target's equilibrium 20 times under
fresh keys. C_a moves by at most 3.3e-16 mol/L, float64 rounding, where the
test asks for more than 1e-6 mol/L, 1 % of the analyser's resolution
(measured).

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `unstable_cstr` | 1200 | 3 | 60 min | 1 in [-1, 1] | 4 | 13 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`, see docs/reward-shaping.md. There
is no running term, since the coolant temperature is a setting with no
tariff.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 5e-4 mol/L | the online analyser's scatter (read, cstr's `precision_floor`), kept because the MPC holds finer. A measured upper bound clamped at the instrument (docs/reward-shaping.md, "How each floor was obtained"). `scripts/measure_hold.py --envs unstable_cstr` records the MPC's mean absolute C_a error over the hold of each block (the hold window below), and the floor is max(5e-4, the lowest per-seed hold). That hold is 6.29e-6 mol/L (measured; the PID's lowest is 1.23e-4), so the clamp binds. |
| `e_tol` | 0 | no product specification for the concentration has been supplied |
| `tracking_exponent` | 2 | quadratic: the owner's cost is off-spec product |
| `failure_cost` | 2.42e6, twice (0.55 / 5e-4)^2 | twice the largest tracking cost an untripped state reaches, with 0.55 mol/L the closed-form `CA_error_max` (derived, `--section params`) |
| `restart_steps` | 1200 (1 h) | cstr's one-hour restart (240 steps of 15 s, marked provisional in cstr) at 3 s steps; a trip costs 1200 x 2.42e6 = 2.90e9 (derived, `test_restart_time_is_cstrs_hour`) |
| `rho_floor_tracking` | 1.99e-5 | (lowest per-seed MPC hold / `e_floor`)^2 = (2.23e-6 / 5e-4)^2, from `scripts/measure_hold.py` (measured; 0.00396 from the CasADi oracle's 6.29e-6 before v3); below 1 because the clamp binds |
| `rho_floor` | 1.99e-5 | equal to `rho_floor_tracking`, since there is no running cost |
| `floor_is_documented_minimum` | False | the plant is disturbed; documented minima are for plants with no disturbance at all |

**The hold window.** `scripts/measure_hold.py` scores this plant's hold
block by block. A fixed settle of 120 steps starts each block's hold at minute
6, and the hold runs to the switch, less the steps at the end of the block in
which the controller already moves toward the next level. Those steps are
found from the squared errors by `target_gym.eval.anticipations`, the rule the
protocol applies to the cost. The script's 3 seeds give 18 blocks and 1440
steps after the settle. The PID's hold scores all 1440 and the MPC's 1513, so
the rule leaves out none of the MPC's (derived from the recorded
`hold_steps_scored`). The per-seed MPC holds are 2.25e-6, 2.23e-6 and
2.69e-6 mol/L, 2.40e-6 over all three seeds (the CasADi oracle's before v3
were 6.33e-6, 6.29e-6 and 6.75e-6), and the PID's 1.39e-4, 1.23e-4 and 1.40e-4 mol/L, 1.34e-4 over all three
(measured, `src/target_gym/data/hold_measurements.json`).
`test_floor_is_the_recorded_mpc_hold` checks `e_floor`, `rho_floor_tracking`
and `rho_floor` against the recorded holds to 1 %, and that the protocol's MPC
hold cost (6.9e-4, measured) is at least 0.98 of the reference.

`rho_floor_tracking` is the oracle's best hold for tracking. It is the lowest
per-seed hold the reference controller demonstrated, divided by `e_floor`
and squared, as the tracking term squares an error. It would be 1 if the
floor were that hold. It is 1.99e-5 here because the floor is clamped at the
analyser resolution, about 45 times the MPC's lowest hold of 2.23e-6 mol/L
(the table above). `rho_floor` is the same with consumption charged in full.
`floor_is_documented_minimum` records whether `e_floor` is a measured or
certified floor or a resolution used as a scale. `failure_cost` is
the per-step cost of a tripped plant, above the largest tracking cost the
envelope can produce, and `restart_steps` the time a restart would take, so a
trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`). The step that leaves the envelope is
charged the trip cost, with tracking zeroed, and the plant restarts at once as
`reset_env` would, on the same clock. `terminated` stays
false unless `continuing = False`, which ends the episode on that step instead;
`info["tripped"]` marks the event for the evaluator. `reward_version = 1`
selects the version-1 reward of section 8 (`precision_floor` and
`CA_error_max` are read only by it).

---

## Conformance notes

- **Check 7** (the plant does not accelerate without input) is skipped
  through `KNOWN_OPEN_LOOP_UNSTABLE`. Every target is a saddle by design, and
  a trip restarts the plant, so the unforced run is a runaway-and-restart
  sawtooth whose ratio depends on where the restarts fall. The instability is
  asserted here instead, from the Jacobian (`test_every_target_is_a_saddle`)
  and by stepping the env (`test_unforced_error_grows_at_the_saddle_rate`).
  Replayed on 600 unforced steps from one reset, with one trip, the
  late-to-early increment ratios are 0.00 to 1.13, under the check's limit of
  8 (measured, `--section invariance`).
- **Check 5** (regimes join smoothly). The validity guard trips the
  non-physical proposals of the off-envelope sweep, so no `KNOWN_SEAMS` entry
  is needed. Replayed, the sweep's largest one-sided Jacobian jump is 0.037
  (T to C_a at T = 359.6 K), against a limit of 0.5 (derived,
  `--section invariance`).
- **Check 8** (the actuator moves the tracked variable). Full heating trips
  the plant from each of the 135 reset corners (five targets, nine corners,
  feed drift −6, 0 and +6 K) in 5 to 20 steps, median 9, 15 to 60 s (derived,
  `--section reach`), the row in `docs/model-review-checklist.md`. Full
  cooling from the same corners never trips it.
- **Seeds.** Each trip restarts from a fresh draw keyed on its step, even
  under a constant rollout key (section 5).
- **The shared floor check cannot fail here.**
  `test_mpc_does_not_beat_a_measured_floor` in `tests/test_reward_contract.py`
  compares the protocol's whole-episode MPC tracking cost, 3.2e3 (measured),
  with `rho_floor_tracking`, 1.99e-5. Tracking is the mean over every step (the
  task has no burn-in), and the five switches cost up to (0.1 / 1e-4)^2 = 1e6
  per step while settling, so it sits far above a reference of at most 1.
  The task's own test, `test_floor_is_the_recorded_mpc_hold` in
  `tests/pc_gym/unstable_cstr/test_unstable_cstr_experts.py`, checks the
  floor and both references against the recorded MPC hold instead, and the
  protocol's MPC hold cost (6.9e-4) against the reference.

## Readings

The observation reports concentration and both temperatures through their
instruments and the live target exactly (`docs/proposals/measurement-noise.md`).
`sensor_noise_scale = 0` gives the noise-free task.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `C_a` | 5e-4 mol/L | none | Online UV-Vis process photometer with flow cell | Provisional | [OMA-300](https://www.ima.co.uk/wp-content/uploads/AA_DS001A_OMA300.pdf) |
| `T` | 0.04 K, 10 s lag | none | Pt100 RTD with HART head transmitter, in a reduced-tip thermowell | Confirmed; the lag is read | [iTEMP TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf), [TST90](https://core.instrumart.com/assets/tst90-datasheet.pdf) |
| `T_j` | 0.04 K | none | Pt100 RTD with HART head transmitter | Confirmed | [iTEMP TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |

`C_a`, the tracked quantity, rests on a weak figure. The sheet gives only a
sensitivity of 0.1 % of full scale. Reading that as 2 sigma and mapping it to a
1 mol/L full scale are ours, and the 1 to 5 s response barely fits the 3 s step.
`T` and `T_j` are upper bounds on scatter: the sheet gives ±0.08 °C, stated as
2 sigma and including non-linearity and repeatability.

**The thermowell's lag is 10 s.** Endress+Hauser's TST90 sheet (TI093T/02/en) gives
T50 7.5 s and T90 20.5 s for the reduced fast-response tip in water at 0.4 m/s, to
IEC 751. A first-order lag has T50 = 0.693 tau and T90 = 2.303 tau, so those are
10.8 s and 8.9 s. The straight tip's 24 and 72 s are about 33 s, so a reactor held
on a saddle would buy the reduced tip, and this was picked before any result. It is
the fast end of the 10 to 20 s the plan allows, and the reactor's agitation has to
match 0.4 m/s for it to hold. `T` keeps its Confirmed status for the scatter.

`T_j` has no lag, only because the plan names `T`. It is the same instrument, and a
well in the jacket's line would lag too, though its flow is faster. The owner may
want that changed.

The expert reads these readings, and so does every tuner. The oracle plans and
acts on the true state, so it stays a full-state bound. `reward`, trips and the
score read the true state. `info["measured_reward"]` is the same cost on the
readings, with any trip taken from the true state, and a learner can train on
it.

A plant resting exactly on target pays 1 per step in measured reward,
`(5e-4 / 5e-4)^2`. `e_floor` and `precision_floor` are the analyser's scatter, so
the ceiling is what this analyser can verify.

The cascade's inner derivative now reads `T` through the shared filter
(`pid.filtered_derivative`), with a filter time of one-tenth of the derivative
time, 0.011 min on the stored gains. On a 0.05 min step that passes 82 % of a
step in `T`, so it cuts the derivative's noise only a little. The linearised
rates at 0.45 mol/L are now +1.15 /min for the plain 6 s gains and -0.03 /min for
the marginal ones. The rates and the episode figures in section 7 that were
measured with the raw difference are measured again with the expert protocols.
