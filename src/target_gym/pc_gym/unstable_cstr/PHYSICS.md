# Unstable CSTR: physics model, provenance and validation

Reference process: the shipped `cstr` reactor, a jacketed continuous
stirred-tank reactor running an irreversible first-order exothermic reaction
A → B, held on its **open-loop unstable middle steady state**. A first-order
jacket lag sits between the coolant command and the jacket, and the feed
temperature drifts.

Contract for `target_gym.pc_gym.unstable_cstr`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

Every number carries a label: **read** (from a source or from shipped code),
**derived** (computed from read values by a script or test named beside it),
**ours** (a design choice), **TUNED** (ours and load-bearing, justified by a
sensitivity table), **measured** (from a repo script run on the built env).
"The numbers script" is `scripts/unstable_cstr_numbers.py`; a flag such as
`--section pnr` or `--sensitivity` names the part of it that prints the
figure. Its tables score a hold on a fixed window of their own, minutes 6 to
8.4 of each block (120 <= block_clock % 200 < 168), which ends before the
MPC's 32-step horizon reaches the next switch. The hold that sets the floor
comes from `scripts/measure_hold.py` instead (the reward section).

---

## 1. Model and provenance

Three states under perfect mixing, and the feed drift:

```
dC_a/dt = (q/V)(Caf − C_a) − r
dT/dt   = (q/V)(Ti + dTi − T) + (−deltaHr) r / (rho C) + UA (T_j − T) / (rho C V)
dT_j/dt = (T_c − T_j) / tau_j

with    r = k0 · exp(−EA_over_R / T) · C_a

dTi[k+1] = a dTi[k] + Ti_sigma sqrt(1 − a²) xi[k],   a = exp(−delta_t / Ti_tau),   xi[k] ~ N(0, 1)
```

| symbol | quantity | unit |
|---|---|---|
| C_a | reactant concentration | mol/L |
| T | reactor temperature | K |
| T_j | jacket temperature, seen by the energy balance | K |
| T_c | coolant command, the action | K |
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

**Added by this task.** The jacket lag (`tau_j` 6 s, TUNED; section 6) and the
feed drift (`Ti_sigma` 2 K, `Ti_tau` 10 min, ours). Neither is sourced.

**Integration.** One RK4 step per env step (`rk4_1`) over `[C_a, T, T_j]`,
with the command and the drift held over the step. The drift is then advanced
by its exact Ornstein-Uhlenbeck update, with the innovation drawn from the
step key folded with the pre-increment `time` (as `glass_furnace` and
`battery` do), so a constant rollout key still gives a zero-mean process.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Aqueous heat capacity (D1) | C = 0.239 J/(g K) is what every source above uses; section 9. |
| Jacket energy balance (D2) | The jacket follows the command through a fixed lag, whatever heat it removes. |
| Analyser dead time (D3) | C_a is observed at once. |
| Feed-concentration disturbance (D4) | Only the feed temperature drifts. |
| Measurement noise (D5) | Observations are exact. |
| Side reactions, catalyst decay | One irreversible first-order step, as in cstr. |
| Volume, boiling, pressure | Constant holdup, liquid phase throughout. |
| The ignited branch above the trip | The 365 K interlock trips first (section 2). |

**Regime of validity.** RK4 at `delta_t` is stable for |lambda| dt < 2.785
on the negative real axis (read). The fastest dynamics are the runaway just
below the trip. Under full heating until the trip, the largest |lambda| dt is
0.711 from the 135 corners of the reset box and 0.989 from 33 extinguished
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
return is the lowest reactor temperature from which full cooling still
reaches the trip (`point_of_no_return`, bisection on the env's own step). The
margin "in pure T" raises T alone, with C_a = L and the jacket at
Tc*(L, dTi). The margin "along the eigenvector" moves along the unstable
direction, where C_a falls as T rises (dC_a/dT −0.0072 to −0.0090
mol/(L K)). The reset corner is (L + 0.01 mol/L, T* + 1.5 K), the box's
hottest.

| L (mol/L) | PNR − T* in pure T, dTi −6 / 0 / +6 K (K) | along the eigenvector, dTi 0 (K) | reset corner margin, dTi 0 / +6 K (K) |
|---|---|---|---|
| 0.45 | 4.009 / 3.304 / 2.476 | 5.098 | 1.341 / 0.511 |
| 0.50 | 4.536 / 3.810 / 2.968 | 5.633 | 1.897 / 1.053 |
| 0.55 | 5.232 / 4.490 / 3.634 | 6.464 | 2.618 / 1.758 |
| 0.60 | 6.126 / 5.370 / 4.494 | 7.646 | 3.529 / 2.650 |
| 0.65 | 7.260 / 6.487 / 5.587 | 9.320 | 4.673 / 3.770 |

The margin grows with the target, so 0.45 mol/L is the tightest at every
drift (`test_point_of_no_return_at_the_hottest_target`).

**The line the controllers read.** `PNR_LINE` is (354.9, −40.7), meaning
T = 354.9 − 40.7 (C_a − 0.45) K. It is the supporting line under the exact
point of no return at dTi +6 K over C_a 0.40 to 0.70, refit as
354.918 − 40.699 (C_a − 0.45), with the slope rounded and the intercept
rounded down to 0.1 K (derived, `--section pnr`). It lies under the exact
curve at drifts of −6, 0 and +6 K. The smallest gap is 0.873 K at
dTi 0, 1.611 K at −6 K and 0.018 K at +6 K, each near C_a 0.545
(`test_pnr_line_is_conservative`). The MPC's soft bound, the line less
`MPC_PNR_MARGIN_K` (1 K, ours), still sits 1.067 K above T* at 0.45, so it
cuts no target off. The PID's setpoint ceiling at 0.45 is the line less
0.3 K, 354.600 K or T* + 1.767 K, and the line is the binding ceiling for
targets up to 0.464 mol/L.

**Boundary to peak.** Started 0.01 K inside the point of no return with the
6 s lag, full cooling turns the excursion at its peak 16 to 19 steps later at
dTi 0 (0.80 to 0.95 min; the highest peak is 363.80 K, at 0.45) and 20 to 21
steps later at dTi +6 K (1.00 to 1.05 min) (derived, `--section pnr`). The
longest, 21 steps at 0.60 and 0.65 mol/L and dTi +6 K, sets the MPC's
horizon (section 7).

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
| `precision_floor` | 1e-4 | mol/L | read | cstr's online-analyser resolution; version 1 only | ✅ |
| `time_unit_seconds` | 60 | s per min | read | cstr's convention: `delta_t` in minutes | ✅ |
| `T_c_min` | 290 | K | read | PC-gym's CSTR action-space lower bound | ✅ |
| `T_c_max` | 310 | K | ours | spans both folds, with at least 4.63 K headroom at \|dTi\| 6 K (derived, `--section targets`) | ⚠️ |
| `tau_j` | 0.1 (6 s) | min | TUNED | not sourced; the sensitivity table in section 6 is its justification | ⚠️ |
| `T_trip` | 365 | K | ours | high-high interlock, 4.49 K above the extinction fold and 12.17 K above the hottest target (derived) | ⚠️ |
| `T_valid_min` | 250 | K | ours | validity guard, unreachable (section 1) | ✅ |
| `Ti_sigma` | 2.0 | K | ours | stationary standard deviation of the feed drift | ⚠️ |
| `Ti_tau` | 10 | min | ours | feed-drift correlation time | ⚠️ |
| `Ti_dev_reset_clip` | 3.0 | sd | ours | clip on the initial drift draw (section 5) | ⚠️ |
| `target_CA_range` | (0.45, 0.65) | mol/L | ours | the middle branch, 5.89 K from the ignition fold and 7.68 K from the extinction fold (derived) | ⚠️ |
| `switch_sizes` | (0.05, 0.05, 0.10, 0.10, 0.10) | mol/L | ours | a fixed multiset, permuted per episode | ⚠️ |
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
| Full heating from the 135 reset corners (five targets, 3 x 3 corners, dTi −6, 0 and +6 K) | trips in 5 to 20 steps, median 9 (15 to 60 s); from 5 to 12 steps at 0.45 up to 9 to 20 at 0.65 | `--section reach`, `test_full_heating_trips_from_every_corner` |
| Full cooling for 60 min from the same corners | no trip; the hottest point is 355.28 K | `--section reach`, `test_full_cooling_never_trips` |
| Re-ignition from the low state at a 300 K jacket (324.48 K) under full heating | 340 K after 18 steps, trip after 25, so 21 s between them | `--section reach`, `test_extinction_is_recoverable` |
| Zero action (a 300 K jacket) from T* ± 0.5 K | leaves ±2 K in 6 to 28 steps; the starts 0.5 K above T* at 0.45 and 0.50 trip at steps 13 and 18, the other eight extinguish to 324.5 K | `--section reach`, `test_zero_action_leaves_the_middle_state` |
| Restart | cstr's 240 steps of 0.25 min are 1200 steps at 3 s | `--section params`, `test_restart_time_is_cstrs_hour` |
| Drift per step | a = exp(−delta_t / Ti_tau) = 0.99501; innovation sd 0.1995 K | `--section params` |

---

## 5. Task design

**Observation** `[C_a, T, T_j, live target]`. Hidden: the drift `Ti_dev`, the
later levels and the block clock. C_a is read instantly and without noise
(D3, D5). T is observed because no P or PI loop on C_a alone can stabilise
the plant (`test_no_p_or_pi_loop_on_c_a_stabilises`). On the linearisation
with the lag, a loop on C_a alone leaves one coefficient of the closed-loop
polynomial at det − tr/tau_j, which is −28.95 at 0.45 and −9.35 at 0.65
(derived, `--section targets`).

**Action.** One entry, raw in [−1, 1] mapped linearly to a coolant command in
[290, 310] K, raw 0 = 300 K. The two signs disagree
(`test_short_term_and_static_signs_are_opposite`). Raising the coolant by
1 K lowers C_a within a step or two, by 1.0e-5 to 1.5e-5 mol/L after one step
and 7.4e-5 to 1.2e-4 after two, since the reactor heats and burns faster.
Holding a higher C_a needs a warmer jacket, with Tc* rising by 13.4 to
17.8 K per mol/L across the band (derived, `--section targets`).

**Schedule.** Six 10-minute blocks. The first level is uniform in (0.45, 0.65)
mol/L. The five switches take the sizes 0.05, 0.05, 0.10, 0.10 and 0.10
mol/L in a random order, each with a fair-coin sign that is flipped when the
move would leave the band (`test_reset_draws_a_stratified_schedule`). The
sizes are fixed so that every episode has the same total squared move,
0.035 (mol/L)² (derived), which keeps the spread of returns across seeds
small. Where each target is drawn on its own, as in cstr, the return is set
mostly by how far the target sits from the start. cstr's recorded MPC
returns run from −165 to −2.35e6 over its 10 seeds, and this task's recorded
PID returns from −6.61e7 to −7.08e7 (both measured by
`scripts/record_baselines.py` and recorded in
`src/target_gym/data/baseline_returns.json`).

**Scoring at a switch.** The reward and the observation of one state use the
same live target, `level[min(block_clock // block_steps, 5)]`, as
`glass_furnace` and `battery` do. The step that enters a new block is scored
against the new level, which that step's observation already shows
(`test_live_target_follows_the_block_clock`); five steps per episode are
scored against a level first shown in their own observation. The last level
holds past step 1200.

**Reset.** A warm start. C_a is within 0.01 mol/L and T within 1.5 K of the
first level's equilibrium, the jacket is already at Tc*(L0, dTi0), and the
initial drift is drawn from its stationary law clipped at 3 standard
deviations (6 K) (`test_reset_is_warm_with_a_clipped_drift`). Every reset is
recoverable. The worst corner of the box (C_a 0.46 mol/L, T*(0.45) + 1.5 K)
is 1.341 K inside the point of no return at dTi 0 and 0.511 K inside at
dTi +6 K, and would cross it only at dTi +9.24 K (4.62 sd), which the clip
excludes (derived, `--section pnr`;
`test_every_reset_is_inside_the_point_of_no_return`). None of 20000
`reset_env` draws starts past it (measured, `--sensitivity`, 6 s row).

**Trip and restart.** The trip is one-sided. T >= 365 K trips the plant, as
does the validity guard, and there is no low trip. The extinguished branch,
below the ignition fold at 335.654 K, is charged as tracking error and is
always recoverable, since at a 310 K coolant the only steady state (383.89 K)
lies above the trip and full heating re-ignites the plant from the low state
in 18 steps (section 4). A trip is part of the kernel (`failure_kernel`). The
step is charged the trip cost, and the plant restarts from the fresh draw
`reset_env` makes from the step's key, with a new schedule and drift and the
block clock at 0, while `time` runs on
(`test_a_trip_restarts_fresh_on_the_same_clock`). With a constant rollout key,
as every shipped rollout helper passes, every trip in an episode restarts from
the episode's own initial state.

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
unstable operating point while its reference moves, 2.5 to 7.3 K below a
point of no return at the lag and drifts it runs at (section 2), where a
failure is one-sided and costs an hour of downtime. On its validation
episodes the shipped cascade comes no closer than 1.336 K to the exact point
of no return, and the MPC no closer than 1.519 K (measured, section 7).

---

## 6. Jacket lag sensitivity

`tau_j` is TUNED. The lag is not sourced, and these two tables are its
justification (`--sensitivity`). lambda+ is 3.158 /min at 0.45 at
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

**Cascade PID** (`make_unstable_cstr_pid`). An outer PI on C_a sets a
rate-limited temperature setpoint around the feedforward T*(L), clipped
between T* − 8 K and min(T* + 2 K, the point-of-no-return line at L − 0.3 K);
an inner PD on T sets the coolant. The setpoint starts at the measured T
(`cascade_init`). On the 50 catches from the extinguished branch that
`--pid-validation` runs, a setpoint started at T* instead trips 98 times in
42 of the 50 episodes (measured). Four gains are tuned. The setpoint limits
are held, since their right value is set by the distance to the point of no
return, which the return sees only once a trip happens.

A stability guard (`check_cascade_gains`) refuses gains whose linearised
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

The shipped gains are TUNED by `scripts/tune_pid.py --envs unstable_cstr`, a
coordinate search on episode return over 24 seeds starting from
`DEFAULT_GAINS`, and stored in `src/target_gym/data/pid_gains.json` under
`unstable_cstr`. The search raises Kp_T from 7 to 9.8 and Kc_Ca from 100 to
140, shortens Ti_Ca from 0.5 to 0.35 min, and keeps Kd_T at 0.4, moving the
return from −6.967e7 to −6.907e7 (measured, printed by that script). The
closed-loop rates at 0.45, 0.50, 0.55, 0.60 and 0.65 mol/L are −0.59, −1.40,
−2.33, −3.33 and −3.37 /min, against −1.73 to −2.76 /min for `DEFAULT_GAINS`,
and the drift does not change them (derived, `closed_loop_rates`,
`--pid-validation` and `--section invariance`). The smallest margin of an
equilibrium setpoint inside its window is 1.45 K.

**PID validation** (`--pid-validation`, 300 episodes, float32 episodes and
float64 margins, measured). The margin is to the exact point of no return;
the line margin is to `pnr_line`.

| episodes | trips | re-extinctions after capture | smallest PNR margin (K) | smallest line margin (K) | mean abs error, minutes 6 to 8.4 of a block (mol/L) | v2 cost per step |
|---|---|---|---|---|---|---|
| 200 random schedules, drift on | 0 | 0 | 1.336 (p1 1.524, median 2.385) | 0.333 | 1.34e-4 | 5.753e4 |
| 50 of the 0.55/0.45 alternation, dTi +6 K held | 0 | 0 | 1.413 | 0.719 | 1.59e-5 | 8.326e4 |
| 50 catches from extinction, dTi −6, 0 or +6 K held | 0 | 0 | 1.389 (median 3.436) | 0.688 | 3.21e-2 | 1.485e6 |

No step goes above the line. The smallest margin, 1.336 K, is above the
0.5 K the numbers script requires before it keeps `GUARD_MIN_DECAY` at
0.5 /min. `test_pid_holds_every_target` builds and runs its own 130 episodes
the same way: the first 64 random schedules, the first 16 alternating ones
and the same 50 catches. It asserts that no episode trips, that each brings
C_a within 0.01 mol/L of its target, that T stays above the ignition fold
(335.654 K) from then on, that T is at most 0.1 K above `pnr_line(C_a)` on
every step, and that at the eight steps of each episode closest to that line
the exact point of no return is at least 0.25 K above T.

**MPC** (`make_unstable_cstr_mpc`). A CasADi NMPC on radau collocation of
degree 3, which keeps the unstable mode from blowing up a propagated rollout.
The stage cost is the version-2 tracking term, rescaled by
`MPC_ERROR_SCALE` (0.01 mol/L, ours), plus `MPC_MOVE_WEIGHT` (1e-3, ours) on
the squared move. The terminal cost is the Riccati weight of the env's
one-step linearisation at 0.45 mol/L (`terminal_weight`,
`test_terminal_weight_is_positive_definite`). T is kept under
`pnr_line(C_a)` less `MPC_PNR_MARGIN_K` (1 K, ours) by a soft bound that
costs `MPC_PNR_PENALTY` (1e4, ours) per K per interval. The cascade PID is
its fallback, its memory tracked to every action the MPC applies
(`cascade_track`, `test_mpc_falls_back_to_the_pid`). Like every MPC in the
suite it is an oracle ceiling. It reads the true state, the drift and the
whole schedule, and it plans on the drift's conditional mean a^k dTi
(`test_mpc_plans_the_drift_mean`, `test_mpc_preview_matches_the_env`). Its
model is the env's velocity to 1e-12 relative
(`test_mpc_model_is_the_env_velocity`). For model-review check 13,
`test_mpc_predicts_one_step_like_the_plant` steps 120 states of a shipped-PID
episode through one interval of that model and through `step_env`, and
requires a mean signed C_a error under a tenth of `e_floor` and no error
above `e_floor`.

**Horizon.** `MPC_HORIZON` is 32 steps (1.6 min, ours). Across the band that
is 2.2 to 5.1 unstable time constants (1.6 min times lambda+ of 1.391 to
3.158 /min, derived). It is 1.52 times the longest boundary-to-peak time, 21
steps at dTi +6 K (section 2), so a plan that starts next to the point of no
return runs past the peak of the excursion it is turning. 32 is the smallest
whole number of steps at least 1.5 times 21. The horizon is shorter than the
PID's longest 90 % closure of a switch, 47 steps or 2.35 min on 0.55 to 0.65
mol/L (derived, `--section settle`), and the terminal weight prices what lies
beyond it. `scripts/audit_mpc_horizons.py --envs unstable_cstr` returns ok.
It finds the PID closing 63 % of the reset error (0.0059 mol/L) in 11 steps,
a ratio of 32 / 11 = 2.91 (measured). That audit times only the reset error,
so it never sees a 0.1 mol/L switch.

**MPC gate** (`--mpc-gate --workers 8`, 110 episodes, float32 episodes and
float64 margins, measured).

| episodes | trips | fallback steps | smallest PNR margin (K) | steps above the soft bound (largest excess) | mean abs error, minutes 6 to 8.4 of a block (mol/L) | v2 cost per step, MPC / PID on the same starts |
|---|---|---|---|---|---|---|
| 100 random schedules, drift on (seeds 1000 to 1099) | 0 | 0 | 1.581 (p1 1.695, median 2.220) | 40 (0.064 K) | 6.57e-6 | 3883 / 5.766e4 |
| 10 of the 0.55/0.45 alternation, dTi +6 K held (seeds 2000 to 2009) | 0 | 0 | 1.519 | 30 (0.000 K) | 1.82e-6 | 7433 / 8.326e4 |

No solve was capped, and IPOPT took 13.2 and 13.5 iterations on average. The
PID also ran every one of these starts without a trip. Solve time per step
over 132000 steps was 28.3 ms at the median, 41.0 ms at p90, 50.9 ms at p99
and 166 ms at most (wall clock on a shared machine, 8 workers). The gate
also measures, on every step, how far a fallback's first action would move
the coolant from the one last applied. The median is 0.086 K and the largest
20 K, but after a step in minutes 6 to 8.4 of a block the largest is 0.469 K,
inside `FALLBACK_STEP_BOUND_K` (0.5 K, ours), which
`test_mpc_falls_back_to_the_pid` asserts.

**Recorded baselines** (`scripts/record_baselines.py --envs unstable_cstr`,
10 seeds, measured). The PID's mean return is −6.857e7 and the MPC's
−5.014e6, so the MPC costs 4178 per step against the PID's 5.714e4 and leads
on all 10 seeds. Neither trips. The MPC made 12000 solver calls, none failed
or capped, at 13.3 IPOPT iterations on average.

**Protocol row** (`scripts/evaluate_baselines.py --envs unstable_cstr`, 3
seeds, measured, recorded in `src/target_gym/data/protocol_results.json`).

| controller | gain (all tracking) | hold | reach cost per change | transient cost per change | NEA | failure rate |
|---|---|---|---|---|---|---|
| PID | 5.74e4 | 2.68 | 1.15e7 | 5.84e5 | | 0 |
| MPC | 4.03e3 (± 476) | 0.00649 | 8.07e5 | 2.52e5 | 0.93 | 0 |

The gain is the mean cost over every step. The MPC's is 4.03e3 against the
PID's 5.74e4, an NEA of 0.93 against the reference `rho_floor` of 0.00396.
The MPC's hold cost is 0.00649 and the PID's 2.68. The hold leaves out the
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
`tests/pc_gym/unstable_cstr/test_unstable_cstr_physics.py`, and D2, D3 and D5
in `tests/pc_gym/unstable_cstr/test_unstable_cstr_env.py`. The numbers below
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

**⚠️ D5. No measurement noise.** Two observations of one state are identical.
In `test_observations_are_noisy` one state stepped with one action under
keys 1 and 2 gives the same observation to the last bit, since the key only
draws the hidden drift's next innovation (measured).

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `unstable_cstr` | 1200 | 3 | 60 min | 1 in [-1, 1] | 4 | 10 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`, see docs/reward-shaping.md. There
is no running term, since the coolant temperature is a setting with no
tariff.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1e-4 mol/L | the online analyser's resolution (read, cstr's `precision_floor`), kept because the MPC holds finer. A measured upper bound clamped at the instrument resolution (docs/reward-shaping.md, "How each floor was obtained"). `scripts/measure_hold.py --envs unstable_cstr` records the MPC's mean absolute C_a error over the hold of each block (the hold window below), and the floor is max(1e-4, the lowest per-seed hold). That hold is 6.29e-6 mol/L (measured; the PID's lowest is 1.23e-4), so the clamp binds. |
| `e_tol` | 0 | no product specification for the concentration has been supplied |
| `tracking_exponent` | 2 | quadratic: the owner's cost is off-spec product |
| `failure_cost` | 6.05e7, twice (0.55 / 1e-4)^2 | twice the largest tracking cost an untripped state reaches, with 0.55 mol/L the closed-form `CA_error_max` (derived, `--section params`) |
| `restart_steps` | 1200 (1 h) | cstr's one-hour restart (240 steps of 15 s, marked provisional in cstr) at 3 s steps; a trip costs 1200 x 6.05e7 = 7.26e10 (derived, `test_restart_time_is_cstrs_hour`) |
| `rho_floor_tracking` | 0.00396 | (lowest per-seed MPC hold / `e_floor`)^2 = (6.29e-6 / 1e-4)^2, from `scripts/measure_hold.py` (measured); below 1 because the clamp binds |
| `rho_floor` | 0.00396 | equal to `rho_floor_tracking`, since there is no running cost |
| `floor_is_documented_minimum` | False | the plant is disturbed; documented minima are for plants with no disturbance at all |

**The hold window.** `scripts/measure_hold.py` scores this plant's hold
block by block. A fixed settle of 120 steps starts each block's hold at minute
6, and the hold runs to the switch, less the steps at the end of the block in
which the controller already moves toward the next level. Those steps are
found from the squared errors by `target_gym.eval.anticipations`, the rule the
protocol applies to the cost. The script's 3 seeds give 18 blocks and 1440
steps after the settle. The PID's hold scores all 1440 and the MPC's 1141, so
the rule leaves out 299 MPC steps, 20 per switch on average over the 15
switches (derived from the recorded `hold_steps_scored`). The per-seed MPC
holds are 6.33e-6, 6.29e-6 and 6.75e-6 mol/L, 6.46e-6 over all three seeds,
and the PID's 1.39e-4, 1.23e-4 and 1.40e-4 mol/L, 1.34e-4 over all three
(measured, `src/target_gym/data/hold_measurements.json`).
`test_floor_is_the_recorded_mpc_hold` checks `e_floor`, `rho_floor_tracking`
and `rho_floor` against the recorded holds to 1 %, and that the protocol's MPC
hold cost, 0.00649, is at least 0.98 of the reference.

`rho_floor_tracking` is the NEA reference for tracking. It is the lowest
per-seed hold the reference controller demonstrated, divided by `e_floor`
and squared, as the tracking term squares an error. It would be 1 if the
floor were that hold. It is 0.00396 here because the floor is clamped at the
analyser resolution, about 16 times the MPC's lowest hold of 6.29e-6 mol/L
(the table above). `rho_floor` is the same with consumption charged in full.
`floor_is_documented_minimum` records whether `e_floor` is a measured or
certified floor or a resolution used as a scale. `failure_cost` is
the per-step cost of a tripped plant, above the largest tracking cost the
envelope can produce, and `restart_steps` the time a restart would take, so a
trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`). The step that leaves the envelope is
charged the trip cost, with tracking zeroed, and the plant restarts at once as
`reset_env` would, on the same clock. `terminated` is never raised;
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
- **Seeds.** Under a constant rollout key every trip in an episode restarts
  from the episode's own initial state (section 5).
- **The shared floor check cannot fail here.**
  `test_mpc_does_not_beat_a_measured_floor` in `tests/test_reward_contract.py`
  compares the protocol's whole-episode MPC tracking cost, 4.03e3, with
  `rho_floor_tracking`, 0.00396. Tracking is the mean over every step (the
  task has no burn-in), and the five switches cost up to (0.1 / 1e-4)^2 = 1e6
  per step while settling, so it sits far above a reference of at most 1.
  The task's own test, `test_floor_is_the_recorded_mpc_hold` in
  `tests/pc_gym/unstable_cstr/test_unstable_cstr_experts.py`, checks the
  floor and both references against the recorded MPC hold instead, and the
  protocol's MPC hold cost (0.00649) against the reference.
