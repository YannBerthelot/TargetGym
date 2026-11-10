# Compressor surge: physics model, provenance and validation

Reference process: a variable-speed compressor feeding a header through a
lumped duct and plenum (Greitzer's compression-system model) with the
Moore-Greitzer cubic characteristic scaled by the fan laws. Gas leaves the
header through the consumers' valve, against their static head, and through a
recycle (anti-surge) valve back to suction. The header pressure follows a
ramped setpoint while the consumers' demand ramps between levels, and the
plant trips when the compressor's flow coefficient falls below the episode's
surge line, a hidden offset right of the map's.

Contract for `target_gym.compressor_surge`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

Every number carries a label: **read** (from a source or from shipped code),
**derived** (computed from read values by a script or test named beside it),
**ours** (a design choice), **TUNED** (ours and load-bearing, justified by a
sensitivity table), **measured** (printed by the repo script or flag named
beside it, run on the built env). "The numbers script" is
`scripts/compressor_surge_numbers.py`; a flag such as `--section params` or
`--sensitivity` names the part of it that prints the figure. A test named
next to a measured number asserts that number or a bound on it; the number
itself is the one the named section prints.

The heavy flags `--hardness`, `--sensitivity`, `--v1` and `--pid-validation`
run the pair at its shipped gains (section 7), and their figures here are at
the measured `c_hold`, 62.27 kW. `--mpc-gate`, which left with CasADi
after 0.7.0, ran before `scripts/measure_hold.py` measured `c_hold`, so the
costs it printed, for the NMPC, the pair and G10, are at the provisional
49.6 kW. The running term only falls as `c_hold` rises, so the measured value
lowers each of those costs by at most its energy share at 49.6 kW. For the
NMPC the run printed that share,
0.39 and 0.41 % at its two candidate margins. Trip counts, margins, recycle
power and valve travel do not depend on `c_hold`.

**Figures that predate the ramps.** Setpoint ramps, the surge-line offset and the
wear term landed on the integration branch in November 2026. Everything labelled
measured in sections 4 to 8 was taken on stepped setpoints and the fixed line, with
the pair and the NMPC as they stood, so it describes that plant and PR 9 measures
again. Section 5's ramp figures are the exception, and say so.

---

## 1. Model and provenance

Three ODEs, two rate-limited actuators, the consumers' demand and the trip:

```
dm_c/dt = (A_c / L_c) (rho01 U^2 Psi_c(Phi) - dp)
d dp/dt = (a01^2 / V_p) (m_c - m_d - m_r)
dN/dt   = (N_ramp - N) / tau_N

U = N U_r,   Phi = m_c / (rho01 A_c U)
Psi_c(Phi) = psi_c0 + H (1 + 1.5 y - 0.5 y^3),   y = Phi / W - 1,   for Phi <= phi_join
           = its tangent line at phi_join,                            for Phi >  phi_join

m_d = k_d u r(dp - dp_out),   m_r = k_r x r(dp),   r(z) = sqrt(s softplus(z / s)),  s = check_width
u   = clip(u_sched(t) + dev, 0.2, 1.0)

N_ramp and x move toward their commands by at most drive_rate h, and
valve_open_rate h opening or valve_close_rate h closing, at the start of each
substep of length h

dev[k+1] = a dev[k] + demand_sigma sqrt(1 - a^2) xi[k],   a = exp(-demand_theta dt),   xi[k] ~ N(0, 1)

trip when  -T log sum_j exp(-Phi_j / T) < 2W   over the step's substeps j,   T = soft_min_temperature
```

| symbol | quantity | unit |
|---|---|---|
| m_c | compressor (duct) mass flow | kg/s |
| dp | plenum (header) gauge pressure; observed in kPa | Pa |
| N | shaft speed, fraction of rated; observed in percent | 1 |
| N_ramp | drive setpoint after its rate limit; observed in percent | 1 |
| x | recycle valve opening; observed in percent | 1 |
| u | consumers' total opening | 1 |
| dev (`demand_dev`) | OU deviation of the consumers' opening | 1 |
| m_d, m_r | delivered and recycled flow | kg/s |
| Phi, Psi | flow coefficient, head coefficient dp / (rho01 U^2) | 1 |
| t | time; one step is `delta_t` = 0.1 s | s |

**Nondimensional form.** At fixed speed, with Psi = dp / (rho01 U^2) and
s = U t / L_c, the first two equations become

```
dPhi/ds = Psi_c(Phi) - Psi
dPsi/ds = (Phi - Phi_T(Psi)) / (4 B^2),   B = (U / 2 a01) sqrt(V_p / (A_c L_c)) = U / (2 omega_H L_c)
```

with Phi_T(Psi) the throttle's flow in Phi units. This is Gravdahl's (1998)
eq. (2.6), Greitzer's (1976) model with time rescaled, with s = xi / l_c in
the thesis' notation (read, thesis pp.14 to 15). The orbit depends only on B,
the characteristic and the throttle. At rated speed B = 1.80 (derived), the
value the thesis uses for surge (read, p.46), which is what makes its
figures 2.8 and 2.10 anchors for this plant (section 2).

**Provenance** (all URLs read unless marked citation only):

| source | what it gives | |
|---|---|---|
| Moore and Greitzer (1985), NASA CR-3878, https://ntrs.nasa.gov/api/citations/19850012807/downloads/19850012807.pdf | the cubic characteristic, eq. (3.13) p.16, with H 0.18 and W 0.25 (p.16, p.44); the peak at (0.5, 0.66) (p.53) | ✅ |
| Gravdahl (1998), dr.ing. thesis, NTNU, https://tomgra.folk.ntnu.no/papers/thesis.pdf | eq. (2.6) and B (2.3), pp.14 to 15; Table D.1 p.137 (psi_c0 0.3, H 0.18, W 0.25); B = 1.8 for surge, p.46; figures 2.8 and 2.10, p.47 | ✅ |
| Gravdahl, Egeland and Vatland (2002), Automatica 38(11), https://tomgra.folk.ntnu.no/papers/Gravdahl_Automatica_2002.pdf | a dimensional variable-speed model and the drive acceleration limit of Table 2, p.1892 (section 6, drive rate) | ⚠️ |
| Mirsky, McWhirter, Jacobson, Zaghloul and Tiscornia (2015), METS III tutorial, https://turbolab.tamu.edu/wp-content/uploads/2018/08/Tutorial_12.pdf | industrial anti-surge practice: a control line about 10 % in flow from the surge line (pp.11, 13), the open-loop line near the middle of the margin (p.13), recycle sizing 1.8 to 2.2 times the surge flow and stroke times, opening under 2 s and closing, under positioner control, in no more than 8 to 10 s (p.23) | ✅ |
| Wilson and Sheldon (2006), Hydrocarbon Processing, August 2006, reprint at https://prod-edam.honeywell.com/content/dam/honeywell-edam/pmt/hps/products/ccc/turbomachinery-automation-systems/ccc-control-software/antisurge-control/hon-ccc-antisurge-product-application-note.pdf | Table 1: anti-surge valve specification (full open within 2 s including dead time, full close in 3 s) | ✅ |
| U.S. Standard Atmosphere, 1976 (NASA-TM-X-74335), https://ntrs.nasa.gov/citations/19770009539 | sea-level density 1.2250 kg/m^3 and speed of sound 340.294 m/s, the table `test_suction_constants_are_isa` checks against | ✅ |
| Gravdahl, Willems, de Jager and Egeland (2000), IEEE CDC, https://tomgra.folk.ntnu.no/papers/Gravdahl_CDC00.pdf | a documented rig (TU/e); read and not used, since its 27.7 Hz Helmholtz frequency would need an 8 ms step | ⚠️ |
| Willems (1996), TU/e report WFW 96.151, https://pure.tue.nl/ws/files/4246406/669083.pdf | Greitzer's model reproduced (secondary) | ✅ |
| Greitzer (1976), Parts I and II, https://doi.org/10.1115/1.3446138 and https://doi.org/10.1115/1.3446139 | the lumped model and B (citation only) | |
| Moore and Greitzer (1986), https://doi.org/10.1115/1.3239887 and https://doi.org/10.1115/1.3239893; Gravdahl and Egeland (1999), https://doi.org/10.1007/978-1-4471-0827-6; Fink, Cumpsty and Greitzer (1992), J. Turbomachinery 114(2) | journal and book versions (citation only) | |

**Integration.** Each 0.1 s step runs 10 substeps of 10 ms
(`integration_method="rk4_10"`). A substep first moves `N_ramp` and `x`
toward the commands by at most one substep's reach, then takes one RK4 step
of `[m_c, dp, N, tau]` with the shared `integrate_dynamics`, where `tau` is
the schedule clock in seconds with dtau/dt = 1, so each RK4 stage reads the
demand ramp at its own time. The deviation is held over the step at its
start-of-step value. After the substeps the trip variable is the soft minimum
of Phi over them, computed max-shifted (the plain form underflows to inf in
float32 and never trips), and the deviation takes its exact OU update, with
the innovation drawn from the step key folded with the pre-increment `time`,
so a constant rollout key still gives a zero-mean process.

**Disturbance order.** The step integrates with the deviation it starts with
and draws the next one after, as `unstable_cstr` does. The NMPC then reads
the deviation the coming step uses, so its first predicted step is exact. For
a policy that does not observe the deviation, this order and the reverse one
(draw, then integrate) give the same stationary process.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Rotating stall (D1) | The trip at the peak keeps the plant on the unstalled branch; section 9. |
| Compressibility in the duct (D2) | Mach up to 0.48 in the duct; section 9. |
| Discharge temperature (D3) | The plenum's speed of sound is taken at suction temperature. |
| Time for surge to develop (D4) | The trip is instantaneous on the substep minimum. |
| A tight check valve (D5) | The smooth square root leaks 0.706 kg/s at zero drop and full opening. |
| Recycle-line volume and cooler | The recycle returns to suction at once, at suction conditions. |
| Drive torque and inertia | The speed follows a rate-limited setpoint through a first-order lag. |
| Efficiency and power balance | The running quantity is the ideal compression power of recycled gas. |
| Flow-element lag | Observations are immediate, with noise (section Readings). |
| Parallel machines, load sharing | One compressor. |

**Regime of validity.** The trip at the peak keeps every continuing state on
the right branch. The dead-head regime, speeds below 82.5 % with the recycle
shut, where the shutoff head is below the consumers' static head (derived,
`test_dead_head_speed`), is reached only after the plant has tripped. RK4 at
10 ms is stable wherever the plant goes. Along full travel high over 20
schedules the stiffest eigenvalue, the duct on the steep right branch, gives
h |lambda| = 1.94, under the real-axis limit 2.785, and every eigenvalue's
RK4 amplification is below 1 (derived, `test_rk4_is_stable_on_the_reachable_set`).
Ten substeps are converged. Over a 5 s ramp ending 2 % from the line, 40
substeps move dp by 4.6e-7 relative and the smallest step-end Phi by 1.2e-7,
float32 rounding (measured, `--section integrator`;
`test_integrator_is_converged` asserts 0.1 % and 0.2 %).

Over 361 right-branch equilibria across the action box the largest RK4
amplification is 0.978 (derived, `--section integrator`). Along a run
through a demand ramp to 2 % from the line, the soft minimum sits 1.5e-3 to
2.3e-3 below the substeps' hard minimum, within its bound T ln 10 = 2.3e-3,
with the substeps reproduced exactly by 10 ms steps of the same step
function (derived, `--section integrator`). The reset's Newton iteration
from Phi 0.70 reaches 1.2e-8 relative after four steps and 5.6e-16 after
five in float64, and float32 rounding (1.2e-7) after four, over openings 0.2
to 1.0; six are used (derived, `--section integrator`). Every state of zero
action and full travel high over 20 schedules is finite in float32
(measured, `--section reach`).

---

## 2. Validation targets

Computed with the env's own functions by the tests named. The closed forms
run with `xp=np` (float64); the velocity, its Jacobian and the step run under
`enable_x64` (float64) unless marked float32.

| Quantity | Target | Model | Test | |
|---|---|---|---|---|
| Suction constants | ISA 1.2250 kg/m^3, 340.294 m/s (read) | 1.22501 kg/m^3, 340.292 m/s (R 287.05 against the standard's 287.05287 puts the density 1.0e-5 high) | `test_suction_constants_are_isa` | ✅ |
| Characteristic | peak (0.5, 0.66) (read) | the cubic; value and slope continuous at the join 0.78 (slope -3.774), zero head at Phi 0.8316 (derived) | `test_characteristic_is_the_moore_greitzer_cubic` | ✅ |
| Gravdahl Fig. 2.8: B 1.8, throttle gain 0.61, from Phi = Psi = 0.6 | deep surge, Phi from the axis floor -0.2 to about 0.75, Psi about 0.21 to 0.67 (read, plot) | Phi -0.194 to 0.761, Psi 0.212 to 0.669 over 2.2 s, flow collapses at 0.52 and 1.63 s (derived, static head 0, recycle shut, opening 0.5626) | `test_gravdahl_fig_2_8_deep_surge_orbit` | ✅ |
| Gravdahl Fig. 2.10: throttle gain 0.65 | flat start at about Phi 0.53, Psi 0.66 (read, plot) | equilibrium (0.5268, 0.6568), eigenvalues -0.137 +- 0.257i in units of s; at gain 0.61 (0.4955, 0.6599), +0.0046 +- 0.276i (derived) | `test_gravdahl_fig_2_10_throttle_is_stable` | ✅ |
| Helmholtz pair on the peak | modulus omega_H, real part -(a01^2 / V_p) dm_out/d dp / 2 (closed form) | -1.058 +- 13.852i /s, modulus 13.8923 against omega_H 13.8924 (derived, rated speed, recycle shut, opening 0.683) | `test_helmholtz_pair_on_the_peak` | ✅ |
| Off-peak pair, 24 kPa, demand 0.8, recycle shut | | Phi 0.5358 at speed 86.53 %: zeta 0.567, damped 2.023 Hz (derived) | `test_off_peak_pair` | ✅ |
| Damping near the line, 24 kPa, recycle shut | falls as the margin shrinks | zeta 0.619 at an 8 % margin to 0.105 on the line, monotone over 41 margins (derived) | `test_damping_collapses_toward_the_line` | ✅ |
| Stability boundaries over the action box | none at or right of the line | 107 (speed, recycle) families reach the peak with opening in [0.2, 1]; the trace on the peak is at most -1.58 /s; the Hopf point sits 0.70 % (105 %, recycle 0.30) to 3.77 % (70 %, recycle shut) left of the peak in the 105 families that meet one (derived) | `test_boundaries_lie_left_of_the_peak` | ✅ |
| Fan laws | flow as N, pressure as N^2 (exact) | to 1e-5 at 70, 87.5 and 105 %, no static head, recycle shut | `test_fan_laws` | ✅ |
| Surge line | dp = K m^2, K = 215.51 Pa/(kg/s)^2 (derived) | the smallest trip-free opening at 80, 90 and 100 % lies on it to 1 % | `test_surge_line_is_a_parabola` | ✅ |
| Plenum mass balance | V_p / a01^2 (dp_end - dp_0) = integral of m_c | agrees to 0.1 % of the integral of \|m_c\| over a 2 s fill into deep surge, valves sealed, a01 from the ISA table | `test_plenum_mass_balance_closes` | ✅ |
| Reset | right of the line, converged | six Newton steps equal a bisection root to 1e-6 in float32; Phi 0.530 to 0.728 over openings 0.2 to 1.0 (derived) | `test_reset_is_surge_safe_and_converged` | ✅ |
| Integrator | `rk4_10` against `rk4_40` | section 1 | `test_integrator_is_converged` | ✅ |

The two figures' flow collapses at xi about 55 and 167 (read, plot) and the
model's at s 26.0 and 81.5 give xi / s of 2.1 and 2.05, one time scale; the
figure's l_c is not given, so this is stated and not asserted.

`--section params`, `--section linear` and `--section anchor` print every
row above from the env. Two more results come from them (derived):

- **No hysteresis of its own.** At rated speed with the recycle shut the
  Hopf point sits at Phi 0.4951, 0.98 % left of the peak. 0.1 % past it the
  plant runs in deep surge (Phi -0.196 to 0.743), and 0.3 % before it a run
  started on that deep-surge cycle returns to the equilibrium. The
  two-state model therefore has no bistable band next to the peak, and the
  trip at the peak stands in for stall (D1).
- **Deep surge in the task's own configuration.** At rated speed, recycle
  shut and opening 0.2, with the trip off, the plant cycles with flow
  reversal, m_c -6.44 to 17.75 kg/s, period 0.835 s, 1.85 Helmholtz
  periods. At 70 % the cycle dies within seconds and the plant parks near
  zero flow (m_c within 0.26 kg/s of it) at 7.07 to 7.37 kPa, the shutoff
  head 7.20 kPa, below the consumers' 10 kPa static head.

---

## 3. Parameter table

The reward's parameters (`e_floor`, `e_tol`, `tracking_exponent`,
`dp_error_max`, `failure_cost`, `restart_steps`, `c_hold`, `running_weight`,
`precision_floor`, `rho_floor_tracking`, `rho_floor`) are in the reward table
at the end, with their sources.

| Symbol | Value | Unit | Label | Source | |
|---|---|---|---|---|---|
| `P01`, `T01` | 101325, 288.15 | Pa, K | read | ISA sea level, as `src/target_gym/plane/PHYSICS.md` checks | ✅ |
| `R_gas`, `gamma` | 287.05, 1.4 | J/(kg K), 1 | read | same; with `T01` they set a01, so the plenum compliance V_p / a01^2, omega_H and B | ✅ |
| `A_c` | 0.10 | m^2 | ours | sets the flow scale rho01 A_c U_r that the valves, the static head and the setpoints are sized against; its dynamic effect is through omega_H and B, which the `V_p` and `L_c` rows of section 6 vary | ⚠️ |
| `L_c` | 4.0 | m | TUNED | moves omega_H and B together (both as L_c^-1/2); section 6 rows at 0.5x and 2x | ⚠️ |
| `V_p` | 15.0 | m^3 | TUNED | gives B = 1.80 at rated speed, the thesis' surge value (read, p.46); moves omega_H and B in opposite directions; section 6 rows at 0.5x and 2x | ⚠️ |
| `U_r` | 200 | m/s | ours | rated tip speed; sets the pressure scale rho01 U_r^2 (rated peak 32.34 kPa, pressure ratio 1.319, derived) | ⚠️ |
| `psi_c0`, `H`, `W` | 0.30, 0.18, 0.25 | 1 | read | CR-3878 pp.16, 44, 53; Gravdahl thesis Table D.1 p.137 | ✅ |
| `phi_join` | 0.78 | 1 | ours | tangent continuation just past the largest Phi the plant reaches, 0.777 (measured, `--section reach`; `test_full_travel_high_never_trips` asserts 0.7775) | ✅ |
| `k_d` | 0.12 | kg/(s Pa^0.5) | ours | consumer valve at opening 1 | ⚠️ |
| `dp_out` | 10.0e3 | Pa | ours | consumers' static head | ⚠️ |
| `k_r` | 0.15 | kg/(s Pa^0.5) | ours, read-consistent | at full opening it passes 2.20 times the surge flow at any pressure (derived, 0.15 sqrt(215.51)), the top of Mirsky et al.'s 1.8 to 2.2 (p.23) | ✅ |
| `check_width` | 50 | Pa | ours | width of the smooth one-sided square root (D5) | ⚠️ |
| `N_min`, `N_max` | 0.70, 1.05 | of rated | ours | speed command range; every schedule corner needs 78.6 to 101.8 % (derived, `test_every_block_is_feasible`) | ✅ |
| `drive_rate` | 0.03 | of rated per s | TUNED | section 6, drive rate | ⚠️ |
| `tau_N` | 1.0 | s | ours | speed lag behind the drive setpoint | ⚠️ |
| `valve_open_rate` | 0.5 | per s | read-consistent | full open in 2 s, opened and checked again in both sources: under 2 s with under 0.4 s of delay (Mirsky et al. p.23; 0.3 s on p.27), 2 s including dead time and 0.4 s of opening dead time (Wilson and Sheldon Table 1). The plant folds the dead time into the stroke (D6) | ✅ |
| `valve_close_rate` | 0.1 | per s | TUNED | full close in 10 s, the slowest closing Mirsky et al. p.23 allow (no more than 8 to 10 s under positioner control); Wilson and Sheldon Table 1 ask for 3 s, where the shipped pair trips 341 times over 256 episodes and the pair at `DEFAULT_GAINS` does not trip (measured, `--sensitivity`, section 6); the shipped gains were tuned and validated at 10 s | ⚠️ |
| `p_ref_range` | (20, 28) | kPa | TUNED | the hardness depends on the moving setpoint (section 6) | ⚠️ |
| `setpoint_block_steps` | 300 (30 s) | steps | ours | four blocks per episode | ✅ |
| `setpoint_ramp_s` | 10.0 | s | ours | linear ramp at the start of every block, from the reset's own pressure in the first: twice the 4.9 s the drive needs for a full 20 to 28 kPa move on the control line (section 4), and under half a block so that the scorer still reads each move | ⚠️ |
| `surge_offset_max` | 0.05 | fraction of flow | ours | the episode's line sits uniformly 0 to 5 % right of the map's. Half the conventional 10 % margin (Mirsky et al. pp.11, 13). Mirsky et al. p.11 list the line's uncertainty among what the margin pays for, and give no figure, so this is ours for the owner to accept | ⚠️ |
| `demand_range` | (0.35, 0.95) | opening | ours | six levels per episode | ⚠️ |
| `demand_block_steps` | 200 (20 s) | steps | ours | | ✅ |
| `demand_ramp_s` | 5.0 | s | ours | linear ramp at the start of blocks 2 to 6; section 6 rows at 2 and 10 s | ⚠️ |
| `demand_clip` | (0.2, 1.0) | opening | ours | total opening after the deviation | ✅ |
| `demand_sigma` | 0.02 | opening | ours | stationary sd of the deviation; the one `noise_fields` entry; section 6 | ⚠️ |
| `demand_theta` | 0.2 | 1/s | ours | reversion rate, a 5 s correlation time; a = 0.980199 per step, innovation sd 0.00396 (derived) | ⚠️ |
| `demand_dev_reset_clip` | 3.0 | sd | ours | clip on the initial draw only (0.06 of opening) | ✅ |
| `N_reset`, `x_reset` | 1.0, 0.35 | of rated, opening | ours | the reset and restart point; surge-safe for every opening 0.2 to 1.0 (section 5) | ✅ |
| `reset_newton_iters` | 6 | 1 | ours | Newton from Phi 0.70; static field | ✅ |
| `soft_min_temperature` | 1e-3 | 1 | ours | the trip is conservative by at most T ln 10 = 2.3e-3 in Phi (derived) | ✅ |
| `delta_t` | 0.1 | s | ours | 4.5 steps per Helmholtz period (0.452 s, derived) | ✅ |
| `max_steps_in_episode` | 1200 (120 s) | steps | ours | four setpoint blocks, six demand blocks | ✅ |
| `reward_version` | 2 | | read | the suite's current reward, `docs/reward-shaping.md` | ✅ |
| `floor_is_documented_minimum` | False | | ours | the plant is disturbed | ✅ |

---

## 4. Figures of merit

All derived from the parameter table (`--section params` and
`--section steady`, float64).

| figure | value | |
|---|---|---|
| rho01, a01 | 1.22501 kg/m^3, 340.292 m/s | `test_suction_constants_are_isa` |
| omega_H, f_H, Helmholtz period | 13.892 rad/s, 2.211 Hz, 0.452 s | `test_helmholtz_pair_on_the_peak` |
| B at 70 / 87.5 / 100 / 105 % | 1.260 / 1.575 / 1.800 / 1.890 | |
| rated peak, pressure ratio | 32.34 kPa, 1.319 | |
| surge constant K, surge flow at rated | 215.51 Pa/(kg/s)^2, 12.25 kg/s | `test_surge_line_is_a_parabola` |
| surge flow at 20 / 24 / 28 kPa | 9.633 / 10.553 / 11.398 kg/s | |
| zero-head Phi of the tangent line | 0.8316 | `test_characteristic_is_the_moore_greitzer_cubic` |
| deliverable flow at 105 %, 20 / 28 kPa | 18.70 / 17.08 kg/s against 12.00 / 16.10 at full opening | `test_every_block_is_feasible` |
| blocks that force recycle (setpoint and demand uniform) | 66 % with no margin | `test_most_blocks_force_recycle` |
| fixed-recycle band | at least 0.256 to keep off the line, at most 0.071 for 105 % to hold 28 kPa at demand 0.95 | `test_no_fixed_valve_is_both_safe_and_able` |
| recycle opening that alone carries the surge flow | 0.454, at any pressure | `test_zero_action_never_trips` |
| lowest pressure an untripped plant holds | 7.32 kPa (70 %, recycle and consumers fully open) | `test_restart_and_failure_cost_are_derived` |
| duct Mach number | 0.29 on the rated peak, 0.48 at Phi 0.777 and 105 % | D2 |
| blocks that force recycle at a 5 / 10 / 15 % margin | 73 / 80 / 88 % | |
| fixed-recycle band at a 5 / 10 / 15 % margin | at least 0.280 / 0.306 / 0.335 to keep off the line | |
| zero action's equilibria (87.5 %, recycle 0.5) | Phi 0.632 to 0.731, dp 13.40 to 21.44 kPa over openings 0.2 to 1.0 | |
| reset equilibria (100 %, recycle 0.35) | Phi 0.530 to 0.728, dp 18.06 to 32.14 kPa over openings 0.2 to 1.0; Phi from 0.581 over the drawable first openings 0.29 to 1.0 | `test_reset_is_surge_safe_and_converged` |
| t63 of dp from the zero-action point at opening 0.9 | speed 87.5 to 88.5 %: 1.28 s; 87.5 to 105 %: 4.93 s; recycle 0.5 to 0.45: 0.41 s, to 1.0: 0.53 s, to 0.0: 3.57 s | |
| a full 20 to 28 kPa move | on the surge line 14.4 % of rated speed, 4.8 s at 3 %/s plus the 1 s lag; on the 15 % control line 14.7 %, 4.9 s | |

---

## 5. Task design

### Task: `compressor_surge`

Hold the header pressure of a variable-speed compressor on a ramped setpoint
while the consumers' demand ramps between levels, with the drive speed and an
anti-surge recycle valve. Flow below the surge line trips the plant, and
neither the reward nor any observation reads the surge margin directly, so the
policy has to learn to keep it from the flows it does observe.

| obs | meaning |
|---|---|
| 0 | header pressure (kPa) |
| 1 | `m_c`, compressor flow (kg/s) |
| 2 | `m_d`, flow delivered to the consumers (kg/s) |
| 3 | drive speed, percent of rated |
| 4 | drive speed setpoint, percent of rated |
| 5 | recycle valve opening (%) |
| 6 | target header pressure (kPa) |

**Observation.** The compressor flow m_c is the anti-surge flow element. m_d is the
consumers' flow at the state's clock with the deviation the coming step
integrates with, so the total opening can be inferred from it. Hidden: how
that opening splits into block level and deviation, both schedules' future
values, the clock and `phi_min`. The surge margin,
`obs[1] / (12.25 obs[3] / 100) - 1` against the map, has no channel of its own and
is left for the agent to learn. The true margin is up to `surge_offset_max` less,
and nothing observed says how much.

**Action** (2): speed command in [70, 105] % and recycle command in [0, 1],
from raw [-1, 1]. Raw 0 is 87.5 % speed and a half-open recycle. A recycle
opening of 0.454 carries the surge flow alone at any pressure (derived), so
zero action never trips. Over 20 schedules its smallest Phi is 0.610
(measured, `--section reach`; `test_zero_action_never_trips` asks for 0.55).
The lowest Phi zero action can see is the reset's, 0.581 at the lowest
drawable first opening; a drop to the clip floor 0.2 straight after that
reset reaches 0.568 (derived, `--section reach`).

**Schedules.** Four header-pressure levels, iid uniform in 20 to 28 kPa, held
30 s each, with a 10 s linear ramp into each; six demand levels, iid
uniform in 0.35 to 0.95, 20 s each, with a 5 s linear ramp into blocks 2 to 6.
A step cost the pair 94.6 to 99.6 % of its tracking cost in the 10 s after it
(mean 98.3 %, seeds 200 to 209, past the burn-in, measured), because a move the drive
cannot follow is paid by everyone and the ranking read the slew. On the same
schedules ramps take the pair's tracking cost past the burn-in from 5.8e5 to 2.6e4,
and leave its hold where it was, 22.6 and 22.1 per step (seeds 100 to 109, measured).
The verdict reads the whole episode, so the opening move ramps too. The setpoint moves
because the hardness depends on it.

**The opening is now the pair's own.** A ramp from the reset's pressure asks nothing
of the drive that it cannot do, and the pair's whole-episode tracking cost still falls
mostly in the first 5 s: 5.1e5 of 5.3e5 on seed 200 (measured). The reset's margin is
0.32 to 0.40 against the 15 % control line, so the anti-surge PI shuts the valve at
its closing rate while the drive is still slowing, and the header rises 3 to 5 kPa
over a target moving by under 2 kPa. That is coordination between the two loops, not
slew, so it stays in the score. PR 9 sees it when it retunes the pair. With the setpoint held at 24 kPa the chaser at bias
0.3 trips in none of 256 episodes, against 11.3 % with the steps, and G10
costs 1.76e5 per episode against 5.45e5 (measured, `--sensitivity`,
section 6). The live setpoint and demand are functions of `block_clock`,
which `reset_env` sets to 0.

**Reset.** Equilibrium at 100 % speed and recycle 0.35 for the first
opening, drifted by an initial deviation drawn from its stationary law and
clipped at 3 sd. Every such equilibrium is right of the line (Phi 0.530 to
0.728, `test_reset_is_surge_safe_and_converged`). The reset is off the first
level. Its pressure spans 18.06 to 32.14 kPa against setpoints of 20 to 28 kPa
(derived, `--section steady`), so the target opens on that pressure and ramps onto
level 0 in 10 s, an approach the hold measurement's burn-in excludes
(`test_the_first_move_ramps_from_the_resets_own_pressure`). Over 256 resets the first step
never trips, under zero action or full travel low (smallest phi_min 0.584,
measured, `--section reach`).

**Switch step.** The reward and the observation of one state use the same
live setpoint, and a step on a ramp is scored against where the ramp has reached
(`test_live_target_follows_the_block_clock`). The scorer reads a ramp as one move, at
its start (`test_a_compressor_ramp_starts_one_cycle`), so the 100-step fixed settle
is the ramp itself and the hold opens as the ramp ends. The pair settles about 147
steps after a ramp starts (training seeds 100 to 109), so PR 8's settle should cover
that.

**Trip and restart.** The plant trips when the soft minimum of Phi over a
step's substeps falls below the episode's line, 2W (1 + `surge_offset`). The map's
line, offset 0, is 2W = 0.5, the peak at every speed; in (m, dp) terms the parabola
dp = 215.51 m^2 (`test_trip_reads_the_substep_minimum`,
`test_surge_line_is_a_parabola`). The offset is drawn per episode, uniform up to
`surge_offset_max`, and hidden, so a controller has the map and a margin, as a real
one does (D7, `test_the_trip_reads_the_episodes_line`). The oracle reads it, as it
reads the rest of the true state. A dip between control samples trips. The
step is charged the trip cost and the plant restarts at once from a fresh
draw of the reset, schedule clock at 0 (`base.failure_kernel`,
`test_a_trip_restarts_fresh_on_the_same_clock`); with `continuing = False`
the episode ends on that step instead.
Each restart draws from a key of its own step (`base.restart_key`), so
under a constant rollout key a run of trips does not replay the episode's
initial state. On seeds
0 to 31 the recycle shut under a pressure PI, a fixed recycle of 0.10 and
the margin-blind chaser at bias 0.2 each trip at least once per episode on
average (`test_v1_ranks_tripping_below_safe` asserts it). On seeds 1000 to
1127 they trip 36.3, 26.2 and 6.1 times per episode (measured, `--v1`,
section 8).

**Deviation.** An OU process on the consumers' opening, sd 0.02 and a 5 s
correlation time, zero mean under a constant key
(`test_demand_deviation_is_a_zero_mean_ou_process`).

**What the task adds.** The trip is reachable and sits on a variable the
reward does not track, and the actuator that prevents it, the recycle valve,
has no tracked output of its own. On seeds 0 to 63 the recycle shut under a
pressure PI trips in every episode, and a setpoint chaser that ignores the
margin, at bias 0.3, trips in at least one
(`test_setpoint_chasing_without_the_margin_trips` asserts both). On 256
other seeds the two trip in 100 % and 11.3 % of episodes (measured,
`--hardness`, the table below). A fixed valve keeps every block off the line
only from an opening of 0.256, and above 0.071 the drive at 105 % cannot
hold 28 kPa at demand 0.95 (derived, section 4). Recycle energy is 0.22 % of
the cost of the cheapest policy that did not trip in the hardness table
(measured, `--hardness`), so the tracking error sets nearly all of it. No
reward term reads the margin, so running close to the line pays only through
the recycle power it saves.

**What one innovation costs near the line** (derived, `--section
disturbance`, float64, on the substeps' hard minimum). At steady points held
exactly on a 2, 3 and 5 % margin (36 each, setpoints 20 to 28 kPa, demand
0.20 to 0.95), a deviation 5 innovation sd below the plan costs 1.98, 1.85
and 1.63 margin points in its first step with the commands held, and 5.33,
4.57 and 3.58 over ten steps. With the valve commanded open in the step the
deviation first acts in, the loss is 0.10, 0.09 and 0.09 points. Under this
env's disturbance order a controller that reads the deviation, as the NMPC
does, sees it before that step.

Uniform random actions never tripped in 64 episodes (smallest margin 0.17,
measured, `--section reach`).

**Hardness** (`--hardness`, 256 episodes per policy, seeds 1000 to 1255,
float32, measured). The fixed policies and the chaser run the heuristic
speed PI (`HEURISTIC_SPEED`) with the recycle held, or set to bias + 0.2
(dp - setpoint) in kPa with no regard for the margin. The guarded heuristic
sets the recycle to 0.2 + 0.4 (dp - setpoint), never below an anti-surge PI
(gains 3 and 0.3) at its own control line, G5 at 5 % and G10 at 10 %, with
its override at 0.4 of that line. The pair runs at its shipped gains
(section 7). Cost is the version-2 cost per episode, trips included, and
"/ G10" is that cost over G10's. The smallest margin, phi_min / 2W - 1, is
over every untripped step of the 256 episodes. Recycle power is the mean over untripped steps, the
energy share is the running term's share of the cost, and valve travel is
the sum of the valve's moves per episode, in openings.

| policy | trips per episode | episodes with a trip | cost per episode | / G10 | smallest margin | recycle power (kW) | energy share | valve travel |
|---|---|---|---|---|---|---|---|---|
| recycle shut, speed PI | 36.85 | 100 % | 3.75e11 | 6.88e5 | 0.0000 | 91.3 | 0.00 % | 8.41 |
| fixed recycle 0.10 | 27.26 | 96.9 % | 2.78e11 | 5.09e5 | 0.0000 | 92.0 | 0.00 % | 5.26 |
| fixed recycle 0.20 | 10.78 | 69.5 % | 1.10e11 | 2.01e5 | 0.0000 | 102.8 | 0.00 % | 1.65 |
| fixed recycle 0.25 | 0.523 | 22.7 % | 5.33e9 | 9787 | 0.0000 | 112.3 | 0.00 % | 0.15 |
| fixed recycle 0.30 | 0 | 0 | 5.64e6 | 10.35 | 0.0401 | 130.9 | 0.02 % | 0.05 |
| fixed recycle 0.50 | 0 | 0 | 3.58e7 | 65.67 | 0.2079 | 177.4 | 0.01 % | 0.15 |
| chaser, bias 0.2 | 6.109 | 74.2 % | 6.22e10 | 1.14e5 | 0.0000 | 114.4 | 0.00 % | 14.16 |
| chaser, bias 0.3 | 0.238 | 11.3 % | 2.43e9 | 4452 | 0.0012 | 131.9 | 0.00 % | 14.42 |
| chaser, bias 0.35 | 0.062 | 3.9 % | 6.37e8 | 1169 | 0.0011 | 145.4 | 0.00 % | 14.55 |
| chaser, bias 0.4 | 0.008 | 0.8 % | 8.03e7 | 147.4 | 0.0033 | 155.6 | 0.00 % | 14.51 |
| chaser, bias 0.45 | 0 | 0 | 1.06e6 | 1.95 | 0.1522 | 162.6 | 0.18 % | 14.49 |
| guarded, 5 % line (G5) | 0 | 0 | 5.36e5 | 0.98 | 0.0132 | 123.0 | 0.22 % | 18.42 |
| guarded, 10 % line (G10) | 0 | 0 | 5.45e5 | 1.00 | 0.0591 | 123.3 | 0.22 % | 18.41 |
| the pair | 0 | 0 | 2.26e6 | 4.15 | 0.0907 | 63.7 | 0.02 % | 2.78 |
| zero action | 0 | 0 | 1.16e8 | 213.4 | 0.2079 | 123.4 | 0.00 % | 0.15 |
| random actions | 0 | 0 | 2.47e8 | 453.7 | 0.2036 | 125.6 | 0.00 % | 19.71 |

A valve fixed at 0.25, just under the static bound of 0.256, trips in
22.7 % of episodes, and 0.30 is the smallest fixed opening in the table that
never trips. The chaser stops tripping only at bias 0.45, where it spends
162.6 kW on recycle. The cheapest policy with no trip in these episodes is
G5, 5.365e5 per episode, of which recycle energy is 0.22 %, and a trip costs
1.9e4 of its episodes (measured, `--hardness`). G5 is a rare tripper, with
4 trips in 5120 episodes (`--v1`, section 8). G10 costs 1.6 % more and did
not trip here. The pair costs 4.15 times G10 and spends about half its
recycle power (63.7 against 123.3 kW), with 2.78 openings of valve travel per
episode against G10's 18.41. The pair tracks the header pressure with speed
alone, and the guarded heuristic also opens the valve to track it.

---

## 6. Sensitivity of the values labelled ours

`--sensitivity` re-runs five policies of the hardness table on the env with
one value changed per row (256 episodes per cell, seeds 1000 to 1255,
float32, measured): the recycle shut under the speed PI, the chaser at bias
0.3, a fixed recycle of 0.30, the shipped pair and G10. Costs are per
episode, trips included. The pair's and G10's trips are counts over the 256
episodes.

| row | f_H (Hz), B | recycle shut: trips per episode | chaser 0.3: episodes with a trip | fixed 0.30: cost / G10 (episodes with a trip) | pair: cost / G10, trips, smallest margin | G10: cost, trips, smallest margin |
|---|---|---|---|---|---|---|
| as shipped | 2.21, 1.80 | 36.85 | 11.3 % | 10.35 (0 %) | 4.15, 0, 0.0907 | 5.449e5, 0, 0.0591 |
| drive 1.66 %/s | 2.21, 1.80 | 35.28 | 22.3 % | 12.33 (0 %) | 6.20, 0, 0.0907 | 5.925e5, 0, 0.0601 |
| drive 10.4 %/s | 2.21, 1.80 | 42.66 | 4.7 % | 9.24 (0 %) | 1.70, 0, 0.0859 | 5.007e5, 0, 0.0563 |
| valve close 33 %/s (3 s full close) | 2.21, 1.80 | 94.07 | 24.6 % | 0.13 (0 %) | 310.40, 341, 0.0000 | 4.376e7, 1, 0.0030 |
| valve open 100 %/s | 2.21, 1.80 | 36.85 | 12.1 % | 5.45 (0 %) | 2.18, 0, 0.0942 | 1.035e6, 0, 0.0791 |
| deviation sd 0.01 | 2.21, 1.80 | 36.72 | 12.9 % | 10.65 (0 %) | 4.31, 0, 0.0981 | 5.218e5, 0, 0.0689 |
| deviation sd 0.04 | 2.21, 1.80 | 37.13 | 10.5 % | 592.18 (3.1 %) | 3.76, 0, 0.0745 | 6.145e5, 0, 0.0516 |
| demand ramp 2 s | 2.21, 1.80 | 36.85 | 11.7 % | 11.66 (0 %) | 3.94, 0, 0.0907 | 5.847e5, 0, 0.0470 |
| demand ramp 10 s | 2.21, 1.80 | 36.76 | 10.5 % | 9.39 (0 %) | 4.28, 0, 0.0907 | 5.278e5, 0, 0.0664 |
| constant setpoint 24 kPa | 2.21, 1.80 | 36.65 | 0.0 % | 11.95 (0 %) | 6.19, 0, 0.0907 | 1.758e5, 0, 0.0632 |
| setpoint range 22 to 26 kPa | 2.21, 1.80 | 36.74 | 3.1 % | 12.15 (0 %) | 5.34, 0, 0.0907 | 2.459e5, 0, 0.0642 |
| `V_p` 0.5x | 3.13, 1.27 | 36.26 | 8.6 % | 9.90 (0 %) | 3.97, 0, 0.0931 | 5.638e5, 0, 0.0721 |
| `V_p` 2x | 1.56, 2.54 | 37.01 | 15.6 % | 9.21 (0 %) | 3.68, 0, 0.0812 | 6.234e5, 0, 0.0613 |
| `L_c` 0.5x | 3.13, 2.54 | 37.01 | 14.5 % | 10.36 (0 %) | 4.11, 0, 0.0915 | 5.477e5, 0, 0.0660 |
| `L_c` 2x | 1.56, 1.27 | 36.48 | 11.7 % | 10.43 (0 %) | 4.19, 0, 0.0889 | 5.420e5, 0, 0.0588 |

**Restart break-evens.** Over the twelve policies of the `--v1` set on the
same 256 seeds, version 2 ranks every safe policy above every tripping one
from `restart_steps` = 429, and version 1 from 2771 (measured,
`--sensitivity`). The shipped 9000 is above both.

**Setpoint.** `p_ref_range` and the moving setpoint are TUNED by the
setpoint rows, which ran on steps. Held at 24 kPa, the setpoint lets the margin-blind chaser
run without a trip and cuts G10's cost to a third; over 22 to 26 kPa the
chaser trips in 3.1 % of episodes. The steps, and their size, are what make
a policy that ignores the margin trip.

**Ramps and the offset.** The table above ran on steps. Share of 256 episodes with a
trip on the shipped plant (seeds 1000 to 1255, measured):

| chaser bias | steps, offset on | 5 s ramp | 10 s ramp, as shipped | 10 s ramp, offset off |
|---|---|---|---|---|
| 0.20 | 77.7 % | 70.3 % | 69.1 % | 64.5 % |
| 0.25 | 38.7 % | 25.0 % | 19.9 % | 12.9 % |
| 0.30 | 12.5 % | 2.7 % | 0.8 % | 0.0 % |
| recycle shut | 100 % | 100 % | 100 % | 100 % |

Ignoring the margin still trips the plant, but the near-safe end of that family is
much safer, because a falling setpoint no longer asks the drive for a full-rate cut.
Bias 0.3 is close to a fixed valve that clears every block, so
`test_setpoint_chasing_without_the_margin_trips`
moved to bias 0.25. Ramp length matters little between 5 and 10 s, and the offset adds
about a third to the trips at 0.20 to 0.25, which is what it is for. Figures elsewhere
that quote the chaser at 0.3 are the stepped plant's.

**Helmholtz scale.** The `V_p` and `L_c` rows move f_H over 1.56 to
3.13 Hz and B over 1.27 to 2.54. Across them the recycle-shut policy trips
36.3 to 37.0 times per episode, the chaser in 8.6 to 15.6 % of episodes, and
the pair costs 3.68 to 4.19 times G10; neither trips, and the pair's smallest
margin stays within 0.081 to 0.093. These rows move the figures less than
the drive and valve rows do.

**Demand.** A 2 s ramp leaves the pair's smallest margin at 0.0907, its
value as shipped, and takes G10's from 0.0591 to 0.0470; neither trips. A
deviation sd of 0.04 makes the fixed 0.30 trip in 3.1 % of episodes and
takes the pair's smallest margin to 0.0745, just under its override line
(0.075), with no trip.

**Drive rate.** Gravdahl, Egeland and Vatland (2002) Table 2 (p.1892) list a
maximum acceleration "dN/dt_max = 17.4 1/s^2" beside a maximum speed of
12000 rpm and, in Table 1, a design speed of 10000 rpm (read). The paper does
not define the unit of N in the table; its shaft state is omega in rad/s and
its figures plot rpm. Read as rev/s^2 the limit is 10.4 %/s of design speed;
read as rad/s^2 it is 1.66 %/s (derived). The paper's torque budget favours
rad/s^2. At the design point the drive has 8000 - 7100 = 900 N m over
35.9 kg m^2, 25.1 rad/s^2 or 2.39 %/s (derived), so a 10.4 %/s limit could
never bind there. That is an inference the source does not state. The source
machine is a 7.5 MW, 10000 rpm pipeline compressor (read), and this plant is
a low-pressure-ratio blower of about 0.5 MW (ours, an estimate: 17 kg/s
heated by about 30 K at c_p 1005 J/(kg K)), so the rate here is ours,
3 %/s, and TUNED by the 1.66 and 10.4 %/s rows. At 1.66 %/s the chaser
trips in twice as many episodes (22.3 % against 11.3 %) and the pair costs
6.20 times G10; at 10.4 %/s the chaser trips in 4.7 % and the pair costs
1.70 times G10. In neither row do the pair or G10 trip.

**Valve closing.** The task ships a 10 s full close, the slowest Mirsky et
al. p.23 allow. Wilson and Sheldon's 3 s full close is the 33 %/s row. There
the recycle-shut policy trips 94 times per episode against 37, and G10 trips
once in the 256 episodes, with its margin down to 0.003. The pair as shipped
before 2026-10-28 (kp_surge 1.4, ki_surge 6.0; this row was not re-run with the
current gains) trips 341 times over the 256 episodes, 1.33 per
episode, with its smallest margin at 0, and costs 310 times G10. The pair at
`DEFAULT_GAINS` (kp_surge 2.0, ki_surge 3.0) ran the same row with no trip
and a smallest margin of 0.0469 (measured, `--sensitivity` with no
`compressor_surge` entry in `src/target_gym/data/pid_gains.json`, so that
`load_gains()` returns `DEFAULT_GAINS`). The tuner and the guard's battery
run at the shipped 10 s close, and at that close the shipped pair has no trip
in the 4096 validation episodes (section 7). A change to `valve_close_rate`
therefore calls for tuning and validating the pair again.

---

## 7. Experts

Both controllers live in `experts.py`, outside the version stamp; the env
reaches them only inside `make_pid` and `make_mpc`.

### The pair

The arrangement industrial anti-surge systems use (Mirsky et al. 2015,
pp.11 and 13), on the observation:

- **Performance PI.** Speed command = I_N + kp_speed e, with e the setpoint
  less the header pressure in kPa, clipped to 70 to 105 %. The integral
  moves by ki_speed e dt unless the command is saturated in the direction
  of e.
- **Anti-surge PI.** The flow-coefficient margin sm = m_c / (12.25 N) - 1
  comes from the flow and speed channels (the speed channel is in percent,
  so N = obs[3] / 100). Its only model knowledge is the surge line, which a
  vendor gives every anti-surge controller. Valve command = I_x + kp_surge
  (surge_line - sm), clipped to [0, 1]. Before each update the integral is
  clipped to within `reset_band` of the observed valve (external reset), so
  it cannot wind up while the valve closes at its 10 %/s limit; it also
  stops at the stops.
- **Override.** A margin under `override_line` opens the valve fully and
  holds it open for `override_hold` after the margin recovers, 20 steps
  counting the triggering one.

The memory starts at the observed drive setpoint and valve (`pair_init`), so
the speed loop takes over bumplessly from any state and the valve loop from
its control line. It persists through a trip (the suite's convention).

The stored values come from section 9 of
`docs/proposals/expert-protocols-process.md`, run by
`scripts/tune_process_experts.py --envs compressor_surge`. Each test starts at the
pair's rest at the levels the schedule ends on, with no stored gain in the way.

| value | stored | label and source |
|---|---|---|
| `kp_speed`, `ki_speed` | 0.0289, 0.0164 | SIMC on speed to pressure: k = 47 kPa per unit speed, tau1 = 1.77 s, theta = 0.65 s. A drive, so no dead band. 8 of 10 tests: two seeds trip in the hold, on a hidden line right of 0.04 |
| `kp_surge`, `ki_surge` | 1.0, 4.68 | SIMC on recycle to margin: k = 1.16, tau1 = 0.214 s, theta = 0.05 s, 1.02 capped at Mirsky's 1. N7 stops at Kc,min. 6 of the 8 points with the valve open |
| `Tf_surge` | 0.021 s | N5's cap. Under the 0.1 s step, so it barely filters |
| `surge_line` | 0.10 | read, Mirsky's 10 %. The schedule check passes it: worst margin 0.061, no trips on seeds 100 to 109 |
| `override_line` | 0.05 | read, half the control margin |
| `override_hold` | 2 s | derived, the valve's full-open stroke |
| `reset_band` | 0.014 of opening | derived, C8 |

**Pairing.** RGA(dp on N) is 0.91 at the median rest with the valve open, so speed
holds pressure and the valve holds the margin, as shipped.

**Held-out check** (seeds 200 to 209). It holds 12.8 per step (worst seed 24.9)
against a ceiling of 1, with no trips. The best null is constant 0.5, ratio 0.011,
0 of 10 wins. The valve travels about 0.3 % per step in hold, from 0.71 %, still over
the positioner's 0.25 %, since the filter's cap is under one step. The true margin
over each episode's hidden line falls to 0.014 at worst. The opening still costs 57
to 92 % of an episode: the anti-surge PI shuts the valve while the slower speed loop
catches up. That is the recipe's result, and the structure has no term to coordinate
them. Over 1024 validation seeds it does not trip, and its margin reaches 0.0496,
under the override line: a finding, `test_the_pair_holds_the_validation_seeds`.

The tests in `tests/compressor_surge/test_compressor_surge_experts.py` hold the
pair. `test_default_gains_are_finite_and_tuned_ones_nonzero` and
`test_gains_are_read_under_the_task_key` check the gains and where they are
read from. `test_shipped_gains_pass_the_guard` and
`test_guard_refuses_known_bad_gains` check the guard below.
`test_override_opens_fully_and_holds`, `test_integrators_do_not_wind_up`
and `test_reset_clears_the_pair` check the override, the anti-windup and the
reset. `test_functional_core_matches_the_class` checks that the functional
core (`pair_step`, which the numbers script runs under `vmap`) agrees with
the stateful controller.

### The guard

`check_pair_gains` reports gains that fail any of three checks, and since D2 of the
process protocols the factory does not call it
(`check_pair_gains`, which raises `UnsafePairGains`):

1. **Linear decay.** At five points where the anti-surge loop is active
   (20, 24 and 28 kPa at opening 0.35; 24 kPa at 0.50; 28 kPa at 0.60), the
   Jacobian of one closed-loop step (`pair_step`, then the env's
   `compute_next_state` with the deviation off) in (m_c, dp, N, N_ramp, x,
   I_N, I_x) must have its slowest mode decaying at 0.2 /s or faster
   (`GUARD_MIN_DECAY`, ours). It is taken at the closed loop's steady state
   in closed form (`settled_point`: on the control line with the valve at
   0.148 to 0.335 at these points), where the rate limits, the override and
   the reset band are inactive. Float64.
2. **Margin.** A stress battery of eight runs of 410 steps on the env from
   settled points, each through a setpoint step and a 5 s demand ramp at
   the same 60 s boundary: demand drops from 0.95 to 0.35 at 20, 24 and
   28 kPa and with the setpoint moving 28 to 20 and 20 to 28 kPa, the two
   setpoint moves alone at demand 0.35, and the clip corner, 0.95 to 0.20 at
   24 kPa. No run may trip, and the smallest margin phi_min / 2W - 1 after
   the move must be at least 0.098 (`GUARD_MIN_MARGIN`). Float32, as the env
   ships.
3. **Valve travel.** In each battery run, the valve command's travel (the
   sum of its step-to-step changes) over the last 100 steps
   (`GUARD_TRAVEL_STEPS`, 30 to 40 s after the move) must be at most 0.01 of
   opening (`GUARD_MAX_TRAVEL`, ours). The decay check linearises with every
   rate limit inactive. Once the valve's rate limits act, a loop that passes
   it can still ring, with the valve ratcheting open and shut against its
   limits in a limit cycle, and this check refuses such a loop.

`GUARD_MIN_MARGIN` is the battery minimum of `DEFAULT_GAINS`, 0.1133, less
0.015 (ours), rounded down to 0.001, and never below the override line: 0.098
(derived, `--section guard`). It is frozen with the physics, and the 0.015
leaves the tuner room to trade margin for tracking.

| gains (kp_speed, ki_speed, kp_surge, ki_surge) | slowest decay at the five points (/s) | battery minimum | largest valve travel, last 100 steps | verdict |
|---|---|---|---|---|
| `DEFAULT_GAINS` (0.08, 0.04, 2, 3) | 0.451 to 0.461 | 0.1133 | 0.0000 | passes |
| anti-surge (3, 3) | 0.451 to 0.461 | 0.0983 | 2.5232 | refused on travel |
| anti-surge (3, 0.3) | 0.083 | 0.1036 | 2.3523 | refused on decay, travel |
| a slow pair (0.01, 0.005, 1.5, 3) | 0.195 to 0.220 | 0.1146 | 0.0001 | refused on decay (0.195 at 20 kPa) |
| anti-surge (0.5, 0.05) | 0.045 | 0.0618 | 0.0094 | refused on decay, margin |
| anti-surge (1, 1) | 0.453 to 0.462 | 0.0838 | 0.0000 | refused on margin |
| anti-surge (12, 1.2) | -20.8 to -19.6 (grows) | 0.1064 | 12.9731 | refused on decay, travel |
| speed (0.3, 0.3) | -3.9 to -3.0 (grows) | 0.1127 | 0.0716 | refused on decay, travel |

(decay derived, battery minimum and travel measured, `--section guard`;
`test_guard_refuses_known_bad_gains` asserts the refusals of the anti-surge
sets (3, 0.3), (0.5, 0.05), (12, 1.2), (3, 3) and (1, 1) and of speed
(0.3, 0.3)). The shipped gains pass all three checks
(`test_shipped_gains_pass_the_guard`).

The pair's float64 host controller and its float32 functional core, run on
the same seed through a whole episode, keep the header pressure within
0.01 kPa of each other at every step
(`test_functional_core_matches_the_class`). The battery ranks gains, and the
validation on 4096 episodes below bounds the tail.

**Cost.** Every process that builds the pair runs the guard once.
`check_pair_gains` takes 0.9 to 1.1 s cold in a fresh process with
`OMP_NUM_THREADS=1` (compile and run; importing `target_gym` takes 1.7 to
2.3 s more), against a budget of 15 s, and 0.01 s for each further gain set,
since the gains are traced (measured, `--section guard`, five runs on a
machine at load average 2.7 to 12).

**Tuning.** `scripts/tune_industrial_experts.py --envs compressor_surge`
runs the coordinate search of `scripts/tune_pid.py` on episode return over
the training seeds 100 to 109, from `DEFAULT_GAINS`, on the four tuned gains.
A candidate the guard refuses scores minus infinity. The search multiplied
`ki_surge` by 0.25 (3 to 0.75) and `kp_speed` by 1.4 (0.08 to 0.112) and
kept `ki_speed` and `kp_surge`, which moved the training return from
-2.003e6 to -1.938e6 (measured, printed by that script). The gains it
replaces (kp_surge 1.4, ki_surge 6.0) came from the same search over seeds 0
to 23, which include the scored ones. On the held-out seeds 200 to 209 the
pair holds 3.93 a step, the previous gains 5.29, against a ceiling of 1:
0.59 decades of room. The shipped gains are stored in
`src/target_gym/data/pid_gains.json` under `compressor_surge`.

**Validation** (`--pid-validation`, 4096 episodes, seeds 1000 to 5095,
float32, measured). The shipped pair trips in none of them, a one-sided 95 %
bound of 7.3e-4 trips per episode. Its smallest margin per episode is 0.0951
at the minimum, 0.0992 at the 0.1 % quantile, 0.1035 at 1 %, 0.1071 at 5 %
and 0.1144 at the median, and no episode goes under the override line. Per
episode it averages 65.1 kW of recycle power, a mean absolute error of 0.379
kPa, 4.76 openings of valve travel and a cost of 2.072e6 (the previous gains:
0.0877 at the minimum, 0.386 kPa, 2.81 openings, 2.119e6). On the first 1024
episodes, which `test_the_pair_holds_the_validation_seeds` runs, the smallest
margin is 0.0954. Under the decision rule, a trip or a margin under the
override line would raise `GUARD_MIN_MARGIN` by 0.01 and tune the pair once
more, and a second failure would ship `DEFAULT_GAINS` (`--pid-validation`
prints the rule with its verdict). The validation passes, so the tuned gains
ship. Its smallest margin, 0.0951, is under the battery's floor of 0.098,
which ranks gains on eight moves from settled points, and above the override
line, which the rule reads.

### The oracle

The MPC slot, built by `make_compressor_surge_mpc`, is a problem definition,
not a tuned controller (docs/proposals/oracle-audit.md): the shared solver's
oracle (`target_gym.oracles`), brought under the rule by the smoothing
protocol's decision S10. It reads no expert code and no gains, and never
hands a step to the pair.

**Execution.** Once every T_c, Cooper's sample time from the horizon
protocol (one step, `data/oracle_horizons/compressor_surge.json`), it plans
P = 187 moves of both commands from the true state, fewer near the episode's
end, and applies the first. A cold start puts every planned state at the
measured one and holds the drive setpoint and the valve where they are, a
later solve starts from the last plan shifted one
step, and a warm solve that fails or hits the cap is solved again cold,
and one that still fails applies the last plan's next move.

P is Cooper's 123 plus the 64 samples the drive's rate limit takes to carry the
header pressure across its 8 kPa target range (H21): a bump test is too small
to reach that limit, and at a short horizon the plan saw a large setpoint step too late
to start the drive.

**Problem.** The plan maximises the episode reward over the horizon. The
model is the env's step written again on the env's own functions
(`compressor_step`): ten substeps, each moving the drive setpoint and the
valve toward their commands at their rate limits and then taking one RK4
step, with the consumers' deviation on its noise-free mean path, a^k times
its measured value. The rewrites, by the smoothing protocol's inventory:

- the commands' clips are replaced by bounds on the moves (T1);
- the tracking term is written as (e / e_floor)^2, the reward itself since
  e_tol is 0 and p is 2, and the running cost max(P_rc - c_hold, 0) / c_hold and
  the wear term through epigraph rows on every planned step, also exact (T2);
- the trip, phi_min below the episode's line, is a hard limit on every planned step, elastic
  at the trip cost per unit of Phi (T3), planned 0.0040 inside the line
  (K11): the smoothing's gap bound on phi_min, 6.7e-5, and the 1 % quantile of
  one innovation of the demand deviation, 0.0039, at the pair's states, plus
  the solver's tolerance. Planned on the line, the plant tripped, even
  without noise;
- the characteristic's switch to its tangent past `phi_join` is left as it
  is, and each plan's substep Phi is checked against it (T4, K9);
- each command lies within the travel its actuator covers over the step,
  from where it stands (T1r, the smoothing protocol's Amendment 1): past it
  the drive or valve runs at full rate in every substep, as at it, so the
  bound is exact. Without it a command past the reach changed nothing in
  the model, IPOPT certified plans that did nothing, and the oracle lost to
  the pair 20 to 60 times over;
- the corner left inside the reach, where an actuator arrives mid-step, is
  smoothed by a softplus at its K5 width: the largest of
  s/10, ..., s/1e8, with s the clipped interval's span, whose one-step effect
  on the header pressure stays under half the transmitter's resolution at
  the states the pair visits on training seeds 100 to 109
  (`scripts/measure_oracle_widths.py`, `data/oracle_widths/compressor_surge.json`).
  A cold solve reaches those widths by continuation from a tenth of each
  span (K7).

Two moves track one output, and while recycle power is under `c_hold` the
reward does not say how to split the work, so the solve adds a tie-break on
each move's change, sized to cost at most 0.25 reward units over an episode
(G10 in docs/proposals/oracle-solver.md). It picks one plan among equal ones
and stays out of the plan's cost and the record.

The hand-set terms of the NMPC it replaces are gone: the error scale, the
move and terminal weights, the frozen 49.6 kW power reference, the 5 %
surge margin and the 0.1 rounding of the clips. Its reach constraints
return as T1r.

**Solver and certificate.** IPOPT's algorithm solves it with exact Hessians
to a scaled KKT error of 1e-8, or stops at 150 iterations. Every solve's
certificate (KKT error, iterations, violation, and the plan replayed on
`step_env` from the same start, which measures what the smoothing costs) is
summed into the record.

**Recorded results.** Both controllers' returns over 10 seeds
(`scripts/record_baselines.py`) are on the
[environment page](../../../docs/environments/compressor_surge.md#baselines), and their
protocol row over 3 seeds (`scripts/evaluate_baselines.py`: gain, hold, reach
and transient cost per change, failure rate) is on the
[results page](https://yannberthelot.github.io/TargetGym/baseline-results/). Both are generated from the records in
`src/target_gym/data/`, so neither is copied here to go stale. The NMPC's
solver statistics are in its row of
`src/target_gym/data/baseline_returns/compressor_surge.json`. The gain is the mean
cost over every step.
The reach cost is a controller's cost above its own hold level from a change
until it settles, and the transient cost is the cost summed over the
change's transient (`target_gym.eval`).

---

## 8. Version-1 reward and its exploit

Version 1 (ours) is log-scaled tracking between `precision_floor` and
`dp_error_max`, 1 at zero error and 0 at 20.68 kPa, with no recycle term, and
exactly `-restart_steps` (-9000) on a tripped step. It is therefore not
bounded in [0, 1] on this task.

With 0 on the tripped step, a restart would put the plant back on its reset
equilibrium at no cost, and policies that trip often would outscore safe
ones. `test_v1_ranks_tripping_below_safe` checks both scorings on seeds 0
to 31 with six policies: the recycle shut under a pressure PI on speed, a
fixed recycle of 0.10 and the margin-blind chaser at bias 0.2, which each
trip at least once per episode on average, and a fixed recycle of 0.30,
zero action and the pair at `DEFAULT_GAINS`, which never trip. With 0 on a
tripped step the chaser scores above the fixed 0.30. As shipped, each of the
three tripping policies scores below zero action, zero action below the
fixed 0.30, and the fixed 0.30 below the pair.

**The exploit, measured** (`--v1`, 128 episodes per policy, seeds 1000 to
1127, float32). The set is the test's six policies, with the pair at its
shipped gains (section 7) in place of `DEFAULT_GAINS`, and six more: G10,
full travel low, a fixed recycle of 0.20, 0.25 and 0.50 and the chaser at
bias 0.3. The rest of this section reports this run.

| policy | trips per episode | v1 per step, 0 on a trip | v1 per step, as shipped | v2 cost per step |
|---|---|---|---|---|
| recycle shut, speed PI | 36.26 | 0.3701 | -271.6 | 3.076e8 |
| fixed recycle 0.10 | 26.23 | 0.4714 | -196.3 | 2.226e8 |
| fixed recycle 0.20 | 10.73 | 0.5835 | -79.92 | 9.106e7 |
| fixed recycle 0.25 | 0.44 | 0.6311 | -2.650 | 3.715e6 |
| chaser, bias 0.2 | 6.06 | 0.6853 | -44.78 | 5.143e7 |
| chaser, bias 0.3 | 0.23 | 0.7029 | -0.996 | 1.922e6 |
| full travel low | 39.28 | 0.2873 | -294.3 | 3.332e8 |
| fixed recycle 0.30 | 0 | 0.5908 | 0.591 | 4929 |
| fixed recycle 0.50 | 0 | 0.3421 | 0.342 | 3.052e4 |
| zero action | 0 | 0.1526 | 0.153 | 9.779e4 |
| the pair (shipped gains) | 0 | 0.7809 | 0.781 | 1815.5 |
| guarded, 10 % line (G10) | 0 | 0.6902 | 0.690 | 474.7 |

With 0 on a tripped step, 17 of the 35 (tripping, safe) pairs rank the
tripping policy higher, and the Spearman correlation with version 2 over the
twelve policies is +0.483. A tripped step must score below -2915 to reverse
all 17. The shipped -9000 does, and as shipped the correlation is +0.993.
Version 1 then orders the twelve policies as version 2 does, except that it
puts the pair above G10.

A policy that trips rarely can still rank above a safe one under version 1.
Version 1 prices a trip at 9000 steps of its best score, 7.5 episodes
(derived, 9000 / 1200). Version 2 prices it at `failure_cost` x
`restart_steps` = 1.018e10 (derived), 1.79e4 episodes of G10, the cheapest
safe policy of the set (measured, `--v1`). Two rare trippers run 5120
episodes each (seeds 1000 to 6119, measured, `--v1`; intervals are Poisson
95 %):

| rare tripper | trips (per episode) | v1 per step, 0 on a trip / as shipped | v2 per episode | v1 charge that ranks it below fixed 0.30 | trip rate at which v2 ranks it level with fixed 0.30 | v1 charge with that break-even |
|---|---|---|---|---|---|---|
| guarded, 5 % line (G5) | 4 (7.8e-4, 2.1e-4 to 2.0e-3) | 0.6921 / 0.6862 | 8.502e6 | 1.56e5 (6.07e4 to 5.71e5) | 5.27e-4 | 2.31e5 |
| chaser, bias 0.4 | 13 (2.5e-3, 1.4e-3 to 4.3e-3) | 0.6487 / 0.6297 | 2.660e7 | 2.74e4 (1.6e4 to 5.14e4) | 5.07e-4 | 1.37e5 |

The fixed 0.30 scores 0.5908 per step under version 1 and costs 5.915e6 per
episode under version 2 on the 128 episodes above. Version 1 as shipped
ranks both rare trippers above it, and version 2 ranks both below it. For
version 1 to agree at their measured rates, a trip would have to cost 2.74e4
(the chaser) and 1.56e5 (G5) steps, 3.0 and 17 times the shipped 9000. The
charge stays `-restart_steps`, the `unstable_cstr` convention.

---

## 9. Known deviations

Each is a strict xfail in
`tests/compressor_surge/test_compressor_surge_physics.py`, which fails the
suite if the model starts doing it. The numbers below are what the model
gives in each test (derived, `--section deviations`).

**⚠️ D1. No rotating stall.** Left of the peak a real compressor runs in
rotating stall, whose annulus-averaged pressure rise lies below the
unstalled characteristic. Here the model has one branch. The stable sliver at
70 % with the recycle shut holds an equilibrium 2 % left of the peak, and
10 s of stepping leave dp on the cubic (15836.5 Pa against 15836.5), where
`test_equilibrium_left_of_the_peak_sits_below_the_cubic` asks for 5 % below
it. The trip at the peak keeps every continuing state right of it.

**⚠️ D2. Incompressible duct.** At the largest flow coefficient the plant
reaches, 0.777 at 105 % with both valves open, the duct's throughflow is Mach
0.48, where the isentropic density is 10.6 % below the suction density
(derived). The model's flow per unit velocity and area is the suction
density; `test_duct_flow_is_compressible` asks for at least 5 % less.

**⚠️ D3. Plenum speed of sound at suction temperature.** The plenum holds gas
compressed to about 1.32 times suction pressure, at least 23.7 K warmer
(isentropic, derived), which raises its speed of sound and the Helmholtz
frequency by about 4 %. The env's pair on the peak has modulus 13.8923 rad/s
against the suction value 13.8924; `test_helmholtz_frequency_sees_the_discharge_temperature`
asks for at least 3 % more.

**⚠️ D4. The trip is instantaneous.** Surge takes time to develop, and a
machine carried across its line for less than a Helmholtz period recovers.
Here, from an equilibrium 5 % right of the line at rated speed with the flow
knocked to 1 % left of it, the plant is back right of the line within one
period, and the env trips on the first step;
`test_a_short_excursion_does_not_trip` asks for no trip.

**⚠️ D5. The check valve leaks.** The smooth one-sided square root passes
k_d sqrt(s ln 2) = 0.706 kg/s at zero drop and full opening (derived), where a
real consumer valve passes nothing; `test_check_valve_is_tight` asks for
under 1e-3 kg/s.

**⚠️ D6. The valve has no dead time of its own.** Both sources allow up to 0.4 s, and
Wilson and Sheldon count it inside their 2 s full opening, so the plant opens at a
steady 50 %/s from the first substep. A small command therefore answers up to 0.4 s
sooner than a real valve would. A first-order lag on the command would stand in for
it, at the cost of a state in the oracle's model and a new reach bound there.

**⚠️ D7. The offset moves the trip, not the characteristic, and only to the right.**
A real line moves with the machine, and either way. Here the line stays at or right
of the peak, so no continuing state sits on the unstable branch left of it, which the
regime of validity needs. The price is that the map's line is the best case, so a
controller knows which way the error runs.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `compressor_surge` | 1200 | 0.1 | 2 min | 2 in [-1, 1] | 7 | 23 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + wear + failure)`, see
docs/reward-shaping.md. Tracking is the header-pressure error in kPa over
`e_floor`, squared. Running is the ideal compression power spent on recycled
gas, m_r dp / rho01, the avoidable part of the drive's consumption, charged in
full as a share of `c_hold`, in the contract's dimensionless form. It is not
priced, since a header-pressure deviation has no tariff that would put tracking in
the same currency. With a scalar `c_hold` under a moving schedule, blocks where
recycle is forced pay a running cost no controller can avoid.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.0275 kPa (derived: the resolution clamp binds) | `max(precision_floor, lowest per-seed MPC hold)`, with the hold from `scripts/measure_hold.py --envs compressor_surge`. The NMPC's lowest per-seed hold is 1.03e-4 kPa (measured; the pair's lowest is 0.043 kPa), 267 times finer than the transmitter's 0.0275 kPa, so the clamp binds and `e_floor` stays at the transmitter accuracy, a measured upper bound clamped at the instrument resolution (docs/reward-shaping.md, "How each floor was obtained") |
| `e_tol` | 0 (ours, provisional) | no header-pressure specification has been supplied |
| `tracking_exponent` | 2 (ours) | a pressure deviation has no linear settlement price |
| `dp_error_max` | 20.68 kPa (derived) | the 28 kPa top level against the 7.32 kPa lowest pressure an untripped plant holds (`--section params`, `test_restart_and_failure_cost_are_derived`); `test_the_error_envelope_is_certified` checks it against random bang-bang on both actuators over 512 episodes; the version-1 envelope and the base of `failure_cost` |
| `failure_cost` | 1.131e6 (derived), twice (20.68 / 0.0275)^2 | twice the largest tracking cost an untripped state reaches (derived) |
| `restart_steps` | 9000 (15 min, ours, provisional) | no source gives a restart time; a trip costs 9000 x 1.131e6 = 1.018e10 (derived, `--section params`, `test_restart_and_failure_cost_are_derived`) |
| `c_hold` | 62 270 W (62.27 kW, measured) | the NMPC's mean recycle power over its hold, from `scripts/measure_hold.py --envs compressor_surge` (the pair's is 66.1 kW). It replaced a provisional 49.6 kW; the oracle reads `c_hold` through the reward |
| `running_weight` | 1 (ours, provisional) | one floor-width of tracking error is worth once the hold-phase recycle power |
| `travel_free` | 0.0025 per step | read: the least a real valve follows, a digital positioner's hysteresis plus dead band, 0.25 % of span (Fisher DVC6200, `docs/proposals/measurement-noise.md`, the same figure as `eval.REVERSAL_DEAD_BAND`). Wear is charged above it. Not a measured hold, since the pair's own valve travels 0.0071 per step in hold (measured, seeds 100 to 109), mostly flow-noise chatter through `kp_surge` |
| `travel_weight` | 1 (ours, provisional) | twice the free travel in one step is worth one floor-width of tracking error. A full stroke at 0.05 per step costs 19 per step |
| `precision_floor` | 0.0275 kPa (derived) | the header-pressure transmitter's reference accuracy, 0.055 % of calibrated span (read: Yokogawa EJA530E gauge transmitter, GS 01C31F01-01EN, p.1, capsule A at spans of 20 kPa or more, https://web-material3.yokogawa.com/GS01C31F01-01EN.pdf; Siemens SITRANS P320, catalog FI 01 09/2026, p.1/14, 1 bar cell at a turn-down of 5 or less, https://support.industry.siemens.com/cs/attachments/109765052/sitransp_p320_p420_fi01_en.pdf), times a 0 to 50 kPa calibrated span (ours). It covers linearity, hysteresis and repeatability together, and the Yokogawa sheet states it as a 3-sigma bound, so an error below it is one the instrument cannot tell from a smaller one. Neither sheet gives a finer measuring resolution; the one resolution found, the current output's quantisation (Endress+Hauser Cerabar PMP71B, TI01509P, p.24, https://bdih-download.endress.com/file/11181feb66c9cd03e4256402f1eff641/TI01509PEN_1025-00.pdf), is finer than the repeatability and is not used. The value scales with the span. Version 1's floor and the clamp on `e_floor`, so while the clamp binds it sets the version-2 scale |
| `rho_floor_tracking` | 1.40e-5 (derived from the measured hold) | (lowest per-seed MPC hold / `e_floor`)^2 = (1.03e-4 / 0.0275)^2, from `scripts/measure_hold.py`; below 1 because the clamp binds |
| `rho_floor` | 1.40e-5 (derived from the measured hold) | equal to `rho_floor_tracking`, from when recycle power at `c_hold` cost nothing. It gains the running term, about `running_weight`, when PR 9 re-measures the hold |
| `floor_is_documented_minimum` | False | the plant is disturbed |

**The hold window.** `scripts/measure_hold.py` scores this plant's hold
over 3 seeds of 1200 steps. The first 150 steps are burn-in, three of the
plant's slowest cost-bearing time constants of about 5 s (the deviation's
correlation time, and the 4.93 s t63 of the header pressure after a speed
step, section 4), which leaves out the approach from the off-target reset.
In each setpoint block the hold starts 100 steps (10 s, ours) after the
change and runs to the next one, less the steps in which a controller
already moves toward the next level (`target_gym.eval.anticipations`). The
demand ramps inside a block stay in the hold, since they are the
disturbance the task holds against. The NMPC's hold scores 2165 steps and
the pair's 2250. The per-seed NMPC holds are 2.07e-4, 2.89e-4 and 1.03e-4
kPa, 2.0e-4 pooled, and the pair's 0.043, 0.048 and 0.083 kPa, 0.058 pooled.
Over the hold the NMPC spends a mean recycle power of 62.27 kW and the pair
66.1 kW (measured, `src/target_gym/data/hold_measurements.json`).
`test_floor_is_the_recorded_mpc_hold` checks `e_floor`, `c_hold`,
`rho_floor_tracking` and `rho_floor` against the recorded hold to 1 %, and
that the protocol's NMPC hold tracking cost, 0.0018, is at least 0.98 of the
reference.

`rho_floor_tracking` is the oracle's best hold for tracking: the lowest per-seed
hold the reference controller demonstrated, divided by `e_floor` and
squared, as the tracking term squares an error. It is 1.40e-5 here because
the floor is clamped at the transmitter accuracy, 267 times the NMPC's
lowest hold. `failure_cost` is unchanged by the measurement, since
`e_floor` is.

---

## Conformance notes

- **Check 3** (write-only fields). Every field is read: `phi_min` by the
  trip, `N_ramp` by the dynamics and the observation, `block_clock` and both
  level arrays by the schedules, `demand_dev` by the demand.
- **Check 5** (regimes join smoothly). Replayed as the conformance test runs
  it, from the PRNGKey(0) reset at zero action, every one of 847 next states
  is finite, and the largest one-sided Jacobian jump is 0.214 against the
  limit of 0.5, from `x` to `dp` at x = 0.503, where the recycle valve's
  rate limit reaches its command inside the step (measured, `--section
  invariance`). No `KNOWN_SEAMS` entry is needed.
- **Check 7** (the plant does not accelerate without input). Stable inside
  its envelope and zero action never trips; late-to-early increment ratios
  over 600 zero-action steps are at most 1.13 (the deviation, a stationary
  process) against the limit of 8 (measured, `--section invariance`).
- **Check 8** (the actuator moves the tracked variable). Full travel low
  (70 %, recycle shut) trips the plant from the PRNGKey(0) reset at every
  demand level from 0.20 to 0.95, in 3 steps at 0.20 to 117 at 0.95, and
  never at full opening; over 20 drawn schedules it trips in every one, first
  between 14 and 153 steps (measured, `--section reach`;
  `test_full_travel_low_trips` asserts the trips and their order). Full
  travel high never trips (`--section reach`,
  `test_full_travel_high_never_trips`).
- **Seeds.** Each trip restarts from a fresh draw keyed on its step, even
  under a constant rollout key (section 5). A valve loop that rings
  against the valve's rate limits amplifies float32 rounding: the battery's
  valve travel for the anti-surge sets (3, 3) and (3, 0.3) differs between
  float32 and float64 runs (measured, `--section guard`). Code that rounds
  differently can then differ in single trajectories and trip counts while
  it agrees on statistics over many episodes. The guard's travel check keeps
  such a loop out of the pair (section 7). This page takes every closed-loop
  figure from the numbers script, whose heuristic policies (`policy` and
  `_act`) the env tests also run.
- **The shared floor check cannot fail here.** It compares the protocol's
  whole-episode MPC tracking, 60.4 (measured), which includes up to
  (8 / 0.0275)^2 = 84628 per step (derived) while settling after a setpoint
  move, with a reference of 1.40e-5. `test_floor_is_the_recorded_mpc_hold`
  in `tests/compressor_surge/test_compressor_surge_experts.py` compares the
  floor with the recorded MPC hold: `e_floor`, `c_hold`, `rho_floor_tracking`
  and `rho_floor` against the NMPC's hold in `hold_measurements.json`, and
  the protocol's MPC hold tracking, 0.0018, against that reference.

## Readings

The observation reports the plant through its instruments, and the live setpoint
and the drive's own rate-limited reference `N_ramp` exactly
(`docs/proposals/measurement-noise.md`). `sensor_noise_scale = 0` gives the
noise-free task, rounding included.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `dp` | 9.17 Pa | none | Smart gauge pressure transmitter | Confirmed | [EJA530E](https://web-material3.yokogawa.com/GS01C31F01-01EN.pdf) |
| `m_c` | 0.021 kg/s | none | Annubar DP flow meter with fast DP transmitter | Provisional | [Rosemount 485](https://core.instrumart.com/assets/product-data-sheet-rosemount-dp-flow-meters-primary-elements-en-87598.pdf) |
| `demand_dev`, carries `m_d` | 8.4e-4 opening | none | Delivered-flow DP meter | Provisional | [Rosemount 485](https://core.instrumart.com/assets/product-data-sheet-rosemount-dp-flow-meters-primary-elements-en-87598.pdf) |
| `N` | 5.8e-5 of rated | 1e-4 of rated (1 rpm) | Speed probe with machinery tachometer module | Confirmed | [3500/50M](https://core.instrumart.com/assets/3500-50M-Datasheet.pdf) |
| `x` | 1.5e-3 | none | Recycle valve travel feedback in digital positioner | Ceiling | [DVC6200](https://www.emerson.com/is/content/emerson/en/final-control/flow-controls/documents/d103415x012.pdf) |

The delivered flow has no state field: `m_d` is computed from `dp` and the
opening. Noise on `dp` alone gives 0.03 % of flow, so the meter reads through
`demand_dev`. 8.4e-4 is 0.14 % of flow at the nominal opening of 0.6, and the
measured copy computes `m_d` from both, so that channel's scatter scales with
the flow.

Three figures are weak. `m_c` and `demand_dev` split the sheet's ±0.80 % of flow
into ±0.75 % and a remainder of 0.28 %, taken as 2 sigma. That split is ours, the
sheet calls the ±0.75 % the primary's best accuracy and not a bias, and turbulence
in the DP signal, which is real short-term scatter, is left out. `x` is a ceiling:
±0.3 % of output span is where the positioner puts the valve, not the scatter of
its travel reading.

`dp` is an upper bound: ±0.055 % of the 50 kPa span is a 3-sigma conformance figure
covering linearity, hysteresis and repeatability. `N` is accuracy treated as a
uniform count error, 1e-4 / √3, and this repository has no rated speed, so the
1 rpm step assumes the source's 10,000 rpm.

The expert reads these readings, and so does every tuner. The oracle plans and
acts on the true state, so it stays a full-state bound. `reward`, trips and the
score read the true state. `info["measured_reward"]` is the same cost on the
readings, with any trip taken from the true state, and a learner can train on
it.

`phi_min` has no instrument, so the surge trip stays true. The surge margin the
pair reads comes from `m_c` and `N`, so it carries their noise, and it reads the map,
so it also carries the episode's hidden offset.

**Timing and actuators.** A status per figure, for the owner to accept the ones
marked ours. The tracked readings the noise proposal left Provisional keep that
status in the table above.

| figure | value | status |
| --- | --- | --- |
| Recycle valve, full open | 2 s, dead time included | read, twice: Mirsky et al. p.23, Wilson and Sheldon Table 1 |
| Recycle valve, full close | 10 s | TUNED, read: the limit of Mirsky et al. p.23. Wilson and Sheldon ask for 3 s and note 10 s is also specified (section 6) |
| Recycle valve, dead time | none of its own | read and folded in, D6 |
| Drive rate | 3 %/s | TUNED, section 6 |
| `tau_N` | 1 s | **ours** |
| Control step | 0.1 s | **ours** |
| Sensor dead time and sampling | none | **ours.** No source gives one for these signals, so readings arrive within the step |

A plant resting exactly on target pays 0.11 per step in measured tracking cost,
`(9.17 Pa / 27.5 Pa)^2`.

## Written exception to check 2

The credibility check on a sourced real figure has nothing to read here. The owner
accepted a class exception on 2026-11-06: no published figure could be found or opened
after two searches with a re-fetch check. The plan in docs/proposals/industrial-fixes.md
says what was tried.
