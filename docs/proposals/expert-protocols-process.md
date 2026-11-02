# Expert protocols: process and industrial experts

Design document for owner review, 2026-11-02. Nothing here has been run or
implemented. It replaces the searched or hand-set values the
[audit](expert-tuning-audit.md) lists for the process and industrial experts.
The MPC oracles are out of scope.

**Owner decisions, 2026-11-02.** Every recommendation in section 8 is
accepted (D1 to D10). Where no source gives a number, the value comes from a
rule fixed here before any run and labelled ours.

> I want a data agnostic protocol to select this value, not a post-hoc
> selection. The protocol can include collecting data if it is reasonable and
> matches real life protocols, otherwise we stick to what the protocol gives
> us. The protocol has to come from real life practices. The goal is not to
> have the best results by tuning the controller, but to have the performance
> that the best available tuning protocol/heuristic from real life gives us.

Every value below comes from one fixed recipe, run once on bump tests from
training seeds 100 to 109. No score, no held-out seed and no menu picks a
value. Labels: **read** (a source states it), **derived** (the script
computes it from a read value or a plant datasheet figure), **ours** (no
source gives it, so it is decided here and the script holds no choice).

## 1. Choices shared by every recipe

| # | Choice | Decided value | Label and ground |
| --- | --- | --- | --- |
| C1 | Test data | One bump test per training seed (100-109) at that seed's state; response is the stepped run minus an unstepped run on the same noise | read: bump test (Shinskey and Gerry 2006); as the scripts do today |
| C2 | Step size | Existing steps stay. A new test steps 5 % of the actuator range, away from the nearer limit | ours; 5 % is what first_order, cstr, the unstable_cstr inner loop and the kiln already use |
| C3 | Model | First order plus delay; integrating (with lag) when tau1 > 8 theta | read: SIMC 2012 section 5.2 |
| C4 | Delay | theta = fitted delay + h/2, h the sample time | read: SIMC 2012 eq. 5.12 |
| C5 | Which fit | Spread = max\|k\| / min\|k\| over the ten fits. Spread <= 1.5: the median fit. Else: the fit with the largest \|k\|, with its own tau1 and theta | read: Shinskey, tune "at the operating point where the gain is highest", and a gain variation under about 50 % costs little (the 1.5). Derived: SIMC's gain margin is 3.14 (first order) or 2.96 (integrating), 2012 Table 5.2. Tuned on the median, a point 1.5 times higher keeps a margin near 2 and one 2 times higher (pH) near 1.5; tuned on the highest gain, every point keeps 2.96 or more. \|k\| is k, or the slope k' |
| C6 | tau_c | tau_c = theta. No menu | read: SIMC eq. 5.31, "tight" for active constraints |
| C7 | Rule | PI: Kc = tau1 / (k (tau_c + theta)), tau_I = min(tau1, 4 (tau_c + theta)). Integrating: Kc = 1 / (k' (tau_c + theta)), tau_I = 4 (tau_c + theta). If tau1 < theta, tau1 becomes tau1 + theta/3. Second order with tau2 > theta: PID, tau_D = tau2 | read: eq. 5.27-5.29, Table 5.1, eq. 5.38-5.39 |
| C8 | Tracking time | T_t = sqrt(tau_I (tau_c + theta)) wherever a clamp is sized from it | read: T_t above T_d and below T_i (Astrom and Hagglund 2006, 3.5). Ours: their sqrt(Ti Td) with T_d replaced by the loop's faster time scale, as these loops have no D |
| C9 | Poor result | It ships. The check run prints held-out hold, trips and the ceiling for the PR, and changes nothing | the owner's rule |
| C10 | Record | Each `pid_gains.json` entry stores gains, model (k, tau1, theta, chosen seed), tau_c, rule and source | ours |

The "tune for the largest gain" guidance is Shinskey's. I did not find it in
either SIMC text; C5 cites it from him and adds the margin argument.

## 2. ph_neutralization

**Source.** Shinskey, "Characterizers for control loops": the pH set point sits where the titration curve is steepest, the gain "must be adjusted for stability there", and a varying curve is matched at its most nonlinear. SIMC for the rule.

**Recipe.**
1. Run the bump test below on each training seed and fit first order plus delay in demand units (`reagent_demand`, nominal-buffer curve, as today).
2. Take the highest-gain model (C5), then theta + 2.5 s.
3. Kp = tau1 / (k 2 theta), Ti = min(tau1, 8 theta). Done.

**Test.** +0.1 mL/s on the base flow from each seed's reset state, 120 steps (600 s), against an unstepped run on the same buffer noise. Unchanged from today. The buffer is not forced: the ten resets draw it from its stationary law, and the low-buffer seeds are the steep ones.

**Free choices.** "The highest-gain point of the titration curve" is the seed with the largest \|k\| among the ten fits. No fit is dropped, whatever its residual.

**Poor result.** Expectation, not a result: PHYSICS.md section 7 puts the low-buffer demand gain near twice the median, so Kp should land near 8.7 and Ti near 20 s (today 8.79 and 40 s). If a seed cycles, it ships.

**Code.** `tune_ph_neutralization` in `scripts/tune_process_experts.py`: drop the `_pick_tau_c` call, add the C5 selector. Re-records `ph_neutralization`.

## 3. compressor_surge

**Source.** Mirsky et al. 2015 (METS III tutorial): the margin is "impossible to predict" for a given system, 10 % of flow is the common expectation and site testing sets it; the open-loop line sits "near the middle of the surge control margin"; a PI "nominal gain of less than 1"; the valve opens in under 2 s and closes in 8 to 10 s. McMillan 2012 (Control Global, cascade checklist): slow valves need external reset from the valve readback. Astrom and Hagglund 2006, 3.5: back-calculation. SIMC for the gains.

**Recipe.**
1. Operating points: each training seed's first setpoint and first demand level, at the pair's rest there (`settled_point`, surge_line 0.10).
2. Speed loop: valve held, speed command +1.75 % of rated (5 % of 70 to 105 %), header pressure against an unstepped run, 100 steps (10 s). Fit, C5, C7. `kp_speed` = Kc, `ki_speed` = Kc / tau_I.
3. Anti-surge loop: speed held, recycle command +0.05, at the points whose rest opening is above 0.02 (control line active), margin (`surge_margin_from_obs`) against an unstepped run, 100 steps. Fit, C5, C7. `kp_surge` = min(Kc, 1.0), `ki_surge` = `kp_surge` / tau_I.
4. `surge_line` 0.10, `override_line` 0.05, `override_hold` = 1 / valve_open_rate = 2 s, `reset_band` = `kp_surge` (1 - T_t / tau_I) `surge_line`.

| Value | Decided | Label and ground |
| --- | --- | --- |
| `surge_line` | 0.10 | read: Mirsky's 10 % of flow; the margin here is a flow-coefficient margin |
| `override_line` | 0.05 | read: Mirsky, open-loop line mid-margin |
| gain cap | 1.0 | read: Mirsky's "less than 1", on the margin scale PHYSICS.md already compares it to (Mirsky's DEV is not exactly our margin); D1 |
| `override_hold` | 2 s | derived: the valve's full-open stroke (env 0.5 per s; Mirsky under 2 s), equal to today's value. Ours: hold equals stroke |
| `reset_band` | derived | Astrom and Hagglund: at saturation the integrator sits K T_t / T_i times the error from the actuator; taken at the largest error, `surge_line` (margin 0, the trip). It is 0.05 when `kp_surge` is 1 and tau_I = 4 (tau_c + theta). Ours: C8 |
| step sizes | C2 | ours |

**Poor result.** The guard (`check_pair_gains`) has five thresholds that are all ours, so it cannot veto a protocol value (D2). It becomes a report: the battery of eight moves prints beside the held-out table, and a trip in it ships.

**Code and records.** `tune_compressor_surge` is rewritten on shared SIMC helpers and no longer calls `tune_pid.py`. `DEFAULT_GAINS`, `HELD_GAINS` and `SURGE_CONTROL_LINE` go; rendering reads the stored line. Re-records `compressor_surge`: the pair, the NMPC fallback to it (`pair_track`) and the oracle rows. PHYSICS.md section 7 tables that quote the pair are re-measured.

## 4. unstable_cstr

**Source.** SIMC for the loops (as today). McMillan 2012: a cascade's secondary setpoint limits must match the primary's output limits. DCS PID blocks carry setpoint high, low and rate limits (Foundation Fieldbus and Rockwell PlantPAX manuals). SIMC 2012 eq. 5.35 (input usage: first move within the actuator range) for the ramp. Astrom and Hagglund: a P-only loop leaves an offset.

**Recipe.**
1. Loop gains: SIMC as today (inner PD from the unstable-lag fit at the 37 training levels, outer PI at every third level), with C5 and C6.
2. `Tc_bias` = `steady_coolant(0.55, 0)`: the output at the middle of the 0.45 to 0.65 band, no drift.
3. Inner offset e_eq = max over training levels L and drift d in {-6, 0, +6} K of \|Tc*(L, d) - `Tc_bias`\| / Kp_T.
4. `sp_line_margin` = max over the same of (`Tc_bias` - Tc*)+ / Kp_T: how far the P loop holds T above its setpoint.
5. `sp_below` = `sp_above` = Kc_Ca x 0.10 + e_eq: the window never clips the outer PI's own move on the largest scheduled switch (0.10 mol/L). The point-of-no-return line stays the hot limit.
6. `sp_rate` = (smallest coolant headroom / Kp_T) / (tau_c + theta) of the inner loop: the steepest ramp that loop follows without saturating the coolant, since a ramp lags by rate x (tau_c + theta).

**Test.** No new test. Items 2 to 6 use the plant's steady map and the fitted gains. Drift bound 6 K (3 sd, what `pnr_line` and the guard use) and the 0.10 switch size are read from the plant; items 3 to 6 are ours, as no source gives numbers. I could not confirm Seborg et al. on ramps (section 8).

**Illustration, not a result.** Kp_T 8.49 and the 4.63 K headroom of PHYSICS.md give 0.55 K and, with 0.05 min, about 11 K/min against 3 today. The window grows from (8, 2) K to about 10 K each side, bounded by `pnr_line` less the margin. The most consequential item here (D4).

**Poor result.** More trips are possible. It ships; the validation and held-out tables report them. `check_cascade_gains` becomes a report (D2).

**Code and records.** `tune_unstable_cstr` writes all nine entries; `write()` stops keeping old held values; `HELD_GAINS` goes. Re-records `unstable_cstr`: the cascade, the MPC fallback and the oracle rows.

## 5. cement_kiln

**Source.** Jantzen 1998: a fuzzy controller made linear has triangular sets crossing at 0.5, input universes large enough to avoid saturation, and gains matched to a linear controller (GE x GU = Kp; an incremental controller integrates). The Swain and Subudhi rule base stays as published. SIMC for the equivalent loop.

**Recipe.**
1. Burning-zone fuel PI: unchanged (SIMC from a fuel step, tight). It passes.
2. `T_bz_base` = burning-zone reading after 400 steps (8 residence times) of nominal fuel, speed and feed, feed noise off. PHYSICS.md gives 1481 C, 1754 K.
3. `sp_trim_max` = half the gap from `T_bz_base` to the nearer interlock (`T_bz_min` 1620 K, `T_bz_max` 1900 K): about 67 K.
4. Supervisory test: fuel PI closed, trim frozen, setpoint +5 % of 2 `sp_trim_max` (about 6.7 K), 240 steps (2 h, about 5 residence times), free lime as a fraction against an unstepped run. Fit, C5 and C7 give Kc and tau_I; K_I = \|Kc / tau_I\|, in K per s per unit of lime.
5. `lime_halfwidth` = `lime_band` (0.006, the task's spec band). `speed_halfwidth` = 3 sd of the speed change over the trend window. `trend_steps` = 10, a quarter of the feed's correlation time (1205 s, 300 s).
6. `step` = K_I x `lime_halfwidth` x 30 s / 2. On the rule base's linear part (trend OK) the trim moves 2 `step` e / `lime_halfwidth` per sample, an integrator of that gain (Jantzen's incremental form).

**Free choices.** Item 2, 3, 5 (3 sd after Jantzen's no-saturation universe; the quarter window) and the integral-only equivalence in item 6 are ours. The grid is gone.

**Poor result.** If the fitted tau1 exceeds 4 (tau_c + theta), integral-only is thin on margin; the script prints that gain margin and ships. A trip costs a 24 h restart in the reward, and ships too.

**Code.** `tune_cement_kiln`: delete the 3 x 4 grid and its `train_cost` loop. `trend_steps` and `sp_trim_max` become stored values. Re-records `cement_kiln`.

## 6. battery, boiler_drum, glass_furnace

### battery

**Source.** SIMC improved PI for delay-dominant processes (2012 eq. 5.38-5.39). BMS power derating is an onset SoC plus a linear slope to the limit (Orion BMS manuals); neither source gives numbers.

**Recipe.**
1. Test: each training seed in manual at its first dispatch level (SoC 0.35 to 0.75, guard inactive), command +0.1 raw (5 % of the range, 100 kW), 20 steps (100 s), delivered power against an unstepped run. Fit first order plus delay, C5. The first sample already shows the whole step, so the fitted delay is 0 and theta is the half sample, 2.5 s (C4).
2. The plant is a one-sample delay with tau1 near 0, so C7 uses tau1 = theta/3: Kp = (theta/3) / (k 2 theta) = 1 / (6 k), Ki = Kp / (theta/3) = 1 / (2 k theta). Unrun expectation: Kp about 0.17, Ki about 0.2 per s (today 0.5 and 0.02); Ki x 5 s = 1 is a deadbeat integral.
3. `guard_margin` = SoC drawn by two dispatch blocks at rated power, 2 x 1 MW x 300 s / pack energy (1692 kWh), 0.0985 (today 0.12). Ours: one block to see a step, one to ramp the derate.

**Poor result.** SIMC at tau_c = theta may chase the dispatch noise. It ships. **Code.** New `tune_battery` in `tune_industrial_experts.py`, and a `battery` entry in `pid_gains.json` (none today). Re-records `battery`.

### boiler_drum

**Source.** Astrom and Hagglund 3.5: conditional integration, off when the output is saturated and the update would saturate it more; actuator limits are the limit.

**Recipe.** Delete `trim_max`. The level integral freezes when `q_feed_raw` differs from the clipped `q_feed` (the actuator, 0 to 200 kg/s), not when the trim exceeds 25 kg/s. The bump tests stay (firing +0.02, feedwater trim +2.0 kg/s, from the settled expert state, 120 steps). Both loops stay lambda-tuned with lambda = theta, the value the menu chose, so the stored gains are expected unchanged (ours: no confirmed lambda default; D7).

**Poor result.** Ships. **Code.** `boiler_drum_pid_step`; `pick()` deleted. Re-records `boiler_drum`.

### glass_furnace

**Source.** SIMC Table 5.1 (integrating with lag), half rule eq. 5.12, PID where tau2 > theta.

**Why the old test missed the delay.** `env.py` has `FUEL_DEAD_TIME_STEPS` = 2 at 30 s (60 s), a 120 s thermocouple lag and a 15 min crown. The test fitted a pure delay plus integrator to 120 steps of a concave response, so the fit put theta at its bound, 0, and only the half sample (15 s) entered.

**Recipe.**
1. Keep: 400 steps under the expert (8 whole 25 min reversal cycles, so the step lands just after a firing interruption), firing in manual at the last-cycle average, +0.05 raw.
2. Record 40 samples (20 min), the longest window that ends before the next interruption (about sample 49).
3. Observed dead time = the leading samples with \|dy\| under 0.1 % of the window's largest \|dy\|, times 30 s (above float32 rounding, 2e-4 K). Expected 60 s.
4. Fit k' e^(-theta s) / (s (tau2 s + 1)), theta from the observed dead time up to +60 s, tau2 in [0, 1800 s]; C5 on k'.
5. theta = fitted theta + 15 s. If tau2 <= theta: PI, theta + tau2/2 (half rule). Else PID with tau_D = tau2 (series form, converted by eq. 5.30). Then C6, C7: tau_c = theta, Kc = 1 / (k' (2 theta)), tau_I = 8 theta.

**Poor result.** Expect a Kp far under 12 and a slower hold. It ships. **Code.** `tune_glass_furnace`; the lag fit moves to the shared SIMC module (section 7). Re-records `glass_furnace`.

## 7. Code, records and cross-cutting fixes

| Fix | Change |
| --- | --- |
| Menus | Delete `_pick_tau_c` (`tune_process_experts.py:158`), `pick()` (`tune_industrial_experts.py:147`) and four_tank's inline loop. All tuners call the rule with tau_c = theta |
| Shared module | `scripts/simc_rules.py`: fitters, C5 selector, C7 rules. Both scripts have copies of `fit_fopdt` and `simc_pi` today |
| `tune_pid.py` | Remove the glass_furnace, reactor and compressor_surge rows and their tuners; `tune_industrial_experts.py` stops importing it. The aircraft rows leave with the aircraft PR |
| `_meta` | Replace the relay/AMIGO text with "written by scripts/tune_*.py; each entry names its rule" |
| `reactor` entry | Delete (nothing reads it). It enters the reactor fingerprint, so the reactor stamp moves with values unchanged |
| `gain()` fallbacks | Process and industrial calls drop their defaults; a missing key raises and names the file (D8). Same for `_p.get(..., default)` in the kiln and boiler |
| Docs | `docs/using-the-pids.md` line 26, `docs/baselines.md`, `CONTRIBUTING.md`, `Makefile`, `tests/test_registry_rules.py`, `tests/experts/test_tuning.py`, `tests/experts/test_pid_tuning.py` mention `tune_pid.py`; PHYSICS.md expert sections are rewritten |

**Records.** Re-record ph_neutralization, compressor_surge, unstable_cstr, cement_kiln, battery, boiler_drum and glass_furnace, plus reactor for the stamp. first_order, cstr, four_tank and distillation re-record only if C5 moves them (spread over 1.5); the script prints whether it does. Held-out seeds 200 to 209 report only. Land after the CasADi port, as the audit plans, so the two oracle rows move once. The reward constants `c_hold` and `e_floor` were measured with the shipped PID and MPC (`scripts/measure_hold.py`) and sit in the env version stamp: the run reports their drift and moves nothing (D10).

**Size budget.** Net about +250 lines in `scripts/` (the module and the new tuners, less the menus and the grid) and about 60 in `src/`. Measure after.

## 8. Open decisions

| # | Decision | Recommendation |
| --- | --- | --- |
| D1 | Cap `kp_surge` at 1.0 (Mirsky) above the SIMC value | Cap: it is the only vendor-side gain statement |
| D2 | Guards become reports, not gates (compressor, unstable_cstr) | Yes; a gate with thresholds of ours vetoes the protocol after the fact |
| D3 | Ship `surge_line` 0.10, or a commissioning ladder (0.10, 0.125, 0.15 until the battery has no trip) | Ship 0.10, per the rule; the ladder is the real-life variant if you want it |
| D4 | Adopt the unstable_cstr limit rules (about 3.6 times the ramp, wider window), or keep the hand values labelled ours | Adopt, and read the validation table before merging |
| D5 | Apply C5 to the passing tasks | Yes, as a check; they move only past a spread of 1.5 |
| D6 | glass_furnace becomes PID if tau2 > theta | Yes, as SIMC says |
| D7 | boiler_drum stays lambda (gains stay) or moves to SIMC integrating | Stay for now |
| D8 | `gain()` raises on a missing key | Yes |
| D9 | `sp_trim_max` half the interlock gap (67 K, ours) or keep 120 K | Half gap; 120 K has no source either |
| D10 | Leave `c_hold` and `e_floor` as stamped, report drift only | Yes; moving them re-stamps the env versions |

**Source check.** PDFs: WebFetch cannot parse them, so I extracted text with pdftotext from the saved files and re-read each cited passage in a second pass. Web pages: fetched twice with different questions. Checked this way: Skogestad and Grimholt 2012 (eq. 5.12, 5.27-5.31, 5.32, 5.35, 5.38-5.39, Tables 5.1 and 5.2); Shinskey's "Characterizers" (Valmet PDF; Automation.com returns 403, so a search summary of it was the second read); Mirsky (10 % remark, open-loop line, gain under 1, stroke times); Jantzen 1998 (design checklist, GE x GU); Astrom and Hagglund 3.5 (T_t bracket, conditional integration, K T_t / T_i offset); McMillan 2012 (twice); Orion BMS (form only, no numbers). Read once: Shinskey and Gerry 2006 (bump test). **Not confirmed on a second fetch:** the Rockwell PlantPAX page (403; Fieldbus parameter names come from vendor-manual snippets in a search); Seborg et al. on setpoint limits and a ramp tied to tau_c + theta (found nowhere; the audit's attribution is unverified); Mirsky page numbers (taken from PHYSICS.md); any varying-gain guidance by Skogestad (absent from the 2003 paper, the 2012 chapter and his 2025 ABB slides).
