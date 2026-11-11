# Nuclear reactor: physics model, provenance and validation

Reference plant: a **3 GW thermal pressurised-water reactor** with a two-point core:
iodine and xenon in the top and bottom halves, an axial offset, rods that enter from
the top, soluble boron that reaches the core late, thermal feedback, and a steam
generator driven by the turbine.

Contract for `target_gym.reactor`. Design notes and measured results:
`docs/proposals/reactor-axial.md`.

Each number is **read** (opened and read twice), **derived** (computed from read
numbers, the sum shown) or **ours** (a choice, with its reason).

---

## 0. The control property

**Axial offset.** Rods act in seconds but disturb the power's axial shape, and that
shape drives a xenon oscillation near 29 h long. Boron leaves the shape alone but
reaches the core 5 to 15 minutes late. A plant running Mode A depends on both, which
is why OAPS names the operator's struggle with the offset as the real limit on its
load following.

Tavg is held by reactivity from either actuator. Delta I is moved only by the rods.
So the rods are spent on the offset, and boron carries Tavg through the ramps, late.

---

## 1. Model and provenance

**Total power** is point kinetics with Keepin's six delayed groups:

```
dn/dt   = ((rho - beta) / Lambda) n + sum_i lambda_i C_i
dC_i/dt = (beta_i / Lambda) n - lambda_i C_i          i = 1..6
```

**Reactivity:**

```
rho = rod_worth(y) - rod_worth(y_park) + boron_worth (boron - boron_ref)
    + alpha_fuel (T_fuel - T_fuel_ref) + alpha_coolant (T_avg - T_coolant_ref)
    - rho_Xe_full (Xe_hat - 1)
rod_worth(y) = -1500 pcm (y - sin(2 pi y) / (2 pi))
```

`y` is how far in the rods are, as a fraction of the stroke. The curve is the
integral worth of a cosine flux, so a shallow bank is worth little.

**The two-point core** is the Onega and Kisner model, as Mastroianni's 1979 thesis
prints it (eqs 2.15 to 2.18). Flux, iodine and xenon are each a cosine plus a first
harmonic, `psi = n (cos bz + A sin 2bz)`, with amplitudes found by integrating over
the two halves of the core. `A` is then exactly (top - bottom) / (top + bottom), the
axial offset, and each half is `mean +- tilt`:

```
dI_hat/dt  = lambda_I (n - I_hat)
dI_tilt/dt = lambda_I (n A - I_tilt)
dXe_hat/dt = a (r n + I_hat) - (lambda_Xe + s n (1 + A Xe_tilt / Xe_hat)) Xe_hat
dXe_tilt/dt = a (r n A + I_tilt) - lambda_Xe Xe_tilt - (8 / 3 pi) s n (A Xe_hat + Xe_tilt)

A = -(8 b + 10 eps) / (N + sqrt(N^2 + 64 b (b + 5/4 eps)))
b = S Xe_tilt
N = 45 pi^3 D / (8 H^2) - S Xe_hat + 7 n Phi alpha nu_Sigma_f
```

Here `s` is `sigma_phi0`, `r` is `gamma_ratio`, `a = (lambda_Xe + s) / (1 + r)` and
`S` is the full-power xenon's absorption, `sigma_x X0`. With `eps = 0`, `Xe_hat = 1`
and `n = 1` this is the thesis's `A = v - sqrt(v^2 + 1)`, `v = N / (8 S Xe_tilt)`.

**What we added to the thesis, by its own method.** Projecting the diffusion equation
onto the first harmonic gives the thesis's `v`, down to `D`, `S` and the denominator,
so the same projection places an absorption difference `eps` between the halves. A
term of 0.25 % of absorption, the thesis's stand-in for the rod move, then leaves a
tilt of 0.19 once it ends, which is the plant's reading. Two sources feed `eps`:

```
eps = -0.5 rho_ext_min nu_Sigma_f sin(pi y)^3                     rods from the top
    + alpha_tilt nu_Sigma_f (rise_full - (T_hot - T_cold)) / 4     part-power shift
```

The rods' weight is their overlap with the harmonic: nothing when out, and nothing
all the way in. The moderator's term is the coolant's rise up the core: the two
node means sit a quarter of the rise either side of Tavg, and a smaller rise at part
load moves power up the core, the direction Krsko describes. Boron enters no `eps`:
it is uniform, and the APR1400 paper finds its effect on axial shape negligible.

**Boron** is a feed and bleed of well-mixed coolant. A command joins a queue, and
reaches the core `boron_delay` control steps later:

```
d(boron)/dt = Q / M (C_feed - boron)
Q = 10 kg/s dilution (C_feed 0), 3 kg/s boration (C_feed boric_acid_ppm)
```

The heat path is unchanged: fuel, hot leg, cold leg, steam generator, with the
turbine taking `P_ref load`. Energy balance fixes steady power at the load. Thermal
states and xenon step explicitly, neutrons by TR-BDF2, 4 solves in each of 6 sub-steps
of 5 s.

**Left out:** the second axial harmonic and radial modes, grey rods, boron's effect
on the moderator coefficient, burn-up, samarium, decay heat, loop transport delay.

---

## 2. Validation targets

| Quantity | Source figure | Model | |
|---|---|---|---|
| Axial mode period, full power | 28.3 h measured (Ringhals-4); 31.5 to 39.3 h in Pedersen et al.'s models | **32.7 h** | in the models' range |
| Axial mode decay, full power | 0.05 per hour (Ringhals-4) | **0.070 per hour** | near |
| Axial mode at half power | damped (DP-1239: feedback weakens, xenon saturates) | **-0.057 per hour** | yes |
| Undamped xenon cross section | 3.02e-18 cm2 (thesis, section 3.2) | 2.92e-18 | yes |
| Tilt after the Oconee pulse | 0.19 (fig. III-1) | **0.19 +- 0.04** | yes |
| Rods 34 steps in at 50 %, xenon at 110 of 96 | delta I near +5 % (Krsko figs 3, 5, 6) | **+5.4 %** | yes |
| Rate that full flows allow at 1690 ppm | 1 to 2 %/min (OAPS) | **about 2 %/min** | yes |
| Rods steps for delta I = 0 at 50 % | 34 steps (Krsko fig. 3) | **42 steps** | yes |
| Inhour at +100 pcm | tens of seconds | 55 s | yes |
| Coolant rise | 30 to 40 K | 33.3 K | yes |
| Peak xenon worth after a scram | 1800 to 3200 pcm, test band | **1419 pcm** | **no** |

**The peak xenon miss.** The two-point model's flux, 2.69e13 with its pi / 4, burns
xenon at 5.75e-5 per second, against the 7.95e-5 this page used before. The flux is
tied to the period, so it stayed, and the old test fails on it. Reported, not moved.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | Status |
|---|---|---|---|---|
| `beta_i`, `lambda_i` | 6-group | | Keepin (1965) | earlier |
| `Lambda_gen` | 1e-4 | s | PWR range | earlier |
| `alpha_fuel` | -3.0e-5 | 1/K | Doppler, PWR range | earlier |
| `alpha_coolant` | -12e-5 | 1/K | The start-of-cycle MTC, where the core sits (1690 ppm, OAPS's start-of-cycle ramps). It is near zero there and grows more negative as boron falls (NRC GFE PWR 192004, nuclear-power.com): BEAVRS measures the isothermal coefficient, Doppler included, at -1.75 and -1.71 pcm/F at hot zero power and 975 and 1405 ppm, and specifications keep it at or below 0 above 50 % power. No hot-full-power start-of-cycle figure was found; the reviewer's judgment is -9 to -18 pcm/K (physics review, finding 2). The range was 0 to -20 until the family sweep (2026-11-10): at -2.9 to -7 the rods cannot carry a 5 %/min return, so the owner took the reviewer's range. The sweep draws it with `boron_worth`, the most negative MTC with the largest worth | ours, range -9 to -18 pcm/K |
| `alpha_tilt` | -30e-5 | 1/K | The part-power axial shift's lumped coefficient (moderator and Doppler redistribution), kept at the value that reproduces Krsko's +5 % delta I at half power when the MTC moved | ours |
| `T_fuel_ref` | 1051.4 | K | Zero feedback at full power on program | derived |
| `rho_ext_min` | -0.015 | | A 1500 pcm bank, kept from the earlier plant, no figure opened | ours |
| `rod_speed` | 72 / 60 / 228 | 1/s | 72 steps/min over 228 steps, NRC HRTD 8.1, read by an earlier session | earlier |
| `rod_park` | 3 / 228 | | D bank at 225 steps, Krsko fig. 3 | read |
| `boron_ref` | 1690 | ppm | Krsko fig. 3 | read |
| `boron_worth` | -8e-5 | 1/ppm | BEAVRS (Tables 22, 23) measures 10.8 to 10.9 pcm/ppm at hot zero power and 975 to 1405 ppm; boron is worth less at full power and higher boron, and two NRC papers gave 6.73 and 8.15 pcm/ppm (search summaries, not opened) | ours, range -6.7 to -10.9 pcm/ppm |
| `dilution_flow`, `boration_flow` | 10, 3 | kg/s | OAPS, section 2.1 | read |
| `boron_delay_range` | 10 to 30 | steps | 5 to 15 min, OAPS, section 2.1, drawn at reset | read |
| `boric_acid_ppm` | 7800 | ppm | A vendor's recycling concentrate, seen in a search summary, the PDF would not open | ours |
| `primary_mass` | 2.5e5 | kg | The water behind `C_coolant` and `C_cold`. 60 ppm of 1690 then takes 9 t, against Krsko fig. 4's 6 t | ours |
| `sigma_x` | 2.72e-18 | cm2 | Mastroianni, Table III-1 | read |
| `D_diffusion` | 4.99 | cm | 283 pcm, the first axial harmonic's eigenvalue separation of Ringhals-4 in Pedersen et al.'s homogeneous model (table 3), as D x 3 pi^2 / (H^2 nu Sigma_f). The thesis's 0.405 cm gives 23 pcm, halves coupled 12 times too weakly (docs/proposals/reactor-coupling.md). Owner's decision, 2026-11-08 | derived |
| `alpha_power` | 3.64e-16 | cm2 s | Table III-1 | read |
| `Sigma_f` | 0.14 | 1/cm | Table III-1 | read |
| `gamma_x`, `gamma_I` | 0.0046, 0.061 | | Table III-1 | read |
| `H_core` | 365.8 | cm | Table I-1 | read |
| `nu_Sigma_f` | 2.44 x 0.16 | 1/cm | Derived, see below | derived |
| `Phi_full` | 2.02e13 / 0.75 | 1/cm2/s | Thesis section 3.1, the test's flux at 75 % power | read |
| `sigma_phi0` | 5.75e-5 | 1/s | `sigma_x Phi_full pi / 4` | derived |
| `gamma_ratio` | 0.0754 | | `gamma_x / gamma_I` | derived |
| `rho_Xe_full` | 0.025 | | 2500 pcm, kept | earlier |
| `P_thermal_ref` | 3.0 | GW | Large PWR | earlier |
| `C_fuel`, `UA`, `m_dot_cp` | as in `env.py` | | The 33 K rise, a 5 s fuel lag | earlier |
| `C_coolant`, `C_cold`, `UA_sg`, `C_sg` | as in `env.py` | | No source for the volumes | ours |
| `T_avg_no_load`, `T_avg_full_load` | 564.8, 580.2 | K | NRC HRTD 5.0, read by an earlier session | earlier |
| `n_max` | 1.09 | | Remembered trip, not opened | ours |
| `T_avg_min`, `floor_allowed` | 556 K, 900 s | | The minimum temperature for criticality, 541 F at Salem, a limit with 15 min to act, not a trip. No EDF figure is public; the NRC copies could not be opened | ours |
| `load_ramp_rate` | 2 %/min | | Top of OAPS's Mode A range | read |
| `load_min`, `hold_range` | 0.5, 3 to 8 h | | The URD's cycle, Chari and Rohr (1997) | earlier |
| `ramp_down_range` | 0.5 to 2 h | s | | ours |
| `demand_theta`, `demand_sigma` | 300 s, 0.5 % | | A small frequency-control wobble | ours |
| `burnup_range` | 2.3 to 3.3 | ppm/day | BEAVRS (MIT), Table 24: letdown 623 to 0 ppm over 52 to 326 EFPD in cycle 1 (2.27 a day), 918 to 104 ppm over 13 to 257 EFPD in cycle 2 (3.34 a day) | read |
| `initial_tilts`, `initial_tilt_share` | ten pairs, 0.108 | | Yesterday's end-of-day iodine and xenon tilts, from the reference on training seeds, drawn on load following's share of days | derived |
| `carried_days_range` | 0 to 1 | day | Burnup not yet diluted at reset: the operator dilutes at least once a day | ours |
| `max_steps_in_episode` | 2880 | steps | A day, so the xenon's swing shows after the return | design |

**`nu_Sigma_f`.** Table I-1 prints Sigma_a 1.53 and nu Sigma_f 1.56, and with those
the thesis's equations are heavily overdamped, which its own results are not.
Sigma_a is not one of the varied parameters. Criticality asks Sigma_a = nu Sigma_f,
and its fission estimate of 0.16 with Chalmers' nu of 2.44 gives 0.39. That value
reproduces its undamped cross section to 3 % and its plant's period to 2 %. It was
chosen on the thesis's printed results, not on anything of this task's.

---

## 4. Task design

### Task: `reactor`

Run a day of a pressurized water reactor at the start of its fuel cycle, with control
rods and soluble boron, knowing the day's load plan: hold the average coolant
temperature within 1.6 K of its program on either side and the axial offset within
0.05 of its reference, keep the rods above their insertion limit, and dilute for the
fuel's burnup, at the least cost in lost output, rod wear and effluent. Six scenarios
weigh baseload days against five kinds of load-follow day (REALITY.md sections 6
and 7).

Each episode starts at full load, rods parked, xenon at the balance of a recent power
of 0.9 to 1.0, and Tavg lined out below its program by 0 to 1 day of burnup not yet
diluted. Yesterday's axial xenon carries over (the physics review's finding 3): on
10.8 % of days, load following's share (REALITY.md section 7), yesterday was a
load-follow day and the iodine and xenon tilts start as one of ten end-of-day pairs
the reference left on training seeds 0 to 9 (`scripts/reactor_tilt_table.py`,
xenon tilts -0.0067 to +0.0085); otherwise yesterday was baseload, whose days end
with tilts under 3.1e-4, and the tilts start at 0. On a load-follow day the load starts down in 0.5 to 2 h,
holds at 50 % and comes back, at the scenario's rate, with a small wobble.

| obs | meaning |
|---|---|
| 0 | `n`, power relative to rated |
| 1 | hot-leg temperature (K) |
| 2 | rod position, +1 fully out, -1 fully in |
| 3 | turbine load |
| 4 | `T_avg` (K) |
| 5 | `T_ref` (K) |
| 6 | `delta_I`, top minus bottom, a fraction of rated |
| 7 | boron in the core (ppm), as the boronmeter read it 20 min ago |
| 8 | hours until the planned ramp down starts, 0 once it has |
| 9 | hours until the planned ramp up starts, 0 once it has |
| 10 | the recent power the starting xenon is balanced with (rule 16: operators know their power history) |

Actions: `[rod demand, boron flow]` in [-1, 1]. Rod demand reads as obs 2. Boron flow
is -1 for full dilution, +1 for full boration. Xenon, iodine, their tilts, the boron
on its way and the delay's length are hidden; the power history in obs 10 is what an
operator's xenon tool starts from.

**Limits** (charter rule 14, Amendment 3):

| limit | kind | form | source |
|---|---|---|---|
| power, fuel temperature, hot leg (610 K, ours: under the 618.0 K saturation at 2250 psia, IAPWS-IF97) | trip | the envelope; a scram, 48 h of lost output | ours, family 24 to 70 h (Doel-3, INIS 1ypr9-dk595; APR1400, KNS 20A-065) |
| Tavg under 556 K | timed | 900 s allowed, then hot standby, 15 h of lost output | ours, inside the family's 15 to 30 min to restore; 15 h from Doel-3's 6 h down and 18 h reduced |
| axial offset outside +-0.05 | penalty clock | time outside counts at every power; 1 h in 24 h is free; past it, the lost output of a derate to 50 % | EDF's band (IAEA NP-T-3.23, annex II-4.3); the 1 h in 24 h and the derate from Westinghouse specifications (D.C. Cook, RAOC), pooled in since EDF's was not found |
| Tavg above its DNB limit | timed | the full-power program plus 5.1 F (2.83 K) at every power; 2 h to restore, then MODE 2 within 6 h, so hot standby | STS LCO 3.4.1 (US EPR GTS and AP1000 bases, from search summaries); the margin from Braidwood's COLR, 588.0 F nominal against 593.1 F (a search summary, not opened: nrc.gov blocks automated reads), the only plant found with both numbers |
| rod insertion limit | timed | a straight line in power, 22 % of the stroke in at full power to 78 % at half; 2 h to restore (STS 3.1.6), then hot standby | KAIST's lead for the points; the family's shape |

The clock starts each day at 0 (ours: the episode is the 24 h window).

---

## 5. Known deviations

**D1: the load goes straight to heat.** Steam pressure, valves and the secondary's
controls are left out.

**D2: the halves' coupling is not the thesis's.** D is set from Ringhals-4's eigenvalue
separation, not from the thesis's one-group constants, which coupled the halves 12
times too weakly and made the half-power mode neutral. The thesis cross-checks run on
the thesis's own D. With the new D, Krsko's half-power state comes out at +5.4 %
against +5 %.

**D3: rods from one side.** A bank reaches only the top, so at full power with rods
out nothing can push power up the core. Xenon tilted to the top can drive delta I
down and past the band, and only time brings it back. Real plants have more than one
bank.

**D4: no scored peaking.** Delta I is scored on its penalty clock alone, with no link
to local power or to a trip, and the outer AFD envelope is not modelled.

**D6: an overrun is priced, not flown.** Past the offset's allowed hour the cost is a
derate's lost output, but the load does not drop: the plant keeps following the plan.

**D8: EDF Mode A cannot hold the band on these days.** With the offset band as a
penalty clock, the Mode A pairing (the rod program on Tavg, the operator's boron on the
offset) spends about 9 h a day outside the band on seeds 100 to 109, overruns the
allowed hour on every seed and loses to the parked-rods null ($671k against $658k,
docs/proposals/reactor-ladder-a3.json). That is the family's own experience: EDF
developed Mode G, with grey power-shimming groups, for deep load following, and 48 of
its 56 units run it (Muniglia et al., ICAPP 2016, hal-01355257; arXiv 2107.11201).
Mode G is not modelled (docs/proposals/reactor-coupling.md). The owner adopted the
Westinghouse procedure instead on 2026-11-08: bank D on delta I and boron trimming Tavg
(Krsko, NENE 2022, section 3; KNS 18A-534), if it keeps the rod insertion limit.

**D7: the boronmeter reads exact.** It is 20 min late (Areva US 8,489,338) but carries
none of the patent's 7 % noise.

**D5: no decay heat.** A scram removes all heat at once.

**D10: no steam relief.** At part power the hot side of a PWR is bounded by its steam
side: a Tavg high enough to lift the steam relief and safety valves (1176 psig, 570.0 K
of steam at V.C. Summer, a search summary) relieves heat and then trips. The model
has no relief and no steam trip. The two-sided dead band prices any time past 1.6 K
over the program, which at the 50 % hold is about 1065 psig of steam (interpolated in the review's IAPWS-IF97 table), so a hot
excursion costs output long before it would lift a valve, and the DNB limit stays as
the timed bound (physics review, finding 1, option B not taken).

**D9: burnup by the clock.** The fuel's burnup is worth boron at the day's drawn rate
for every second since reset, as if at full power. Real burnup follows the energy
made, so on a day with a 50 % hold it is overstated by at most half the hold's share
of the day: under 0.1 ppm at 3.3 ppm a day, 0.02 K of Tavg. It keeps the oracle's
model exact without a new state. The day starts lined out below its program by the
burnup earlier days left undiluted, 0 to 1 day of it, 0 to 0.8 K. Burnup's bite is
small within a day (0.56 to 0.8 K of Tavg at the BEAVRS rates), so it matters mostly
together with that carried debt.

**Retired tests (owner, 2026-11-08, TESTING.md section 10).**
`test_peak_xenon_worth_is_in_the_published_band` (the task has no scram; the load cut's
swing is tested against Krsko instead), `test_the_reactors_winning_null_has_a_written_exception`
(the old plant's written exception is gone; acceptance, J2, still checks the nulls) and
`test_a_day_has_two_moves_for_the_scorer` (the old scorer's move detector, which the
weighted-saving headline does not use).


---

## 6. The expert

The expert is the reference of `scripts/reactor_ladder.py`: the Westinghouse procedure
(Krsko, NENE 2022, section 3; KNS 18A-534), kept over EDF's Mode A, which loses to the
null on these days (D8). Bank D holds the axial offset and never sits deeper than its
insertion limit; boron trims Tavg past the rods' 1.6 C dead band (IAEA NP-T-3.23, annex
II-5.1), fed forward from the day's plan, led by the boron line's delay, and from
xenon's balance by the operator's xenon tool (textbook iodine and xenon run on the
observed power from the power history, `xenon_feedforward`). Both loops are SIMC PIs
on bump tests at full power and at the half-power hold, keeping the least gain. The
owner adopted it on 2026-11-08, on condition it keeps the insertion limit, which it
does on every scenario. Results and room are in `docs/proposals/reactor-axial.md`.

**The rods' action band (owner, 2026-11-08), fixed before any rerun.** On a baseload
day a plant's rods sit still: the target band is a limit operators keep, and they move
bank D only when the offset heads for it. Without a band the rods chased the
frequency-control wobble and the sensor noise, 30 to 38 steps a day with the offset
never past 0.004, and the parked-rods null beat the reference there. No source gives
the action threshold; the family's range runs from 0 (act on any offset) to the whole
target band (act only once outside, which spends the penalty clock). The owner fixed
it at half the band: the rods' loop sees no error while the offset is within 0.025 of
its reference, half of EDF's 0.05 (2.5 % of delta I at full power, half the +-5 %
delta I band of Krsko and the Westinghouse specifications), and past it the error is
the excess. It is a band on the offset, like the target band, so in delta I it
narrows with power as the target band does; read as 2.5 % of delta I at every power
it would equal the whole band at the half-power hold. **ours,** family range 0 to
0.05, chosen at half, and not to be adjusted after results (J1).

**Rods on Tavg through a fast ramp (owner, 2026-11-10), fixed before any rerun.** On
the start-of-cycle core (reactor-v8) a load cut releases about 820 pcm of Doppler and
moderator feedback per 50 % of load, and boron moves at most 32 to 35 pcm a minute,
so a 5 %/min ramp outruns it: with the rods on the offset, Tavg ran 20 to 29 K hot
and the hot leg tripped. In the family, fast ramps are carried by rods on Tavg
(Westinghouse's automatic rod control, NRC HRTD 8.1; EDF's grey rods), while Krsko's
procedure, rods on the offset and boron on Tavg, is a slow-ramp one (8 %/h). So the
reference switches by one test: while the day's plan ramps faster than boron can
follow, the rods run Westinghouse's rod program on Tavg; otherwise bank D holds the
offset. Boron can follow when the ppm a second the ramp needs, the plant's power
defect per unit of load (from `rod_balance` on the program) times the planned load
rate, is within the line's full dilution (`dilution_flow / primary_mass * boron`) or
full boration (`boration_flow / primary_mass * (acid - boron)`), at the boronmeter's
reading. The handback is bumpless: the offset loop starts from where the rods are.
The rod insertion limit holds in both modes. No new number.

**The xenon tool's play (owner, 2026-11-10), fixed before any rerun.** The rule: a
controller acts on what the cost charges and on what predicts a charge, never on a
deviation that is free. The xenon tool broke it on baseload: it moved 13.6 ppm of
boron, $90 of effluent a day, to cancel a xenon drift that left alone costs $0.80
(diagnostic, `docs/proposals/reactor-axial.md`). So the tool now has play. It keeps
the boron it did not move as a debt in ppm, and reads the debt as the Tavg shift it
would leave at steady power, `debt * boron_worth / (alpha_coolant + alpha_fuel)`
(at a fixed power the fuel sits a fixed step above the coolant, so it cools with it
and both feedbacks absorb the reactivity; 0.53 K per ppm). While that
shift stays inside Tavg's free zone, the tool does nothing; past it, it moves only
the excess. The free zone is what the cost leaves free: the 1.6 K dead band on
either side of the program (on the hot side up to the DNB limit until the physics
review's finding 1, 2026-11-10). Each edge is pulled in by the
shift xenon causes over the boron line's lead (600 s, the same lead as the plan's
feedforward), so the tool acts before the edge, not after it. No new number: the
dead band, the lead and the plant's two worths are all set elsewhere.
The rods and the Tavg PI are unchanged. Every rung of the ladder uses the same tool,
so the comparison between rungs stays fair.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `reactor` | 2880 | 30 | 24.0 h | 2 in [-1, 1] | 11 | 107 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per 30 s control step

`compute_reward = -(tracking + running + failure)`, with Tavg past the dead band on either side of its program and the offset clock's derate in `tracking`, and rod travel and effluent in `running`; each physics sub-step charges its 5 s and `step_env` sums the six, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.05 K | the RTD reading's resolution, about its 0.045 K scatter: a documented minimum, not a certified floor |
| `e_tol` | 1.6 K | the black rods' dead band: EDF's rods move only past 1.6 C off program (IAEA NP-T-3.23, annex II-5.1). CEA's papers on Mode G's R group give 0.8 C with slow motion up to 1.7 C |
| `tracking_exponent` | 1 | linear, as lost output is |
| `imbalance_multiple` | 1.25 x 0.0023 per K of `spot_price_per_MWh` on `P_electric_GW` | **ours.** Off program, steam conditions cost output. The relative Carnot slope at 547 K steam and 305 K condenser is 305 / (547 x 242) = 0.23 %/K. At the one imbalance price of the priced plants, $100/MWh (1.25 x the $80 spot), on `P_electric_GW` = 1.0 GW, a third of the 3 GW thermal, that is 100 x 1000 x 0.0023 = $230/h per K, so **$1.917 per K over a 30 s step**. It prices a hot plant as a cold one, which the slope does not say: the band is a procedure band both cited procedures keep on either side (IAEA NP-T-3.23 annex II-5.1; NRC HRTD 8.1), priced like the offset band by the output it costs (physics review, finding 1, owner 2026-11-10, reversing Amendment 3 decision 7) |
| `rod_step_price` | $1 per step | **ours,** kept by the owner on 2026-11-07. Rod travel, as the steps the rods actually moved |
| `ao_band`, `ao_ref` | 0.05, 0 | EDF: during load follow the axial offset stays within 0.05 of its reference (IAEA NP-T-3.23, annex II-4.3). The reference is the offset at full power, xenon balanced, rods out: 0 in this core. In delta I the band is 0.05 times the power |
| `ao_allowed`, `derate_to`, `derate_price_per_MWh` | 3600 s, 0.5, $80/MWh | the penalty clock: 1 h in 24 h, then a derate to 50 %, its lost output priced at $80/MWh (inside the family's $80 to $100) on `P_electric_GW` (Westinghouse specifications, D.C. Cook and RAOC, pooled in) |
| `ril_full`, `ril_half`, `ril_allowed` | 0.22, 0.78, 7200 s | the rod insertion limit, a straight line in power (KAIST's lead), 2 h to restore (STS 3.1.6) |
| `effluent_price` | $0.0628 per kg injected | **ours.** Every kg injected displaces a kg to treat (OAPS). Boiling it off takes 2.26 MJ/kg, 0.628 kWh, at $100/MWh |
| `failure_cost` | 666.67 $ per step | a step of lost production: 1 GWe at $80/MWh, inside the family's $80 to $100 |
| `restart_steps` | 5760 (48 h) | a scram's restart, **ours,** family 24 to 70 h: Doel-3 ran 18 h at reduced power after a 6 h scram (INIS 1ypr9-dk595), APR1400's restart windows are 22 to 70 h (KNS 20A-065) |
| `standby_steps` | 1800 (15 h) | hot standby after a timed limit's overrun, **ours:** Doel-3's 6 h down and 18 h reduced, about 15 h of full output lost |

A step at the floor costs `rho_floor_tracking`, $0.0958: the price per K over the 30
s step, times `e_floor`.

`rho_floor_tracking` is the oracle's best hold for tracking -- the lowest per-seed
hold cost the reference controller demonstrated, in the reward's units, which
is 1 per term where the floor is that hold and less where the floor is clamped
at the instrument resolution -- and `rho_floor` the same with consumption
charged in full; `floor_is_documented_minimum` records whether `e_floor` is a
measured/certified floor or a resolution used as a scale, and where it is a
resolution on a deterministic plant both references are 0, since exact hold is
achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce, and `restart_steps` the time a restart would take,
so a trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`): the step that leaves the envelope is
charged the trip cost, with tracking and running cost zeroed, and the plant
restarts at once as `reset_env` would, on the same clock. `terminated` stays
false unless `continuing = False`, which ends the episode on that step instead; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).

## Readings

The observation reports power and the coolant temperatures through their instruments
(`docs/proposals/measurement-noise.md`). Rod position, the load and Tref stay exact, and boron reads 20 min late:
the rods are the controller's own rate-limited command, known by counting steps.
`sensor_noise_scale = 0` gives the noise-free task. Tavg is the mean of the two RTDs,
so it scatters by 0.045 K over sqrt(2), and shares a draw with the hot-leg channel.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `n` | 1.9e-4 of rated | | Ex-core power-range ionization chambers | Corrected | [OSTI 7260785](https://www.osti.gov/servlets/purl/7260785) |
| `delta_I` | 2.7e-4 of rated | | Two ex-core sections, each as noisy as `n`, summed. Ours, from that one figure | ours | as `n` |
| `T_coolant`, `T_cold` | 0.045 K | | Pt100 RTD with TMT82-class transmitter | Corrected | [TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |

**Weak figures.** Both are bounds, from opposite sides.

- `n` is an upper bound. The paper's broadband 5.5 and 19 mil are mostly real
  flux common to all channels, driven by rod actions and a feedwater controller,
  so the refuter kept only its 2 to 15 Hz band, at most 1.9 mil at 0.01 % per
  mil. That scatter is a fraction of signal, so it is proportional to `n`. A
  reading's sigma is a constant, so this task uses the figure at full power at
  every power. It overstates the bound by up to 2 times at the lowest load of 0.5,
  and understates it only above full power.
- `T_coolant` is a lower bound. It covers the transmitter alone, at 307 C. Hot-leg
  streaming is real and likely larger, and no figure for it was opened. NRC
  documents on power-range accuracy could not be read.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true state. `step_env` sums `info["measured_reward"]` over the six
sub-steps as it does `reward`. Each sub-step's true error is shifted by the draw
in the observation that step returns, because a controller gets one reading per
step, and any trip comes from the true state. The cost is linear, so noise has no
floor of the proposal's quadratic form here.
