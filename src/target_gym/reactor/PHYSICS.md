# Nuclear reactor: physics model, provenance and validation

Reference plant: a **3 GW thermal pressurised-water reactor** with a two-point core:
iodine and xenon in the top and bottom halves, an axial offset, rods that enter from
the top, soluble boron that reaches the core late, thermal feedback, and a steam
generator driven by the turbine, whose loading rate the operator sets. The core sits
at a drawn point of its fuel cycle.

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

**How fast the unit follows.** Dilution removes boron in proportion to the boron
there is, and boron falls through the cycle, so what a ramp up asks of the line grows
as the cycle goes on: Mode A carries 1 to 2 %/min at the start of a cycle and 0.1 to
0.3 %/min at 80 % of it (OAPS, section 2.2). The grid's plan is what the dispatch
wants, the turbine takes what the operator loads, and the difference is lost output.
Getting back to the plan sooner means having the reactivity ready: diluting ahead,
holding Tavg with the rods meanwhile, inside the offset's limits. That is what the
family's predictive assistants are built to do (REALITY.md section 3).

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

**The core through its cycle** (owner, 2026-11-15). `c` is the fraction of the cycle
the core has burnt: where it stood at reset or its last refuelling, plus the burnup
since, a ppm of boron being `1 / 1690` of the cycle. One coordinate moves the three
numbers that follow the core's boron, in straight lines (`env.core_at`):

```
boron_ref(c)     = 1690 ppm (1 - c)
alpha_coolant(c) = -12 pcm/K + (-60 - -12) pcm/K c
boron_worth(c)   = -8 pcm/ppm + (-10.9 - -8) pcm/ppm c
```

The straight line for boron is BEAVRS's letdown (Table 24). The other two are ours,
as one joint choice (charter B4): less boron makes the moderator coefficient more
negative and each ppm worth more. Everything else in the reactivity is as above.

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

**The turbine** (owner, 2026-11-15). The load is the turbine's, and the turbine's
moves at the rate the third action asks for:

```
d(load)/dt = action x 5 %/min        never above the dispatch's plan,
                                     never below the restart load (15 %)
```

At +1 it follows the plan. After a trip's restart the fuel's conditioning ramps cap
the loading until the turbine meets its plan (section 4, the stream).

The turbine takes no more load than its valves pass wide open on the steam it has
(owner, 2026-11-15). The steam is saturated at the steam generator's temperature,
a condensing turbine passes flow in proportion to its inlet pressure over the square
root of its inlet temperature (Stodola's cone law), and wide open the valves pass
105 % of rated flow at rated steam, the full-load program's 558.2 K (6.92 MPa):

```
load <= 1.05 x p(T_sg) / p(558.2 K) x sqrt(558.2 K / T_sg),     ln p = a - 4618.5 K / T
```

On its program the limit never binds: the steam is warmer at part load, so the
valves pass 1.134 at 15 % of load, 1.099 at half load and 1.05 at full load. It
binds when Tavg runs under its program: 3.5 K under at full load, 5.2 K at 97 %,
11 K at 90 %. The load then follows the steam's pressure, the plan not taken is
lost output, and the turbine reloads at the rate asked once the steam is back. It is
there because a planner free of Tavg's band loaded the turbine at 97 % of rated
load on steam 8 K colder than its program's, which no turbine takes
(docs/proposals/reactor-axial.md).

The turbine does not load while Tavg is 10 F (5.556 K) or more under its program
at the turbine's load, and loads again once Tavg is back (interlock C-16; owner,
2026-11-16). The stop holds the load: it does not unload the turbine, and a
turbine on a flat plan stays on it. It is the stop for a return to power on the
moderator alone, faster than the dilution follows: at the start of the cycle, with
the rods and the boron still, Tavg falls 109.6 K under its program per unit of
load, so the stop sits 5.07 % of rated load up, and the plant's heat lags the
turbine by some 80 s, so a turbine loading at 5 %/min overshoots it by a few
percent before it stops. Below about 96.5 % of load the stop comes before the
valves' limit, and above it the valves' limit comes first.

The steam generators' safety valves relieve the steam over their set pressure
(owner, 2026-11-17): the lowest valve's 1176 psig, 8.21 MPa, which on the steam
curve above is 570.0 K. They are the heat sink when the reactor makes more heat
than the turbine takes, so they bound how hot the primary can sit at part load:

```
steam relieved = (p(T_sg) / p(570.0 K) - 1) / 0.03   of the full-power steam, never under 0
```

They open in proportion to the pressure over the set pressure and pass the
full-power steam 3 % over it, so the steam sits 2.1 K over 570.0 K per unit of
power relieved. On its program the steam never reaches them: 564.8 K at no load,
558.2 K at full load. With the steam held there, Tavg is at most 570.0 K plus 22 K
per unit of the power the steam generator carries: 7.7 K over its program at half
load, and reactivity added past that raises the power and blows steam, not Tavg.
The steam relieved is heat the turbine does not take; it is not charged.

**Left out:** the second axial harmonic and radial modes, grey rods, samarium, decay
heat, loop transport delay, and the cycle's effect on the axial mode (D11).

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
| Rate that full dilution allows at 80 % of the cycle | 0.1 to 0.3 %/min (OAPS) | **0.37 %/min** | near |
| That rate, start of the cycle over 80 % of it | 3.3 to 20 (OAPS's two ranges) | **5.3** | yes |
| Rods steps for delta I = 0 at 50 % | 34 steps (Krsko fig. 3) | **42 steps** | yes |
| Inhour at +100 pcm | tens of seconds | 55 s | yes |
| Coolant rise | 30 to 40 K | 33.3 K | yes |
| Peak xenon worth after a scram | 1800 to 3200 pcm, test band | **1419 pcm** | **no** |

**The rate at 80 % of the cycle.** At the cycle point's full-power boron, 338 ppm,
full dilution is worth 8.37 pcm/min (10 kg/s into 250 t, at 10.32 pcm/ppm), and the
program asks 22.4 pcm per % of load (730 pcm of Doppler and 388 of moderator per
50 %), so 0.37 %/min, a quarter over the top of OAPS's range, as the start of the
cycle sits at the top of its own. The ratio between the two is the source's. The
primary's mass and the pumps' flows set both; they were not moved.

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
| `alpha_coolant` | -12e-5 | 1/K | The start-of-cycle MTC (1690 ppm, OAPS's start-of-cycle ramps). It is near zero there and grows more negative as boron falls (NRC GFE PWR 192004, nuclear-power.com): BEAVRS measures the isothermal coefficient, Doppler included, at -1.75 and -1.71 pcm/F at hot zero power and 975 and 1405 ppm, and specifications keep it at or below 0 above 50 % power. No hot-full-power start-of-cycle figure was found; the reviewer's judgment is -9 to -18 pcm/K (physics review, finding 2). The range was 0 to -20 until the family sweep (2026-11-10): at -2.9 to -7 the rods cannot carry a 5 %/min return, so the owner took the reviewer's range. The sweep draws it with `boron_worth`, the most negative MTC with the largest worth | ours, range -9 to -18 pcm/K |
| `alpha_tilt` | -30e-5 | 1/K | The part-power axial shift's lumped coefficient (moderator and Doppler redistribution), kept at the value that reproduces Krsko's +5 % delta I at half power when the MTC moved | ours |
| `T_fuel_ref` | 1051.4 | K | Zero feedback at full power on program | derived |
| `rho_ext_min` | -0.015 | | A 1500 pcm bank, kept from the earlier plant, no figure opened | ours |
| `rod_speed` | 72 / 60 / 228 | 1/s | 72 steps/min over 228 steps, NRC HRTD 8.1, read by an earlier session | earlier |
| `rod_park` | 3 / 228 | | D bank at 225 steps, Krsko fig. 3 | read |
| `boron_ref` | 1690 | ppm | Krsko fig. 3, the start of the cycle; it falls to 0 at the end in a straight line, BEAVRS's letdown (Table 24: 623 to 0 ppm and 918 to 104 ppm, both straight in burnup) | read |
| `cycle_range` | 0 to 0.9 | of the cycle | Where a day's core sits, drawn evenly: a core burns at a steady rate, and EDF's units keep their full load-following range up to 90 % of the cycle (IEA 2019, EDF, slides 6 and 12) | read; the even draw derived |
| `alpha_coolant_end` | -60e-5 | 1/K | The MTC at the end of the cycle, 0 ppm, with a straight line from the start's. The family's runs from 0 to -80 pcm/K "depending on the concentration of boron", -80 being a typical specification limit (nuclear-power.com, opened); no hot-full-power end-of-cycle figure was opened | ours, range -45 to -75 pcm/K |
| `boron_worth_end` | -10.9e-5 | 1/ppm | The worth at the end of the cycle, with a straight line from the start's: BEAVRS's 10.8 to 10.9 pcm/ppm at lower boron, the top of the range above. It moves with the MTC on the cycle's one coordinate | ours, with `boron_worth` |
| `turbine_rate` | 5 | %/min | The turbine's fastest loading and unloading: the EUR's fastest return (REALITY.md section 6) | read |
| `valve_margin` | 1.05 | of rated steam flow | What the turbine's valves pass wide open at rated steam: 105 % of the manufacturer's guaranteed rating at Wolf Creek, 105 % of rated steam flow at River Bend and Clinton (REALITY.md section 3, [S19]: search summaries, the reports not opened) | read (summaries) |
| `steam_slope` | 4618.5 | K | Saturated steam, `ln p = a - b / T` through the steam table's 5.503 MPa at 270 C and 8.588 MPa at 300 C ([S20], opened). It gives 6.417 and 7.443 MPa at 280 and 290 C, the table's 6.417 and 7.442 | derived |
| `loading_stop` | 5.556 | K | The turbine's loading stop (interlock C-16): no loading while Tavg is this far under its program at the turbine's load. What it does is Vogtle's safety report's ([S21], read in full); the 10 F is a training card's for a plant it does not name ([S22], opened), not a plant document | source, setpoint flagged |
| `relief_temperature` | 570.0 | K | The steam's saturation temperature at the lowest safety valve's set pressure, 1176 psig (8.21 MPa): 569.96 K on the steam curve above. What the valves do is the standard specifications' ([ST1] Bases B 3.7.1, read in full); the set pressure is North Anna 1's and V.C. Summer's, from search summaries ([S24]), units whose no-load Tavg is this plant's 557 F | source, set pressure flagged |
| `relief_accumulation` | 0.03 | of the set pressure | The valves pass the full-power steam this far over their set pressure. The code's accumulation for steam safety valves, from the reviewer's memory, not opened | ours |
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

Run a day of a pressurized water reactor somewhere in the first 90 % of its fuel
cycle, with control rods, soluble boron and the turbine's loading rate, knowing the
day's load plan: take as much of the plan as the core can carry, hold the average
coolant temperature within 1.6 K of its program on either side and the flux
difference within 5 % of its target, keep the rods above their insertion limit, and
dilute for the fuel's burnup, at the least cost in lost output, rod wear and
effluent. Six scenarios weigh baseload days against five kinds of load-follow day
(REALITY.md sections 6 and 7).

Each episode starts with the core at a point of its cycle drawn evenly from 0 to
0.9 (observation 13), with that point's boron, moderator coefficient and boron worth
(section 1). It starts at full load, rods parked, xenon at the balance of a recent
power of 0.9 to 1.0, and Tavg lined out below its program by 0 to 1 day of burnup
not yet diluted. Yesterday's axial xenon carries over (the physics review's finding 3): on
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
| 3 | turbine load: the plan's, or while the turbine reloads after a trip's restart, the reload's |
| 4 | `T_avg` (K) |
| 5 | `T_ref` (K) |
| 6 | `delta_I`, top minus bottom, a fraction of rated |
| 7 | boron in the core (ppm), as the boronmeter read it 20 min ago |
| 8 | hours until the planned ramp down starts, 0 once it has |
| 9 | hours until the planned ramp up starts, 0 once it has |
| 10 | the recent power the starting xenon is balanced with (rule 16: operators know their power history); after a trip's restart, the power whose balance is the restart's xenon, 1 when it is above full power's |
| 11 | the day's ramp rate, % of rated load a minute, as the dispatch gives it (owner, 2026-11-12) |
| 12 | the day's minimum load, a fraction of rated, as the dispatch gives it |
| 13 | the fraction of its cycle the core has burnt, which the operator's curve book is indexed by |
| 14 | the dispatch plan's load now, a fraction of rated: what the turbine may take, where observation 3 is what it takes |
| 15 | the offset's penalty deviation time over the last 24 h, in hours, as the process computer keeps it (NUREG-1431, B 3.2.3A) |

Observations 11 and 12 joined on 2026-11-12 and 13 to 15 on 2026-11-15, so the
observation grew from 11 to 16 channels: a learner sized for fewer inputs must be
rebuilt. The first 11 are as they were. The task's version goes up with the
re-record pass (step 4), not before.

Actions: `[rod demand, boron flow, turbine loading rate]` in [-1, 1]. Rod demand
reads as obs 2. Boron flow is -1 for full dilution, +1 for full boration. The
turbine's rate is -1 to unload at 5 %/min, +1 to load at it, 0 to hold; the turbine
never takes more than the plan, so +1 follows it, and the plan it does not take is
charged as lost output. The third action joined on 2026-11-15: a policy with two
outputs must be rebuilt. Xenon, iodine, their tilts, the boron on its way and the
delay's length are hidden; the power history in obs 10 is what an operator's xenon
tool starts from.

**Limits** (charter rule 14, Amendment 3):

| limit | kind | form | source |
|---|---|---|---|
| power, fuel temperature, hot leg (610 K, ours: under the 618.0 K saturation at 2250 psia, IAPWS-IF97) | trip | the envelope; a scram, 48 h of dead time, then the turbine's reload | ours, inside APR1400's 22 to 70 h from the scram to the rods' withdrawal (KNS 20A-065) |
| Tavg under 556 K | timed | 900 s allowed, then hot standby, 6 h of dead time, then the turbine's reload | ours, inside the family's 15 to 30 min to restore; 6 h, Doel-3's time down (INIS 1ypr9-dk595) |
| flux difference outside its band, 5 % of rated delta I either side of a target that scales with power | penalty clock | penalty time counts a second a second at 50 % of rated power and above, half a second a second from 15 % to 50 %, not below; 1 h in 24 h is free; past it, the lost output of a reduction to 50 % | NUREG-1431 Rev. 3.0, Bases B 3.2.3A, Notes 2 and 3 and Required Action C.1; the band's 5 % is EDF's and Krsko's |
| the same, at 90 % of rated power and above | timed | 30 min outside without a break (15 to restore, 15 to reduce), then the lost output of a reduction to 90 % | the same bases, Required Actions A.1 and B.1 |
| flux difference outside the acceptable operation limits, from 11 % of rated delta I at 90 % power to 31 % at 50 %, between those powers | timed | 30 min outside without a break, then the lost output of a reduction to 50 % | the same bases, Figure B 3.2.3A-1 ("for illustration only": a plant's core report gives its own) and C.1's completion time |
| Tavg above its DNB limit | timed | the full-power program plus 5.1 F (2.83 K) at every power; 2 h to restore, then MODE 2 within 6 h, so hot standby | STS LCO 3.4.1 (US EPR GTS and AP1000 bases, from search summaries); the margin from Braidwood's COLR, 588.0 F nominal against 593.1 F (a search summary, not opened: nrc.gov blocks automated reads), the only plant found with both numbers |
| rod insertion limit | timed | a straight line in power, 22 % of the stroke in at full power to 78 % at half; 2 h to restore (STS 3.1.6), then hot standby | KAIST's lead for the points; the family's shape |

The offset's clock looks back 24 h, in whole control hours: at the start of each
hour the hour just done is filed and the hour filed a day ago leaves the count, so
time outside the band a day ago stops counting ("1 hour during the previous 24
hours", the CAOC wording; REALITY.md section 8 [ST1]). A window is one day and
counts from 0, as before. Where two reductions are due the unit makes the deeper
one, so their prices do not add. Until 2026-11-15 the band was 0.05 of the offset,
so half as wide in delta I at half power, time outside counted in full at every
power, and nothing bounded how far outside the flux difference sat.

### The stream (docs/proposals/long-strata.md, S1)

As a continuing plant (`target_gym.stream`) each load-follow scenario is a day:
it lasts 24 h on average and brings one day's program on average, EDF's 39.5
days a year (REALITY.md section 7), and baseload days last 24 h too, so each day is a
load-follow day at load following's share (0.108, REALITY.md section 7). The
dispatch announces each day (`start_event`): the load starts down a time drawn in
`day_notice_range` (the window's own 0.5 to 2 h) ahead, ramps at the scenario's rate
to its minimum, holds a time drawn in `hold_range`, and comes back. The day carries
its rate and depth, so it runs to its end through a switch, and the plan's clock is
rebased on each new day, so it stays exact however long the plant runs. A day drawn
while one is under way, from its announcement to the end of its return ramp, is
absorbed (ours), so a visit draws days faster than one a visit, at the rate that
delivers one on average by the day's own shape (owner, 2026-11-11; `scenarios.py`):
1.21 a day for `edf_short_hold` to 1.77 for the Krško days, whose ramps keep the
dispatch busy 13.75 h. One draw a visit delivered 0.70 to 0.86 days a visit, 30 a
year. Draws are memoryless, so about two visits in five bring no day and a quarter
bring two or more. A scoring run pinned to a load-follow row is one long visit and
draws at its own rate, 0.051 to 0.098 an hour, which delivers one day per 24 h there
(`pinned_events_per_h`; owner, 2026-11-12). A stream starts with no day planned: nothing ahead and full load.

Burnup runs as in a window and moves the core along its cycle, its coefficients
with it. When the core reaches the end of its load-following range, 90 % of the
cycle (`cycle_range`), the stream refuels it (`refuels`; owner, 2026-11-11 and
2026-11-15): it is renewed at the range's start as at reset, uncharged, the outside
world carrying on, and burnup counts again from there. At 2.3 to 3.3 ppm a day the
range lasts 460 to 660 days. A window never refuels. Until 2026-11-15 the stream
refuelled every 30 days, to stay at the start of the cycle.

**The restart after a trip** (long-strata R8; owner, 2026-11-11 and 2026-11-12).
The trip's lump prices the dead time before the restart: 48 h after a scram
(`restart_steps`, ours, inside APR1400's 22 to 70 h from the scram to the rods'
withdrawal) and 6 h after a timed limit's hot standby (`standby_steps`, Doel-3's
time down). Iodine and xenon, means and tilts, are what that time at zero power
left (`env.xenon_after_shutdown`, the plant's own rates at n = 0 solved exactly):
from balance at full power, xenon is about 0.20 of balance after 48 h and builds
back over the next two days, and about 1.55 after 6 h, near its peak, and burns
out. The plant restarts at 15 % of rated power (`restart_load`, Doel-3's return),
or the plan's load if it is lower, lined out on its program with the rods parked
and critical on boron with the burnup since the last refuelling diluted, as
operators line out a restart. The core's burn rate, boron delay and point in its
cycle stay.

The turbine then reloads from the restart load, as fast as the third action asks
and no faster than 15 %/h to 50 % of rated power and 3 %/h above (the fuel's
conditioning ramps: Exelon's practice below the threshold, KEPCO NF's guideline
above it), until it meets the plan; from there the ramps no longer bind. From 15 % to full load
that is 19 h, close to Doel-3's 18 h at reduced power. The plan the turbine does
not take meanwhile is lost output, charged in tracking at the trip's own price
($666.67 a step at full shortfall), as the offset's derate is, so the lump and the
dynamics each pay their part once. Before any trip the plan is the load and the
charge is 0, so every window is unchanged up to its first trip.

Observation 3 is the turbine's load, the reload's while it lasts; the dispatch's
timing stays in observations 8 and 9. This differs from `boiler_drum`, whose
load-now stays the plan's and whose steam flow shows the reload: here Tref is
programmed on the turbine's load and the rod program reads it, so the turbine's
load is the one a control room reads. The turbine's load and Tref therefore left
the task's outside world: they follow the plant's own history.

Before, the plant restarted at the plan's full load with the xenon 15 h of standby
left, about 1.34 of balance, and the expert re-tripped 5 of 5 times about 18 min
in, the burnout driving the hot leg past 610 K. Earlier still, a trip restarted at
once in established operation, as `reset_env` would.

---

## 5. Known deviations

**D1: the load goes straight to heat, under one limit.** The turbine's valves wide
open are in (section 1). The governor, the steam's moisture, the steam generator's
level and the rest of the secondary's controls are left out.

**D2: the halves' coupling is not the thesis's.** D is set from Ringhals-4's eigenvalue
separation, not from the thesis's one-group constants, which coupled the halves 12
times too weakly and made the half-power mode neutral. The thesis cross-checks run on
the thesis's own D. With the new D, Krsko's half-power state comes out at +5.4 %
against +5 %.

**D3: rods from one side.** A bank reaches only the top, so at full power with rods
out nothing can push power up the core. Xenon tilted to the top can drive delta I
down and past the band, and only time brings it back. Real plants have more than one
bank.

**D4: no scored peaking.** Delta I is scored on its band, its penalty clock and the
acceptable operation limits, with no link to local power or to a trip. The limits
are the standard's illustration, not one plant's core report.

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

**D11: the axial mode does not change through the cycle.** The cycle's coordinate
moves boron, the moderator coefficient and the boron worth, not the axial mode's
damping, which stays the start of the cycle's (section 2). Twelve-foot cores grow
less stable axially with burnup; no figure for it was opened, so it is left out, and
late in the cycle the offset is easier to hold here than in the family. The starting
tilts are likewise the start-of-cycle reference's.

**D12: one conditioning rule.** The fuel's conditioning ramps cap the turbine only
from a trip's restart until it meets its plan. A turbine the operator holds low for
hours reloads at 5 %/min. Plants also recondition after long low-power holds; no
figure for the hold was found, and the task's holds last hours.

**D13: the plan is the grid's wish at every point of the cycle.** A day's plan keeps
its scenario's ramp rate, up to 5 %/min, whatever the core's cycle point, though a
unit late in its cycle would be dispatched within what it can carry. The plan here
is what the grid would take, and what the unit does not deliver is priced as lost
output: that is the family's own account of the stake (REALITY.md section 1).

**D14: the loading stop's setpoint is a training card's, and its second condition
is left out.** The turbine does not load while Tavg is 10 F (5.556 K) or more under
its program at the turbine's load, and loads again once Tavg is back (interlock
C-16, section 1; owner, 2026-11-16). What the interlock does is read from Vogtle's
safety report, which gives no number; the 10 F is from a training card for a plant
it does not name, whose vocabulary is a newer design's (REALITY.md section 4). The
card's second condition, a Tavg at or under 552 F (562.0 K) whatever the program,
is left out: it is the stricter one only below 18 % of load. The stop reads the one
Tavg the model has, where a unit reads its coldest loop's. Units count a return on
colder water as a design transient (2,000 over a unit's life, a search summary):
the plant does not count them.

**D10: one bank of safety valves, and the steam they blow is free.** The steam
generators' safety valves relieve the steam over 1176 psig (section 1; owner,
2026-11-17), which bounds a hot primary at part load. A unit has five valves a
steam generator on staggered set pressures and atmospheric valves an operator can
open below them; the model has one bank at the lowest set pressure, opening in
proportion to the overpressure. A lift is not counted and the steam lost is not
charged, where a unit loses feedwater inventory and reports the lift. Under the
valves a Tavg over its program is priced by the two-sided dead band past 1.6 K and
bounded by the DNB limit, as before. The condenser steam dump is left out: it is
unblocked only by a load rejection of 10 % in a step or 5 %/min sustained
(REALITY.md section 4).

**D9: burnup by the clock.** The fuel's burnup is worth boron at the day's drawn rate
for every second since reset, as if at full power. Real burnup follows the energy
made, so on a day with a 50 % hold it is overstated by at most half the hold's share
of the day: under 0.1 ppm at 3.3 ppm a day, 0.02 K of Tavg. It keeps the oracle's
model exact without a new state. The day starts lined out below its program by the
burnup earlier days left undiluted, 0 to 1 day of it, 0 to 0.8 K. Burnup's bite is
small within a day (0.56 to 0.8 K of Tavg at the BEAVRS rates), so it matters mostly
together with that carried debt.

**Retired tests (owner, 2026-11-08, TESTING.md section 10).**
The scram's peak xenon worth test (the task has no scram; the load cut's
swing is tested against Krsko instead), `test_the_reactors_winning_null_has_a_written_exception`
(the old plant's written exception is gone; acceptance, J2, still checks the nulls) and
`test_a_day_has_two_moves_for_the_scorer` (the old scorer's move detector, which the
weighted-saving headline does not use).


---

## 6. The expert

The expert is the reference: the Westinghouse procedure (Krsko, NENE 2022, section 3;
KNS 18A-534), kept over EDF's Mode A, which loses to the null on these days (D8), run
from the operator's curve book at the day's cycle point (owner, 2026-11-15). Bank D
holds the axial offset and never sits deeper than its insertion limit. Boron trims
Tavg, fed forward from the day's plan and from the operator's tool for xenon and the
fuel's burnup. The turbine loads at the rate the boron line carries. The owner
adopted the procedure on 2026-11-08, on condition it keeps the insertion limit.

**From the curve book (2026-11-15).** The reference reads the core's cycle point
(observation 13) and takes that point's boron worth and moderator coefficient
(`experts.curve_book`, the plant's own lines of section 1). What it works out for
boron is in ppm, turned into a flow by the pumps' rates at the boron read, so its
loops have the same gain diluting and borating, at any point of the cycle. Until
then its gains and its feedforward were fitted at the start of the cycle.

- *The plan's boron.* The program asks `ppm_per_load` of boron per unit of load, the
  power defect on the program over the boron's worth: 205.6 ppm at the start of the
  cycle, 216.7 at 80 % of it. It is counted, not timed: what the load the turbine
  will take asks for, a lead ahead (600 s, the middle of the line's delay), is added
  to what is still owed, and the line sends what it can of it each step. So a ramp
  faster than the line still gets all its boron.
- *The Tavg loop.* A SIMC PI past the 1.6 K free band, on what is charged (the
  owner's rule of 2026-11-10): `Kc = 1 / (k 2 theta)` and `Ti = 8 theta`, with `k`
  the curve book's K of Tavg per ppm and `theta` = 615 s, the boron line's dead time
  to Tavg from the recipe's bump tests.
- *The operator's tool.* Xenon's rate, and the burnup read off the cycle point (a
  ppm of boron is 1 / 1690 of the cycle), kept as a debt with the play of
  2026-11-10 below. Its edges sit two standard deviations of Tavg inside the free
  band, the zone's back-off (charter J1, k = 2): a Tavg that close to the edge
  predicts a charge. Under the load's 0.5 % wobble, with rods and boron still,
  Tavg's error spreads by 0.548 K at the start of the cycle and 0.209 K at 80 %
  (`experts.tavg_spread`; the figure is quasi-static, and the days recorded at the
  start of the cycle show 0.43 K). Each edge is pulled in by what xenon moves Tavg
  over the line's longest delay, 15 min, since an edge is a limit. The debt starts
  from where Tavg reads and is corrected toward it at the Tavg loop's integral
  time, so what the plan's boron and the line's delay leave behind does not build
  up unseen.

Two numbers still come from the recipe's bump tests (`scripts/reactor_ladder.py`):
the rods' loop on the offset and the line's dead time. Nothing was set from a cost.

**The turbine (2026-11-15).** The reference loads the turbine at the rate boron
carries (`experts.carried_rate`): full dilution at the boron read, plus xenon's
rate, over the ppm the program asks per unit of load. That is 1.97 %/min at the start
of the cycle with 1690 ppm and 0.50 %/min at 80 % of it with 450 ppm, the limits OAPS
calls theoretical (REALITY.md section 3). Tavg's free band is reactivity too: every
kelvin Tavg sits warmer than the cold side of the backed-off band is load the core
can pick up, 0.9 % of rated load at the start of the cycle and 2.4 % at 80 % of it
(a kelvin over `tavg_per_ppm x ppm_per_load`), and the turbine takes it over the
line's lead. So the turbine loads at the carried rate with Tavg at that cold side,
faster the warmer Tavg is, and tapers to a stop as Tavg cools from there to the free
band's cold edge (`experts.turbine_demand`). While the reference holds the turbine
under its plan, xenon's burnout carries load with the dilution, so it comes off the
boron still owed and the tool does not borate it away.

The reference does not dilute ahead of a return with the rods holding Tavg, and it
does not spend the offset's allowance: it is the standard strategy, which keeps the
offset on its reference (OAPS, section 3.1). What a planner gains over it is the
task's room, and the family's own account of what a predictive assistant adds
(REALITY.md section 3).

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
the excess. **ours,** family range 0 to 0.05, chosen at half, and not to be adjusted
after results (J1). It follows the target band's units: since 2026-11-15 the band is
5 % of rated delta I at every power (section 4), so the action band is 2.5 % of rated
delta I at every power; until then both were bands on the offset, which narrow in
delta I with the power.

**Rods on Tavg through a fast ramp (owner, 2026-11-10), fixed before any rerun.** On
the start-of-cycle core (reactor-v8) a load cut releases about 820 pcm of Doppler and
moderator feedback per 50 % of load, and boron moves at most 32 to 35 pcm a minute,
so a 5 %/min ramp outruns it: with the rods on the offset, Tavg ran 20 to 29 K hot
and the hot leg tripped. In the family, fast ramps are carried by rods on Tavg
(Westinghouse's automatic rod control, NRC HRTD 8.1; EDF's grey rods), while Krsko's
procedure, rods on the offset and boron on Tavg, is a slow-ramp one (8 %/h). So the
reference switched by one test: while the day's plan ramps faster than boron can
follow, the rods run Westinghouse's rod program on Tavg; otherwise bank D holds the
offset. Boron can follow when the ppm a second the ramp needs, the plant's power
defect per unit of load (from `rod_balance` on the program) times the planned load
rate, is within the line's full dilution (`dilution_flow / primary_mass * boron`) or
full boration (`boration_flow / primary_mass * (acid - boron)`), at the boronmeter's
reading. The handback is bumpless: the offset loop starts from where the rods are.
The rod insertion limit holds in both modes. No new number.

Since 2026-11-15 the same switch is read off the plan's boron count: the rods run
the rod program on Tavg while boration is still owed for a ramp down, more than the
line sends in a step, and for the line's lead after the last of it is sent, so the
handback waits for the boron to reach the core. On the way up the turbine waits for
the boron instead (above), and the rods stay on the offset.

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

**Each day's dispatch (re-record pass, step 2; owner, 2026-11-11 and 2026-11-12).**
The reference read the plan's hours to the ramps on its first step only, and planned
every day at the default 2 %/min down to 50 %. So in a stream it never saw a later
load-follow day, and on Krsko's days (8 %/h, and 4 %/h to 75 %) and the EUR's
5 %/min it fed forward a ramp the plant did not run: it tripped 3 times in every
krsko_slow and krsko_shallow window and up to twice in fast_return's. It now reads
the dispatch on its first step and again whenever the hours ahead show a new day
once the day it holds has ended, as an operator reads each day's program when it
comes, and plans with the day's own rate and minimum, which the dispatch now gives
(observations 11 and 12). A ramp the dispatch reads 0 for has already run, so a
reference started mid-day takes it as done rather than starting now.

**The load the turbine takes (owner, 2026-11-12).** After a trip's restart the
turbine reloads (section 4, the stream) and does not follow the plan until it meets
it. The reference fed the plan forward all the same: a day announced just before a
trip made it borate for a ramp down the reloading turbine never made, and Tavg fell
21 K under its program and tripped on its floor. It now feeds forward the load the
turbine will take. When the turbine's load reads more than `reload_band` under the
plan, it is reloading, and the reference predicts it climbing from where it reads at
the plant's conditioning ramps, which the operators run and so know, until it meets
the plan. `reload_band` is 0.05, **ours:** ten times the frequency wobble's 0.5 %
spread, which a turbine on its plan never leaves.

Both are structural; no gain moved. Windows on edf_daily, edf_short_hold and
baseload are unchanged, bit for bit. The Krsko and fast_return windows move and stop
tripping (the re-record pass, step 5, records them again).

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `reactor` | 2880 | 30 | 24.0 h | 3 in [-1, 1] | 16 | 140 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per 30 s control step

`compute_reward = -(tracking + running + failure)`, with Tavg past the dead band on either side of its program, the offset's reductions and the plan the turbine does not take in `tracking`, and rod travel and effluent in `running`; each physics sub-step charges its 5 s and `step_env` sums the six, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.05 K | the RTD reading's resolution, about its 0.045 K scatter: a documented minimum, not a certified floor |
| `e_tol` | 1.6 K | the black rods' dead band: EDF's rods move only past 1.6 C off program (IAEA NP-T-3.23, annex II-5.1). CEA's papers on Mode G's R group give 0.8 C with slow motion up to 1.7 C |
| `tracking_exponent` | 1 | linear, as lost output is |
| `imbalance_multiple` | 1.25 x 0.0023 per K of `spot_price_per_MWh` on `P_electric_GW` | **ours.** Off program, steam conditions cost output. The relative Carnot slope at 547 K steam and 305 K condenser is 305 / (547 x 242) = 0.23 %/K. At the one imbalance price of the priced plants, $100/MWh (1.25 x the $80 spot), on `P_electric_GW` = 1.0 GW, a third of the 3 GW thermal, that is 100 x 1000 x 0.0023 = $230/h per K, so **$1.917 per K over a 30 s step**. It prices a hot plant as a cold one, which the slope does not say: the band is a procedure band both cited procedures keep on either side (IAEA NP-T-3.23 annex II-5.1; NRC HRTD 8.1), priced like the offset band by the output it costs (physics review, finding 1, owner 2026-11-10, reversing Amendment 3 decision 7) |
| `rod_step_price` | $1 per step | **ours,** kept by the owner on 2026-11-07. Rod travel, as the steps the rods actually moved |
| `ao_band`, `ao_ref` | 0.05, 0 | the target band: 5 % of rated delta I either side of the target at every power (EDF's 0.05, IAEA NP-T-3.23, annex II-4.3; Krsko's 5 %; NUREG-1431's figure draws the band 10 % wide at every power). The target is the offset at full power, xenon balanced, rods out, times the power: 0 in this core |
| `ao_allowed`, `ao_half_power`, `ao_free_power`, `derate_to`, `derate_price_per_MWh` | 3600 s, 0.5, 0.15, 0.5, $80/MWh | the penalty clock: a second a second at 50 % of rated power and above, half a second from 15 % to 50 %, none below; 1 h in 24 h, then a reduction to 50 %, its lost output priced at $80/MWh (inside the family's $80 to $100) on `P_electric_GW` (NUREG-1431 Rev. 3.0, B 3.2.3A, Notes 2 and 3, Required Action C.1) |
| `ao_strict_power`, `ao_strict_allowed` | 0.9, 1800 s | at 90 % and above the flux difference must be in its band: 15 min to restore it and 15 min to be under 90 %, then that reduction's lost output (the same bases, Required Actions A.1 and B.1) |
| `ao_limit_high`, `ao_limit_low`, `ao_limit_allowed` | 0.11, 0.31, 1800 s | the acceptable operation limits between 50 and 90 %: 11 % of rated delta I at 90 % power to 31 % at 50 %, read off Figure B 3.2.3A-1, which is "for illustration only"; outside them 30 min without a break, then a reduction to 50 % (C.1's completion time) |
| `turbine_rate` | 5 %/min | the third action's full scale; the plan the turbine does not take is lost output at `derate_price_per_MWh`, $666.67 per step per unit of rated load, the trip's own price |
| `ril_full`, `ril_half`, `ril_allowed` | 0.22, 0.78, 7200 s | the rod insertion limit, a straight line in power (KAIST's lead), 2 h to restore (STS 3.1.6) |
| `effluent_price` | $0.0628 per kg injected | **ours.** Every kg injected displaces a kg to treat (OAPS). Boiling it off takes 2.26 MJ/kg, 0.628 kWh, at $100/MWh |
| `failure_cost` | 666.67 $ per step | a step of lost production: 1 GWe at $80/MWh, inside the family's $80 to $100 |
| `restart_steps` | 5760 (48 h) | a scram's dead time, **ours,** inside APR1400's 22 to 70 h from the scram to the rods' withdrawal for the restart (KNS 20A-065). The reload after it is played out and charged in tracking |
| `standby_steps` | 720 (6 h) | hot standby's dead time after a timed limit's overrun: Doel-3's time down (INIS 1ypr9-dk595). It was 15 h, Doel-3's 6 h down and its 18 h reduced, until the reload was played out (owner, 2026-11-12) |
| `restart_load` | 0.15 | the load a restart comes back at: Doel-3 held its return at 15 % after a 6 h scram, its offset off band from the xenon the shutdown built (INIS 1ypr9-dk595) |
| `reload_knee`, `reload_rate` | 0.5, 3 %/h | the turbine's reload above 50 % of rated power: KEPCO NF's fuel conditioning guideline, 3 %/h above a 50 % threshold, 2 to 4 %/h tested (KNS 15A-200); 3 %/h above the threshold at most plants (EPRI, REALITY.md [ST2], [ST3]) |
| `reload_fast_rate` | 15 %/h | the reload below the knee: Exelon's practice of about 15 %/h below its threshold (REALITY.md [ST3]); KNS 15A-200 finds the rate there does not matter to the fuel (5 to 50 %/h tested). From 15 % the reload takes 19 h, Doel-3's 18 h. It was first 50 %/h, **ours,** the "no limit" KNS 15A-200 simulates; the expert then re-tripped on Tavg's floor after every scram, the turbine loading faster than boron can follow, so the owner took the practice rate (2026-11-12) |

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
restarts at the restart load with the xenon its dead time left, and the turbine reloads (`restart_plant`,
section 4, "The stream"), on the same clock. `terminated` stays
false unless `continuing = False`, which ends the episode on that step instead; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).

## Readings

The observation reports power and the coolant temperatures through their instruments
(`docs/proposals/measurement-noise.md`). Rod position, the load and Tref stay exact, as do the dispatch's hours, rate and minimum (plan values, not sensors), and boron reads 20 min late:
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
