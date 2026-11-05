# Expert protocols on noisy plants

A proposal written on 2026-11-05. It amends
[expert-protocols-process.md](expert-protocols-process.md) and
[expert-protocols-aircraft.md](expert-protocols-aircraft.md) before they run, so
that experts are tuned on the readings they will have.

## Status

The owner made the design choices on 2026-11-05, listed under
[Decided](#decided). The recipes are accepted only after the
[measurement](#measure-first) has been read on the fixed plants.
[industrial-fixes.md](industrial-fixes.md) reversed noise decisions 1, 5 and 7 after
this was written: analysers now go stale, so N8 must allow for dead time and true
sampling, and PR 9 of that plan applies these amendments with its own.

## Why

Sensor noise is now part of the scored tasks, and a later study retunes experts
and policies at several noise levels to see which methods adapt best. The
protocols were written for a noise-free plant:

- **C1** subtracts an unstepped twin run on the same noise. A real plant cannot
  do that, and it cancels sensor noise, so experts would be tuned as if their
  sensors were perfect.
- **C6** fixes `tau_c = theta`, with no noise term.
- Nothing measures noise, and the aircraft protocol does not mention it.

## What the sourcing found

**Engineers rarely tune from measured noise.** Most settings in this suite are
fixed by practice: TECS filters, the wind turbine's 0.25 Hz speed filter, the
rod deadband, G36's zone PI. A refutation pass confirmed 2 of 24 candidate rules
as stated, and found that 13 would not change the outcome between 0.5× and 2×
noise. The amendments below keep what survived, with its corrections.

So an expert that stays the same across noise levels is often faithful practice,
not a bias against it. Aircraft, wind, HVAC and the reactor fall here, because
their noise sits far below fixed thresholds. The expert changes where an engineer
would change it: on process loops, the bearing-only estimator and analyser-fed
loops.

The research data is in `docs/research/noise_tuning_practice.json` in ExpertBench.

## Amendments

| # | Replaces | Change | Label | Responds to noise? |
|---|---|---|---|---|
| N1 | C1 | One run: hold, record a baseline, step, fit. No twin. `unstable_cstr` and drum level test closed loop, since nobody holds output there | read | Fit scatter grows |
| N2 | New | Noise band `NB` = 6 sd of the reading about a fitted line over the baseline, floored at one rounding step. Slow analysers pool seeds | read | Yes, where sensors reach the baseline |
| N3 | C2 | Step is the larger of C2 and what moves the reading by 5 `NB` (30 sd). Rerun once. If an actuator limit blocks it, it ships and the ratio prints | read | Where it binds |
| N4 | C3, C4 | Delay is read at the first exit from the band, with a persistence count, and held in the fit. Integrating loops use `NB / (2 |k'| du)` | read | Small |
| N5 | New | PV filter sized so noise-driven output stays inside the actuator's dead band: `Kc NB sqrt(a/(2-a)) <= DB`, capped at 10% of `Ti` | derived | Until the cap |
| N6 | C4, C7 | `theta_eff = theta + Tf`, then recompute. Fixed pass count | read | Through N5 |
| N7 | C6 | If output noise still exceeds `DB` at the cap, lower `Kc` to `DB / NB_f`, never below Skogestad's `Kc,min` | derived | Yes: `Kc` scales as 1/noise |
| N8 | New | Analyser loops count updates, not steps, and act only on moves past `NB` | derived | Yes |
| N9 | Aircraft 3.5 | The bearing-only estimator becomes a real two-state filter, set from measurement and process noise | derived | Yes: lag about 9, 16, 28 s |

Fixed lengths that no source gives, such as the baseline, the persistence count
and the pass count, are set here before any run and labelled ours. N3 cancels
part of N1 and N4 wherever it binds, because the step grows with noise. That is
what an engineer does, so it stays.

## Rejected

| Candidate | Reason |
|---|---|
| Fixed filter from `alpha` (Segovia 2014) | A research rule, not field practice, with no noise term, and usually below one sample |
| Dropping derivative action as noise rises | The filter cap means it never triggers |
| Error gap or deadband for noise | Practice uses it for limit cycles, not noise |
| Filter on TECS speed rate | ArduPilot sets these filters as fixed values, with no noise estimate |
| Aircraft rate-loop filter | Rate noise is rounding, which does not scale |
| Wider rod deadband, rod lag, HVAC band, wind filter | Noise stays inside fixed settings at 2× |
| Battery trim gate | No battery source, and setpoint jitter would switch the trim off |

## Measure first

The first measurement ran on `feat/sensor-noise` at `509fd5b`. It settled each
expert, held the output and took the spread about a fitted line at 0.5×, 1× and 2×.
A growth of 4 means sensors set the band, and 1 means load swings do.

| Task | Reading | Growth | Noise share at 1× | Result |
|---|---|---|---|---|
| `first_order` | `x` | 4.00 | 1.00 | Follows sensors |
| `battery` | power | 4.00 | 1.00 | Follows sensors |
| `cstr` | `C_a` | 4.01 | 1.00 | Follows sensors |
| `distillation` | `yD`, `xB` | 3.71, 3.35 | 0.89, 0.86 | Follows sensors |
| `four_tank` | `h1` | 2.81 | 1.00 | Follows sensors, with a rounding floor |
| `glass_furnace` | crown | 1.42 | 0.23 | Load swings dominate |
| `ph_neutralization` | pH | 1.40 | 0.28 | Load swings dominate |
| `boiler_drum` | level, pressure | 1.28, 1.03 | 0.11, 0.03 | Load swings dominate |
| `compressor_surge` | `dp` | 2.76 | 0.15 | Load swings at 1×; trips distort 2× |
| `unstable_cstr` | `T` | 1.54 | 0.57 | Not readable: it runs away when held |
| `cement_kiln` | free lime | 1.14 | 0.17 | Not readable: noise held past the window |

So N3, N5 and N7 respond to sensor level on five tasks. On pH, boiler, glass and
compressor they size from what an engineer would see, which is mostly load swing,
so those experts stay nearly the same across levels. That is faithful practice.
`unstable_cstr` needs N1's closed-loop test, and the kiln needs pooled seeds.

The plants change under [industrial-fixes.md](industrial-fixes.md), so measure again
after its plant PRs. Step, filter and gain tables wait for that. Accept the recipes
only after reading them.

## Gains per noise level

- Each level's recipe output is stored separately, so the scored gains at 1× are
  not touched by a study at another level.
- `make_expert(name)` keeps its meaning, and takes the level when one is asked
  for.
- A level's gains are part of the expert fingerprint for that level.

## Known limits

- `glass_furnace` and the `distillation` composition test may be too small to
  read: stored gains suggest the step cannot clear the noise inside the window.
- The dead band comes from valve practice. Non-valve actuators (pumps, kiln
  feeder, converter, compressor drive) have no sourced figure, so N5 and N7 do not
  apply to them.
- Sources were read through automated fetches, and several core references could
  not be opened: Seborg, Shinskey's book, Cooper's book, Astrom and Hagglund's
  books, ISA-TR5.9, the NREL report and the full G36 text.

## Size budget

About 250 lines across `scripts/` and the experts, on top of the existing
protocols' budgets. Measure it after.

## Decided

1. **The dead band in N5 and N7 is 1% of span,** the sourced limit for a
   well-engineered valve. Tighter figures could detune `cstr` 16 to 25 times, for
   valve wear the plant does not model.
2. **C5 uses the median unless the spread exceeds what the measured noise
   explains.** Noise inflates the largest of ten fits, so 2× would otherwise
   detune for a false reason. This test is ours.
3. **Tests too small to read ship, with the signal-to-noise ratio printed,** as
   the protocols' poor-result rule says.
