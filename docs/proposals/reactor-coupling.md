# Reactor: the core's halves are coupled too weakly

Status: finding for the owner, 2026-11-08. No code changed. It explains why the EDF
rebuild's reference cannot hold the offset band at half power.

## What was seen

At half power the rods move the axial offset by about 0.032 per step, against 0.0034
at full power, by the ladder's own bump tests. EDF's band of +-0.05 is then less than
two rod steps wide, and every rung of the rebuilt ladder lets the offset reach its
clip. The axial mode is also near neutral at half power (-0.0012 per hour), against
-0.035 per hour at full power.

## The env is faithful to its model

The thesis's equations (Mastroianni 1979, eqs 2.15 to 2.18), rebuilt outside the env
with Sigma_a = 0.39, give the same trend:

| Flux | Tilt from the same pulse | Period | Decay |
|---|---:|---:|---:|
| full | 0.456 | 29.4 h | -0.013 /h |
| 75 % | 0.561 | 28.7 h | -0.002 /h |
| half | 0.696 | 26.9 h | +0.002 /h |

The trend has a known cause. DP-1239 (Savannah River) says the upper bound of the
instability region "arises because xenon burnout saturates with increasing flux while
the effect of the power coefficient does not". So lowering the power moves a core that
feedback holds stable at full power toward instability. In the thesis's offset formula
everything goes through N = leakage - xenon + feedback. The feedback halves with power,
the xenon term falls less, and the offset's response to a tilt goes as 1/N.

## The fault: the leakage term is 12 to 25 times too small

The leakage term is the coupling between the two halves. With the thesis's one-group
constants, D = 0.405 cm and nu Sigma_f = 0.39 /cm, the first axial harmonic's
eigenvalue separation is D x 3 pi^2 / (H^2 nu Sigma_f) = **23 pcm**. For Ringhals-4,
Pedersen, Demaziere and Vinai (Progress in Nuclear Energy 202, table 3) give **579 pcm**
from their heterogeneous model and **283 pcm** from their homogeneous one. A D of 0.4 cm
is a thermal-group figure. The coupling between the halves is carried mostly by fast
neutrons, which migrate much further. The thesis made up for the weak coupling with its
fitted feedback, which held at full power but not at half.

## What matching the coupling does

The thesis model, rebuilt outside the env, with D set to match each separation:

| Coupling | D | Full power: tilt, period, decay | Half power: tilt, period, decay |
|---|---:|---|---|
| thesis, 23 pcm | 0.41 | 0.456, 29.4 h, -0.013 /h | 0.696, 26.9 h, +0.002 /h |
| 2G-HOM, 283 pcm | 4.99 | 0.315, 33.5 h, -0.056 /h | 0.436, 35.7 h, -0.061 /h |
| 2G-HET, 579 pcm | 10.2 | 0.226, 38.3 h, -0.095 /h | 0.286, 44.8 h, -0.081 /h |

At 283 pcm, the separation of the homogeneous model, closest to this homogeneous
two-point core:

- The full-power decay is -0.056 per hour, against Ringhals-4's -0.05.
- The period is 33.5 h. That sits inside the 31.5 to 39.3 h the Chalmers paper's models
  give, but above Ringhals-4's 28.3 h.
- Half power is stable at -0.061 per hour, not neutral.
- The offset's static response at half power falls by N's ratio, 0.0046 to 0.0106, so
  about 2.3 times.

## What it would change

D in the env goes from 0.405 to 4.99, cited to Pedersen's table 3, and labelled as
derived from that separation.

- Every measured property needs re-measuring: the mode at each power, the rods and
  boron moves, and the ladder.
- Some current matches move. The full-power period goes from 29.4 h to about 33.5 h,
  and the thesis checks would be compared at the thesis's own D.
- The EDF rebuild's ladder is then rerun before any oracle.

The owner decides whether to make this change.

## Done, 2026-11-08

The owner chose 283 pcm. D is now derived from it in `env.py` (4.99 cm). The thesis
cross-checks run on the thesis's own D. Two new tests pin the separation and a damped
half-power mode. Remeasured with `scripts/reactor_properties.py`:

| Load | Period | Decay |
|---|---:|---:|
| full | 32.7 h | -0.070 /h |
| 75 % | 33.7 h | -0.050 /h |
| half | 34.3 h | -0.057 /h |

At full power, 60 rod steps move delta I by -0.34 (it was -0.47), and the same worth in
boron by about 3e-6.

The EDF ladder on this core: rung 3, the cascade, is the best clean rung at $103,386,
with no trip. It is still weak:

- **Offset:** past the band on 863 of 2880 steps, peaking at 0.90.
- **Tavg:** past the dead band on 2090 steps.
- **Null:** rods parked with no boron costs $161,561.

The trace shows two causes:

- **Half-power hold:** rods, boron and offset cycle with about a 2 h period.
- **Back at full power:** the rods sit on their top stop for hours with Tavg 10 K low.
  They are parked at the top, so they cannot withdraw, and boron is slow to make up.

EDF's temperature group sits in a manoeuvring band inside the core, so it can move both
ways. No public figure was found for it. A US NRC document says the lead bank at full
power stays more than 190 of 228 steps withdrawn; it was not opened.

## The temperature rods' band: what the sources give

Two open papers by CEA and EDF authors on EDF's 1300 MW units in Mode G, both read in
full:

- Muniglia et al., ICAPP 2016, *A multi-physics PWR model for the load following*:
  <https://hal.science/hal-01355257v1/file/paper_ICAPP.pdf>, section II.C.
- Muniglia et al., arXiv 2107.11201: <https://arxiv.org/pdf/2107.11201>, section 2 and
  its figure 2.

What they say:

- **Two groups, two jobs.** The power-shimming rods (grey G1 and G2, black N1 and N2)
  run open loop, set from the electrical power by a calibration. The temperature group
  R, 9 black rods, runs closed loop on Tavg.
- **R's band.** "the regulation rods are shut within a maneuvering band which height is
  27 steps in the upper part of the core". The core is 260 steps of about 1.65 cm. The
  second paper says "about 50 centimeters in the upper part of the core".
- **R's speed program.** 0 steps/min inside +-0.8 C. 8 steps/min from 0.8 to 1.7 C.
  Linear from 8 to 72 steps/min between 1.7 and 2.8 C. 72 steps/min past 2.8 C.
- **Boron.** It takes over when R is at the edge of its band. To lower a high offset,
  "the operator reduces the boron concentration (dilution) to increase the temperature.
  Automatically, the regulation rods will insert". That is item 4's mechanism, now
  sourced.
- **Make-up.** Boric acid at 7700 ppm (ours was 7800, unsourced). At most 30 t/h for
  dilution and 10 t/h for boration, 8.3 and 2.8 kg/s, against OAPS's 10 and 3.

## What this means for the task

- **The band is Mode G's.** In Mode G, R is a small group in a narrow band because the
  shimming rods take the power defect. This task has one bank holding all 1500 pcm, as
  in Mode A (OAPS: one black group on Tavg). No band for Mode A was found.
- **Parking one whole bank in a 27-step band does not work.** Scaled to 228 steps it is
  24 steps from the top, worth about 11 pcm on this bank's S-curve. Far too little to
  regulate.
- **The dead band conflicts.** The CEA papers give +-0.8 C, with slow motion up to
  1.7 C. The IAEA report's 1.6 C is the frequency-control threshold for moving the
  black rods. The CEA program is the more complete source. It also matches 0.8 C plus
  slow speed to 1.7 C.

So a sourced band needs the task to follow Mode G, which 48 of EDF's 56 units run
(OAPS 2.2). That means two rod groups: shimming rods on a calibration of power, and an
R group in its 27-step band on Tavg with the speed program above. Staying with one
Mode A bank keeps the rods parked at the top or puts them at an unsourced depth.

## Mode G, first build (2026-11-08): the plant fails its own line-out

What was built, on the work-in-progress commit:

- Two groups and three inputs.
- The shimming group is worth the power defect from 100 % to 0 %, 1922 pcm (derived),
  set by a calibration of position against power, at 60 steps/min.
- Group R sits in its 27-step band, worth 168 pcm (ours), at 72 steps/min.
- R's position is observed. 20 contract tests pass.

Measured with no controller:

- At full power the axial mode has a period of 33.3 h and decays at -0.070 /h.
- At 75 % and 50 %, lined out with the shimming group on its calibration, the offset
  sits at its -0.99 clip.
- Moving R across its band at full power shifts delta I by -0.39.
- The same reactivity in boron shifts it by -0.04 and arrives 11 min later.

The cause: the model turns each pcm of rods into about 0.0024 of offset. Half the
shimming group's worth, one bank in the middle of the core, then tilts the core
completely. EDF spreads that worth across grey groups inserted in sequence, so that "as
less perturbations as possible" are produced (ICAPP 2016, II.B). That design is the
point of Mode G, and this model has no data for it: no group worths, no grey strength,
no calibration curve were found.

The core physics itself is not the problem now. At Krsko's half-power state (D bank 34
steps in, xenon 110 of 96), the model with the 283 pcm coupling and the old single bank
gives delta I +0.054, against the published +0.05.
