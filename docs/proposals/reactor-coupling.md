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
