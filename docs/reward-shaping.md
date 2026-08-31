# Reward shaping

TargetGym exists to ask a specific question: **can a learned policy hold a
setpoint better than a PID or an MPC?** That question lives entirely in the
reward. A reward that saturates once the error is "small enough" scores a
policy holding 1 m the same as one holding 10 m, and the comparison the
benchmark was built to make becomes invisible — not wrong, invisible.

So the reward is not a detail to be tuned afterwards. It is the measuring
instrument, and this page records how it was chosen.

## The constraint nobody escapes

A bounded per-step reward has a fixed amount of dynamic range to spend, and an
environment has a wide error range to cover. The aircraft's altitude envelope
is 12 km; the tracking that distinguishes a good controller from a great one
happens over metres. **Every reward shape is a decision about where to spend
that range.**

- **Linear in the error** spends it uniformly per metre. With 12 000 m to cover,
  the last 10 m receive 0.08 % of the range.
- **A band** (Gaussian or rational, `e/band`) spends nearly all of it within a
  few multiples of the band, and almost none inside it or far outside.
- **Logarithmic** spends it uniformly *per decade* — the same amount separating
  1000 m from 100 m as separating 1 m from 0.1 m.

Only the last is scale-free, and scale-free is what "closer is better" has to be
if it is to mean the same thing near the target and far from it.

## What was measured

One family of controllers — the shipped cascaded PID, detuned by scaling its
outer altitude gain — flown on the 2D aircraft over three seeds. The *same*
trajectories scored under four reward shapes, each normalised so that reward is
1 at the target and 0 at the edge of the envelope, which makes the columns
comparable.

| `Kp_alt` | settled \|err\| | original `(1-e/span)¹⁰` | pseudo-Huber | band `1/(1+(e/b)²)` | log-scaled |
|---|---|---|---|---|---|
| ×0.15 | 311.75 m | 0.7883 | 0.9775 | 0.2386 | 0.4491 |
| ×0.30 | 63.17 m | 0.9516 | 0.9966 | 0.6234 | 0.6512 |
| ×0.60 | 17.42 m | 0.9861 | 0.9994 | 0.8868 | 0.7633 |
| ×1.00 | 13.36 m | 0.9892 | 0.9997 | 0.9016 | 0.7920 |
| ×1.60 | 9.41 m | 0.9924 | 0.9998 | 0.9330 | 0.8297 |
| ×2.40 | 6.49 m | 0.9947 | 0.9999 | 0.9578 | 0.8613 |

All four rank the controllers correctly. Ranking is not the hard part. What
matters is **how much reward the last improvement is worth**, because that is
the signal an optimiser has to find and hold on to:

| shape | 312 m → 6.5 m | 9.4 m → 6.5 m | share earned in the last step |
|---|---|---|---|
| original `(1-e/span)¹⁰` | 0.2065 | 0.0024 | 1.1 % |
| pseudo-Huber | 0.0224 | 0.0001 | **0.4 %** |
| band `1/(1+(e/b)²)` | 0.7193 | 0.0249 | 3.5 % |
| **log-scaled** | 0.4122 | **0.0317** | **7.7 %** |

The log-scaled reward pays the most, in absolute terms, for the most precise
improvement — and it is the only one where refining an already-good controller
is worth a meaningful share of the total.

## Why pseudo-Huber did worst, which was not expected

Quadratic near zero and linear far away is the standard robust-regression
answer, and its *shape* is reasonable. It came last anyway, separating the
whole family by 0.022 out of 1.

The reason is the normalisation, not the shape. Made bounded over a 12 km
envelope, its near-linear far field consumes essentially the entire range, so
everything below the transition scale compresses into ≈ 1. It fails for the
same reason the original did: it spends its dynamic range on the far field.

This is worth stating plainly because it is the trap. Reasoning about a reward's
*local* shape — "quadratic near the target, so it is sensitive there" — says
nothing until you ask what fraction of the bounded range that region receives.

## The chosen shape

```python
# doc: skip -- the formula as it appears in plane/env.py, not a runnable snippet
reward = 1 - log1p(error / precision_floor) / log1p(span / precision_floor)
```

1 at the target, 0 at the edge of the envelope, and every halving of the error
worth the same: 0.0835 from 1600 m to 800 m, 0.0824 from 100 m to 50 m, 0.0678
from 6.25 m to 3.13 m.

Two parameters, and the distinction between them matters:

- **`span`** normalises the result into [0, 1]. Unlike the shapes above, it does
  *not* set the sensitivity — a logarithm has no scale, so widening the envelope
  rescales the reward without changing what it prefers.
- **`precision_floor`** is the error below which more precision stops *meaning*
  anything. It is a resolution, not a tolerance: the reward pays for every
  halving down to this point and only flattens beneath it. It should be a
  physical limit — a barometric altimeter reads to about a metre, so tracking
  tighter would reward chasing measurement noise. It is also what keeps the
  reward bounded, since an unfloored logarithm diverges at zero error.

A **band** is the thing to avoid here, and the distinction is easy to lose: a
band says "get inside this tolerance", a floor says "keep getting closer, until
closer stops being measurable".

## Status

Applied to the whole aircraft family: the 2D plane, all three 3D tasks (heading,
circle, figure-8) and the lead term of the patrol formation, all going through
one `log_scaled_reward` in `utils.py` rather than four transcriptions of the
formula. The remaining environments still use their original shapes — two
Gaussians on bands, the rest assorted forms built on `abs()` — and converting
them is the reward-shaping phase in the roadmap, not a sweep: each needs its
floor chosen from that plant's own instrumentation.

Three floors are in use, each a sensor resolution rather than a tolerance:

| Floor | Value | Source |
| --- | --- | --- |
| `precision_floor` | 1 m | barometric altimeter resolution |
| `heading_precision_floor` | 0.0087 rad (0.5°) | AHRS / compass resolution |
| `position_precision_floor` | 3 m | civil GPS horizontal accuracy |

### What converting the 3D tasks turned up

The two path-following tasks scored proximity with a Gaussian a tenth of the
path radius wide — σ ≈ 1 km for a 10 km circle. Across the entire band those
controllers actually operate in, that reward is flat: from a 100 m cross-track
error down to 0.1 m it moves by 0.005. The log-scaled form moves by 0.43 over
the same range.

Fixing the shape then exposed a defect underneath it. The figure-8 finds its
cross-track error by `argmin` over 400 samples of a 44 km curve, so the number
is quantised by the sample spacing: an aircraft flying the commanded curve
*exactly* was reported up to **66 m** off it — more than the figure-8 expert's
own settled error, meaning the reward had been measuring its own discretisation.
Projecting onto the two adjacent chords brings the floor below a millimetre and
makes a known 1 m offset read as 1.000 m. A reward can only pay for precision
its error metric can see; that is now check 11 on the model review checklist.

Two things found while doing the aircraft, recorded so the phase starts from
them:

- **The existing bands are guesses.** Five of eleven are not mentioned in their
  own `PHYSICS.md` at all, against a convention that marks 235 parameter rows
  as sourced and 54 as `TUNED`. Where a real operational tolerance exists —
  ASHRAE comfort ranges, drum-level trip bands, free-lime quality specs — the
  floor can be sourced rather than guessed.
- **Four-tank's D1 already records this defect**: *"the reward band was three
  times the operating range … originally normalised by the full tank span."*
  The aircraft was an unfixed instance of a failure mode this repository had
  already diagnosed once, which is the argument for writing the convention down
  rather than rediscovering it a third time.
