# How TargetGym builds and validates environment physics

The goal is **simple yet faithful** simulation: the smallest model that still
reproduces the behaviour that makes the control problem what it is. This
document is the recipe for getting there in a domain nobody on the project is
an expert in, and for proving afterwards that we did.

It was derived from a pilot on `target_gym.plane` — the most mature model in the
library — which is worth stating plainly: applying this method to the plane
surfaced three real defects that its existing test suite had passed over for its
entire life. See `src/target_gym/plane/PHYSICS.md` §5.

---

## 1. The core distinction

Not all claims about a model are the same kind of claim, and they cannot be
tested the same way.

| Kind | Example | Needs a source? | How it's tested |
|---|---|---|---|
| **Structural** | Energy balance closes. More fuel ⇒ hotter crown. Drag ∝ V². | No | First principles / invariants |
| **Parameter** | `c_p,glass = 1250 J/(kg·K)`; `k = 0.045` | Yes — citation *or* derivation | Range check against the cited value or the geometry it must be consistent with |
| **Behavioural** | Crown thermal time constant ≈ 15 min; L/D ≈ 17 | Yes — reference operating point | Numeric assertion with tolerance |

Most modelling errors are **parameter** or **behavioural** errors, and a test
suite made only of structural checks — which is what TargetGym had — cannot see
them. The plane's `k` (induced drag) and `cl_alpha` (lift slope) are both
individually plausible numbers; they are only *wrong together*, because they
imply different wings. Only a consistency check catches that.

## 2. The cardinal rule for tests

> **Never assert a formula by restating it.**

A test that recomputes its subject's own expression validates transcription, not
correctness — if the formula is wrong, the test is wrong identically. The old
`test_newton_second_law` re-derived the function's own force sum and asserted
equality. It could never fail for a physics reason.

Assert **emergent** consequences instead: figures of merit, equilibria,
invariants, published table values, scaling laws.

---

## 3. Procedure for a new environment

### Step 0 — Choose a model with a reference, if one exists
Strongly prefer systems that are already published control benchmarks, because
then validation data exists and the sourcing burden collapses. Where no
benchmark exists, a model derived from first principles is acceptable
(**physically defensible**), but §4's consistency obligations get stricter.

### Step 1 — Write `PHYSICS.md` *before* the code
Next to the env module. Sections:

1. **Scope** — what is modelled, and an explicit table of what is *not*, with
   the rationale. Omissions are design decisions and must be legible.
2. **Governing equations** — the ODEs, with symbols and units.
3. **Parameter table** — every constant: symbol, value, unit, and
   source/derivation. Each entry is exactly one of:
   - a citation,
   - a derivation from geometry or first principles (show the arithmetic),
   - `TUNED — not sourced`, stated honestly.
4. **Figures of merit** — derived quantities comparable to reality
   (time constants, efficiencies, steady-state gains, dimensionless groups).
5. **Known deviations** — quantified, each with an `xfail` test.

Writing the table first is what forces the questions. The plane's duplicate
`M_crit` and its unreachable `CL_max` both become obvious the moment you try to
fill in a "source" column.

### Step 2 — Derive reference operating points
Compute, by hand or in a scratch script, what the system *should* do at one or
two documented conditions. These become the behavioural assertions. Do this
**before** looking at the model's output, so the expectation is independent.

### Step 3 — Probe the implementation against them
Run the model at those conditions and compare. Disagreements are findings, not
noise — resolve each into either a parameter fix or a documented deviation.

### Step 4 — Codify as tests
Every row of the parameter and figure-of-merit tables becomes an assertion, so
`PHYSICS.md` cannot drift from the code.

---

## 4. The standard test battery

Reusable across environments; each env implements the applicable subset.

| Test | What it catches |
|---|---|
| **Equilibrium** — at the documented steady state, all derivatives ≈ 0 | Most parameter errors. Very high value per line. |
| **Conservation** — energy/mass balance closes to <0.1 % over an episode | Sign errors, missing/double-counted terms |
| **Reference point** — published operating point reproduced within tolerance | Wrong constants, wrong units |
| **Figure of merit** — L/D, τ, efficiency, steady-state gain in a plausible band | Individually-plausible-but-jointly-wrong parameters |
| **Cross-parameter consistency** — two params that encode the same physical object must agree | The `k` vs `cl_alpha` class of defect |
| **Scaling** — double a capacitance ⇒ halve a rate | Structural/dimensional errors |
| **Monotonicity** — more heat in ⇒ hotter; more drag past M_crit | Sign errors |
| **Integrator convergence** — solution invariant to `delta_t`/substeps | "The physics is actually truncation error" |
| **Dimensional sanity** — quantities land in physically possible ranges | Unit slips (kg vs t, °C vs K, s vs h) |

Two of these deserve emphasis because they are cheap and catch the most:
**equilibrium** and **cross-parameter consistency**.

---

## 5. Handling deviations

When the model disagrees with reality and fixing it is out of scope or too
disruptive:

1. Quantify it — the number, not "approximately right".
2. Document it in `PHYSICS.md` §5 with the physical consequence.
3. Write the test that *should* pass, marked `@pytest.mark.xfail(strict=True)`.

`strict=True` matters: if someone fixes the physics, the xfail becomes an
XPASS **failure**, forcing the doc and marker to be updated. The deviation can
never be silently resolved or silently worsened.

---

## 6. Scope discipline: what "simple yet faithful" means

Fidelity is per-regime, not global. A model is faithful if it is accurate **in
the regime the task occupies**, and its inaccuracy elsewhere is documented.

The plane is a clean example: its low-speed behaviour is wrong (stall speed
~228 kt vs a real ~150 kt), but the altitude-hold task lives at cruise, where
the model validates well. That is an acceptable simplification *because it is
written down* — and it immediately tells you the model must not be reused for an
approach-and-landing task without fixing D1 first.

So for each environment, state the **regime of validity**, and check that the
task's operating envelope sits inside it. An undocumented simplification is a
bug; a documented one is a design decision.

## Test the regime an optimiser will find, not the one you designed for

The aircraft carried a defect for a long time that every physics test agreed
with. `CD = cd0 + k·CL²` models drag as a consequence of lift, which is true in
attached flow and false once the wing separates; past the stall the model
collapsed lift and so collapsed drag with it, leaving a stalled wing with *less*
drag than in cruise. A departed aircraft then had almost no aerodynamic forces
at all.

Thirty-eight physics tests did not catch it, and the reasons generalise.

**Testing a derived quantity where it is derived tests nothing.** All seven drag
tests probed attached flow, where drag is *defined* from lift. Inside that
regime the formula is self-consistent, so no amount of testing there can reveal
that it has no separated-flow term.

**Weak assertions pass degenerate answers.** Two tests did reach past the stall.
One asserted lift "collapses" — and a collapse to exactly zero is a maximal
collapse, so the defect satisfied it emphatically. The other swept angle of
attack to 30°, straight through the broken region, and asserted only `isfinite`
and `cd > 0`: a wing producing 0.02 of drag is finite and positive. It went to
the right place and asked the wrong question.

**An optimiser goes looking for the region nobody modelled.** This is what makes
a benchmark different from ordinary simulation. An engineer drives a model
around its design point; an MPC or an RL agent searches, finds the unmodelled
corner where the physics is cheap, and exploits it. The MPC found this one by
stalling the aircraft and discovering that departure cost nothing.

So contracts should be stated over the **reachable** state space, not the
intended one. `test_angles_stay_bounded_under_extreme_actions` in the
conformance suite is the general form: drive each plant to its action limits and
require that its state stay physical or its episode end. It found the aerodynamic
gap, and it is still finding one — the aircraft has no pitch-rate damping, so a
departed airframe tumbles indefinitely even now that the forces are right.

### Anchor the check outside the model

The common thread in every defect found this way is that the quantity was
tested where it is *derived*. Drag was tested only in attached flow, where
`CD = cd0 + k·CL²` defines it from lift and is self-consistent whatever the
values. An energy audit has the same weakness in a subtler form: checking
`dE/dt = T·V − D·V` closes by construction, because `D` is the model's own
drag. It tests the integrator, not the physics.

What discriminates is an anchor the model does not get to move:

- **A published performance number.** The terminal velocity of a falling
  airframe, `√(2mg/ρSC_D)`, was 693 m/s under the defect and is 86 m/s now.
  The check needs no hypothesis about what is missing — only that 693 is absurd.
- **An exact analytic result.** A flat plate's lift-to-drag ratio at 45° is
  exactly 1, since `CL = C_N cos α` and `CD = C_N sin α` are equal there
  whatever `C_N` is. It cannot be moved by any coefficient the model chose.
- **A quantity spanning several mechanisms.** L/D sees defects in lift or drag;
  each coefficient looked defensible alone, and the ratio did not.

A useful test for a proposed contract: *could the model be wrong and this still
pass?* If the assertion is computed from the same expression as the behaviour,
the answer is yes.

### Energy: bound it, and watch the seams

Two forms of energy check are worth having, and they catch different things.

**Bounds on both sides, not a budget.** Energy may only enter through the
actuator, so the total cannot rise faster than the actuator can supply it — for
the aircraft, `ΔE ≤ T·V·Δt` at full thrust. And energy removed has to be
accounted for too: the only sink is drag, so it cannot fall faster than the
largest drag the geometry admits, `ΔE ≥ −½ρSC_D,max·V³·Δt`.

Neither side references the model's instantaneous forces, which is what makes
them independent. A sign error, a bad integration step, a regime switch that
quietly injects energy, or a clip silently discarding it all break one bound or
the other, and none of them have to be anticipated. Contrast the weak form,
`dE/dt = T·V − D·V`, which closes by construction because both sides use the
model's own drag — it tests the integrator, not the physics.

**Continuity at the seams.** A model assembled from regimes — attached and
separated flow, laminar and turbulent, charging and discharging — has a
boundary between them, and that boundary is exactly where an optimiser will
sit, because it is where the plant is being pushed. If the descriptions do not
join, the dissipated power steps at the seam and energy appears or vanishes for
no modelled reason. Measured as the largest step in dissipated power relative
to the typical step across a fine sweep: the aircraft's blend gives 4.7×, a
naive piecewise switch gives 32×, and drag collapsing to `cd0` past the stall
gives infinity.

The second is the more general of the two. Any environment built from more than
one regime has seams, and they are cheap to sweep.
