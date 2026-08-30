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
