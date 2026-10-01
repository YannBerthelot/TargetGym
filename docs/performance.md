# Throughput

Steps per second for a vmapped, jitted, scanned rollout: the way an RL loop
actually drives these environments, not one Python-level step at a time.
Reproduce with:

```
uv run python -m target_gym.benchmark_speed --batch 4096 --steps 500
```

Measured on arm64 CPU, Darwin, single process, no GPU.

**These are not properties of the plants.** They move with the machine and with
the batch size, which is why they are measured here rather than claimed in the
seventeen `PHYSICS.md` contracts, where every other number is a fact about the
process being modelled. Nothing in CI checks them, because a check that fails on
somebody else's laptop is worse than no check.

| environment | batch 4096 | batch 256 |
| --- | --- | --- |
| `first_order` | 694.72 M | 773.12 M |
| `cstr` | 201.55 M | 176.35 M |
| `four_tank` | 111.44 M | 108.26 M |
| `unstable_cstr` | 14.44 M | 12.42 M |
| `hvac` | 12.75 M | 12.39 M |
| `boiler_drum` | 11.89 M | 10.85 M |
| `battery` | 11.31 M | 9.23 M |
| `wind_turbine` | 11.23 M | 12.44 M |
| `glass_furnace` | 4.10 M | 3.34 M |
| `ph_neutralization` | 3.97 M | 3.77 M |
| `compressor_surge` | 2.53 M | 1.57 M |
| `plane` | 2.13 M | 1.39 M |
| `plane_energy` | 2.13 M | 1.39 M |
| `plane_sine` | 2.13 M | 1.41 M |
| `plane3d_heading` | 1.55 M | 1.00 M |
| `plane3d_figure8` | 1.50 M | 0.99 M |
| `cement_kiln` | 1.19 M | 0.61 M |
| `plane3d_circle` | 1.18 M | 0.98 M |
| `plane3d_racetrack` | 1.18 M | 0.97 M |
| `reactor` | 1.17 M | 1.30 M |
| `distillation` | 0.82 M | 0.38 M |
| `patrol_bearing_only` | 0.64 M | 0.54 M |
| `patrol` | 0.62 M | 0.53 M |

The `unstable_cstr` row was measured later than the others (2026-09-30), with
the command above and with `--batch 256` at the default 1000 steps, on an
arm64 Darwin machine under a shared load. `cstr`, measured in the same runs,
came out at 199.91 M and 169.22 M, within about 4 % of its row, so the new row
compares with the rest to about that accuracy.

The `compressor_surge` row was measured on 2026-10-01 with the same two
commands, on an arm64 Darwin machine at a load average of about 2.5, with
`cstr` in the same runs. `cstr` came out at 210.89 M and 174.35 M, within
about 5 % of its row. A second pair of runs gave 2.55 M and 1.48 M for
`compressor_surge`, so the two pairs differ by up to 6 %.

## What this means for RL

The protocol's largest sample budget is 1e7 environment steps. At the slowest
environment here that is about 16 seconds of environment time; at the fastest
it is under a millisecond. Network forward and backward passes dominate any
learning loop by two to three orders of magnitude, so **the environment is never
the bottleneck at these budgets**, and none of the twenty-three is unfit on
speed.

The spread is arithmetic per step, not anything fixable. The distillation column
integrates 41 states through 16 RK4 substeps, and its contract establishes that
16 is a stability requirement rather than a refinement. Patrol integrates two
complete 3D aircraft. The cement kiln carries 16 axial zones of 4 states plus a
sequential gas sweep. The three-order-of-magnitude leaders are one to four states
with a single substep.

**Which environment is slowest depends on the batch size**: patrol at 4096, the
distillation column at 256. Two contracts used to claim the title outright, and
both were wrong at one of the two batch sizes.
