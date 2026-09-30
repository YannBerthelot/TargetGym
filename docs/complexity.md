# Complexity ladder

There is a wide difficulty range in this suite, so it can be used as a
curriculum and not only as a benchmark. Tiers weigh both the **dynamics**
(linearity, coupling, stiffness) and the **RL side** (dimensionality,
horizon, partial observability).

| Tier | Environment | Obs | Act | Dynamics | Key RL challenges |
|---|---|---|---|---|---|
| 1 (Trivial) | First Order System | 2 | 1 | Linear SISO | Baseline sanity-check |
| 2 (Medium) | CSTR | 3 | 1 | Nonlinear SISO | Exponential Arrhenius kinetics, stiff dynamics, exothermic runaway risk |
| 3 (Hard) | Building HVAC | 7 | 1 | Linear RC network | **Partial observability** (thermal mass hidden), 43 h time constant, setback anticipation, comfort/energy trade-off |
| 3 (Hard) | Four Tank | 6 | 2 | Nonlinear MIMO | **Non-minimum phase** (gamma1+gamma2 = 0.4): the RGA element is *negative*, so the obvious diagonal pairing is unstable and the loops must be crossed |
| 3 (Hard) | Grid Battery | 5 | 1 | Nonlinear ECM | **Finite budget**: tracking now costs the ability to track later; irrecoverable charge limits, state-dependent efficiency |
| 4 (Very Hard) | Wind Turbine | 6 | 2 | Nonlinear aero-elastic | Turbulent unmeasured inflow, region switching, drive-train torsion, thrust/power trade-off |
| 4 (Very Hard) | pH Neutralisation | 3 | 1 | Implicit algebraic | 45x steady-state gain variation across the range, unmeasured buffering, same pH from different states |
| 4 (Very Hard) | Binary Distillation | 6 | 2 | Stiff nonlinear MIMO | **Ill-conditioned** (condition number ~140): the two purities move together far more easily than apart |
| 4 (Very Hard) | Plane 2D | 10 | 2 | 2D aerodynamics | Coupled nonlinear aerodynamics, very long horizon |
| 4 (Very Hard) | Glass Furnace | 5 | 1 | Nonlinear radiation (T^4) | **Partial observability** (6/9 states hidden), regenerator reversal cycle, multi-hour transients, batch-blanket nonlinearity |
| 4 (Very Hard) | Nuclear Reactor | 4 | 1 | Stiff multi-timescale | **Partial observability** (7/11 states hidden), xenon memory trap, 86k-step horizon |
| 5 (Extreme) | Boiler Drum | 7 | 2 | Nonlinear, two-phase | **Non-minimum phase**: the level's first move is the wrong way. Integrating output, irrecoverable trips both sides, hidden riser voidage |
| 5 (Extreme) | Plane 3D Heading | 15 | 3 | 3D aerodynamics | Multi-objective (altitude + heading), roll/pitch/yaw coupling |
| 5 (Extreme) | Plane 3D Circle | 17 | 3 | 3D + path following | Sustained coordinated banked turns, km-scale circular path |
| 5 (Extreme) | Plane Patrol | 26 | 3 | 3D + moving target | **Non-stationary manoeuvring reference**, relative-frame observation, collision (irrecoverable) |
| 6 (Extreme+) | Cement Kiln | 8 | 2 | Distributed (1D advection + Arrhenius) | **Transport delay**: half the response to a fuel change takes a full 25-min residence time. 64 hidden states behind 8 measurements, one input that moves the delay itself |
| 6 (Extreme+) | Plane 3D Figure Eight | 19 | 3 | 3D + twisted lemniscate | 3D path with altitude crossovers, direction reversal |
| 6 (Extreme+) | Plane Patrol MARL | 18 / 26 | 3 + 3 | 3D two-body | **Multi-agent coordination**, non-stationary co-player, shared collision state |

The target-pattern variants (`plane_sine`, `plane_energy`) and
the 3D holding pattern (`plane3d_racetrack`) share their base aircraft's plant
and therefore its tier; what differs is the reference they must track.
