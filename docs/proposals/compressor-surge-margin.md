# compressor_surge: a margin the plant earns

Status: the owner chose option 1, the lag and the hidden line shift, its four values,
and a shift local to the line (2026-11-23). It is built in the plant.

## The finding

After the reset fix (#223), what is left of the task's room is the reference's 10 %
control margin. The plant as it is does not ask for it:

- placed on the surge line itself, the reference never surges on normal and slow
  days, and costs 8.59 kJ per step on slow days against a balance floor of 8.47;
- only fast days need a margin: under 7 % the reference surges there (2 of 10 seeds
  at 6 %). Fast days weigh 0.05.

So most of the room is a fixed offset an agent can find, on a line it reads exactly.

## What the sources say the margin is for

`compressor_surge/REALITY.md`, "Why the margin is kept" (third search, 2026-11-23):

- the line's position is not known exactly and moves: field and shop curves over
  10 % apart [C1], a drift with the machine's condition that cannot be measured
  [C10], 2.6 to 7 % left by invariant coordinates at low suction pressure [C14];
- the transmitters are not instant: flow dP 100 to 150 ms, at most 200 ms; pressure
  300 to 400 ms [C10];
- the margin covers both, with the disturbance's size and speed [C10, p. 11].

The plant has neither: its readings have noise and no lag, and its surge line is the
map's. Version 2 had a hidden line offset, removed in version 4 for lack of a source.

## Prototype

`scripts/compressor_surge_margin_causes.py`. The ladder's override rung, commissioned
on the nominal plant, on seeds 100 to 109 of each scenario. `J` in kJ per step, its
floor on the variant's own surge line, and the fast-day seeds with a surge, of 10. No
run tripped.

| variant | margin | J | J floor | fast-day seeds with a surge |
|---|---|---|---|---|
| plant as it is | 0.10 / 0.08 / 0.07 / 0.06 / 0.05 | 3.04 / 2.76 / 2.64 / 2.54 / 2.47 | 1.74 | 0 / 0 / 0 / 2 / 4 |
| readings lag 0.2 s | 0.10 / 0.09 / 0.08 / 0.07 / 0.05 | 3.07 / 2.93 / 2.81 / 2.70 / 2.54 | 1.74 | 0 / 0 / 0 / 1 / 4 |
| true line 5 % right of the map | 0.10 / 0.12 / 0.15 | 2.98 / 3.28 / 3.84 | 2.10 | 1 / 0 / 0 |
| line 5 % right, lag 0.2 s | 0.10 / 0.12 / 0.15 | 3.01 / 3.31 / 3.86 | 2.10 | 1 / 0 / 0 |
| true line 5 % left of the map | 0.10 | 3.27 | 1.48 | 0 |
| line 5 % left, lag 0.2 s | 0.15 | 4.95 | 1.48 | 0 |
| inlet 268 K, map at 288 K | 0.10 | 2.77 | 1.85 | 1 |
| inlet 308 K | 0.10 | 4.16 | 1.67 | 0 |

- The line shift scales the cubic's `W` by 1.05 or 0.95 while the controller keeps the
  nominal map.
- Inlet temperature moves the true margin by about 3.7 points for 20 K at a constant
  header pressure. Practice cancels it with reduced coordinates [C10, C11], and the
  task has no suction temperature reading, so on its own it is a weak cause.

## The set-point rule on this task

Owner's rule of 2026-11-23: on a fixed grid, the lowest set point at which the
reference leaves its limit on none of its seeds. Here the priced limit is the surge
line, so "no exit" reads as no counted surge. On a 1 % grid it keeps 0.07 on the plant
as it is, 0.08 with the lag, and 0.12 with the true line 5 % right of the map (0.11
was not run). With the sourced causes in the plant the rule lands in practice's 10 to
15 % [C1], and the margin is then the plant's own.

Two cautions. Only the fast day decides the placement, one seed apart between grid
points. And on the header pressure itself the reference leaves its zone on about half
the fast-day seeds even at the zone's centre, so there the rule has no grid point
without an exit.

The table was measured on the plant before the change, on the branch of the ceiling
fixes. The script now starts from that plant (`BARE`); on a later head of the
integration branch its rows come out within 2 % of these, with the same surges.

## The decision

1. **Lag and a hidden line shift (recommended).** A first-order lag on the flow and
   pressure readings, and a per-episode shift of the true surge line against the map,
   drawn either way. The lag's value is sourced [C10]. The shift's size is ours inside
   [C14]'s 2.6 to 7 %. The property becomes a limit whose position is hidden: an agent
   shaves the margin only by reading how the plant behaves near its line.
2. **Lag only.** Every number sourced, a small effect (0.07 to 0.08).
3. **Neither.** The rule places the margin at 0.07 and the room on J is about 1.5.

Not decided here: the shift's law and range, whether a restart redraws it, and
whether the observation gains a suction temperature channel [C4, C5].

## As built, and what it gives (2026-11-23)

Decided by the owner: the shift is uniform within 5 % either side of the map; a
restart keeps the machine; no suction temperature channel; the flow reading lags
0.15 s and the pressure reading 0.3 s [C10]; and the shift is local to the line.

**Local to the line.** A first build stretched the whole map, and the speed that held
24 kPa at a given flow then gave the shift away (0.7 % for 5 %, seventy steps of the
speed reading), which a plant cannot do [C10]. Now the machine differs from its map
only inside the control margin: from the map's control line on (10 % right of the
map's surge line, the line the plant is established on and holds [C6; C1]) its head
is the map's exactly; at and left of its true line it is the map moved; between the
two the move fades smoothly (the fade's shape is ours). Two machines with opposite
shifts then read alike to within their instruments' resolution in normal operation,
and part as the line nears: on a demand fall that carries the map's own machine
onto its line, the one whose line is 5 % right surges and the one whose line is 5 %
left does not (`tests/compressor_surge/test_compressor_surge_hidden_line.py`).

**The reference's margin by the owner's rule**, on seeds 100 to 109: no surge on any
seed from 0.13 down to 0.09, two fast-day seeds at 0.08. So 0.09.

**Room** on held-out seeds 200 to 209, the reference at 0.09 against the ceiling on
the true line, paired by seed, kJ per step, with a 95 % interval from a bootstrap
over seeds within each scenario:

| scenario | reference | ceiling | balance floor | room |
|---|---|---|---|---|
| normal | 0.722 | 0.116 | 0.103 | 6.22 [3.67, 15.99] |
| slow_approach | 11.10 | 8.75 | 8.64 | 1.27 [1.23, 1.32] |
| fast_approach | 12.69 | 9.45 | 9.34 | 1.34 [1.27, 1.43] |
| load_swings | 5.24 | 3.51 | 3.40 | 1.49 [1.36, 1.73] |
| J | 3.04 | 1.95 | 1.91 | 1.55 [1.46, 1.66] |

The ceiling costs less than the reference on 40 seeds of 40, trips on none, and sits
2 % above the balance floor on J. 34 of its 40 solves end on the 1200-iteration cap.
The fast and load_swings solves ran on JAX 0.6.2, the other two on 0.11.2.

**What the rule's margin is worth on machines it was not placed on.** The reference
on 200 other machines per scenario (seeds 1000 to 1199):

| margin | fast days with a surge, of 200 | other days with a surge, of 600 | trips | J |
|---|---|---|---|---|
| 0.09 (the rule's, from 10 seeds) | 11 | 0 | 0 | 3.04 |
| 0.10 | 7 | 0 | 0 | 3.19 |
| 0.12 | 0 | 0 | 0 | 3.53 |

Every surging machine has its line right of its map: none of 97 with it left, 2 of 46
within 2.5 % right, 9 of 57 beyond. A margin placed on ten machines surges on about
one fast day in twenty elsewhere, never trips, and the margin that surges on none of
200 is 0.12, inside practice's 10 to 15 % [C1], at 16 % more recycle energy.

