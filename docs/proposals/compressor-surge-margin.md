# compressor_surge: a margin the plant earns

Status: prototype, nothing registered. Open for the owner (2026-11-23).

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
