# Reactor: the task against real plants

Status: report for the owner, 2026-11-08. No code changed. Each source below was opened
and the quoted lines checked a second time, except where it says otherwise.

## Verdict

The plant's physics holds up: the measurement, the boron path and its delay, the ramp
rates and the shape of the day all match. What does not match is the job. Five
mismatches sit at the origin of the design. The first four explain the problems found
so far: the published rod program trips every seed, the Mode A pairing "cannot work",
and the cost piles up in the half-power hold.

1. **The operator knows the day's schedule. The agent does not.** OAPS (section 1):
   operators "receive grid instructions several hours in advance" and plan with
   simulation tools. Boron arrives 5 to 15 min late, and after a power reduction "the
   operator must initiate a dilution" for the xenon (section 2.2). Both only work with
   foresight. The task draws the ramp times at random and observes only the present
   load (`obs = [n, T_hot, rod, load, T_avg, T_ref, delta_I, boron]`).
   *Symptom:* the published rod program with boron trim trips on all ten seeds, always
   on the way back up. On seed 100 at 9.8 h the rods are fully out, boron sits at 1785
   ppm, never diluted ahead, and Tavg falls 18 K under its program into the 556 K floor.
   Seed 104 does the same at 5.3 h.
   *Change:* put the schedule in the observation: the next ramp's time and level, or a
   window of the planned load. The foresight oracle then cheats only on the noise.

2. **The 556 K floor is not a trip.** In the code it is "the minimum temperature for
   criticality", labelled `ours` and "remembered". In Westinghouse technical
   specifications that is a limiting condition with a time to act. It is restored within
   minutes or the plant goes to hot standby. No reactor trip fires. Plants differ: 515 to
   568 F.
   *Source:* search results for plant specifications hosted by the NRC (Salem, Calvert
   Cliffs, Byron). The NRC site blocks automated readers, so these were not opened, and
   the values are unconfirmed.
   *Change:* make it a priced limit with an allowed time, not a trip, or model the trip
   a low Tavg really leads to, if any.

3. **Where the delta I band applies.** Here the band is +-5 % of delta I around 0,
   charged at every power, including the 50 % hold, where most of the reference's cost
   was. Two plant families say otherwise, and the task mixes them:
   - **Westinghouse CAOC** (Krsko is a Westinghouse 2-loop):
     - The Korean paper quotes the 2-loop specification. The band is "applied MODE 1
       above 50 percent of rated thermal power". Under RAOC "there is usually no limit
       below 50% RTP", and "there is no delta I limit below at 50% power" (KNS 2018,
       sections 6 and 7.2).
     - The band is +-5 % at low burnup and +3/-12 % later. It is set around a target
       measured at full power, not around 0.
     - Westinghouse's own load-follow patent (US 4,057,463) also says this. Above 90 %
       power the band is about +-5 %. Deviations under 1 hour "should not have any
       adverse effect". The allowed offset deviation "is inversely proportional to the
       power level".
   - **EDF**: the IAEA report (Annex II-4.3) gives the French rule. During load follow,
     "the axial offset should be kept within +-0.05 of the reference axial offset". That
     is a band on the offset, so in delta I it narrows with power: +-2.5 % at half power.

   The task takes Krsko's band (Westinghouse) but EDF's operation (OAPS Mode A), and
   applies the band at every power around 0.
   *Change:* pick one family. The Westinghouse rule makes the 50 % hold free of the band
   and moves the difficulty to the return to power. The EDF rule tightens the hold.
   Either way, centre the band on the target.

4. **In real Mode A, the rods hold Tavg and the operator steers the offset with boron.**
   - OAPS (section 2.2): the black rods are "assigned to ACT regulation", the average
     coolant temperature. The operator "must manually adjust the boron concentration" to
     keep the rods from moving too far.
   - IAEA (Annex II-5.1): the black rods move only "if |Δθ| > 1.6°C".
   - US 4,057,463: rods run on "an average coolant temperature control system", and
     "Flux difference limits should be maintained by boration".

   Boron reaches delta I through the rods' Tavg loop: boron shifts Tavg, and the
   automatic rods move to restore it. The ladder's finding that boron's relative gain on
   delta I is 0, so the Mode A pairing "cannot work", read boron's direct gain with the
   rods held. That is not how the plant is wired. The reference then paired rods with
   delta I, which no plant does.
   *Change:*
   - Rebuild the Mode A rungs the way plants run them: the published rod program on
     Tavg with its deadband, and boron on delta I or rod position, fed forward from the
     schedule (needs item 1).
   - Charge Tavg only past a deadband, or as a limit. Today it is charged from 0.05 K,
     which punishes the 1.6 C the plant's own rods ignore.
   - The cost is also the same on both sides of the program. Its reason is the Carnot
     slope of the steam conditions, but above the program the steam is hotter, which
     helps.

5. **No rod insertion limits.** Plants hold a power-dependent rod insertion limit, for
   shutdown margin (KNS 2018, section 7.2 and its figure 2). The task lets the bank go
   anywhere. *Change:* add the limit as a priced constraint. That needs a sourced curve,
   which was not opened here.

## What matches

| Point | Task | Plant | Source |
|---|---|---|---|
| Delta I | n times the offset, top minus bottom over rated | the same, from a two-section ex-core detector | nuclear-power.com, axial flux difference |
| Boron delay | 5 to 15 min, drawn | about 5 to 15 min | OAPS 2.2 |
| Dilution, boration | 10 and 3 kg/s | typically 10 and 3 kg/s | OAPS 2.2 |
| Ramp | 2 %/min | Mode A 1 to 2 %/min at the start of a cycle; 2 %/min generally | OAPS 2.2; IAEA 2.2.4, II-5.2 |
| Day | one 100-50-100, holds 3.6 to 8.3 h | at most two deep moves a day, plateaus of at least 2 h | IAEA II-4.3 |
| Xenon lag | about 6 to 7 h to peak | "6 or 7 hour delay period" | US 4,057,463 |
| Boron and offset | boron barely moves the offset | "little effect on power distribution, but their action is time delayed" | IAEA II-3 |

Outside the task's scope: dilution weakens through the cycle, to 0.1 to 0.3 %/min at 80 %
of the cycle (OAPS 2.2). The task models the start of a cycle only, which is fine if
stated.

## What it would change

With 1 to 4 fixed, the hard part of the day moves from the 50 % hold to the return to
power. There the rods run out, dilution is slow and late, xenon is still burning out, and
the band applies again above 50 %. That is the part OAPS and the Westinghouse patents
describe as the real limit. Room, the ladder and the oracle would all need redoing. The
plant model itself, from Onega and Kisner, stays.

## Sources

Opened, and each cited line checked twice:

- IAEA Nuclear Energy Series NP-T-3.23 (2018), *Non-baseload operation in nuclear power
  plants*: <https://www-pub.iaea.org/MTCD/Publications/PDF/P1756_web.pdf>. Section 2.2.4
  and Annex II.
- Barbier et al., OAPS, ICAPP 2025: <https://arxiv.org/pdf/2605.20169>. Sections 1 and 2.
- Moon et al., KNS Autumn 2018, 18A-534, delta I control of a Westinghouse 2-loop plant:
  <https://www.kns.org/files/pre_paper/40/18A-534문상래.pdf>.
- US 4,057,463 (Westinghouse), constant axial power distribution in load follow:
  <https://patents.google.com/patent/US4057463A/en>.
- US 4,222,822 (Westinghouse), load follow with constant axial distribution:
  <https://patents.google.com/patent/US4222822A/en>. It confirms the ex-core band
  practice of the time and the 5 %/min return from 50 %.
- nuclear-power.com, axial flux difference: <https://www.nuclear-power.com/?p=18413>.
  Secondary, used only for the definition.

Not opened, because the NRC site and EDF's HAL block automated readers, including the
in-app browser, and that block was not bypassed: the NRC Westinghouse Technology
course chapters (rod control, CVCS, nuclear instruments), the plant technical
specifications on the 541 F limit, and EDF's *Nuclear power plant flexibility at EDF*.
The rod control and CVCS diagrams the owner asked for are in those chapters. They need
a person to open them, and they would confirm items 2 and 5.
