# Plant reality record: four_tank

The real plant is Johansson's quadruple-tank laboratory process (Lund, 1996)
and the copies of it built at other universities. Every statement below comes
from a cited source. "Not found" means no source we could read says it.
This record was written without reading the task's code or docs.

## Sources

Each source was fetched twice: once to draft, once to check. Figures were
read as page images.

- [S1] K.H. Johansson, J.L.R. Nunes, "A multivariable laboratory process with
  an adjustable zero", Proc. American Control Conference, Philadelphia, 1998,
  pp. 2045-2049. https://people.kth.se/~kallej/papers/quadtank_acc98.pdf
  Read: tables on p. 2046 and 2047, Figures 2 to 6. Checked twice.
- [S2] K.H. Johansson, "Relay feedback and multivariable control", PhD thesis
  TFRT-1048, Lund Institute of Technology, 1997. Paper 4 is the long form of
  [S1]. https://lup.lub.lu.se/search/files/4558587/8840256.pdf
  Read: Paper 4 text, Tables 4.1 and 4.2 (thesis p. 122 to 124), Figure 11
  (p. 134). Checked twice.
- [S3] K.H. Johansson, A. Horch, O. Wijk, A. Hansson, "Teaching multivariable
  control using the quadruple-tank process", IEEE CDC, Phoenix, 1999.
  https://people.kth.se/~kallej/papers/quadtank_cdc99.pdf
  Read: Figures 3 and 4 (p. 5). Checked twice.
- [S4] A. Hansson, O. Markusson, M. Akerblad, revised by J. Wijk, "Laboratory
  experiment: the four-tank process", KTH course EL2745 lab manual, v.2014.
  https://www.kth.se/social/upload/532f2cf5f276541ce4968f4a/laboratory.pdf
  Read: pp. 4 to 17, including the interface Figure 2 (p. 15). Checked twice.
- [S5] E.P. Gatzke, E.S. Meadows, C. Wang, F.J. Doyle III, "Model based
  control of a four-tank system", Computers and Chemical Engineering 24
  (2000) 1503-1509. Author copy:
  https://cse.sc.edu/~gatzke/publications/00pse3.pdf
  Read: Table 3 (p. 2), Figure 2 (p. 3), Figure 4 (p. 4), Figures 5 and 6
  (p. 5). Checked twice.
- [S6] T. Roinila, M. Vilkko, A. Jaatinen, "Corrected mathematical model of
  quadruple tank process", IFAC World Congress, Seoul, 2008.
  https://skoge.folk.ntnu.no/prost/proceedings/ifac2008/data/papers/2172.pdf
  Read: abstract and introduction (p. 11678). Checked twice.
- [S7] L. Pedroso, P. Batista, quadruple-tank setup repository README (parts
  list of a copied rig, 2021).
  https://github.com/decenter2021/quadruple-tank-setup Checked twice.

## The plant

Four interconnected water tanks and two pumps. The inputs are the two pump
voltages; the outputs are the levels of the two lower tanks [S1, p. 2045].
Each pump feeds one lower tank and the upper tank on the other side; the
split is set by a valve before an experiment [S1, p. 2046; S3, p. 2]. The
process is two standard double-tank processes connected together [S1,
p. 2045; S4, p. 12]. Lund rig sizes: tank cross-sections 28 cm2 (tanks 1, 3)
and 32 cm2 (tanks 2, 4); outlet holes 0.071 cm2 and 0.057 cm2 [S1, table
p. 2046]. KTH rig: all tanks 15.52 cm2, and 100 % on the level plot is
25 cm [S4, p. 5 and p. 15]. Delaware copy: four five-gallon tanks of
730 cm2 cross-section, outlet 2.3 cm2 [S5, Table 3 and p. 2]. Tank height of
the Lund rig: not found.

## What operators read

- Lower tank levels y1, y2 as voltages from level measurement devices,
  y = kc h [S1, p. 2046]. Lund kc = 0.50 V/cm [S1, table p. 2046]. KTH
  kc = 0.2 V/cm [S4, p. 5].
- Upper tank levels: the KTH lab reads all four levels by eye from a cm scale
  on the tanks [S4, Exercise 2.2.1, p. 6]. The Lund interface shows y1 to y4
  [S2, Figure 11, p. 134]. Only the two lower sensors are wired to the I/O
  card in the KTH four-tank setup [S4, p. 12].
- Sensor type on the Lund and KTH rigs: not found. A 2021 low-cost copy uses
  a "Continuous Fluid Level Sensor PN-12110215TC-12" per tank, VMA421 water
  pumps and an Arduino Nano [S7, parts list]. The KTH manual warns of "offset jumps" in the
  level signals and says the scale and screen values differ because the
  sensors are not calibrated exactly [S4, footnotes 3 and 4, p. 6].
- Sensor delay or time constant: not found.
- Delaware copy: levels in cm and pump inputs in %, on an ABB-Bailey
  Freelance DCS with an operator screen [S5, p. 2 and Figure 6, p. 5].

## What they move, and in which mode

- Two pump voltages, flow k v proportional to voltage [S1, p. 2046]. Lund
  pump gains about 3.1 to 3.4 cm3/(V s), varying slightly with operating
  point [S1, table p. 2046; S2, p. 123]. KTH: 100 % is 15 V [S4, p. 15].
  Delaware: pump speed in %, nominal 60 % [S5, Table 3, p. 2].
- Valve split gamma1, gamma2: set by hand before the run, not moved during
  it [S3, p. 2]. At KTH the setting is made by plugging tubes and screwing
  in small or medium outlet plugs on the upper tanks [S4, A.2, pp. 13-14].
- Modes: manual and automatic, switched in the interface [S4, p. 15].
  Bumpless manual to auto transfer is described as crucial [S3, p. 6].
  Controllers run on the rigs: decentralized PI [S1, Section 4], statically
  decoupled PI and Glover-McFarlane [S3, p. 4-5; S4, Section 4], IMC and
  DMCplus MPC [S5, pp. 4-5]. Cascade, ratio, feedforward: not found.

## Limits

- Level floor: h >= 0 [S2, p. 121]. Level ceiling: the tank is full at
  25 cm on the KTH rig [S4, p. 15]; tank level 1 means a full tank in [S3,
  Figure 4]. Overflow trip or interlock: not found.
- Pump: 0 to 100 % of maximum voltage, 15 V maximum at KTH [S4, p. 15].
  Students are told to keep steps and load disturbances small so the
  control signals do not saturate [S4, p. 9]. This is a soft rule, not a
  trip.
- Operating window used for identification: Delaware PRBS kept levels within
  +8 cm and -10 cm of the nominal point [S5, p. 3].
- The steady state exists for any target levels only if gamma1 + gamma2 is
  not 1 [S2, p. 123].
- Cost: Delaware DMC put a cost on pump speed with a minimum level
  constraint [S5, p. 6]. Otherwise not found.

## Time scales

- Linearized time constants from physical data: minimum phase P- (T1,T2) =
  (62, 90) s, (T3,T4) = (23, 30) s; non-minimum phase P+ (63, 91) s and
  (39, 56) s [S1, table p. 2047].
- Identified grey-box time constants: P- (96, 99, 32, 39) s; P+ (77, 112,
  53, 55) s [S1, p. 2048].
- Delaware: T1 to T4 = 53.8, 48.0, 38.5, 31.1 s [S5, Table 3].
- Zeros: P- at -0.060 and -0.018; P+ at -0.057 and +0.013 [S1, table
  p. 2047]. Delaware: +0.0324 and -0.088 [S5, p. 4].
- Settling time with PI is about ten times longer in the non-minimum phase
  setting [S1, p. 2048]. Designed crossover 0.1 rad/s (minimum phase) and
  0.006 rad/s (non-minimum phase) [S3, p. 4].
- Pump or valve stroke time: not found. Sampling period: not found for Lund
  and KTH. Delaware PRBS switched at most every 35 s [S5, p. 3].

## A normal day and a bad day

Normal day on the rig: set the valves, start pumps at 50 % (7.5 V at KTH),
wait for steady state, read levels [S4, pp. 5-6]. Then step tests from one
input at a time, manual tracking of a 15 cm (60 %) level, PRBS
identification, and setpoint steps of about 5 percentage points under
automatic control [S4, pp. 6 and 9; S1, Section 3; S5, p. 3].

Disturbances used on the rigs:
- Pour a cup of water into a lower tank; open an extra outlet in an upper
  tank [S4, Exercise 4.1.1, p. 9]. In the non-minimum phase case the extra
  outlet can cause problems [S4, p. 10].
- Delaware: unmeasured flow out of tanks 3 and 4, made by submersible pumps
  that pump water from the upper tanks to the reservoir [S5, pp. 1-2].

Bad day:
- Air in the tubes even with pumps on [S4, footnote 2, p. 5].
- Tubes ending at different heights change k and gamma [S4, footnote 10,
  p. 13].
- Level signal offset jumps; sensor scale not calibrated [S4, p. 6].
- Unequal pumps: the widely copied linear model is wrong when the two pumps
  are not identical [S6, abstract].
- Non-minimum phase setting: students stopped runs thinking their design was
  broken, because responses are so slow [S3, p. 6]. Manual control may not
  succeed within 10 minutes [S4, p. 6].

## The operator's job in one sentence

Hold the levels of the two lower tanks at their setpoints with the two pump
voltages [S1, p. 2046], in a setting where each pump also fills the far upper
tank.

## Proposed behaviour tests source list

| Source and figure | Quantity | Value as printed |
|---|---|---|
| [S1] table p. 2046 | Steady state P- | h1,h2 = 12.4, 12.7 cm; h3,h4 = 1.8, 1.4 cm; v = 3.00, 3.00 V; gamma = 0.70, 0.60 |
| [S1] table p. 2046 | Steady state P+ | h1,h2 = 12.6, 13.0 cm; h3,h4 = 4.8, 4.9 cm; v = 3.15, 3.15 V; gamma = 0.43, 0.34 |
| [S1] eq. (2), (3) p. 2047 | Static gains G-(0) | 2.6, 1.5, 1.4, 2.8 (V/V) |
| [S1] eq. (3) p. 2047 | Static gains G+(0) | 1.5, 2.5, 2.5, 1.6 (V/V) |
| [S1] table p. 2047 | Time constants | see Time scales |
| [S1] Fig. 3, p. 2047 | PRBS validation, P- | outputs about -1.5 to +1 V deviation over 0 to 1500 s |
| [S1] Fig. 5, p. 2048 | PI step in r1, P- | y1 from about 6.2 V to about 7 V within about 50 s; y2 small bump near 6.2 to 6.4 V; u1 peaks near 6 V then settles near 3.5 to 4 V; time axis 0 to 300 s |
| [S1] Fig. 6, p. 2049 | PI step in r1, P+ | y1 from about 6.5 V to 7 V; y2 rises to about 8 V and returns over about 1000 s; time axis 0 to 3000 s |
| [S1] p. 2048 | PI tunings used | P-: (K,Ti) = (3.0, 30), (2.7, 40); P+: (1.5, 110), (-0.12, 220) |
| [S3] Fig. 4 upper, p. 5 | Decoupled PI, minimum phase | normalized y2 0.72 to 0.82, y1 0.45 to 0.35 steps; settle within about 100 s; axis 0 to 1500 s |
| [S3] Fig. 4 lower, p. 5 | Decoupled PI, non-minimum phase | normalized y1 0.36 to 0.46, y2 0.57 to 0.67; settle in several hundred s; axis 0 to 2200 s |
| [S5] Table 3, p. 2 | Delaware steady state | h = 14.1, 11.2, 7.2, 4.7 cm at 60 % pump; gamma = 0.333, 0.307 |
| [S5] Fig. 2, p. 3 | PRBS response | lower levels between about 0 and 30 cm; inputs 40 to 80 % |
| [S5] Fig. 4, p. 4 | IMC setpoint tracking | levels on an axis of about 15 to 19 cm; pumps between about 30 and 100 % |

Sign rules that a simulation must reproduce: minimum phase for 1 < gamma1 +
gamma2 <= 2, non-minimum phase for 0 < gamma1 + gamma2 <= 1 [S1, p. 2047;
S3, p. 3]; RGA lambda = gamma1 gamma2 / (gamma1 + gamma2 - 1) [S3, eq. (2);
S4, Exercise 2.1.6].

## For the owner to open by hand

- K.H. Johansson, "The quadruple-tank process", IEEE TCST 8(3), 2000,
  pp. 456-465, https://doi.org/10.1109/87.845876 (paywall; typeset.io copy
  blocked with 403). Check: tank heights, sensor type, sampling period, pump
  limits, and the step response figures.
- I. Alvarado et al., "A comparative analysis of distributed MPC techniques
  applied to the HD-MPC four-tank benchmark", J. Process Control 21 (2011).
  Report copy https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/11_025.pdf
  (connection refused). Check: Seville plant tank heights, level sensor type,
  flow ranges, sampling time, level limits.
- Seville educational plant paper, https://core.ac.uk/download/322845363.pdf
  (Cloudflare check). Check: the instrumentation section.
- L. Pedroso, P. Batista, J. Process Control 118 (2022) 82-94,
  https://doi.org/10.1016/j.jprocont.2022.08.010. Check: tank dimensions,
  sensor, sampling period, identified parameters.
