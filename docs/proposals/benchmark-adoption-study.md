# Benchmark adoption study

What made classic control, MuJoCo and Atari the standard RL benchmarks, the bar TargetGym would have to reach, and a goal for its first benchmark version.

*A research session wrote this report on 2026-11-27. It was read-only: nothing in the code changed and nothing was published.*

**Status, 2026-11-27.** The owner adopted a [vision](../vision.md) and a [task protocol](../task-protocol.md) that take up this study's recommendation. They differ from section 7 in one way: each task becomes v1 on its own, in a fresh namespace, as it passes the protocol, instead of a suite frozen at the 0.8.0 tag.

## 1. Summary

Classic control, MuJoCo locomotion and Atari became RL's defaults without being realistic; this study rates all three 1 to 2 on a five-point scale. Each arrived with a paper or a long pedigree, and MuJoCo and Atari came with reference agents or a landmark result, such as DQN's 49 Atari games. Gym, now Gymnasium, carried all three, and every RL library checked runs them. Their task sets stayed fixed, with changes under new version names. Runs were cheap, or a cheap protocol grew beside the costly one, as Atari 100k did beside 200M frames. Realistic suites such as Tennessee Eastman, the wastewater BSMs and BOPTEST became standards only inside their own fields, through a frozen plant and institutional care. None appears in a general RL library.

TargetGym already has rich baselines, and three scored plants with no open RL equivalent. It lacks a frozen task set (versions reached v10 in a year), a public score matching its charter, any published learned result, a paper and a cheap protocol. An arcade suite would enter a crowded niche and give up those plants. This study recommends the realistic course with two arcade habits: a cheap protocol tier on the same tasks, and one fixed number per task.

The goal that holds still is "Benchmark v1". It freezes the four scored tasks at the 0.8.0 tag with one scorer and frozen lines, then adds PPO and SAC results at 0.9.0. Any later change to a frozen item makes v2 beside v1.

## 2. Inventory

The tables cover every benchmark in the study's notes. Citation counts come from OpenAlex on 2026-11-27, which merges a preprint into its published record ([OpenAlex](https://api.openalex.org/works?filter=doi:10.48550/arxiv.1606.01540%7C10.48550/arxiv.1207.4708%7C10.1613/jair.3912%7C10.48550/arxiv.1312.5602%7C10.1038/nature14236%7C10.1109/IROS.2012.6386109&select=doi,title,publication_year,cited_by_count)). The notes' Semantic Scholar counts differ, sometimes widely: Gym 5,771 against 9,566 ([S2](https://api.semanticscholar.org/graph/v1/paper/arXiv:1606.01540?fields=title,citationCount,year)), Henderson et al. 2,760 against 628 ([S2](https://api.semanticscholar.org/graph/v1/paper/arXiv:1709.06560?fields=title,citationCount,year)). This report follows the gap-fill note's OpenAlex counts, so read comparisons as orders of magnitude. Stars and push dates come from GitHub on the same day.

Realism runs from 1 (arcade or toy) to 5 (faithful to a validated real plant), with 2 for stylised physics, 3 for a simplified literature model and 4 for an engineering model of a plant type. All realism figures and "why" columns are this study's judgement. This study found no count of RL benchmark use by year. The closest source, a 2020 survey by Stapelberg and Malan, catalogues benchmark tasks without usage counts ([arXiv](https://arxiv.org/pdf/2011.13577.pdf)).

### 2.1 General RL suites

**Table 1a. General RL suites: goal, fate and adoption**

| Benchmark | Stated goal | Stuck or faded, and why (judgement) | Adoption signals (2026-11-27 unless dated) |
|---|---|---|---|
| Classic control (CartPole, Acrobot, MountainCar, Pendulum) | "Classic RL problems based on real-world physics" ([Gymnasium](https://github.com/Farama-Foundation/Gymnasium)) | The debugging and teaching floor, built from tasks of 1983 to 1996 and bundled with Gym ([CartPole docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/)) | Gym paper 9,566 cites; Gymnasium 12.9k stars; in RL Zoo, CleanRL, RLlib, EnvPool and Open RL Benchmark ([RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md)) |
| MuJoCo locomotion (Gym, now Gymnasium) | Answer the "lack of a common benchmark" for continuous control ([Duan et al.](https://arxiv.org/abs/1604.06778)) | Stuck through rllab's benchmark paper with code and the Gym wrapper, despite a paid licence until 2021 to 2022 | MuJoCo paper 9,070 cites; engine 18.5k stars ([GitHub](https://github.com/google-deepmind/mujoco)); in every library checked |
| Atari (ALE) | "Both a challenge problem and a platform and methodology for evaluating" general agents ([Bellemare et al.](https://arxiv.org/abs/1207.4708)) | Stuck. Many human-comparable games, then DQN's 49-game result; the DQN Nature paper has 40,558 cites | ALE 3,719 cites; 2.5k stars; ale-py 0.12.1 on 2026-08-16 ([PyPI](https://pypi.org/project/ale-py/)); in every library checked |
| DeepMind Control Suite | Tasks "with a standardised structure and interpretable rewards" ([Tassa et al.](https://arxiv.org/abs/1801.00690)) | Stuck as the default for pixel and model-based control. Bounded rewards make returns comparable | 1,815 cites; 4.7k stars; dm-control 1.0.52 on 2026-11-11 ([PyPI](https://pypi.org/project/dm-control/)); in CleanRL and EnvPool, absent from RL Zoo |
| Procgen | "Benchmark both sample efficiency and generalization" ([Cobbe et al.](https://arxiv.org/abs/1912.01588)) | Stuck as the generalisation testbed. The paper fixed the budgets and the score | 313 cites; 1,199 stars, last push 2026-03-27 ([GitHub API](https://api.github.com/repos/openai/procgen)); in CleanRL and EnvPool |
| MinAtar | Let researchers "thoroughly investigate behavioural challenges" like ALE's ([paper](https://arxiv.org/abs/1903.03176)) | Stuck as a cheap Atari proxy, now mostly through gymnax | 185 cites; 342 stars, last push 2024-12-19 ([GitHub API](https://api.github.com/repos/kenjyoung/MinAtar)) |
| bsuite | "Clear, informative and scalable problems that capture key issues" ([paper](https://arxiv.org/abs/1908.03568)) | Cited more than reported. A targeted search found no paper that reports its radar ([search example](https://arxiv.org/html/2309.11975v1)). Its tasks live on in gymnax | 239 cites; 1,566 stars; PyPI 0.3.6 on 2026-05-08, five years after 0.3.5 ([PyPI](https://pypi.org/pypi/bsuite/json)) |
| Meta-World | "Benchmarking multi-task and meta reinforcement learning" ([README](https://raw.githubusercontent.com/Farama-Foundation/Metaworld/master/README.md)) | Stuck through fixed named splits (MT1/10/50, ML1/10/45). Farama maintains it | 1,878 cites; 1,910 stars; tasks now at -v3 |
| D4RL, now Minari | Benchmarks "specifically designed for the offline setting" ([paper](https://arxiv.org/abs/2004.07219)) | Stuck as offline RL's first shared datasets with a 0 to 100 score. Being retired to Minari after others reported saturation and flaws ([D5RL](https://arxiv.org/pdf/2408.08441)) | D4RL 2,028 cites, last push 2024-11-18; Minari 1,734 stars, pushed 2026-11-24 ([GitHub API](https://api.github.com/repos/Farama-Foundation/Minari)) |
| Safety Gym, now Safety-Gymnasium | "Tools for accelerating safe exploration research", then "the evaluation and comparison of safety performance" ([paper](https://arxiv.org/abs/2310.12567)) | Safety Gym was archived. The successor is active and bundles 16 algorithms in SafePO | Safety Gym archived, last push 2023-04-02 ([GitHub API](https://api.github.com/repos/openai/safety-gym)); Safety-Gymnasium 274 cites, NeurIPS 2023 datasets track |
| NetHack (NLE), MiniHack | "A scalable, procedurally generated, stochastic, rich, and challenging environment" ([NLE](https://arxiv.org/abs/2006.13760)) | NLE stuck as a cheap unsolved challenge; symbolic bots took the top three places in the 2021 challenge ([Hambro et al.](https://arxiv.org/pdf/2203.11889)). MiniHack stayed niche | NLE 262 cites, original repo archived; MiniHack 145 cites, archived ([GitHub API](https://api.github.com/repos/facebookresearch/minihack)) |
| Crafter, Craftax | "Benchmarking the spectrum of agent capabilities"; Craftax adds "limited computational resources" ([Craftax](https://arxiv.org/abs/2402.16801)) | Craftax grows on JAX speed and a README leaderboard. Crafter is quiet | Crafter 272 cites, last push 2024-01-23; Craftax 125 cites, ICML 2024 spotlight ([GitHub API](https://api.github.com/repos/MichaelTMatthews/Craftax)) |
| Brax | "A focus on performance and parallelism on accelerators" ([paper](https://arxiv.org/abs/2106.13281)) | Stuck as the JAX physics engine, shipping PPO, SAC and ES | 735 cites; 3,268 stars ([GitHub API](https://api.github.com/repos/google/brax)) |
| gymnax, with PureJaxRL | "RL Environments in JAX" ([README](https://raw.githubusercontent.com/RobertTLange/gymnax/main/README.md)) | The de facto JAX environment API; Craftax reuses it | 944 stars, pushed 2026-11-27; PureJaxRL 1,129 stars ([GitHub API](https://api.github.com/repos/luchris429/purejaxrl)); no paper count |
| Jumanji | Environments "designed to be fast, flexible, and scalable" ([paper](https://arxiv.org/abs/2306.09884)) | Moderate uptake, with no shared headline score | 117 cites; 877 stars |
| MiniGrid | Grid-world tasks (goal not quoted in the notes) | Stuck as a cheap Farama testbed | 238 cites; 2,547 stars ([GitHub API](https://api.github.com/repos/Farama-Foundation/Minigrid)); in RL Zoo and EnvPool |
| PettingZoo | "Gym for Multi-Agent Reinforcement Learning" | An API standard with no scored task set | 345 cites; 3,561 stars ([GitHub API](https://api.github.com/repos/Farama-Foundation/PettingZoo)) |
| SMAC, SMACv2 | StarCraft II multi-agent tasks (goal not quoted in the notes) | Stuck, then redesigned as SMACv2 and rebuilt as SMAX in JaxMARL | SMAC 1,613 cites, SMACv2 104; both repos frozen since early 2024 ([GitHub API](https://api.github.com/repos/oxwhirl/smac)) |
| XLand-MiniGrid, MuJoCo Playground, JaxMARL | "Democratizing large-scale experimentation with limited resources" ([paper](https://arxiv.org/abs/2312.12044)); "streamlining simulation, training, and sim-to-real transfer onto robots" ([paper](https://arxiv.org/abs/2502.08844)); GPU-efficient multi-agent RL ([paper](https://arxiv.org/abs/2311.10090)) | Too new to judge. Playground was rewritten on MuJoCo Warp with its first JAX tasks frozen; JaxMARL includes SMAX | 75, 159 and 134 cites; 347, 2,353 and 880 stars |
| RL-Glue and the RL Competitions (faded) | A language-independent API and yearly competitions from 2008 ([AI Magazine](https://ojs.aaai.org/index.php/aimagazine/article/view/2227)) | Faded; no reason found. A protocol and a contest without a standard task and number | A later report counts 51 teams of 90 members ([2014 report](https://typeset.io/pdf/the-reinforcement-learning-competition-2014-3q7s7fzqsh.pdf)) |
| Universe, Gym Retro, Roboschool (faded) | OpenAI platforms for many games, or for free physics | Faded as the host moved on; no stated reasons found | Universe deprecated for Retro ([repo](https://github.com/openai/universe)); Retro archived 2026-05-29 ([repo](https://github.com/openai/retro)); Roboschool deprecated for PyBullet 2019-09-27 ([repo](https://github.com/openai/roboschool)) |

**Table 1b. General RL suites: realism, cost, scoring, baselines and API**

| Benchmark | Realism (estimate) | Cost of one run | Scoring and reporting | Baselines shipped | API |
|---|---|---|---|---|---|
| Classic control | 1 to 2 | Not found. A CleanRL CartPole DQN run is 500k steps ([CleanRL DQN](https://docs.cleanrl.dev/rl-algorithms/dqn/)); gymnax steps 1M transitions in 0.05 to 0.2 s on an A100 | Return; Gymnasium's docs give a reward threshold of 500 for CartPole-v1 | None found | Gym, then Gymnasium |
| MuJoCo locomotion | 2 | TD3's protocol: 1M steps and 10 seeds per task ([Fujimoto et al.](https://ar5iv.labs.arxiv.org/html/1802.09477)); wall-clock not found | Return at a budget, with no normalisation; versions differ (v4 and v5 through many changes), so scores across versions are not strictly comparable ([docs](https://gymnasium.farama.org/environments/mujoco/)) | rllab reference algorithms | Gymnasium v5, since 2024-10-08 ([Farama](https://farama.org/Gymnasium-v1.0)) |
| Atari (ALE) | 1 | Rainbow: 200M frames, about 10 days on one GPU per game ([Hessel et al.](https://ar5iv.labs.arxiv.org/html/1710.02298)). Atari 100k: 400k frames ([Kaiser et al.](https://ar5iv.labs.arxiv.org/html/1903.00374)) | Human-normalised score, 100 x (agent - random) / (human - random); medians over games; sticky actions since 2018 ([Machado et al.](https://arxiv.org/abs/1709.06009)) | Benchmark agents in the 2013 paper | C++, Python, Gymnasium |
| DeepMind Control Suite | 2 | Pixel runs at 100k and 500k steps on six PlaNet tasks ([CURL](https://arxiv.org/pdf/2004.04136)); wall-clock not found | Reward in [0, 1] per step over 1,000 steps, so 1,000 at most; BENCHMARKING and EXTRA sets ([Tassa et al.](https://ar5iv.labs.arxiv.org/html/1801.00690)) | Results for several algorithms in the paper | dm_control |
| Procgen | 1 | Easy: 25M steps, about 3 GPU-hours. Hard: 200M steps, about 24 GPU-hours ([paper](https://arxiv.org/pdf/1912.01588)) | Normalised return per game, mean over 16 games; train on 200 levels | PPO | Gym |
| MinAtar | 1 | 10x10 frames; wall-clock not found | Raw return | DQN-style (not verified) | Gym-like |
| bsuite | 1 | Small tasks; not timed | Per-experiment scores and a capability radar from its notebook ([README](https://raw.githubusercontent.com/google-deepmind/bsuite/main/README.md)) | A baselines folder | dm_env, Gym wrapper |
| Meta-World | 2 | Not found | Success rate (convention, not verified) | A separate results repository | Gymnasium |
| D4RL, Minari | 2 | Not found | 0 at random, 100 at an expert reference return ([README](https://raw.githubusercontent.com/Farama-Foundation/D4RL/master/README.md)) | The paper evaluates existing algorithms | Gym, now Gymnasium |
| Safety Gym, Safety-Gymnasium | 2 | Not found | Return beside episodic cost; normalised cost comes from follow-up work | SafePO | Gym, Gymnasium |
| NLE, MiniHack | 1 | Curves run to 1e9 steps; speed not stated | In-game score, no normalisation | IMPALA-style (not verified) | Gym |
| Crafter, Craftax | 1 | Craftax: 1B PPO steps in under an hour on one GPU | Percentage of the maximum reward; the best 1B-step entry reaches 19.5 % ([README](https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/README.md)) | PPO and RNN variants, RND, ICM, E3B and others | gymnax-style JAX |
| Brax | 2 | Training "in minutes" | Return | PPO, SAC, ES | JAX |
| gymnax | 1 to 2 | MinAtar Asterix: 1M steps in 0.9 s on an A100 | Each task's own return | None as a suite | gymnax (six outputs since v1.0) |
| Jumanji | 2 | Not found | No shared headline | Actor-critic per environment | JAX |
| MiniGrid, PettingZoo, SMAC | 1 to 2 (PettingZoo is an API) | Not found | SMAC: win rate (convention, not verified); others not found | Not found | Gymnasium, PettingZoo, own |
| XLand-MiniGrid, MuJoCo Playground, JaxMARL | 1; 2 to 3; 1 to 2 | "Millions of steps per second"; policies "in minutes on a single GPU"; up to 12,500 times faster when vectorised | Not found | In each; Playground uses RSL-RL PPO | JAX; Playground now on MuJoCo Warp |

### 2.2 Control and industrial suites

**Table 2a. Control and industrial suites: goal, fate and adoption**

| Benchmark | Stated goal | Stuck or not, and why (judgement) | Adoption signals (2026-11-27 unless dated) |
|---|---|---|---|
| PC-Gym | An "open-source tool for developing and evaluating reinforcement learning (RL) algorithms in chemical process control" ([arXiv](https://arxiv.org/abs/2410.22093)) | Modest uptake inside process systems engineering; see section 2.3 | 27 cites; 107 stars, 16 forks; last PyPI release 2024-12-03 ([PyPI](https://pypi.org/project/pcgym/)) |
| safe-control-gym | "The tools to equitably compare the approaches proposed by the controls and reinforcement learning communities" ([arXiv](https://arxiv.org/abs/2109.06325)) | The best-adopted control-versus-RL suite here. Baselines from both camps ship on symbolic models that suit MPC, and a robotics journal carried it | 134 cites; 938 stars ([GitHub API](https://api.github.com/repos/utiasDSL/safe-control-gym)); IEEE RA-L, October 2022 |
| controlgym | Test "the scalability of RL algorithms to high- and potentially infinite-dimensional systems" ([arXiv](https://arxiv.org/abs/2311.18736)) | Little uptake. It ships no baselines, and its scope was reframed between versions | 25 cites; 57 stars ([GitHub](https://github.com/xiangyuan-zhang/controlgym)); L4DC 2024 |
| gym-electric-motor | A Python toolbox for simulating and controlling electric motors ([GitHub](https://github.com/upb-lea/gym-electric-motor)) | A niche success in power electronics, with three papers in good venues | 94 cites (TNNLS) and 24 (JOSS); 439 stars, last push 2025-12-19 |
| Industrial Benchmark | Bridge "the gap between freely available, documented, and motivated artificial benchmarks" and real industrial problems ([arXiv](https://arxiv.org/abs/1709.09480)) | Used for offline and model-based RL. It is synthetic by design | 80 cites; 152 stars; the README lists dozens of papers ([GitHub](https://github.com/siemens/industrialbenchmark)) |

**Table 2b. Control and industrial suites: realism, cost, scoring, baselines and API**

| Benchmark | Realism (estimate) | Cost of one run | Scoring and reporting | Baselines shipped | API |
|---|---|---|---|---|---|
| PC-Gym | 3 | 50k SB3 steps; wall-clock not reported | Per task: median return and its deviation, the gap to an NMPC oracle, a constraint-violation rate; no headline | NMPC oracle; SAC, PPO, DDPG at SB3 defaults | Gymnasium |
| safe-control-gym | 2 | Not found | No headline; performance, data efficiency and safety compared | PID, LQR, iLQR, linear MPC, GP-MPC, SAC, PPO, DDPG, safety filters | Gym-style; tags reproduce each paper |
| controlgym | 2 | Not found | Not stated | Not stated | Gymnasium |
| gym-electric-motor | 3 to 4 | Not found | Not defined in the README | PI, linear feedback, an MPC example, SB3 notebooks | Gymnasium |
| Industrial Benchmark | 2 | Not found | Two opposing reward terms; no metric defined | A script, with no results | Java, Python, Gym wrapper |

### 2.3 PC-Gym in depth

PC-Gym matters more to TargetGym than any other suite, since TargetGym's cstr, first_order and four_tank tasks were adapted from it and checked against it term by term. The two repositories are the same age, created on 2023-08-22 ([GitHub API](https://api.github.com/repos/MaximilianB2/pc-gym)) and 2023-11-08.

**Authors and status.** Nine authors from the process systems community wrote the paper, posted on arXiv between 29 October and 5 December 2024 ([arXiv](https://arxiv.org/abs/2410.22093)). The notes conflict on publication. The arXiv page lists no journal reference, while the gap-fill note found OpenAlex merging the preprint into a 2025 *Computers & Chemical Engineering* record ([OpenAlex](https://api.openalex.org/works?filter=doi:10.48550/arxiv.2410.22093%7C10.48550/arxiv.2109.06325%7C10.1016/0098-1354%2893%2980018-I&select=doi,title,publication_year,cited_by_count)). Following the gap-fill note, a journal version probably exists; this study did not open it.

**Tasks.** The paper has four environments: a CSTR, a five-stage extraction column, a potassium sulfate crystalliser and a four-tank system ([paper](https://arxiv.org/html/2410.22093v4)). The README now lists the CSTR, the extraction column, a first-order system, nonsmooth control and a cement precalciner, which an outside contributor added on 2026-11-14. An open pull request of 2026-11-25 proposes physics and evaluation fixes and a surge-tank task ([commits](https://api.github.com/repos/MaximilianB2/pc-gym/commits?per_page=5)).

**Protocol.** The CSTR holds three concentration set points over 60 steps under a 321 K to 327 K temperature constraint. Training draws the inlet temperature from 330 K to 370 K. Tests step it to 363 K, inside that range, and to 375 K, outside it. The reward is a normalised quadratic tracking error plus an input-change penalty, weighted to match the oracle, and constraints enter only through reward shaping.

**Scoring and results.** The paper reports median return, median absolute deviation, an optimality gap to an NMPC oracle and an empirical constraint-violation probability. It gives no headline number and no aggregate. The oracle runs do-mpc, CasADi and IPOPT, is called a "performance ceiling", and knows the realised uncertainty inside its horizon. The agents are SB3's SAC, PPO and DDPG at default settings, trained for 50,000 steps with one shared seed. Optimality gaps range from 0.002 (SAC on extraction) to 0.950 (DDPG on four-tank). Shaping cut the constraint-violation rate from 0.0727 to 0.0003. The authors name no winning algorithm, and out-of-distribution disturbances reversed the ranking.

**Health on 2026-11-27.** The repository has 107 stars, 16 forks, 339 commits and no GitHub releases ([GitHub](https://github.com/MaximilianB2/pc-gym)). Semantic Scholar lists 20 citing papers, mostly in process systems engineering ([citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2410.22093/citations?fields=title,year&limit=100)). No RL library, competition or leaderboard listing was found. The last PyPI release is 0.1.8 of 2024-12-03, and an issue of 2026-07-10 says the package is behind main. Issue #58 reports that the paper's four-tank table now reproduces only with a pre-2025 data file. Other open issues say that reset ignores its seed, episodes end one step early, and parameter changes after creation have no effect ([issues](https://api.github.com/repos/MaximilianB2/pc-gym/issues?state=all&per_page=30)).

**What it shows TargetGym (judgement).** PC-Gym sets the adoption band for a realistic process-control suite, with tens of citations and about a hundred stars in two to three years. It also shows what a moving main branch does once a paper has printed numbers. Its oracle gap is the idea closest to TargetGym's room, and its one-seed protocol is the weakness that TargetGym's 30-seed intervals already avoid.

### 2.4 Energy and grid suites

**Table 3a. Energy and grid suites: goal, fate and adoption**

| Benchmark | Stated goal | Stuck or not, and why (judgement) | Adoption signals (2026-11-27 unless dated) |
|---|---|---|---|
| BOPTEST, BOPTEST-Gym | "A framework for building control performance benchmarking" ([site](https://ibpsa.github.io/project1-boptest/)) | An in-field standard with shared emulators and KPIs under IBPSA. RL users reach it through wrappers such as Beobench ([Beobench](https://beobench.readthedocs.io/en/stable/envs.html)) | 329 cites; 189 stars, 3,879 commits ([GitHub](https://github.com/ibpsa/project1-boptest)); no NeurIPS, ICML or ICLR paper found through BOPTEST-Gym |
| Sinergym | An "open-source Python-based virtual testbed for large-scale building simulation" ([arXiv](https://arxiv.org/html/2412.08293v1)) | Adopted as a tool. Users set the reward, so there is no fixed score | 86 cites; 242 stars ([GitHub](https://github.com/ugr-sail/sinergym)) |
| CityLearn | A shared Gym environment so that demand-response RL can be compared, plus a yearly challenge ([arXiv](https://arxiv.org/abs/2012.10504)) | Grew through NeurIPS competitions. Each edition changed the dataset or task | 217 cites; 663 stars; 2021 challenge 5 teams, 2022 about ten times more ([UT Austin](https://cockrell.utexas.edu/news/researchers-challenged-to-optimize-building-energy-use-with-ai/)), 2023 600+ participants and 2,500+ submissions ([AIcrowd](https://discourse.aicrowd.com/t/winner-announcement/9387)) |
| Grid2Op, L2RPN | "Develop a catalyst to foster faster progress in the field" ([retrospective](https://ar5iv.labs.arxiv.org/html/2103.03104)) | Competitions with one benchmark-defined score drew outsiders, and a Baidu RL team without power-systems background won in 2020. Each edition was a new challenge | 125 cites (L2RPN) and 176 (retrospective); 494 stars ([GitHub](https://github.com/Grid2Op/grid2op)); NeurIPS 2020: 300+ participants, 64 teams |
| Gym-ANM | A framework for active network management environments ([arXiv](https://arxiv.org/abs/2103.07932v2)) | Small uptake | 61 cites; 175 stars ([GitHub](https://github.com/robinhenry/gym-anm)) |
| PowerGridworld | A "lightweight, modular, and customizable framework" ([ar5iv](https://ar5iv.labs.arxiv.org/html/2111.05969)) | A framework whose authors provide no benchmarks | 13 cites; stars not found |

**Table 3b. Energy and grid suites: realism, cost, scoring, baselines and API**

| Benchmark | Realism (estimate) | Cost of one run | Scoring and reporting | Baselines shipped | API |
|---|---|---|---|---|---|
| BOPTEST | 4 | Not found | KPIs computed by the run-time (discomfort, energy, cost, emissions); no headline | Each case's embedded control | REST over HTTP; Gym wrapper |
| Sinergym | 4 | 40 one-year episodes per agent in the paper; wall-clock not reported | Weighted energy and comfort reward, user-tunable | Default and rule-based control, SAC, TD3, PPO | Gymnasium |
| CityLearn | 3 | Not found | A KPI scorecard normalised to a baseline ([GitHub](https://github.com/intelligent-environments-lab/CityLearn)); a ranking score per challenge | A business-as-usual agent | Gymnasium |
| Grid2Op | 3 to 4 | Not found; the 2020 winner trained on 500 cores for 30 minutes | -100 at blackout to 100, with do-nothing at 0, averaged over 24 weekly scenarios | Do-nothing, an expert system, past winners | Gym-compatible |
| Gym-ANM, PowerGridworld | 3 | Gym-ANM not found; PowerGridworld ran MADDPG for 350 iterations, with agent calls in series | None found | PowerGridworld: MADDPG and RLlib PPO case studies | Gym; RLlib multi-agent |

### 2.5 Control-community classics

**Table 4a. Control classics: goal, longevity and adoption**

| Benchmark | Stated goal and what is fixed | Why it lasted (judgement) | Adoption signals (2026-11-27 unless dated) |
|---|---|---|---|
| Tennessee Eastman (Downs and Vogel 1993) | "A plant-wide industrial process control problem". A Fortran model with disturbance switches, built on a disguised real plant ([Ricker archive](https://depts.washington.edu/control/LARRY/TE/download.html)) | Free code archives, and a fault-detection community that needs labelled faults | 3,660 cites; a C update in January 2015 |
| BSM1, BSM2 (IWA) | Benchmark control strategies for wastewater plants. The layout, influent, default control and evaluation criteria are fixed, and ring tests check implementations across platforms ([Lund](https://lup.lub.lu.se/search/publication/61891cd7-6fd7-4db7-8bc3-a95fc8166b19)) | Stewardship by an IWA task group, with new models added beside the old (BSM1, BSM1_LT, BSM2) | More than 300 related publications; IWA report No. 23 in 2014; repository 60 stars ([GitHub](https://github.com/wwtmodels/Benchmark-Simulation-Models)) |
| Shell heavy oil fractionator (1987) | A Shell workshop problem with unmeasured disturbances, model uncertainty and constraints ([MATLAB Central](https://www.mathworks.com/matlabcentral/fileexchange/65888-shell-control-problem)) | A constrained MPC test bed | Not found |
| IFAC PID 2012 boiler | Two boiler control setups for IFAC PID'12, March 2012 ([PID'12](https://skoge.folk.ntnu.no/prost/proceedings/PID-12/papers/0018.pdf)) | Used by papers at that workshop | Not found |

**Table 4b. Control classics: realism, cost, scoring, baselines, API and RL use**

| Benchmark | Realism (estimate) | Cost of one run | Scoring and reporting | Baselines | API | RL use found |
|---|---|---|---|---|---|---|
| Tennessee Eastman | 4 | Fast with the MEX code; a pure MATLAB port is about 100 times slower | No single headline; detection and false-alarm rates in fault detection | None found | Fortran, C, MATLAB, Simulink | A safe RL control paper in 2023 ([arXiv](https://arxiv.org/pdf/2310.14788v1.pdf)) |
| BSM1, BSM2 | 4 | Not found | Effluent quality and cost indices with violation counts (not verified) | The default control | MATLAB | TD3 cut aeration and pumping energy by 14.3 % against BSM1's default control (2023); another agent gained 8.3 % (2024) ([list](https://www.mdpi.com/2076-3417/13/8/4752)) |
| Shell fractionator, PID 2012 boiler | 3 | Not found | Not found | Not found | Shell: a linear model in MATLAB | Not found |

## 3. Ranking from TargetGym's perspective

TargetGym competes on two fronts. For users and plants, its rivals are the realistic control and energy suites. For attention and tooling, it sits beside the general suites that set the bar. The tiers order benchmarks by how directly they compete with TargetGym's scored tasks.

| Tier | Benchmark | Overlap with TargetGym | How it fared, and why (counts of 2026-11-27; why is judgement) |
|---|---|---|---|
| 1. Direct rivals | PC-Gym | cstr, first_order and four_tank came from it; its NMPC "ceiling" mirrors TargetGym's | Modest (27 cites, 107 stars). One seed and no headline, and main drifted from the paper |
| 1 | BOPTEST, BOPTEST-Gym | hvac overlaps bestest_hydronic_heat_pump (practitioner review) | An in-field standard (329 cites) under IBPSA, with no route into RL libraries |
| 2. Same framing, other plants | safe-control-gym | Classical and RL baselines on physical models | The best of this group (134 cites, 938 stars), helped by baselines from both camps and a robotics venue |
| 2 | CityLearn | battery, an Extra task, overlaps it (practitioner review) | Competition-driven, 600+ participants in 2023; no fixed number across editions |
| 2 | Grid2Op, L2RPN | An operator's job scored from a do-nothing zero | Hundreds of competitors per edition; each edition moved the task |
| 2 | gym-electric-motor | Engineering plants with PI and MPC baselines | A niche success (94 and 24 cites) |
| 2 | Industrial Benchmark, Sinergym, controlgym, Gym-ANM, PowerGridworld | Industrial or energy control without a fixed score | 13 to 86 cites; used as tools |
| 3. Stewardship models | BSM1, BSM2 | A fixed plant scored against a default controller | 300+ publications; RL papers report a saving over default control |
| 3 | Tennessee Eastman | A disguised real plant, frozen in code | 3,660 cites since 1993, mostly in fault detection |
| 4. The bar and the carriers | Classic control, MuJoCo, DMC, ALE | They set what RL users expect of a benchmark | In every library checked; realism 1 to 2 |
| 4 | gymnax, Brax | TargetGym's JAX peers; gymnax is its API model | gymnax 944 stars; Brax 735 cites |
| 4 | bsuite, Safety-Gymnasium, D4RL and Minari | Precedents for TargetGym's radar, trip reporting and planned offline data | bsuite's radar is cited, rarely reported; Safety-Gymnasium reports cost beside return; D4RL was retired after saturation |

Two readings follow, both this study's judgement. First, no realistic or industrial suite reached the general RL libraries; the pages checked list none of BOPTEST, Sinergym, CityLearn, Grid2Op, PC-Gym or safe-control-gym ([Open RL Benchmark](https://arxiv.org/html/2402.03046), [EnvPool](https://github.com/sail-sg/envpool)). The best realistic outcome on record is standard status inside a field. By 2026-11-27, Tennessee Eastman had 3,660 citations over 33 years and the BSM family more than 300 related publications, while BOPTEST had gathered 329 citations in five years. TargetGym itself had 13 stars and 1 fork that day, and no paper. Competitions brought crowds to CityLearn and L2RPN, yet each edition changed the task, so no number lasted from one year to the next.

Second, BSM is TargetGym's closest model. It fixes a plant, its influent, a default controller and the criteria, and ring tests check every implementation. RL papers on it report a percentage saving over the default control. TargetGym's charter headline, the saving over a frozen reference, already has that shape. PC-Gym is the warning beside it, a peer of the same age whose printed table no longer reproduces from its own code.

## 4. The bar

The evidence is correlational. No source tests why a benchmark was adopted, and no year-by-year count of RL benchmark use was found. Koch et al. show that ML research concentrates on fewer and fewer datasets within each task community ([arXiv](https://arxiv.org/abs/2112.01716)). Dehghani et al. show that switching tasks alone can reorder methods ([arXiv](https://arxiv.org/abs/2107.07002)). Concentration makes the winners' traits worth copying; whether those traits caused the win is untested.

| Candidate condition | Evidence for (counts of 2026-11-27) | Limits and counter-cases | Verdict |
|---|---|---|---|
| A task set that never changed | Changes came as new names: CartPole v0 to v1; MuJoCo v4 kept "for reproducibility" beside v5; D4RL's v2 relabel; safe-control-gym tags that reproduce each paper; BSM's fixed layout and ring tests | None of them literally froze. PC-Gym changed in place and its table drifted (#58); CityLearn and L2RPN changed every edition | Kept, restated as frozen versions |
| One number to cite | ALE's human-normalised score; Procgen's normalised return at fixed budgets; D4RL's 0 to 100; DMC's 1,000-point return; L2RPN's -100 to 100 | MuJoCo stuck with return at a budget by convention. bsuite's radar and BOPTEST's KPI set were cited more than reported | Kept |
| A fixed protocol (budget, seeds, statistic) | Machado et al. repaired ALE's protocol in 2018. Agarwal et al. found point estimates from few runs misleading and proposed intervals and the IQM ([arXiv](https://arxiv.org/abs/2108.13264)) | Two sets of 5 TRPO runs on HalfCheetah differed significantly ([Henderson et al.](https://ar5iv.labs.arxiv.org/html/1709.06560)); ALE papers diverged although the benchmark shipped a method | Kept, merged into the number |
| Cheap runs | Classic control and MinAtar are cheap. Cheaper protocols grew beside costly ones: Atari 100k (885 cites), DMC at 100k steps, and Procgen's easy mode at an eighth of the hard mode's GPU time. Full Rainbow on 57 games took about 34,200 GPU hours ([Google Research](https://www.research.google/blog/reducing-the-computational-cost-of-deep-reinforcement-learning-research/)) | MuJoCo needed a paid licence until 2021 to 2022 and still stuck | Kept, with a cheap tier allowed beside a costly one |
| A spread of difficulty | DMC's BENCHMARKING and EXTRA sets; Procgen's easy and hard modes; Gymnasium's toy tasks "useful for debugging"; Atari's headroom lasted until Agent57 beat the human benchmark on all 57 games in 2020 ([Agent57](https://ar5iv.labs.arxiv.org/html/2003.13350)) | NetHack and Craftax stayed unsolved and were adopted anyway; later papers counted D4RL's saturation as a flaw | Kept, as visible headroom |
| Shipped baselines | ALE's agents, rllab, DMC's results, Procgen's PPO, D4RL's evaluations, Craftax's baselines, Grid2Op's do-nothing and expert system | Classic control shipped none but had a pedigree | Kept, with a learned agent among them |
| A standard API and library integration | Gym's paper has 9,566 cites, and the Gymnasium paper targets the "lack of standardization" ([arXiv](https://arxiv.org/abs/2407.17032)). Classic control, Atari and MuJoCo are in RL Zoo, CleanRL, RLlib, EnvPool and Open RL Benchmark; no industrial or energy suite is | RL-Glue was a standard API and faded without a task set and number. Libraries add what users already run | Kept, as one condition |
| A paper | ALE, DMC, Procgen, D4RL and Meta-World each launched with one; the DQN Nature paper has 40,558 cites | A landmark result is outside any project's control | Kept, with the question stated |
| Versioning and stewardship | Farama now maintains Gymnasium, ALE, Meta-World, MiniGrid and D4RL's successor; IWA and IBPSA steward BSM and BOPTEST | Single-lab suites mostly froze (an inference from push dates) | Kept |
| Community, realism | Competitions lifted CityLearn about tenfold in a year. Realism is absent from the go-to suites (see below) | The go-to suites needed neither; a community follows adoption | Dropped as conditions |

The go-to benchmarks did without realism. This study rates classic control 1 to 2, MuJoCo locomotion and DMC 2, and Atari 1, and none is validated against a real system. Arulkumaran et al. credit MuJoCo with "relatively realistic dynamics" for multi-joint control, a relative claim ([arXiv](https://arxiv.org/pdf/1708.05866)). They did without free tools as well. MuJoCo licences cost an institution at least $3,000 a year, and up to $15,000 by one account, until DeepMind bought the engine in 2021 ([Freethink](https://www.freethink.com/?p=37825), [DeepMind](https://deepmind.google/blog/opening-up-a-physics-simulator-for-robotics/)). Their early results used few seeds, 5 for SAC ([Haarnoja et al.](https://ar5iv.labs.arxiv.org/html/1801.01290)) and 10 for TD3, and the statistical repairs came later. MuJoCo never had a normalised score, and none of the three needed a competition. All of them had a fixed definition and a place in the tools that everyone used.

**The bar.** The go-to benchmarks had the eight conditions below in common, though some met one only by convention or pedigree: MuJoCo's number was return at 1M steps by habit, and classic control shipped no baselines of its own. A new benchmark has neither that habit nor that pedigree to lean on, so this study treats all eight as necessary for TargetGym. Meeting them does not guarantee adoption, which also depends on a landmark result or a crowd that no project controls.

- A frozen task set, versioned by name, so that every number names the version it was measured on.
- One number per task, defined and computed by the benchmark under a fixed budget, seed count and statistic.
- Runs cheap enough for many seeds on common hardware, with a cheap tier when the full protocol is costly.
- Shipped baselines with published results at the protocol, including at least one learned agent.
- A standard API that the main RL libraries already run.
- A paper that states the question the suite answers, its tasks, its protocol and its first results.
- Visible headroom from launch: a measured spread between a floor, the baselines and a ceiling.
- Stewardship: maintained releases and a written rule for how versions change.

## 5. TargetGym against the bar

TargetGym meets none of the eight in full. It is partway on six, and the two it misses outright are the frozen task set and the paper.

| Condition | Meets today? | What it lacks | Cheapest way to meet it |
|---|---|---|---|
| 1. Frozen, versioned task set | No | Task versions reached v10 in a year, and each scored plant took 14 to 39 commits in November. The scored set fell from five to four on 2026-11-27. ph_neutralization-v6's file notes behaviour changes on 2026-11-24 and 2026-11-25 under one version number | Treat the 0.8.0 tag as the freeze. Copy the four fingerprints into one manifest, and let a CI test fail whenever one moves |
| 2. One citable number per task | Partly | Rule J4 defines the saving over the reference with a 95 % interval, but the public `tg.score_policy` compares against a different controller, configuration, seeds and statistic. `tg.score` is still being built | Ship `tg.score` as the only benchmark scorer (already planned). Cite the per-task saving, and hold the suite aggregate until 8 to 10 tasks across three domains, as the RL research review advises |
| 3. Cheap runs, with a cheap tier | Partly | Steps are cheap (55k to 580k per second on two cores as measured; the docs claim 0.6M to 700M), but the full protocol needs more than 200 runs per task and algorithm (see below) | Freeze a short tier: one budget, 30 seeds, shipped settings and no tuning, with its wall-clock published |
| 4. Baselines with published results | Partly | Expert, reference, null, MPC and ceiling lines exist but are stale on the candidate. No learned result is published. At 200k steps PPO lost to the best constant action on 7 of 8 tasks, and a 5M-step PPO lost to the pH reference by 14 to 24 % | Re-record at the freeze (planned for 0.8.0). Publish PPO and SAC at the short tier whatever they show (the 0.9.0 slice) |
| 5. Standard API and library reach | Partly | gymnax API and a Gymnasium wrapper, but no Gymnasium registry IDs, so `gymnasium.make` fails. No RL library lists it, like every industrial suite checked | Register the frozen IDs with Gymnasium, and ship one training script per framework |
| 6. Paper with a stated question | No | A @misc BibTeX entry; the paper is planned for 0.11.0. The question exists: the practitioner review rated novelty 7 of 10 and found no open RL equivalent for PWR load follow, anti-surge control or permit-bound neutralisation | Write a v1 page in the repository that states the question, the tasks, the protocol and the results, and point `CITATION.cff` at it |
| 7. Visible headroom | Partly | Null, reference and ceiling lines bracket each task and the room is reported, but the lines are stale and no causal line exists (planned for 0.9.0) | Covered by rows 1 and 4 |
| 8. Stewardship and a versioning rule | Partly | Versioned docs and PyPI releases exist, with one author. The plan writes "a scored version never changes" only at 1.0.0 | Adopt that sentence at 0.8.0 and keep frozen versions installable and scorable |

The protocol's cost is the one gap that per-step speed does not close. The arithmetic here is this study's estimate. Scaling the RL research review's pH timing, one 1e7-step PPO run takes about 150 to 175 s on four cores. The protocol asks for 30 seeds at three budgets plus 64 tuning trials of 3 seeds, more than 200 runs per task and algorithm. If tuning runs at 1e7 steps, one algorithm on one task needs about 9 to 11 hours of four cores, and two algorithms on the four tasks about 75 to 90 hours. With tuning at 1e6 steps, the first figure drops to about 2 to 2.5 hours. A short tier of 30 seeds at 1e6 steps with shipped settings would take about 7 to 9 minutes per task. Only pH was timed. TD3's MuJoCo protocol, for comparison, is 10 seeds at 1M steps per task, and Agarwal et al. warn that costly benchmarks lead to few runs per task.

## 6. Arcade versus realistic

Three options are open. A stylised arcade suite would offer few fast tasks with clear dynamics and one normalised score, possibly simplified from TargetGym's plants. The current course keeps the realistic plants. A two-track project would run both.

| Criterion (counts of 2026-11-27) | A. Arcade suite | B. Realistic course | C. Both tracks |
|---|---|---|---|
| Who holds the niche | Classic control in every library; DMC (1,815 cites); MinAtar, gymnax, Brax, Craftax, XLand-MiniGrid; for stylised control, controlgym (25 cites) and safe-control-gym (134) | PC-Gym, BOPTEST, CityLearn and Grid2Op, each inside its field. The practitioner review found no open RL suite for PWR load follow, anti-surge control or permit-bound neutralisation | Both niches |
| What TargetGym brings | JAX speed, which gymnax, Brax and Craftax already offer | Sourced physics contracts, a written reference recipe, a labelled foresight ceiling and costs in money | Both, with one author's attention split |
| Adoption band in the evidence | Stylised control suites: 25 to 134 cites | Realistic control and energy suites: 27 to 329 cites, plus BSM and Tennessee Eastman over decades | No precedent pairs two realism levels |
| Room under the charter | Likely small. Simplified linear plants suit LQR and MPC, so a principled controller leaves little room (judgement) | Exists, and is measured against a ceiling | The arcade track would need its own rule |
| Holding still | Easier, with no real plant to chase | Possible with frozen versions, with realism going into new versions | Two things to freeze |
| Cost to TargetGym | New tasks, references, ceilings, docs and story; recording the benchmark's lines already takes days on shared servers | Already built; freeze, publish and document | The sum of both |

This study recommends option B, the realistic course, with habits borrowed from the arcade niche. The reasons are judgement on the evidence above. The arcade niche is crowded with suites that libraries already run, and an arcade suite would discard the plants that no other open suite has. Under the charter it would also struggle to keep any room, because simplification strips out the constraints, prices, disturbances and noise where a learned policy can beat a principled controller.

Option C has precedents only in a narrower form. DMC's two task sets, Procgen's easy and hard modes, Atari 100k beside 200M frames and Craftax-Classic beside Craftax each keep one task family and one score at two costs. None pairs a stylised suite with a realistic one, so the borrowing should take the narrower form.

- **A cheap tier on the same frozen tasks.** One budget, 30 seeds, shipped settings and no tuning, with its wall-clock published. The 0.9.0 RL slice, with "published default settings", is close to this already.
- **One number per task before any aggregate.** The saving over the reference, with its interval. The suite aggregate waits for more tasks.
- **A labelled debugging task.** Gymnasium keeps toy tasks "useful for debugging" ([Gymnasium](https://github.com/Farama-Foundation/Gymnasium)). TargetGym's first_order fixture could carry that label, unscored, so that users can check an agent learns at all. With PPO losing to constant actions on seven of eight tasks at 200k steps, users cannot otherwise tell a broken agent from a hard task.

What option B takes is section 7's checklist. The 0.10.0 realism batch then becomes benchmark v2 beside v1, so realism work continues without moving v1.

The owner's standing positions fit this course, with one adjustment.

| The owner's position | What the evidence says (judgement) | Effect on the recommendation |
|---|---|---|
| Learnability is not a task flaw: "it's the agent's job to solve a real task (where we know a better solution exists)" | Supported. NetHack and Craftax stayed unsolved and were adopted. Craftax launched with learned baselines and a leaderboard, and the NetHack challenge published how far neural agents fell short | Publish PPO and SAC at the protocol, even where they lose |
| Keep tasks where a good classical controller leaves room; never weaken a principled controller to make room; never claim room is reachable | Matches the go-to references: ALE's human score, D4RL's expert, L2RPN's do-nothing, BSM's default control | Keep the reference-relative headline |
| A task represents a plant family and is matched to documented real behaviour | Long-lived realistic benchmarks froze one plant (Tennessee Eastman disguised a real one) and added new models beside it (BSM1, then BSM1_LT, then BSM2) | Freeze one representative per version, and widen the family in later versions |
| A nearly saturated task stays when its room is beyond noise and worth money; glass_furnace stays whatever its room | Saturation counted against D4RL as a whole suite, while a saturated CartPole lives on as a floor | No effect on v1, whose four tasks are fixed; such tasks can join a later version |
| The charter's claim: "Beating the reference on a task means handling the kind of difficulty that plants of its family present" | A validity claim that a frozen version can carry for its representatives | Scope the claim to v1's tasks |

## 7. A goal that holds still

### Why the target kept moving

Between 2026-10-27 and 2026-11-27 the project's record shows rule-level changes on at least seven dates. A metric redesign and a capability radar came first, then an expert-tuning audit, a new oracle rule, an industrial review, measurement noise and charter draft 3. On 2026-11-08, Amendments 1 to 4 and charter Version 2 landed in one day, and the headline moved from an equal-weight "gap closed" to a frequency-weighted saving over the reference. Amendments 5 to 7 and the MVP decision followed. On 2026-11-27, shared gate rules arrived with "no new rules for 0.8.0". Later that day, after two outside reviews, four rule changes were adopted, cstr left the scored set and a new release plan was adopted. Task files show the same motion. ph_neutralization-v6's version file notes behaviour changes on 2026-11-24 and 2026-11-25 and a stamp of 2026-11-26, all under one number because no release had shipped v6. On the candidate branch read for this study, `charter.FLAGSHIP` still lists five scored tasks, cstr among them.

The go-to benchmarks changed their protocols too. Sticky actions arrived in 2018, Atari 100k in 2019, and CartPole moved to v1. Each change landed beside a frozen version, so earlier numbers kept their meaning. In this study's judgement, TargetGym has had no frozen version yet, so each change edits the only definition there is. A promise holds when a file and a test enforce it and when new ideas have somewhere else to go. The goal below supplies both, a frozen artefact with a guard and a v2 that collects every later idea.

### The written goal

> **Benchmark v1.** TargetGym Benchmark v1 measures how much a controller saves over a fixed classical reference on the plant tasks that the 0.8.0 tag scores, under one fixed protocol. It is done when every check below passes on a released package. After that, v1 never changes.

Frozen in v1:

- the task list, at the versions the 0.8.0 tag ships (on the candidate today: reactor-v10, hvac-v6, compressor_surge-v5 and ph_neutralization-v6);
- each task's plant, observation, action, cost terms, scenarios, weights, windows and evaluation seeds, as captured by its version fingerprint;
- the comparator and every recorded baseline line (under the current charter: the reference, expert, nulls, MPC and ceiling);
- the score as `tg.score` computes it (under the current charter, rule J4), with its interval methods and report fields;
- the protocol: budgets, seed counts, the tuning budget and the short tier.

### Definition of done

| # | Item | How to check it |
|---|---|---|
| 1 | One manifest names v1: the four task IDs, each fingerprint, the reference file hashes and the protocol constants | The file exists (for example `src/target_gym/data/benchmarks/v1.json`), and a test compares each fingerprint with `src/target_gym/data/env_versions/<task>-vN.json` |
| 2 | The code has one scored list | A test asserts that `charter.FLAGSHIP` equals the manifest's task list |
| 3 | `tg.score` computes the frozen score from the manifest's constants | A test scores the reference against itself and reads a ratio of exactly 1, the 0.8.0 gate |
| 4 | Every shipped line is current at the frozen fingerprints | The 0.8.0 gate reproduces every scored line in CI from the tag, and no v1 record reads "not measured" |
| 5 | A v1 results table holds every line plus PPO and SAC at the short tier, each with its interval | One script regenerates the table (for example `docs/results/benchmark-v1.md`), and no cell is empty |
| 6 | The tasks load through Gymnasium | A test calls `gymnasium.make` on each v1 ID after importing the package |
| 7 | The documents exist: a card per task, the known-limits page, and a v1 page with the goal, the protocol, the change rule and the short tier's wall-clock | The files exist and the docs build passes |
| 8 | A citable release ships v1 | A PyPI release passes the installed-package gate, extended to run checks 1 to 6 from the wheel, and `CITATION.cff` names benchmark v1 |
| 9 | A freeze guard protects v1 | A CI test fails whenever a v1 fingerprint, reference file or protocol constant changes |

### What later work may add without moving v1

- New task versions, such as the 0.10.0 realism batch, which together form benchmark v2.
- New scored tasks, with per-task numbers at once. They enter a suite aggregate only in a later benchmark version.
- New lines and agents, such as the causal MPC line, a model-based learner or the full RL grid, each as a new row in v1's table.
- Tools that keep the fingerprints: the trip wrapper, training recipes, sensitivity tables, information levels, offline datasets and speed-ups.
- The paper, the documentation and the validation studies.

**The change rule.** Any change to a frozen item makes a new benchmark version beside v1, and v1 stays as it was, installable and scorable. A bug found in v1 goes on its known-limits page and is fixed in the next version, as D4RL relabelled its datasets v2 and Gymnasium kept MuJoCo v4 beside v5 ([Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/)). Until the 0.8.0 tag, the unreleased-versions rule still lets a version change in place; after it, no version does. The plan keeps an old version scored for one more release. This study suggests keeping v1 scorable at least until 1.0.0, since frozen versions are what users cite, and leaves that to the owner.

### How it fits the adopted plan

Most of the checklist is already in the plan. The freeze, the re-record, `tg.score`, `tg.make_reference`, task cards, the known-limits page and the gate that the reference scores exactly 1 are 0.8.0 items. The PPO and SAC slice with published default settings is a 0.9.0 item, and it becomes the short tier. Five items are new and small: the manifest, the freeze guard, Gymnasium registration, the v1 page and the stability promise. The plan holds that promise back until 1.0.0. In this study's judgement, moving it to the 0.8.0 tag is the change that does most to stop the target moving. With it in force from 0.8.0, a scored version can no longer change, so 1.0.0 can concentrate on the API and the full RL grid.

The owner also adopted a development plan on 2026-11-27 (a project decision record). It puts this study first, then one vision page and one task protocol, each decided once, and locks 0.8.0 only after them. The checklist above holds under whichever protocol that page sets, because it freezes what the protocol defines rather than the charter's current rules. The draft protocol in the plan already builds in five of the eight conditions: a task frozen once released, a score in the job's own units over a fixed episode, three baselines, evidence of room from an MPC or a short RL run, and an episode under a second on a CPU. Three things would remain to add: Gymnasium registration, a v1 page that states the question until the paper exists, and published learned results at a fixed cheap tier.

The goal leaves out what the project cannot control: citations, stars, use by other groups, acceptance at a venue and competitions. Those measure adoption. They can follow from meeting the bar, and they cannot decide when v1 is done.

## 8. Sources

All web sources were accessed on 2026-11-27. Citation counts are OpenAlex readings from that day unless the text says otherwise.

**TargetGym repository and study inputs**

- TargetGym facts gathered from the repository on 2026-11-27 for this study: stars, forks, releases, task counts, timings, review scores and the record of goal changes. Accessed 2026-11-27.
- `docs/proposals/release-plan-to-1.0.md` on `origin/int/amendment3`, read with `git show`. Accessed 2026-11-27.
- The development plan adopted on 2026-11-27 (study, vision page, task protocol, then the lock), from the project's decision records; not in the repository. Accessed 2026-11-27.
- `src/target_gym/charter.py` on `origin/int/amendment3` (the `FLAGSHIP` and `FIXTURES` tuples). Accessed 2026-11-27.
- `src/target_gym/data/env_versions/ph_neutralization-v6.json` on `origin/int/amendment3` (fingerprint and re-stamp notes). Accessed 2026-11-27.
- `docs/proposals/task-charter.md` and `docs/proposals/charter-history.md` on `origin/int/amendment3`; `docs/proposals/practitioner-review.md` on `review/practitioner-2026-11-27`; `docs/proposals/rl-research-review.md` on `review/rl-research-2026-11-27`. Content as summarised in the study's repository facts. Accessed 2026-11-27.

**Citation databases**

- OpenAlex batch 1 (Gym, ALE, DQN, MuJoCo): https://api.openalex.org/works?filter=doi:10.48550/arxiv.1606.01540%7C10.48550/arxiv.1207.4708%7C10.1613/jair.3912%7C10.48550/arxiv.1312.5602%7C10.1038/nature14236%7C10.1109/IROS.2012.6386109&select=doi,title,publication_year,cited_by_count. Accessed 2026-11-27.
- OpenAlex batch 2 (PC-Gym, safe-control-gym, Tennessee Eastman): https://api.openalex.org/works?filter=doi:10.48550/arxiv.2410.22093%7C10.48550/arxiv.2109.06325%7C10.1016/0098-1354%2893%2980018-I&select=doi,title,publication_year,cited_by_count. Accessed 2026-11-27.
- OpenAlex title search for the Gymnasium paper: https://api.openalex.org/works?search=Gymnasium:%20A%20Standard%20Interface%20for%20Reinforcement%20Learning%20Environments. Accessed 2026-11-27.
- OpenAlex per-paper records used by the general-suites note: https://api.openalex.org/works?filter=doi:10.48550/arxiv.1912.01588 (Procgen), https://api.openalex.org/works?filter=doi:10.48550/arxiv.1903.03176 (MinAtar), https://api.openalex.org/works?filter=doi:10.48550/arxiv.1908.03568 (bsuite), https://api.openalex.org/works?filter=doi:10.48550/arxiv.1910.10897 (Meta-World), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2004.07219 (D4RL), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2310.12567 (Safety-Gymnasium), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2006.13760 (NLE), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2109.13202 (MiniHack), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2109.06780 (Crafter), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2402.16801 (Craftax), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2106.13281 (Brax), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2306.09884 (Jumanji), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2306.13831 (MiniGrid), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2009.14471 (PettingZoo), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2212.07489 (SMACv2), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2312.12044 (XLand-MiniGrid), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2502.08844 (MuJoCo Playground), https://api.openalex.org/works?filter=doi:10.48550/arxiv.2311.10090 (JaxMARL). Accessed 2026-11-27.
- Semantic Scholar API, Henderson et al. record: https://api.semanticscholar.org/graph/v1/paper/arXiv:1709.06560?fields=title,citationCount,year. Accessed 2026-11-27.
- Semantic Scholar API, Gym record: https://api.semanticscholar.org/graph/v1/paper/arXiv:1606.01540?fields=title,citationCount,year. Accessed 2026-11-27.

**General RL suites**

- Gymnasium repository: https://github.com/Farama-Foundation/Gymnasium. Accessed 2026-11-27.
- Towers et al., Gymnasium paper: https://arxiv.org/abs/2407.17032. Accessed 2026-11-27.
- OpenAI Gym repository (archived): https://github.com/openai/gym. Accessed 2026-11-27.
- Gymnasium CartPole documentation: https://gymnasium.farama.org/environments/classic_control/cart_pole/. Accessed 2026-11-27.
- Gymnasium MuJoCo documentation: https://gymnasium.farama.org/environments/mujoco/. Accessed 2026-11-27.
- Farama, Gymnasium 1.0 announcement: https://farama.org/Gymnasium-v1.0; MuJoCo v5 page: https://farama.org/Gymnasium-MuJoCo-v5_Environments. Accessed 2026-11-27.
- Duan et al. 2016, continuous-control benchmark: https://arxiv.org/abs/1604.06778. Accessed 2026-11-27.
- MuJoCo repository: https://github.com/google-deepmind/mujoco. Accessed 2026-11-27.
- DeepMind blog, MuJoCo acquisition (2021-10-18): https://deepmind.google/blog/opening-up-a-physics-simulator-for-robotics/; open-sourcing (2022-05-23): https://deepmind.google/blog/open-sourcing-mujoco/. Accessed 2026-11-27.
- Freethink on MuJoCo licence prices: https://www.freethink.com/?p=37825. Accessed 2026-11-27.
- Roboschool repository: https://github.com/openai/roboschool. Accessed 2026-11-27.
- Bellemare et al., Arcade Learning Environment: https://arxiv.org/abs/1207.4708. Accessed 2026-11-27.
- Machado et al. 2018, revisiting ALE: https://arxiv.org/abs/1709.06009. Accessed 2026-11-27.
- ALE repository: https://github.com/Farama-Foundation/Arcade-Learning-Environment. Accessed 2026-11-27.
- ale-py on PyPI: https://pypi.org/project/ale-py/. Accessed 2026-11-27.
- Google Research blog on DQN: https://research.google/blog/from-pixels-to-actions-human-level-control-through-deep-reinforcement-learning/; DQN Nature DOI: https://www.doi.org/10.1038/NATURE14236. Accessed 2026-11-27.
- Hessel et al., Rainbow: https://ar5iv.labs.arxiv.org/html/1710.02298. Accessed 2026-11-27.
- Kaiser et al., Atari 100k: https://ar5iv.labs.arxiv.org/html/1903.00374. Accessed 2026-11-27.
- Badia et al., Agent57: https://ar5iv.labs.arxiv.org/html/2003.13350. Accessed 2026-11-27.
- Google Research blog, cost of Rainbow: https://www.research.google/blog/reducing-the-computational-cost-of-deep-reinforcement-learning-research/. Accessed 2026-11-27.
- Beyond the Rainbow: https://arxiv.org/html/2411.03820v2. Accessed 2026-11-27.
- Tassa et al., DeepMind Control Suite: https://arxiv.org/abs/1801.00690 and https://ar5iv.labs.arxiv.org/html/1801.00690. Accessed 2026-11-27.
- dm_control repository: https://github.com/google-deepmind/dm_control; dm-control on PyPI: https://pypi.org/project/dm-control/. Accessed 2026-11-27.
- CURL and the pixel protocol: https://arxiv.org/pdf/2004.04136. Accessed 2026-11-27.
- Fujimoto et al., TD3: https://ar5iv.labs.arxiv.org/html/1802.09477. Accessed 2026-11-27.
- Haarnoja et al., SAC: https://ar5iv.labs.arxiv.org/html/1801.01290. Accessed 2026-11-27.
- Procgen repository via GitHub API: https://api.github.com/repos/openai/procgen; Cobbe et al.: https://arxiv.org/abs/1912.01588 and https://arxiv.org/pdf/1912.01588; Impoola reuse of the score: https://arxiv.org/pdf/2503.05546. Accessed 2026-11-27.
- MinAtar via GitHub API: https://api.github.com/repos/kenjyoung/MinAtar; paper: https://arxiv.org/abs/1903.03176. Accessed 2026-11-27.
- bsuite via GitHub API: https://api.github.com/repos/google-deepmind/bsuite; PyPI: https://pypi.org/pypi/bsuite/json; paper: https://arxiv.org/abs/1908.03568; README: https://raw.githubusercontent.com/google-deepmind/bsuite/main/README.md; search example: https://arxiv.org/html/2309.11975v1. Accessed 2026-11-27.
- Meta-World via GitHub API: https://api.github.com/repos/Farama-Foundation/Metaworld; README: https://raw.githubusercontent.com/Farama-Foundation/Metaworld/master/README.md. Accessed 2026-11-27.
- D4RL via GitHub API: https://api.github.com/repos/Farama-Foundation/D4RL; paper: https://arxiv.org/abs/2004.07219; README: https://raw.githubusercontent.com/Farama-Foundation/D4RL/master/README.md; reported flaws: https://arxiv.org/pdf/2306.03286 and https://arxiv.org/pdf/2408.08441; Gymnasium-Robotics: https://github.com/Farama-Foundation/Gymnasium-Robotics. Accessed 2026-11-27.
- Minari via GitHub API: https://api.github.com/repos/Farama-Foundation/Minari; documentation: https://minari.farama.org/. Accessed 2026-11-27.
- Safety Gym via GitHub API: https://api.github.com/repos/openai/safety-gym; OpenAI report: https://cdn.openai.com/safexp-short.pdf. Accessed 2026-11-27.
- Safety-Gymnasium via GitHub API: https://api.github.com/repos/PKU-Alignment/safety-gymnasium; paper: https://arxiv.org/abs/2310.12567. Accessed 2026-11-27.
- NLE via GitHub API: https://api.github.com/repos/facebookresearch/nle and https://api.github.com/repos/heiner/nle; paper: https://arxiv.org/abs/2006.13760; README: https://raw.githubusercontent.com/facebookresearch/nle/main/README.md; Hambro et al., NetHack Challenge: https://arxiv.org/pdf/2203.11889. Accessed 2026-11-27.
- MiniHack via GitHub API: https://api.github.com/repos/facebookresearch/minihack. Accessed 2026-11-27.
- Crafter via GitHub API: https://api.github.com/repos/danijar/crafter. Accessed 2026-11-27.
- Craftax via GitHub API: https://api.github.com/repos/MichaelTMatthews/Craftax; paper: https://arxiv.org/abs/2402.16801; README: https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/README.md. Accessed 2026-11-27.
- Brax via GitHub API: https://api.github.com/repos/google/brax; paper: https://arxiv.org/abs/2106.13281. Accessed 2026-11-27.
- gymnax via GitHub API: https://api.github.com/repos/RobertTLange/gymnax; README: https://raw.githubusercontent.com/RobertTLange/gymnax/main/README.md; PureJaxRL via GitHub API: https://api.github.com/repos/luchris429/purejaxrl. Accessed 2026-11-27.
- Jumanji via GitHub API: https://api.github.com/repos/instadeepai/jumanji; paper: https://arxiv.org/abs/2306.09884. Accessed 2026-11-27.
- MiniGrid via GitHub API: https://api.github.com/repos/Farama-Foundation/Minigrid. Accessed 2026-11-27.
- PettingZoo via GitHub API: https://api.github.com/repos/Farama-Foundation/PettingZoo. Accessed 2026-11-27.
- SMAC and SMACv2 via GitHub API: https://api.github.com/repos/oxwhirl/smac and https://api.github.com/repos/oxwhirl/smacv2. Accessed 2026-11-27.
- XLand-MiniGrid via GitHub API: https://api.github.com/repos/dunnolab/xland-minigrid; paper: https://arxiv.org/abs/2312.12044. Accessed 2026-11-27.
- MuJoCo Playground via GitHub API: https://api.github.com/repos/google-deepmind/mujoco_playground; paper: https://arxiv.org/abs/2502.08844; README: https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/README.md. Accessed 2026-11-27.
- JaxMARL via GitHub API: https://api.github.com/repos/FLAIROx/JaxMARL; paper: https://arxiv.org/abs/2311.10090. Accessed 2026-11-27.
- RL Competition 2008 report, AI Magazine: https://ojs.aaai.org/index.php/aimagazine/article/view/2227. Accessed 2026-11-27.
- Koppejan and Whiteson, Oxford publication page: https://www.cs.ox.ac.uk/publications/publication9794-abstract.html; Whiteson et al. 2011 on environment overfitting: https://www.cs.ox.ac.uk/publications/publication9770-abstract.html. Accessed 2026-11-27.
- RL Competition 2014 report: https://typeset.io/pdf/the-reinforcement-learning-competition-2014-3q7s7fzqsh.pdf. Accessed 2026-11-27.
- Universe repository: https://github.com/openai/universe; Gym Retro repository: https://github.com/openai/retro. Accessed 2026-11-27.

**Libraries and integration**

- RL Baselines3 Zoo benchmark page: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md. Accessed 2026-11-27.
- CleanRL repository: https://github.com/vwxyzjn/cleanrl; overview: https://docs.cleanrl.dev/rl-algorithms/overview/; PPO: https://docs.cleanrl.dev/rl-algorithms/ppo/; DQN: https://docs.cleanrl.dev/rl-algorithms/dqn/. Accessed 2026-11-27.
- EnvPool repository: https://github.com/sail-sg/envpool; README: https://raw.githubusercontent.com/sail-sg/envpool/main/README.md. Accessed 2026-11-27.
- Open RL Benchmark: https://arxiv.org/html/2402.03046. Accessed 2026-11-27.
- RLlib examples: https://docs.ray.io/en/latest/rllib/rllib-examples.html. Accessed 2026-11-27.
- Atari PPO throughput in another setup: https://arxiv.org/pdf/2206.10558. Accessed 2026-11-27.

**Control and industrial suites**

- PC-Gym paper: https://arxiv.org/abs/2410.22093; HTML v4: https://arxiv.org/html/2410.22093v4. Accessed 2026-11-27.
- PC-Gym on PyPI: https://pypi.org/project/pcgym/ and https://pypi.org/project/pcgym/0.1.1. Accessed 2026-11-27.
- PC-Gym repository: https://github.com/MaximilianB2/pc-gym; GitHub API: https://api.github.com/repos/MaximilianB2/pc-gym; issues: https://api.github.com/repos/MaximilianB2/pc-gym/issues?state=all&per_page=30; commits: https://api.github.com/repos/MaximilianB2/pc-gym/commits?per_page=5. Accessed 2026-11-27.
- PC-Gym citing papers, Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/arXiv:2410.22093/citations?fields=title,year&limit=100. Accessed 2026-11-27.
- Bloor et al., survey of RL in process systems engineering: https://arxiv.org/html/2510.24272. Accessed 2026-11-27.
- safe-control-gym paper: https://arxiv.org/abs/2109.06325; author page: https://adamhall.github.io/publication/2022-08-03-safe-control-gym; repository: https://github.com/utiasDSL/safe-control-gym; GitHub API: https://api.github.com/repos/utiasDSL/safe-control-gym. Accessed 2026-11-27.
- controlgym paper: https://arxiv.org/abs/2311.18736; PMLR: https://proceedings.mlr.press/v242/zhang24b.html; repository: https://github.com/xiangyuan-zhang/controlgym. Accessed 2026-11-27.
- gym-electric-motor repository: https://github.com/upb-lea/gym-electric-motor; GitHub API: https://api.github.com/repos/upb-lea/gym-electric-motor; JOSS paper: https://joss.theoj.org/papers/10.21105/joss.02498; RL paper: https://pdf.arxiv.org/pdf/1910.09434. Accessed 2026-11-27.
- Industrial Benchmark paper: https://arxiv.org/abs/1709.09480; repository: https://github.com/siemens/industrialbenchmark; GitHub API: https://api.github.com/repos/siemens/industrialbenchmark. Accessed 2026-11-27.
- Agyei et al. 2025, deep RL in applied control: https://arxiv.org/pdf/2507.08196. Accessed 2026-11-27.
- Devarakonda et al. 2025, review of RL for chemical process control: https://www.mdpi.com/2227-9717/13/6/1791. Accessed 2026-11-27.

**Energy and grid suites**

- BOPTEST site: https://ibpsa.github.io/project1-boptest/; PNNL record of Blum et al. 2021: https://www.pnnl.gov/publications/building-optimization-testing-framework-boptest-simulation-based-benchmarking-control; repository: https://github.com/ibpsa/project1-boptest. Accessed 2026-11-27.
- BOPTEST-Gym listing: https://awesome.ecosyste.ms/projects/github.com%2Fibpsa%2Fproject1-boptest-gym. Accessed 2026-11-27.
- Beobench environments: https://beobench.readthedocs.io/en/stable/envs.html. Accessed 2026-11-27.
- Sinergym paper: https://arxiv.org/html/2412.08293v1; repository: https://github.com/ugr-sail/sinergym. Accessed 2026-11-27.
- CityLearn paper: https://arxiv.org/abs/2012.10504; repository: https://github.com/intelligent-environments-lab/CityLearn. Accessed 2026-11-27.
- CityLearn Challenge coverage: https://cockrell.utexas.edu/news/researchers-challenged-to-optimize-building-energy-use-with-ai/, https://cockrell.utexas.edu/?p=23836, https://research.tue.nl/en/publications/the-citylearn-challenge-2021/, https://neurips.cc/virtual/2023/competition/66590, https://discourse.aicrowd.com/t/winner-announcement/9387. Accessed 2026-11-27.
- Marot et al., L2RPN retrospective: https://ar5iv.labs.arxiv.org/html/2103.03104; Grid2Op repository: https://github.com/Grid2Op/grid2op. Accessed 2026-11-27.
- Gym-ANM paper: https://arxiv.org/abs/2103.07932v2; repository: https://github.com/robinhenry/gym-anm. Accessed 2026-11-27.
- PowerGridworld paper: https://ar5iv.labs.arxiv.org/html/2111.05969. Accessed 2026-11-27.

**Control-community classics**

- Tennessee Eastman listing: https://cs.paperswithcode.com/dataset/tennessee-eastman-process; Ricker archive: https://depts.washington.edu/control/LARRY/TE/download.html; recent uses: https://arxiv.org/html/2401.10266v3, https://arxiv.org/abs/2308.11247, https://arxiv.org/pdf/2310.14788v1.pdf. Accessed 2026-11-27.
- BSM2 record, Lund University: https://lup.lub.lu.se/search/publication/61891cd7-6fd7-4db7-8bc3-a95fc8166b19; BSM repository: https://github.com/wwtmodels/Benchmark-Simulation-Models; RL results list: https://www.mdpi.com/2076-3417/13/8/4752. Accessed 2026-11-27.
- Shell heavy oil fractionator: https://data.zhenzhi.com/details/1/985671400, https://aiche.org/conferences/aiche-spring-meeting-and-global-congress-on-process-safety/2011/proceeding/paper/61a-performance-investigation-constrained-model-predictive-control-shell-fractionater-problem-0, https://www.mathworks.com/matlabcentral/fileexchange/65888-shell-control-problem. Accessed 2026-11-27.
- IFAC PID'12 boiler benchmark: https://skoge.folk.ntnu.no/prost/proceedings/PID-12/papers/0018.pdf; Astrom and Hagglund benchmark systems: https://lup.lub.lu.se/record/537934. Accessed 2026-11-27.

**Meta-research on evaluation and adoption**

- Henderson et al. 2018: https://arxiv.org/abs/1709.06560 and https://ar5iv.labs.arxiv.org/html/1709.06560. Accessed 2026-11-27.
- Agarwal et al. 2021: https://arxiv.org/abs/2108.13264. Accessed 2026-11-27.
- Dehghani et al. 2021, The Benchmark Lottery: https://arxiv.org/abs/2107.07002. Accessed 2026-11-27.
- Koch et al. 2021, dataset concentration: https://arxiv.org/abs/2112.01716. Accessed 2026-11-27.
- Patterson et al., empirical design in RL: https://arxiv.org/abs/2304.01315. Accessed 2026-11-27.
- Benchmark choice in RL (2024): https://arxiv.org/pdf/2410.08870v1. Accessed 2026-11-27.
- Stapelberg and Malan 2020, survey of benchmark tasks: https://arxiv.org/pdf/2011.13577.pdf. Accessed 2026-11-27.
- Kirk et al., survey of generalisation in RL: https://arxiv.org/pdf/2111.09794v4. Accessed 2026-11-27.
- Arulkumaran et al. 2017, survey of deep RL: https://arxiv.org/pdf/1708.05866. Accessed 2026-11-27.
