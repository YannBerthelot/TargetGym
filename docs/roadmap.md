# Roadmap and known gaps

Moved out of the README, which is the shop window rather than the planning
board. This is the honest state of the project: what is done, what is next, and
what is broken and recorded rather than hidden.

## Roadmap

* [x] Mature the glass furnace and reactor environments (physics, reward shaping, episode lengths).
* [x] Document and test every environment's physics against published data.
* [x] Rebuild every renderer on a shared control-room toolkit, and regenerate
      the gallery clips against it.
* [x] Restore the Plane Patrol baselines with pursuit guidance. Done: both
      variants ship a stateful wrapper around the functional pursuit expert,
      and `patrol_bearing_only` adds a lead-state estimator in front of the
      same law. Neither has an MPC yet; see *Baseline coverage*.
* [ ] Add microburst / spatially-varying wind fields (position-dependent, not just altitude-linear).
* [ ] Provide benchmark results for popular RL baselines.
* [ ] Add random orientation variations to circle and heading tasks.

### Before 1.0

* [ ] **Ship 0.6, the first release meant to be advertised.**
      Deliberately 0.6 rather than 0.9: there will likely be further pre-1.0
      releases, and numbering as though 1.0 were imminent would promise a
      freeze that has not been decided. What 0.6 is for is getting the library
      in front of people and finding out what breaks.

      Comparable projects reached 1.0 by freezing an API rather than by
      finishing features. Gymnasium v1.0 says it "marks the end of major changes
      to the project's central API", and its content was mostly removal and
      consolidation; gymnax 1.0 made the break it knew it needed, shipped a
      legacy wrapper, and promised not to do it again. Both ran a feedback
      period first, Gymnasium's from v1.0.0a1 in February 2024 to v1.0.0 that
      October. 0.6 starts that clock here, as a normal release rather than a
      pre-release, because `pip install` skips pre-releases and this package
      needs the feedback more than the ceremony.

      Everything below is doable on the machine this project is developed on.
      None of it needs a GPU farm, which is the point: the large training grid
      is a later release and is blocked on compute this project does not have.

      **Blocking, and fatal on first contact if missed:**

      - [ ] **Release before advertising.** PyPI is at 0.5.0. The 21
        environments, the restructured README, the gallery and the versioning
        are all unreleased, so anyone acting on a post today installs the old
        package. This also clears the stale PyPI summary, which still mentions a
        "Car" environment that does not exist.
      - [x] **Environment versioning.** Done: `EnvSpec.version`,
        `spec.versioned_name`, `src/target_gym/data/env_versions.json`, and
        `tests/test_env_versions.py`, which fails when an environment's
        fingerprint moves without its version being bumped. Every environment
        ships as `v1`.
      - [x] **Baselines re-recorded and green.** Done: all twenty environments
        with an MPC, recorded in one 39-minute run, every MPC leading its PID,
        no environment terminating early on any seed, and all seven CasADi
        plants reporting 100% solver convergence across 119 600 solves.
        `EnvSpec.mpc_degraded` is empty for the first time, both entries retired
        by fixing their cause. The furnace, which began at 16% behind its own
        PID with one seed running seventy minutes against a three-minute
        median, now leads on all ten seeds.
      - [x] **Host the documentation.** Done: `.github/workflows/docs-deploy.yml`
        builds with `mkdocs build --strict` and deploys via
        `actions/deploy-pages@v4`, with a `workflow_dispatch` trigger so the
        site can be published without a commit. The repository's Pages source
        is set to GitHub Actions.
      - [x] **`CODE_OF_CONDUCT.md`.** Done: Contributor Covenant 2.1
        verbatim, with the maintainer's address as the reporting contact,
        linked from `CONTRIBUTING.md`. Verbatim because it is the text
        GitHub's community profile recognises and contributors already
        know; a house rewording would be neither.

      **Baseline gaps to close or document honestly:**

      - [x] **`plane_steps` MPC terminates on all ten seeds** (656 against the
        PID's 1593). Fixed, by two separate defects, and it was never the
        moving setpoint: freezing the target reproduced the first crash
        exactly, with the aircraft on the ground at t=372 and the first tread
        change still 28 s away.

        The first defect is in the objective. The altitude reward scores one
        thing and the aircraft has two actuators, so over a 90 s planning
        window the best move is a zoom climb, trading airspeed for altitude
        faster than the engines can supply it. It touched the commanded
        altitude at t=90 with 30 m/s of airspeed left and departed.
        `_plane_objective` now carries a barrier on airspeed against the stall
        speed at that mass and altitude, the pattern `make_wind_turbine_mpc`
        already uses. Fencing angle of attack instead does not work: it sits at
        4-8 deg through the whole manoeuvre and crosses 15 deg one step before
        the departure.

        The second defect is in the search, it is the more serious of the two,
        and it belongs to `GradientMPC` rather than to the aircraft. Saturating
        actuators are written with `clip` or `maximum`, whose derivative at the
        kink is exactly zero, and `_optimize` projected onto the closed
        interval, so any overshooting step parked an action exactly on a bound
        and it could never move again. Measured at the plan where the aircraft
        gave up: the true one-sided slope in thrust is +3.0 and autodiff
        returns 0.0, with thrust pinned at -1.000 for 800 steps while the
        elevator went on being optimised normally. Holding iterates 1e-3 inside
        the bounds fixes it, which is what interior-point solvers do and for
        this reason. Twelve of the twenty environments with an MPC use this
        optimiser, so any of them with a saturating actuator could have been
        silently sitting at a limit. Re-recording says none of the four plants
        among them was: they moved by under half a point of return, and
        distillation not at all. It was latent there and real on the aircraft.

        Worth recording that four other explanations were argued from a
        plausible mechanism and refuted by the next measurement: the moving
        setpoint, an energy trade the planner would not make, fuel exhaustion,
        and NaN gradients. Sampling the objective on a grid, which is what
        finally located it, cost five minutes and should have come first.
      - [x] **Tune the `plane3d_racetrack` PID.** Done, and the settled
        cross-track error goes from 3.02 km to 0.31 km against an 8.4 km turn
        radius, with the return from 269.2 to 319.0. The `expert_degraded`
        note is gone: it now clears the effectiveness contract on its own.

        Two things were in the way. The class declared `obs_value_index` and
        no `obs_target_index`, so `runners.rollout` raised on it: the search
        scores candidates inside a `try` and reported every one as `-inf`,
        finishing successfully having changed nothing, and no baseline could
        have been recorded for the environment either. The conformance suite
        now checks both indices across the registry, where the tests that
        covered this named three classes by hand.

        And left free, the search stiffens the roll loop and removes its
        damping, because nothing in the reward objects to how the aircraft is
        banked: +21 of return for an achieved bank of 48 deg against a 30 deg
        command limit, and worse tracking. The roll gains are held, and the
        whole improvement is in the cross-track gain.

      **Presentation, before anyone looks at it:**

      - [x] **The media is stale.** Done: all 21 clips re-rendered, the console
        ones re-quantised through `scripts/make_gallery_clips.py`, and the five
        mosaics rebuilt. Nineteen `mpc_output*.gif` files were deleted with
        them: nothing referenced them, not the page generator, not the mosaics,
        not the Makefile, and they were 114 MB of a 246 MB directory. The
        flagship captions also lost their jargon ("non-minimum phase" told a
        reader nothing; the four-tank's problem is that the obvious valve
        pairing is unstable).
      - [x] **The README is too long.** Done: 410 lines and 3199 words to 331
        and 1718. Reference material moved rather than deleted (the validation
        findings to `PHYSICS_METHODOLOGY.md`, the complexity ladder to its own
        page, the throughput table to a link), the second person removed
        entirely, and the quickstart rewritten on the direct `Plane()` API
        instead of the registry. Four wrong numbers came out with it, including
        a throughput range this roadmap's own trim had invented.
      - [x] **Publishable media reaches the site.** Done, and it was broken:
        `docs/videos` is a symlink into a directory whose `*.gif` files are
        gitignored, so a local `mkdocs build --strict` passed on a working tree
        that happened to have the clips while a CI build from a clean checkout
        would have published twenty-one environment pages of broken images.
        `docs-deploy` now renders them before building. Only the five mosaics
        are committed, which is what the README and the environment index embed.

      **Worth doing, cheap:**

      - [x] A Colab or notebook linked from the README. Done:
        `notebooks/quickstart.ipynb`, with a badge in the README, a link from
        `docs/index.md`, and `tests/test_docs.py` executing it so it cannot
        rot. gymnax and Brax both lead with one, and it turns a reader into a
        user in a click.
      - [x] Decide the `4 - Beta` classifier question. Decided: 0.6 ships as
        `4 - Beta`. Twenty-two environments, a versioning scheme with a test
        enforcing it, a documented public API contract and a conformance suite
        that runs against every environment are not what `3 - Alpha` describes
        to someone scanning PyPI. It is not a promise of an API freeze, which
        stays explicitly deferred past 0.6.

      Left for a later release, explicitly: every learned-policy number, the
      hyperparameter search, and the API freeze decisions (whether `Plane` or
      `Airplane2D` is canonical, whether `step_env` stays public alongside
      `step`, any remaining observation-layout changes).

      That includes the preliminary CPU slice that used to be planned here
      (five environments, PPO and SAC, tabula-rasa, budgets 1e5 and 1e6, 30
      seeds, published defaults). It is out of 0.6 by decision, not by
      accident. What 0.6 is for is putting the environments and their expert
      baselines in front of people; a small, deliberately underpowered RL
      result is the one number a reader would over-read, and publishing it
      alongside "no results yet" invites exactly the comparison the protocol
      was written to prevent. `docs/rl-baselines.md` saying the harness exists
      and nothing is published is a cleaner claim than a preliminary table
      hedged with caveats.

* [ ] **Scope the baseline fingerprint to the code each environment reaches.**
      `provenance.baseline_fingerprint` hashes the whole of `experts/pid.py` and
      `experts/mpc.py`, so *adding* a controller for a new environment marks
      every existing environment's record stale. Registering the racetrack hold
      added 93 lines to `pid.py` and deleted none, and that alone invalidated
      all twenty recorded baselines, though no existing controller's behaviour moved.
      Partly done: each task's oracle now lives in its own package's
      `experts.py`, so oracle work re-records only that package. The PIDs of
      the 21 and the shared MPC machinery still invalidate everything.
      Re-recording is cheap enough today (the aircraft variants dominate the
      cost and needed re-recording regardless), so this was paid rather than
      fixed, but it scales badly: it is a full re-measure per environment added.
      The fix is to hash only the definitions an environment's own controllers
      reach, transitively, plus module-level constants. It must stay
      conservative, because the module's whole point is that it may cry stale when
      nothing changed but must never report fresh when something did. So an
      unresolvable entry point has to fall back to hashing the entire file.

* [x] **Remove the `utils.py` helpers nothing calls, with the next re-record.**
      Done, without waiting for another change to pay for the re-record. The
      helpers, their tests, the two imports, the allowlist entry and the stale
      mentions are gone. The 20 baselines and 21 protocol rows were re-recorded
      and matched the previous records bit for bit apart from their
      fingerprints, and the 21 version stamps were re-stamped in place with a
      note saying so. What follows is the item as it was written.

      `load_or_run_mpc_episode`, `save_comparison_gif`, `save_comparison_figure`
      and `load_or_build_interpolator` lost their last callers when 5458229
      replaced the seven per-environment runners with `runners/runners.py`.
      Only `tests/test_runners.py` and `tests/test_utils.py` still call them,
      and `save_comparison_figure` has no caller at all. `run_episode_headless`
      and `run_episode_headless_with_state` are only reachable through them.

      Deleting them changes no behaviour, but `utils.py` is hashed into both
      provenance fingerprints. On the 21-environment registry the deletion
      would stale all 21 version stamps, all 20 recorded baselines and all 21
      protocol rows, which is about two hours of re-measurement to reproduce
      the same numbers. So they go in the next change that has to re-record
      anyway, or once fingerprints hash only the code each environment reaches
      (the item above). That change should also drop the `pickle` and
      `FigureCanvasAgg` imports, which nothing else in `utils.py` uses, the
      helpers' tests, the `("utils.py", "_, rewards = result")` allowlist entry
      in `tests/test_env_conformance.py`, and the mentions of
      `run_episode_headless` in `plane/dynamics.py` and in that test file. It
      needs a CHANGELOG entry, because `target_gym.utils` is importable.
      `moviepy` stays, since `save_video` uses it.

* [ ] **Find a defensible framing for running cost, then put it back.**
      Six environments carried a consumption term in their reward -- fuel on
      the glass furnace, the boiler drum and the cement kiln, energy on the
      building, reboiler duty on the column, and reagent on the pH loop. All six
      are zero for the 0.6 line, and those tasks score setpoint tracking alone.

      The battery is not among them, though it was for a few hours. Its
      `cost_weight` gates degradation and state-of-charge comfort rather than
      pricing an input, and those keep the problem well posed rather than
      trading against it -- which is the distinction this item has to get right
      before restoring any weight anywhere. The fields and the terms are still there, so
      restoring a weight is a one-line change once there is a reason to pick a
      particular one.

      Running cost is real: nobody operates a furnace without caring what the
      gas costs. The problem is the *weight*. Against a tracking term that is
      already normalised into [0, 1], a cost weight silently chooses a point on
      a Pareto front, and none of the seven had an argument behind its number.
      The glass furnace showed what that costs: a 0.1 fuel weight against the
      MPC's quadratic surrogate made a 3.3 K standing error optimal, so the
      controller sat 6 K cold with fuel at minimum 80% of the time and lost to
      its own PID. The reward and the controller disagreed about the trade, and
      both were defensible readings of an undefined one.

      What a proper framing needs: the two terms in commensurable units rather
      than one normalised and one priced, so the exchange rate is a physical
      statement instead of a tuning constant. For the furnace that is money per
      kelvin-hour of off-spec glass against money per GJ of gas, both of which
      are quotable. It also needs the controller's surrogate to inherit the same
      exchange rate rather than re-deriving it, which is the band item above.

* [ ] **Justify the glass furnace's episode length with something measurable.**
      Its 1600 steps and the 0.99938 discount derived from them rest on a crown
      response of 132 steps, and the physics no longer has one. Since it gained a
      regenerator, a reversal cycle, a 120 s thermocouple lag and a two-step fuel
      dead time, the open-loop step response does not settle at all on this
      timescale: 822 steps at a 1600-step probe window, 1520 at 3200, 2383 at
      5760, 3415 at 11520. It keeps growing with the window, which means there is
      no time constant to quote rather than a large one.

      This does not invalidate the recorded return, which is measured over a
      fixed 1600-step episode either way. What it invalidates is the *reason*
      that episode is 1600. Something has to replace it: the setpoint schedule
      (the crown target steps `N_SETPOINTS` times, so an episode could be defined
      as some number of schedule segments), or the reversal period, or a settling
      criterion measured on the closed loop rather than open loop.

      An attempt to enforce the episode-length rule as a test was withdrawn for
      related reasons, recorded in the episode-length section of
      [rl-protocol.md](rl-protocol.md): `tau_actuator` does not exist for two
      thirds of this suite, and measuring it inside the episode under judgement
      makes the criterion circular.

* [ ] **Measurement noise, anywhere.** Not one of the environments has any.
      Every controller in the suite reads the exact state, filtered only where a
      sensor lag was modelled deliberately (the glass furnace's crown
      thermocouple, and nothing else). Real instruments are noisy, and a
      benchmark that asks "can a learned policy hold a setpoint more precisely
      than a PID" without noise is asking it in the one regime where derivative
      action is free. This is a 1.0 item rather than a 0.6 one: adding noise
      invalidates every baseline and every tuned gain, and the tuning has to be
      redone against it rather than ported.

* [ ] **Cullet ratio on the glass furnace.** Declared as deviation D4 in its
      PHYSICS.md. Batch is a mix of raw materials and recycled glass, and cullet
      melts with roughly 2.5 % less energy per 10 % of the charge; real plants
      see it move by tens of percent as supply changes, usually without
      measuring it well. It is a *gain* disturbance, which is qualitatively
      unlike every disturbance the suite currently carries: integral action
      cancels a load and does not cancel a gain error. It would be the first of
      its kind here.

* [ ] **Transport delay on the aircraft, if it is warranted.** The furnace had
      no dead time and a PID held it thirty times tighter than a real furnace is
      held; the aircraft has the same gap, with first-order lags on throttle and
      elevator but no transport delay, and a real turbofan's 5-8 s spool-up is
      flattered by a single lag. The difference is that the aircraft is a fast
      plant given a slow task, so bandwidth is probably not what binds, and its
      PID is nowhere near the ceiling. Measure before changing anything.

* [ ] **Stop the controller models drifting from the plants.** Seven
      environments give their MPC a separate symbolic re-implementation in
      do-mpc rather than differentiating ``step_env``. That is the right call
      where a faithful symbolic model exists -- it buys a properly constrained
      NLP instead of projected gradient descent, native DAE support for the
      furnace's algebraic flame, and convergence in 15-20 sparse iterations
      instead of 50 differentiated rollouts. What it costs is a second copy of
      the plant, and the glass furnace's has now diverged from its environment
      three separate times: an error band inherited from a deleted reward, a
      regenerator coarsened without the observation changing, and batch charging
      the controller does not know is pulsed. Nothing detects any of that. The
      fix is a conformance check per environment: step both models from the same
      state under the same input and assert the one-step predictions agree to a
      stated tolerance, so a divergence is a test failure rather than a slow
      loss of baseline quality.

* [x] **A check for code that has drifted from its own documentation.**
      These contracts are what a reader consults *instead of* the code, so a
      stale one does not merely age, it actively misinforms. Reviewing all
      twenty-one environments found this to be the repository's most common
      defect, and three of that review's own findings were wrong because of it:
      a deviation claiming post-stall lift decays to zero when the fix had long
      been implemented and a duplicate was written and reverted on the strength
      of it; a docstring saying sub-step rewards are summed when the code takes
      their mean; and a deviation crediting a reward fix to a band the reward
      does not read.

      **Done**, as `scripts/check_doc_drift.py`, run by `make ci-docs`, by CI
      and by `tests/test_docs.py`. Four checks, all mechanical: a contract
      naming a symbol its package no longer defines, allowing sentences that
      say a thing *used to* exist; a parameter whose comment claims the reward
      uses it when no reward function references it; an episode-length comment
      quoting a number that disagrees with the value beside it; and a comment
      containing a second `#`, which is the whole signature of the formatter
      merge that damaged five contract files.

      On its first run, after the manual review had already swept by hand, it
      found three more: `slot_tolerance` still driving a Gaussian in
      `patrol/marl.py` after the single-agent reward migrated away from one an
      hour earlier, a racetrack comment quoting 900 steps against a value of
      650, and the doubled `# bar` the manual pass had noticed and not fixed.

      The fifth check, a deviation whose defect no longer appears in the code,
      needs judgement and is deliberately left out rather than approximated
      badly. The two items below are how to reach it.

* [x] **Generate the derived numbers into each `PHYSICS.md`.**
      `docs/environments.md` has never drifted, because a generator writes it
      and CI checks it. Nothing else gets that treatment, and the drift is
      concentrated where it is missing.

      Extend the generator to emit a per-environment facts block between
      markers, computed from the registry and the params: episode length in
      steps, seconds and time constants; `delta_t`; action bounds; the
      observation layout with which state fields are hidden; terminal
      conditions; the reward's envelope and floor. Then `--check` fails when a
      file's block no longer matches.

      This is *less* maintenance than today, not more: generated text cannot be
      wrong and nobody has to update it.

      **Done**, as `scripts/generate_physics_facts.py`, writing a block into all
      fifteen contracts between markers and checked by `make ci-docs`, CI and
      `tests/test_docs.py`. It carries episode length in steps and in wall time,
      `delta_t`, the action dimension and bounds, the observation width and the
      float-state count, whose gap is what the controller cannot see.

      Deliberately narrow. Only facts that can be *computed* go in the block;
      everything a contract says about why a number is what it is, what the
      plant is, what is omitted and where the model stops being valid stays
      hand-written, because none of it can be derived and all of it is the point
      of the document. Terminal conditions and reward envelopes are not in there
      for that reason: they are expressed as code, not as data, and generating
      them would mean parsing rather than reading.

* [ ] **Make deviations testable, the way validation rows already are.**
      Every row of a validation table names a test that asserts it, and no
      validation row has drifted. Deviations name nothing, and they have
      drifted badly. That is not a coincidence.

      Require each ⚠️ or ❌ deviation to cite a test **demonstrating the
      limitation still exists**. Then repairing the code breaks the test, and a
      deviation cannot outlive its own fix. Both of the worst cases would have
      been caught automatically: the post-stall note the day the Viterna blend
      landed, and the reactor's reward note when the sum became a mean.

      Cost is one test per deviation, and there are roughly sixty across
      twenty-one contracts, so this is a real piece of work rather than an
      afternoon. The honest limit is worth stating too: this forces a deviation
      to stay *true*, but nothing can force its *rationale* to stay accurate.
      The goal is to shrink the surface the drift checker has to police, not to
      eliminate it.

* [ ] **Derive the MPC error bands rather than choosing them one at a time.**
      Every MPC objective normalises its tracking error by a per-plant band:
      `tracking_band` on the four-tank, the distillation column, the pH loop and
      now the glass furnace, `power_band` on the turbine and the battery,
      `comfort_band`, `lime_band`, `reward_band`. The surrogate is deliberate
      and measured -- the log-scaled reward's gradient decays like `1/e`, so the
      pull toward the setpoint is weakest where the controller is furthest from
      it, and a quadratic in the normalised error scored 341.9 against 172.1 for
      the reward itself on the turbine. What is not deliberate is the *sizing*.

      The glass furnace showed what that costs. Its band was `tracking_scale`,
      40 K, inherited from the reward it had before the log-scaled one, and left
      unread by the plant when the reward changed. With the loop operating at
      about 1 K the tracking term was 6e-4 against an O(1) fuel penalty, so the
      objective was nearly flat in the direction being scored: the MPC trailed
      its own PID by 16% on 10 of 10 seeds and IPOPT needed 349 iterations a
      step on the worst one. Re-sized to 10 K, with the controller's regenerator
      coarsened at the same time, it runs 43x faster and beats the PID.

      Measuring each band against the error its PID actually holds puts the
      suite between 0.1x and 45x, with no convention visible. The pass is to
      derive the band from the log reward's own discriminating region -- the
      error at which the tracking term halves, which is a function of
      `precision_floor` and the envelope both already declared -- and to check
      each plant's against it. Two rows of that comparison, the battery and the
      turbine, first need their tracked observation expressed in physical units
      rather than normalised ones, or the ratio means nothing: both report power
      in MW in the observation while their bands are in W, so the measured ratio
      came out a million times off and was discarded. Take the error from the
      state rather than the observation.

      The measured ratios where the units did line up, band against the error
      the PID actually holds: glass furnace 45, distillation 35, pH 13.5,
      four-tank 10.5, reactor 1.3, cement kiln 0.1, HVAC 0.5. Both mis-scaling
      directions are represented. The furnace's band was far too large, which
      flattened its objective; the kiln's is fifteen times *smaller* than its
      operating error, which saturates the surrogate instead. Only the reactor's
      is sized to its own loop.

* [ ] **Stronger model-based baselines: scenario and oracle MPC.**
      Scoped in [certainty-equivalence-study.md](certainty-equivalence-study.md),
      which is the detail; this is the summary and the ordering.

      The point is that "RL beats MPC" is a weak claim when the MPC is
      deterministic. Replacing the disturbance by its mean is exactly optimal
      under LQG assumptions and breaks on constraints, non-quadratic costs,
      nonlinearity and the value of information. Inserting a scenario MPC
      between the two splits the gap into *the value of accounting for
      uncertainty* and *what is left for learning*, and an oracle MPC given the
      disturbance realisation in advance gives the achievable ceiling, so a
      good policy can be told apart from a nearly-saturated problem.

      The scoping found the plants are already stochastic, and that the house
      convention of deriving noise as `fold_in(key, state.time)` makes a
      realisation a pure function of key and time, independent of the actions.
      That makes the oracle arm nearly free on the JAX planners. In rough
      order of effort:

      - [ ] **Distributional metrics.** The point of the study is that the
        *mean* is where certainty equivalence looks fine, and nothing here
        reports a tail. `runners.rollout` already returns what IAE, overshoot
        and settling time need; violation rate and quantiles do not exist. Ten
        seeds is also thin for a tail.
      - [ ] **Oracle arm on `SamplingMPC`.** The premise here has inverted, and
        the warning in the old wording was righter than it knew. Because
        `rollout` drives the plant with `PRNGKey(seed)` and the planners scored
        under a fixed `PRNGKey(0)`, **every** stochastic environment was an
        oracle on seed 0, not just the reactor: on the battery that was worth
        350.4 against an honest 151.8. That is now closed by `plan_params`, so
        an oracle arm is something to re-enable deliberately -- pass the
        evaluation key through `plan_params` rather than around it -- and its
        results must never reach `src/target_gym/data/baseline_returns.json`.
      - [ ] **Scenario arm on `SamplingMPC`.** A vmap over K disturbance keys
        and a mean, on top of the vmap over action samples it already does.
        Then a decision about whether to average the objective or use a risk
        measure.
      - [ ] **True-model planners for pH, glass furnace and reactor.** The real
        cost, and the finding that most changes the plan: on three of the four
        priority environments the shipped MPC is a hand-written CasADi model,
        not the simulator, so "it cannot be model error" does not hold today.
        `SamplingMPC` needs only an objective, and it needs no gradients, which
        matters because pH's bisection solve and the furnace's implicit gas
        solve are both gradient risks and the kiln's adjoint already overflows.
      - [x] **Decide what the deterministic arm actually is.** Decided and
        measured: it plans on the **mean**. `experts.mpc.plan_params` zeroes the
        parameters named in `EnvSpec.noise_fields` for the planner's copy of
        the params, which is certainty equivalence. It controls better, not
        merely more honestly: the wind turbine went from 343.9 to 348.3, and
        the battery MPC from losing 9 seeds in 10 to leading.
      - [ ] **An asymmetric reward variant**, if mechanism 2 is to be isolated
        cleanly rather than merely present. Every reward here is symmetric in
        the error, including the cement kiln's free lime; they are all
        non-quadratic, which is the condition that actually matters, so the
        effect is observable without this. Free lime is the defensible
        candidate (high is a quality rejection, low merely wastes fuel) and
        would need `PHYSICS.md` justification, a version bump and a re-record.
      - [ ] **Assert the reactor's stochastic demand.** It is excluded from
        `disturbance_fields` deliberately and correctly, since that field means
        a zero-mean plant disturbance and this is a drifting setpoint. The
        consequence is that nothing asserts the process at all. It wants its
        own check rather than a registry edit.
      - [ ] **An observation-augmentation wrapper**, if the action-queue
        hypothesis is to be tested. No environment's observation carries any
        history of past actions, only current actuator positions, against
        transport delays of up to 25 minutes on the kiln. There is no frame
        stacking and no wrapper; the aircraft's `observe_wind` is
        constructor-level and specific to that plant.

      A new planner for one task goes in that task's own `experts.py`, which
      re-records only its package. A change to the shared machinery in
      `experts/mpc.py` still re-records every task.

* [x] **Host the documentation.** Done, and this was a duplicate of the 0.6
      blocking entry above; it survived because nothing checks the roadmap
      against itself.
* [ ] **Publish RL baseline results.** The environments claim a learned policy
      has something real to beat; no learned policy's numbers are published yet.
      The harness is in place: `src/target_gym/data/rl_results.json`, written through
      `target_gym.rl_results.record_result` and guarded by a fingerprint of the
      environment, so a result recorded before a reward or dynamics change is
      refused rather than quoted. Training runs outside this package (the
      dependency goes RL-library-to-here, never the reverse); see
      [rl-baselines.md](rl-baselines.md).

      **Do this last, after the environments are frozen.** That fingerprint is
      the reason for the ordering. It covers each environment's own modules plus
      the shared physics, so any change to an environment, the integrator or the
      shared reward helper invalidates every result recorded against it. A
      training run is the most expensive artefact this project produces and the
      easiest to invalidate by accident. Everything else on this list should
      land first, including the outstanding baseline gaps, since re-tuning a
      controller is cheap and re-running the whole RL grid is not.

      **Compute plan: the TPU Research Cloud.** Free access to Cloud TPUs for
      researchers, JAX among the supported frameworks, rolling applications with
      no review committee, and quota granted on accepting the invitation. The
      condition is publishing the work, which this project does anyway. It fits
      because Ajax is JAX-native and these environments run on the accelerator
      beside the agent, so a rollout never leaves the device.

      What that costs is not the TPUs, which are free, but the surrounding
      Google Cloud resources: a boot disk per TPU VM, a storage bucket and
      egress. The artefacts here are learning curves, final returns and small
      MLP checkpoints rather than datasets, so 10-20 GB covers it and storage is
      well under a euro a month. Boot disks dominate: roughly 10 EUR a month for
      five TPU VMs on small standard disks, nearer 40 if left on the default
      100 GB balanced disks or run wide on preemptibles. Budget 10-50 EUR a
      month and delete idle VMs, since a disk bills while it exists even
      stopped. Keep the bucket in the TPU's own region so reads are free.

      **Size of the grid**, from the protocol: 21 environments x 2 headline
      algorithms x 3 arms x 3 budgets is 396 vmapped configurations, and seeds
      vmap almost for free, so 30 seeds costs about what one does. Summing the
      budgets gives roughly 44 billion environment steps for the reported runs,
      plus about 8 billion for the hyperparameter search at 64 trials x 3 seeds.
      The 1e7 budget is around 90% of that total, and the SAC arm at 1.0
      gradient steps per environment step is far more expensive than PPO.

      **One measurement is missing before any of this can be scheduled.** The
      seed-scaling figure quoted in the protocol and in `rl-baselines.md`, 15.8 s
      for one seed against 18.4 s for a hundred, has no step budget attached to
      it, so it cannot be scaled into a wall-clock estimate. Running Ajax on one
      environment at the 1e6 budget with 30 seeds and recording steps per second
      turns the rest into arithmetic, and decides whether this is one month of
      quota or three.
* [x] **Drop the git dependency on `gymnax`.** Gone, and it turned out not to be
      needed. The pin tracked upstream `main` on the reasoning that released
      gymnax 1.0.0 caps `gymnasium<1.2` and that "conflicts with newer
      gymnasium", but nothing in this project requires newer gymnasium. It
      declares `gymnasium>=1.1,<1.4`, and 1.1.1 satisfies that. Resolving from
      PyPI alone gives gymnax 1.0.0 with gymnasium 1.1.1, on which the whole
      suite passes unchanged: 1225 fast, 69 slow, same four and two xfails.
      The tested configuration is now reproducible from PyPI, which was the
      point. `uv.lock` carries no git dependencies at all.
* [x] **A performance phase.** Four defects, all paid by every user and none
      visible to a throughput benchmark, which measures steady state after
      compilation. Every environment returned a *weakly typed* reset state, so
      anything jitted over the state compiled twice. Each new environment
      instance retained a compiled executable, leaking ~2.6 MB per construction.
      The pH solver spent its whole runtime on 44 bisection halvings resolving to
      1e-13. And `runners.rollout` re-jitted the environment on every call, so a
      warmed rollout still spent 0.222 s of 0.355 s compiling. Fast CI 164 s ->
      128 s, `tests/experts` 143 s -> 37 s, warm rollouts ~5x, pH throughput 2x.

      Two restructurings were measured and rejected: vectorising the aircraft's
      three aerodynamic calls into one is 0.76x, and `donate_argnums` on the
      batched rollout does nothing (the carried state is 0.26 MB). The remaining
      slow environments are honestly slow, since distillation needs 16 substeps
      across 41 stages for stability, the cement kiln sweeps 16 zones in
      sequence.

      Two cautions for whoever picks this up. Benchmarks on a laptop vary 41%
      across identical trials, so every number here is a min of many; a
      single-shot measurement produced a confident and wrong conclusion partway
      through this work. And the pass found a *correctness* bug while looking for
      speed (see the integration order note below), which is the main reason
      it was worth doing.

      The table's throughput column has since been re-measured with
      `python -m target_gym.benchmark_speed` (batch 256, best of three, after
      warm-up). Every process plant came back within 10% of its published figure,
      which is what makes the aircraft rows conclusive: all five were about 2x
      optimistic, because the post-stall aerodynamics, the three moment
      decompositions, pitch damping and fuel burn were added to those dynamics
      after the numbers were taken. They now read as measured.

      Throughput is also strongly batch-dependent for the aircraft, which the
      single number does not convey: the 3D plane roughly doubles between batch
      256 and batch 16384. Anyone training on these should batch at 4096 or more.

      The aircraft rows fell again when the integration order was corrected from
      one RK4 substep to two (see `plane3d/PHYSICS.md`). That is the honest cost
      of a converged trajectory: at one substep the altitude was 20 m out over
      150 steps, against a reward that resolves to 1 m.
* [x] **Apply the model review checklist to the other environments.** The
      aircraft work produced twelve checks in
      [model-review-checklist.md](model-review-checklist.md), derived
      from real defects rather than from good intentions. All twelve have now
      been run across all eighteen environments.

      The pass so far: check 3 found two write-only state fields a hand review
      had missed; check 9 found the bank-commanded loop gain varies 2.38x on the
      figure-8, and that removing it changes nothing measurable; check 10's own
      advice cleared the integrator in one run. The circle's path-following
      failure turned out not to be a guidance fault at all. A third of its
      radius range is unflyable at the cruise speed its autopilot holds, and
      trading speed for radius took the worst seed from 1860 m to 73 m, closing
      half of a long-standing strict xfail. Check 12 exists because the first
      version of that fix edited one of *three* copies of the same control law
      and measurably did nothing.

      Checks 5, 7 and 8 each needed a plant-agnostic form to be run at all, and
      each needed its first metric discarded. Check 5 is now done by autodiff:
      comparing the two one-sided Jacobians of a step tells a kink from a steep
      curve, which comparing sample-to-sample steps cannot. That ranked
      Arrhenius above every real seam. Check 7 became an unforced run, with
      linear growth separated from accelerating growth so that an aircraft is
      not flagged for flying forwards. Both come back clean: no plant produces
      energy from its own equations, and the only seams that survive refinement
      are a mass clamped at zero, a power limit binding, and one in the aircraft
      at 308 m/s that full actuator travel cannot reach.

      Checks 3, 4, 5, 7 and 8 now run in
      [the conformance suite](https://github.com/YannBerthelot/TargetGym/blob/main/tests/test_env_conformance.py) against every
      registered environment, each with an allowlist so it reports *new*
      defects rather than restating known-benign ones. A new environment
      inherits them by adding one line to the registry.
* [x] **A reward-shaping phase.** The rewards had been written per environment as
      each was added, and the conventions had drifted: Gaussian versus
      quadratic tracking terms, differing crash penalties, differing treatment
      of the target band, and four environments whose reward was *identically
      zero* across the first three halvings of their error. All eighteen now
      share one contract: `(tracking terms, multiplied) x (1 - weighted costs)`,
      bounded in `[0, 1]`, log-scaled around a floor taken from each plant's own
      instrumentation. Costs multiply rather than subtract, so nothing is earned
      without tracking and no episode can profit by ending early, which made
      the flat crash penalties redundant, and they are gone. See
      [reward-shaping.md](reward-shaping.md).
* [x] **Move off the Alpha classifier.** Done: `pyproject.toml` declares
      `Development Status :: 4 - Beta`, per the decision recorded above.

### Documentation debt

* [x] **Action labels for eight environments.** ~~Eleven of twenty-two pages
      showed action bounds with a blank meaning column.~~ Done, and the original
      diagnosis was only half right. It was not simply eight missing docstring
      lines: four public environment classes (`CSTR`, `FirstOrderSystem`,
      `FourTank`, `GlassFurnace`) had no class docstring at all, `Airplane2D`
      had one without an action line, and three more (`PHNeutralization`,
      `BuildingHVAC`, `GridBattery`) documented their action perfectly well in
      a spelling the generator could not read. Those write
      `Action (1,): base flow, raw in [-1, 1]`, with the meaning *before* the
      bracket, and the parser took the first `[` on the line, read the range
      `-1, 1` as two labels, and matched no single-action environment. So the
      fix was at both ends: docstrings at the source, and a parser that reads
      both spellings. All twenty-two now carry labels. A few still read as
      variable names rather than meanings (`rho_ext_norm`, `pitch_raw`,
      `L_raw`), which is the environment's own wording and a smaller,
      separate tidy.

* [x] **Wire the page generators into the test suite.** ~~`docs/environments.md`
      already has a sync test.~~ Done, as a `docs` CI job that runs both
      generators' `--check` and `mkdocs build --strict`, mirrored by
      `make ci-docs`. It runs as its own job, concurrent with the interpreter
      matrix, so it costs runner minutes but no wall-clock against the ten
      minute budget. It found real drift immediately: both new environment
      pages missing from the nav, twenty-five dangling links (every
      `src/**/PHYSICS.md` and `CONTRIBUTING.md` reference resolves on GitHub
      but 404s on the built site), and mkdocs itself absent from `uv.lock`
      entirely, so the docs were only buildable on a machine that had installed
      it out of band. The original text follows.

      `scripts/generate_env_pages.py --check` and
      `scripts/generate_env_reference.py --check` should both run in CI so a
      parameter change cannot leave eighteen environment pages quietly
      disagreeing with the code. The check costs seconds; the failure mode it
      prevents is documentation that lies.

### Known gaps

The test suite records these rather than hiding them, as `strict` xfail cases
from two markers, plus the patrol baseline notes above:

* **Plane Patrol expert quality**: both patrol variants now ship a PID, but it
  completes roughly half of evaluation seeds. The failure is a lateral bank
  oscillation that sets in once the follower overshoots *ahead* of the slot
  chasing a steeply descending lead: pursuit guidance then commands a turn the
  bank loop cannot make, and it rings between its limits. No lateral gain
  combination clears it, so the guidance law needs energy management (the
  follower cannot shed speed in a descent) rather than further tuning.
* **Four-tank zero is fixed**: the real apparatus is celebrated for letting you
  move the multivariable zero across the imaginary axis by turning two valves.
  Here `gamma1` and `gamma2` are constants, so only the non-minimum-phase
  configuration is available.
* **CSTR target margin**: the bottom of its sampled band needs the coolant
  within a fraction of a kelvin of its stop. Reachable, but with almost no
  authority left for disturbance rejection.

---

