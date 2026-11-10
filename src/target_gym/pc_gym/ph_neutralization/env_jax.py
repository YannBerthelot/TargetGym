from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import (
    FieldReading,
    actuate,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.expert_access import RegistryExperts
from target_gym.pc_gym.ph_neutralization.env import (
    PHParams,
    PHState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    solve_pH,
    start_upset,
    steady_base_flow,
    steady_state_invariants,
)
from target_gym.pc_gym.ph_neutralization.rendering import _render
from target_gym.video import SavesVideo


class PHNeutralization(
    RegistryExperts, SavesVideo, environment.Environment[PHState, PHParams]
):
    """pH neutralisation CSTR (reaction-invariant formulation).

    Observation (8,): [pH, q3_pct, setpoint_pH, excursion_min, pH_s1, trim_pct,
                       q1, basin_pct]
    Action      (2,): stage-1 base flow and stage-2 trim, raw in [-1, 1]
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "target_pH",
        "upset_times",
        "upset_levels",
        "flow_times",
        "flow_levels",
        "q1",
        "q2",
        "upset_base",
        "flow_base",
    )
    #: The record a trip's restart carries over (``restart_plant``): the permit
    #: monitor's excursion clock. It is the plant's own, so a policy moves it,
    #: and it is not part of the outside world (long-strata S2).
    RECORDS = ("excursion",)
    #: Runs as a continuing plant (``target_gym.stream``): its upsets ramp back
    #: and start through ``start_event``; its ramps count steps.
    STREAM_READY = True

    registry_name: str = "ph_neutralization"

    render_ph = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 0  # pH
    tracked_names: tuple = ("pH",)
    obs_target_index: int = 2  # target_pH

    # Glass electrode, provisional, 10 s lag and 10 s dead time, in seconds as
    # ``delta_t`` is (PHYSICS.md section on readings).
    readings: tuple = (
        FieldReading("pH", sigma=0.01, step=0.01, lag=10.0, dead_time=10.0),
        FieldReading("pH_s1", sigma=0.01, step=0.01, lag=10.0, dead_time=10.0),
    )

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (8,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> PHParams:
        return PHParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PHState,
        action: jnp.ndarray,
        params: PHParams = None,
    ):
        if params is None:
            params = self.default_params
        action, state = actuate(self, state, jnp.reshape(action, (2,)), params)

        new_state, _ = compute_next_state(
            action,
            state,
            params,
            key,
            integration_method=self.integration_method,
        )
        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    @measured_obs
    def get_obs(self, state: PHState, params: PHParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: PHState, params: PHParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: PHParams = None
    ) -> Tuple[jnp.ndarray, PHState]:
        if params is None:
            params = self.default_params

        keys = jax.random.split(key, 5)

        # Established operation (charter rule 17): both tanks on the steady
        # state that holds the set point under the buffer flow they meet, the
        # trim valve shut and the basin empty, as a loop in automatic leaves
        # them. The upsets come later, unannounced.
        q2 = jnp.clip(
            params.q2_nominal + params.q2_noise_std * jax.random.normal(keys[0]),
            params.q2_min,
            params.q2_max,
        )
        target_pH = jnp.asarray(params.setpoint_pH, jnp.float32)
        q3 = jnp.clip(
            steady_base_flow(target_pH, q2, params), params.q3_min, params.q3_max
        )
        span = params.max_steps_in_episode

        def schedule(times_key, levels_key, size):
            times = jnp.sort(jax.random.randint(times_key, (2,), 0, span))
            draws = jax.random.uniform(levels_key, (2,), minval=-1.0, maxval=1.0)
            moves = size * draws * params.window_upsets
            return times, jnp.concatenate([jnp.ones((1,)), 1.0 + moves])

        upset_times, upset_levels = schedule(keys[1], keys[2], params.acid_upset_size)
        flow_times, flow_levels = schedule(keys[3], keys[4], params.flow_swing)
        Wa, Wb = steady_state_invariants(q3, q2, params)
        pH = solve_pH(Wa, Wb, params)

        state = PHState(
            time=0,
            Wa=Wa,
            Wb=Wb,
            q2=q2,
            pH=pH,
            q3=q3,
            target_pH=target_pH,
            upset_times=upset_times,
            upset_levels=upset_levels,
            flow_times=flow_times,
            flow_levels=flow_levels,
            excursion=jnp.zeros(()),
            Wa_s1=Wa,
            Wb_s1=Wb,
            pH_s1=pH,
            q3_trim=jnp.zeros(()),
            q1=jnp.asarray(params.q1),
            basin=jnp.zeros(()),
            Wa_basin=Wa,
            Wb_basin=Wb,
        )
        return self.get_obs(state), state

    def restart_plant(self, key: chex.PRNGKey, state: PHState, params: PHParams):
        """After a trip the plant restarts as ``reset_env`` would, and the permit
        monitor keeps its record (``RECORDS``)."""
        fresh = self.reset_env(key, params)[1]
        return fresh.replace(**{f: getattr(state, f) for f in self.RECORDS})

    def start_event(self, state: PHState, key, params: PHParams) -> PHState:
        """The stream's event hook (docs/proposals/long-strata.md, S1): the
        scenario's influent upsets start now, each at a level drawn from ``key``
        within its size, and ramp back ``event_hold`` seconds later."""
        acid_key, flow_key = jax.random.split(key)
        acid = start_upset(
            state.upset_times,
            state.upset_levels,
            state.upset_base,
            state,
            params,
            params.acid_upset_size,
            acid_key,
        )
        flow = start_upset(
            state.flow_times,
            state.flow_levels,
            state.flow_base,
            state,
            params,
            params.flow_swing,
            flow_key,
        )
        return state.replace(
            upset_times=acid[0],
            upset_levels=acid[1],
            upset_base=acid[2],
            flow_times=flow[0],
            flow_levels=flow[1],
            flow_base=flow[2],
        )

    def initial_action(self, state, params):
        """The raw actions that hold the reset valves where they run."""
        u = 2.0 * (state.q3 - params.q3_min) / (params.q3_max - params.q3_min) - 1.0
        trim = 2.0 * state.q3_trim / params.q3_trim_max - 1.0
        return jnp.stack([u, trim])

    def action_space(self, params: PHParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.ones(2), high=jnp.ones(2), shape=(2,), dtype=jnp.float32
        )

    def observation_space(self, params: PHParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: PHParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(PHState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(self, screen, state: PHState, params: PHParams, frames, clock):
        frames, screen, clock = self.render_ph(screen, state, params, frames, clock)
        return frames, screen, clock
