import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym import reward as R
from target_gym.base import canonical_reset, failure_kernel
from target_gym.reactor.env import (
    ReactorParams,
    ReactorState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    steady_state_precursors,
    steady_state_xenon,
)
from target_gym.reactor.rendering import _render
from target_gym.utils import save_video

# Number of physics sub-steps per control step. Constant so JIT can treat it
# as static in `lax.scan(length=...)`. Change here only — `env.py` reads it
# via ``Reactor.control_period``.
CONTROL_PERIOD: int = 10


class Reactor(environment.Environment[ReactorState, ReactorParams]):
    """
    Nuclear reactor (point kinetics + thermal feedback).

    Observation (4,): [n, T_coolant, rho_ext_norm, target_n]
    Action      (1,): [rho_ext_norm] in [-1, 1] (control rod position)
    """

    render_reactor = classmethod(_render)
    screen_width = 700
    screen_height = 900
    # Number of physics sub-steps per env-step. ``max_steps_in_episode`` and
    # ``state.time`` count env steps; ``state.physics_time`` counts sub-steps.
    # Exposed so tooling can convert between the two (a control step lasts
    # ``delta_t * control_period`` seconds).
    control_period: int = CONTROL_PERIOD

    # obs = [n, T_coolant, rho_ext_norm, target_n]
    obs_value_index: int = 0  # n (neutron density / normalised power)
    tracked_names: tuple = ("neutron power (normalised)",)
    obs_target_index: int = 3  # target_n

    def __init__(self, integration_method: str = "tr_bdf2_2"):
        self.obs_shape = (4,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> ReactorParams:
        return ReactorParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        """Cost terms at one physics sub-step ($ per second). ``step_env``
        returns their exact sum over the control period in
        ``info["reward_terms"]``; this is the instantaneous rate."""
        return compute_reward_terms(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ReactorState,
        action: jnp.ndarray,
        params: ReactorParams = None,
    ):
        if params is None:
            params = self.default_params

        rho_raw = action
        if not isinstance(action, float):
            rho_raw = action.reshape(())

        # Action is held constant for `control_period` physics sub-steps. Reward
        # is accumulated across the sub-steps; we freeze the state on natural
        # termination so the scan can still run for a fixed length under jit.
        #
        # Only *natural* termination is checked inside the scan. The time
        # limit is an env-step count (``state.time``, advanced once below), so
        # it cannot fire mid-period; gymnax's ``is_truncated`` reads it off the
        # returned state. Checking truncation on the sub-step clock is what
        # used to freeze this plant at a tenth of its episode.
        def sub_step(carry, _):
            state, cum_reward, cum_terms, terminated = carry
            candidate, _metrics = compute_next_state(
                rho_raw, state, params, integration_method=self.integration_method
            )
            r = compute_reward(candidate, params, xp=jnp)
            terms = compute_reward_terms(candidate, params, xp=jnp)
            term = self.is_terminated(candidate, params)
            # Freeze state once terminated; still accumulate the final-step
            # reward. The `~terminated` guard records only the first crossing:
            # once frozen, `term` keeps firing on the held state.
            next_state = jax.tree.map(
                lambda a, b: jnp.where(terminated, a, b), state, candidate
            )
            live = jnp.logical_not(terminated)
            next_reward = cum_reward + jnp.where(live, r, 0.0)
            next_terms = {
                k: cum_terms[k] + jnp.where(live, v, 0.0) for k, v in terms.items()
            }
            next_terminated = terminated | term
            return (next_state, next_reward, next_terms, next_terminated), None

        zero_terms = {
            k: jnp.float32(0.0) for k in compute_reward_terms(state, params, xp=jnp)
        }
        (new_state, reward, terms, terminated), _ = jax.lax.scan(
            sub_step,
            (state, jnp.float32(0.0), zero_terms, jnp.bool_(False)),
            xs=None,
            length=CONTROL_PERIOD,
        )
        # One environment step has elapsed, whatever the physics clock did.
        new_state = new_state.replace(time=state.time + 1)
        # A SCRAM is part of the kernel: the step that leaves the envelope is
        # charged the trip cost and the plant restarts at once
        # (``base.failure_kernel``). On that step the cost is the trip cost
        # and nothing else, whatever the sub-steps summed to.
        new_state, tripped, _ = failure_kernel(self, key, state, new_state, params)
        trip = R.trip_cost(params)
        reward = jnp.where(tripped, -trip, reward)
        terms = {k: jnp.where(tripped, 0.0, v) for k, v in terms.items()}
        terms["failure"] = jnp.where(tripped, trip, terms["failure"])

        # Mean over the sub-steps rather than the sum. The action is held for
        # ``CONTROL_PERIOD`` physics sub-steps, and summing made one environment
        # step of this plant worth several times a step of any other -- its
        # per-step reward peaked near 4.7 where every other environment caps
        # around 1.0, which is misleading the moment returns are read across
        # environments. Dividing by a positive constant leaves the optimal policy
        # and every within-environment comparison untouched.
        #
        # Terminating mid-period still costs: only the sub-steps before the stop
        # contribute to the sum, so the mean falls with them.
        #
        # Version 2 sums instead: its terms are dollars per physics second, so
        # the sum is the cost of the whole 10 s control step, the unit every
        # other plant's step reward is in.
        obs = self.get_obs(new_state)
        reward = jnp.where(params.reward_version == 1, reward / CONTROL_PERIOD, reward)
        # The version-2 terms summed over the sub-steps, exactly as the reward
        # is, for ``target_gym.eval``'s tracking / running split.
        return (
            obs,
            new_state,
            reward,
            jnp.zeros((), dtype=bool),
            {
                "last_state": new_state,
                "reward_terms": terms,
                "tripped": tripped,
            },
        )

    def get_obs(self, state: ReactorState, params: ReactorParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: ReactorState, params: ReactorParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: ReactorParams = None
    ) -> Tuple[jnp.ndarray, ReactorState]:
        if params is None:
            params = self.default_params

        key, n_key, target_key, demand_key = jax.random.split(key, 4)

        initial_n = jax.random.uniform(
            n_key,
            minval=params.initial_n_range[0],
            maxval=params.initial_n_range[1],
        )
        # Initial demand drawn from target range; OU process evolves it from here.
        initial_target = jax.random.uniform(
            target_key,
            minval=params.target_n_range[0],
            maxval=params.target_n_range[1],
        )
        # Precursors start at steady state for the initial neutron density;
        # without that there is a large transient in the first few seconds.
        initial_C = steady_state_precursors(initial_n, params)

        # Xenon and iodine start at the equilibrium for a *different*, recent
        # power level, not the current one. A reactor that has been
        # load-following is essentially never at xenon equilibrium, and starting
        # it there made the poison a constant bias: with dXe/dt = 0 at t = 0 and
        # a 13.2 h xenon time constant, the term the module docstring calls "the
        # dominant control challenge" contributed nothing an operator would have
        # to trim. Sampling the history instead makes it live from the first
        # step. The range is deliberately narrow, so the offset is one a real
        # unit would carry rather than an extreme.
        key, history_key = jax.random.split(key)
        n_recent = jax.random.uniform(
            history_key,
            minval=params.initial_xenon_power_range[0],
            maxval=params.initial_xenon_power_range[1],
        )
        initial_I_hat, initial_Xe_hat = steady_state_xenon(n_recent, params)

        state = ReactorState(
            time=0,
            physics_time=0,
            n=initial_n,
            C=initial_C,
            T_fuel=jnp.asarray(params.initial_T_fuel, dtype=jnp.float32),
            T_coolant=jnp.asarray(params.initial_T_coolant, dtype=jnp.float32),
            I_hat=jnp.asarray(initial_I_hat, dtype=jnp.float32),
            Xe_hat=jnp.asarray(initial_Xe_hat, dtype=jnp.float32),
            target_n=initial_target,
            demand_key=demand_key,
            rho_ext=jnp.zeros((), dtype=jnp.float32),
            rho_ext_cmd=jnp.asarray(0.0, dtype=jnp.float32),
        )

        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: ReactorParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(ReactorState.__dataclass_fields__), dtype=jnp.float32
        )

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_reactor_pid,
            pid_step,
        )

        params, zero_state = make_reactor_pid()
        return FunctionalExpertPolicy(params, zero_state, pid_step)

    def make_pid(self):
        """Return a ready-to-use StatefulPID for neutron-power tracking."""
        from target_gym.experts.pid import make_reactor_stateful_pid

        return make_reactor_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a CasADi MPC oracle for neutron-power tracking."""
        from target_gym.experts.mpc import make_reactor_mpc

        if params is None:
            params = self.default_params
        return make_reactor_mpc(self, params, **kwargs)

    def save_video(
        self,
        select_action: Callable[[jnp.ndarray], jnp.ndarray],
        seed: int,
        params=None,
        folder="videos",
        episode_index=0,
        FPS=60,
        format="mp4",
    ):
        return save_video(
            self,
            select_action,
            folder,
            episode_index,
            FPS,
            params,
            seed=seed,
            format=format,
        )

    def render(self, screen, state: ReactorState, params: ReactorParams, frames, clock):
        frames, screen, clock = self.render_reactor(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


if __name__ == "__main__":
    env = Reactor()
    seed = 42
    env_params = ReactorParams(
        max_steps_in_episode=2000
    )  # 2000 physics = 200 control steps
    os.makedirs("videos/reactor", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        seed,
        folder="videos/reactor",
        episode_index=0,
        params=env_params,
        format="gif",
    )
