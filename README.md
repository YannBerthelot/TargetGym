# Plane Environment
Plane is a library offering a gymnasium/gymnax reinforcement learning environment of a 2D-simulation of an aircraft. It aims at being :
- Compatible with end-to-end jax training on GPU.
- Reliable: It is covered by unit tests.
- Efficient: Simplification of the dynamics to allow fast computation.
- Close to realistic: Simulation results are close to real life results.
- Challenging: This environment offers challenges on different aspects of control. See the "Challenges" section.

# Characteristics

This environment represents an airplane (modeled after an Airbus A320 at the moment) seen from the side evolving in a 2D plane. The goal is to fly to and maintain a target altitude in a fixed window of time. To do so the agent can control the power (and maybe pitch later) of the engines. 

## Action
- Power level : float between 0 and 1. The amount of engine power to be used (between 0% and 100% of engine power).
state.z,
                state.x_dot,
                state.z_dot,
                state.theta,
                state.gamma,
                state.target_altitude,
## Observation
- z: float between min_alt and max_alt. The atltitude of the plane (in meters). #TODO : change to feets?
- z_dot: float between -inf and +inf. The vertical speed of the plane (in meters per second).
- x_dot: float between -inf and +inf. The horizontal speed of the plane (in meters per second).
- theta: float between ? and ?. The pitch angle (in degrees).
- gamma: float between ? and ?. The slope angle (in degrees).
- target altitude: float between min_target_alt and max_target_alt. The altitude to reach and maintain (in meters).

## Reward
The agent gets a reward of `-((target_alt - z)/max_alt)**2` every timestep and a reward of `-max_number_of_timesteps` if it exits the min_alt/max_alt range.

## Episode termination
The episodes terminate after either 1000 timesteps or if the plane exits the min_alt/max_alt range. Whichever happens first. 

## Initialization
The initial altitude and target altitude are initialized randomly in the initial_altitude_range, and target_altitude range respectively.

# Challenges
This environment was designed to represent real-life control challenges that are not yet vastly studied in reinforcement learning in order to benchmark agents against them. The challenges that it represents are:

- Partial observability: It is not possible to observe some, or even most of the state (either impossible or just too hard/expensive). It may also be noisy. In the airplane that corresponds to not being able to measure wind speed or the forces applied to the plane.
- Delay: The actions that are selected by the agent may take some time to actually take effect (or to observe their impact). In the airplane that correponds to the delay between taking an action and the action actually kicking in.
- Non-stationarity/perturbations: Dynamics may change over time. In the airplane that corresponds to possible wind perturbations around the plane altering its dynamics.
- Competing objectives: Parts of the objective are competing with one another and cannot be fully satisfied at the same time. In the airplane that corresponds to moving as fast as possible to the target altitude and staying as close as possible while minimizing fuel consumption and therefore frequency and intensity of actions.
- Momentum: Some of the actions take some time to show their full impact on the environment. In the airplane that corresponds to setting and power level and needing a few second for engine to reach it completely. 

# Specific challenges
Some of the challenges are more specific to a certain type of real-life applications.

- Multiple-stability: Under no perturbations, reapeating the same action will lead to a stable state. In the airplane that corresponds to setting a power level (given its enough to allow the plane to fly) and having the plane stabilize itself at a new altitude without other controls.

- Irrecoverability*: Certain states are irrecoverable and should be avoided as they are guaranteed to lead to failure. In the airplane that corresponds to having gathered so much momentum so that it's not possible to compensate it, leading to a crash.

The environment tries to stay as close to reality as possible while maintaining a simple simulation to be easily modified and customized. The goal is not to represent exactly the flight dynamics but to offer realistic order of magnitudes of the challenges mentionned above by relying on real physics. It is made so that the effects of challenges can be exagerated in order to benchmark agents on them.

# Installation

TBD

# Usage

TBD

# Benchmark
TBD