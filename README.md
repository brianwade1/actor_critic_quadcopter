# Actor-Critic Reinforcement Learning Controller for a Quadcopter

This project trains an Advantage Actor-Critic (A2C) reinforcement learning agent to control the motor speeds on a quadcopter in order to keep the quadcopter in a stable hover following a random angular acceleration perturbation between 0-3 degrees per second in each of the control axes: pitch, roll, and yaw. The A2C control replaces the traditional controllers (two proportional–integral–derivative (PID) controllers) used for stability and guidance. The A2C controller was able to correct the random acceleration and keep the quadcopter near the initial point for the three second duration of the simulation. This simulation uses a discrete action space so the agent does not have nuanced control and does not achieve a perfect hover.

---

## Folders and Files

This repo contains the following folders and files:

Main Files:

* [RL_main_ActorCritic.m](RL_main_ActorCritic.m) - Main file with user input, setup of the A2C agent, training of the A2C agent, and test of the A2C agent.
* [QuadcopterResetFunction.m](QuadcopterResetFunction.m) - function called by the RL_main_ActorCritic.m file that resets the quadcopter to the origin with a random angular acceleration. This file is called at the start of each episode.
* [QuadcopterStepFunction.m](QuadcopterStepFunction.m) - function called by the RL_main_ActorCritic.m file that receives the A2C agent's commands for the quadcopter motors and then propagates the dynamics of the quadcopter on step in time (0.1 seconds for this simulation). This file is called at each step during the episode.
* [abs_tanh_plots.m](abs_tanh_plots.m) - Visualization of the reward function. This file is not called or used in the training or simulation of the A2C agent.

Folders and Files within:

* [Agents](Agents): Trained A2C agent
  * trained_quadcopter_AC_agent.mat - MATLAB variable file with the actor and critic neural networks and the A2C agent.

* [Images](Images): Images used in the readme file

---

## Simulation Description

This simulation does not include a notion of ground (minimum z-altitude) and does not model the effects of wind or any other outside influences other than gravity (Earth’s gravity at sea level). Additionally, it assumes constant environmental parameters throughout the episode.

### State Space

The state of the quadcopter is defined by a 13-dimensional vector. The first three dimensions are the location of the quadcopter with respect to (wrt) to the initial starting point. The next three dimensions are the linear rates in each direction. The next three dimensions are the angular orientation of the quadcopter expressed as Euler angles. These are followed by three dimensions that express the angular rates. The final dimension is the current simulation time, which starts at 0 and each step increases by 0.1 seconds. The max simulation time for an episode is 3 seconds. A description of the state vector is below:

* state(1:3) – Inertial frame position: x, y, z in meters
* state(4:6) – Inertial frame linear rates: dx, dy, dz (rate of change of x, y, z) in m/s
* state(7:9) - Euler angles: phi, theta, psi (roll, pitch, yaw) in radians
* state(10:12) - Euler rates: dphi, dtheta, dpsi (d(roll)/dt, d(pitch)/dt, and d(yaw)/dt)) in rad/s
* state(13) - simulation time in seconds (0, 0.1, 0.2, ..., 2.9, 3)

### Episode Starting Conditions

The quadcopter begins each episode at the point (0, 0, 0) meaning that state(1:3) are all equal 0. In the simulation, all x, y, z locations are relative to this starting location. The quadcopter can drift in any direction resulting in a positive or negative value for each x, y, and z coordinate (state vector indices 1-3). At the start of each episode, the quadcopter does not have an initial linear velocity (state(4:6) all equal 0) and no initial pitch, roll, or yaw (state(7:9) are all equal to 0). However, each episode begins with a random disturbance, which sets the angular rates to a random value between 0-3 degrees per second (expressed as a radian per second value). At the start of each episode, state(10:12) is set by the following equation:

State(10:12) = ((2 * deviation * rand(3,1) – deviation) * pi/180)

Where deviation is 3 degrees, rand(3,1) is a three-position vector of random numbers drawn from the uniform [0,1] distribution, and the pi/180 term converts the measure from degrees per second to radians per second. These initial condictions are set by the [QuadcopterResetFunction.m](QuadcopterResetFunction) file, which is called at the start of each episode

### Simulation Termination

The simulation will end in a failed episode if the quadcopter drifts more than 3 meters from its start position (i.e. if norm(state(1:3)) > 3 meters). The simulation will also terminate if either roll or pitch (state vector indices 4 and 5) exceed 80 degrees (expressed in radians) or if the yaw (state vector indices 6) exceeds 170 degrees (also expressed in radians). Should any of these conditions occur, the “IsDone” flag is set to 1. See the [QuadcopterStepFunction.m](QuadcopterStepFunction) file, lines 86-98, for the implementation of these termination criteria.

### Action Description

This simulation will used a discretized control input for each of the quadcopter’s four motors. At each time step, the AC agent will choose one of three possible values for each motor: -1,  0, or 1. This number of available actions is defined by the variable ‘action_range’. The min and max of the action space maps to a 2% control authority for each motor that is centered at the thrust required to hover in steady flight for that individual motor (mass*gravity / number_of_motors).  Thus, the action for each motor is defined by that motor’s thrust required to hover in steady flight:

| Actor Action | Action in Simulation from the Individual Motor |
| :---: | --- |
| -1 | 1% less thrust than what is required to hover in steady flight for a single motor |
| 0 | The thrust required to hover in steady flight for a single motor |
| 1 | 1% more thrust than what is required to hover in steady flight for a single motor |

Because there are four motors each motor can perform any of these three actions, the AC agent must choose from 81 possible actions (3^4) at each time step. For example, a possible action might be [0, -1, 1, 0].  This means that motor 1 is set to action 0 (thrust required to hover), motor two is set to action -1 (1% less thrust than what is needed to hover), motor three is set to action 1 (1% more thrust than what is needed to hover), and motor four is set to action 0 (the thrust required to hover). The definition and discretization of the actions is calculated and defined at the top of the main script, [RL_main_ActorCritic.m](RL_main_ActorCritic.m).

---

## A2C Actor and Critic Neural Networks

Both the actor and critic models within the agent are composed of multi-layer feed-forward neural networks. The Actor neural network receives the state vector from the environment and chooses one of the 81 discretized action vectors. The Critic Neural Network also receives a state vector and estimates the value (long-term expected reward following the current policy) for that state. During training, the critic's estimate of the value of a state is used to update the actor's perception of the previous state's value using temporal differencing [[1]](#References). The advantage part of the A2C model simply means that the update for a value is the difference between the actual return (G <sub>t</sub>) and the estimates value (V(S|theta)). In this example, exploration is partially implemented with a stochastic actor.

### Actor

The actor in the A2C model is composed of a two-hidden layer feed-forward neural network with the following hyperparameters:

| Number of Nodes in 1st Hidden Layer (FC1) | Number of Nodes in 2nd Hidden Layer (FC2) | Learning Rate | Gradient Threshold | L2 Regularization Factor | Training Function |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 64 | 1e-4 | 1 | 1e-4 | adam |

The final layer in the actor network (FC4) had 81 nodes (one per available action vector). The final layer in the below diagram (ActorOutput) is the softmax to turn the action selection into a normalized percentage. The actor neural network model is shown below.

![actorNetwork](/Images/actorNetwork.png)

### Critic

The critic in the A2C model is composed of a two-hidden layer feed-forward neural network with the following hyperparameters:

| Number of Nodes in 1st Hidden Layer (FC1) | Number of Nodes in 2nd Hidden Layer (FC2) | Learning Rate | Gradient Threshold | L2 Regularization Factor | Training Function |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 264 | 128 | 1e-3 | 1 | 1e-5 | adam |

The CriticOutput is a single node which represents the value of a given state. The critic neural network is shown below.

![criticNetwork](/Images/criticNetwork.png)

---

## Reward Function

The reward function is the feedback provided by the environment that the A2C model uses to adjust the weights and bias values in the actor and critic so that actions taken maximize the long-term, cumulative reward. The reward function used in this example problem is expressed in the [QuadcopterStepFunction.m](QuadcopterStepFunction) file, lines 100-106. Note that several other potential reward functions are commented out in this section. The reward used in this example included three reward terms:

* r1: reward for remaining at or near the original location (origin - state(1:3) = (0, 0, 0))
* r2: reward for motor actions that are similar - to hover the motors all need to output action 0 (speed required to hover). This is the action vector [0, 0, 0, 0].
* r3: negative reward (punishment) for terminating the simulation early. This helps the agent avoid state-action pairs that lead to the terminal states such as excessive drift or angles.
  
The reward function used is:

r1 = 1-abs(tanh(norm(NextObs(1:3))));

r2 = -0.1*Action_delta;

r3 = -50*IsDone;

Reward = r1 + r2 + r3;

Here, Action_Delta is the difference between the max and min value of the action value (-1, 0, or 1). The IsDone is a flag. If the simulation state is not a [terminal state](#simulation-description) then the value of IsDone is 0. However, if the state is terminal, then the value of IsDone is 1. The r1 term calculates the the drift of the quadcopter from the goal position at the origin (the magnitude of the three-dimensional position vector (state(1:3))) and finds the reward value with the following function:

![Reward_Function](/Images/Reward_Function.png)

Using this reward function, the maximum reward that an agent can receive at a given step is 1. This would occur if the quadcopter was at the origin (state(1:3) = (0,0,0)), thus r1 = 1, the action_delta was 0 meaning that the max and min actions were equal such as what occurs at a hover, and the IsDone flag was 0 meaning that the state was not terminal. This means that the maximum reward obtainable in the simulation is 300 (one point for each of the 300 steps (0.1 seconds) in the 3 second simulation).

---

## Results

The training of the A2C agent took approximately four hours on a standard single-core laptop. the training progress is shown below:

![training_progress](/Images/TrainingHistory_AC.png)

Once training, the main script then simulates the quadcopter in 10 separate simulations with different initial perturbations. The A2C agent achieved an average reward of 292 points. The position history of the quadcopter for one of these simulations is shown below. The quadcopter is making continual attempts to return to the origin and does not drift by more than 0.05 meters. The agent could be improved with a larger action space (discretization into more points) or with a continuous action space.

![TrainingSample](/Images/TrainingSample_AC.png)

---

## MATLAB Libraries

This simulation was built using MATLAB 2020a with the following libraries:

* Statistics Toolbox
* Deep Learning Toolbox
* Optimization Toolbox
* Reinforcement Learning Toolbox

---

## References

[1]  Richard S. Sutton and Andrew G. Barto. 2018. Reinforcement Learning: An Introduction. A Bradford Book, Cambridge, MA, USA. [http://incompleteideas.net/book/the-book.html]
