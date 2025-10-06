## Reinforcement learning approach to safe 3D autonomous navigation of a quadrotor

This is a small applied research project to learn, apply, and demonstrate the use of reinforcement learning to create a controller for safe navigation of a simulated quadrotor in a 3D environment. This project uses the [Mujoco](https://mujoco.org/) physics engine to simulate the quadrotor, and uses Proximal Policy Optimization [(PPO)](https://arxiv.org/pdf/1707.06347), based on the [cleanRL implementation](https://github.com/vwxyzjn/cleanrl).

Before we get into the details, let's take a look at some of the results. Currently, model is trained on a fixed goal start and goal location, and the quadrotor is able to navigate to the goal, and hover over it. 

![Demo: Successful Target Hover with Sparse Obstacles](demo-videos/successful-target-hover-sparse-obstacles.gif)
![Demo: RL Episode 5200 High Quality](demo-videos/rl-video-episode-5200-hq.gif)

### Key features:
1. Custom Gymnasium environment based on the Mujoco [Skydio X2 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2)
2. Parallel environments using AsyncVectorEnv for more efficient data collection
3. Train using [PPO](https://github.com/vwxyzjn/cleanrl)
4. (In progress) Obstacle avoidance based on an optimization approach that learns a compensator, using control barrier functions
5. Formulation and implementation of Euler-Lagrange [quadrotor dynamics](https://vnav.mit.edu/material/06-Control1-notes.pdf) in canonical form $\dot x = f(x) + g(x)u$.
6. Neural-net based residual model to account for differences in Euler Lagrange and Mujoco forward dynamics, as well as incorporating Mujoco [inertia-based aerodynamics modelling](https://mujoco.readthedocs.io/en/latest/computation/fluid.html#flinertia) of drag forces and torques into the analytical equations: $\dot x = f(x) + f_{residual}(x) + g(x, \tau_{d}, f_d)u$
7. Dynamic environment generation during episode rollouts using MjSpec to support future generalizability work

### Details
#### Formulation
- The navigation problem is formulated as a finite horizon (episodic), continuous time MDP
- The quadrotor state is defined as ```x = [p, q, v, w]``` where ```p``` is the position, ```v``` is the velocity, ```q``` is the quaternion, and ```w``` is the angular velocity.

#### Custom Gymnasium environment
- Reward shaping with adaptive difficulty tasks and intermediate rewards to guide learning
- Termination conditions tuned to support efficient data collection
- Wandb hyperparameter sweeps to determine the optimal reward function parameters
- Start location randomization in each parallel env to improve generalization i.e. reach the target from any start location. This will enable future work in which the drone is routed through arbitrary paths
- Dynamic generation of obstacles using the MjSpec API. Can generate new obstacle configurations for each episode during training. 
- Reset noise to improve generalization
- Wrapped the Gym observation normalization wrapper to save normalization statistics learned during training. During evaluation on a new environment, it is important to load these normalization statistics and not calculate new ones.

#### Obstacle Avoidance (in progress)
- Reinforcement learning algorithms estimate the solution to an unconstrained optimization problem, and as such, do not provide mathematical guarantees for safety
- I've used a Control Barrier Function (CBF) formulation based on this [paper](https://arxiv.org/pdf/2110.05415) to ensure safety. Determining the control input that satisfies the CBF constraints requires solving a Quadratic Program (QP) at each time step. 
- The safety layer acts as a compensator to the potentially unsafe RL control input. As such, it may modify the RL control. Since the policy and value networks are updated using these unsafe actions, the modified controls can make the learning process difficult. 
- The authors recommend using a QP solver that is differentiable. To that end, I use the [qpth](https://locuslab.github.io/qpth/) differentiable QP solver library to define a QP layer and integrate it with the actor network.
- The CBF formulation should make the training process independent of obstacle locations. This is a work in progress.

#### Quadrotor Dynamics
- Quadrotors are underactuated systems with nonlinear dynamics. They have 6 degrees of freedom (3 position, 3 orientation), and only 4 direct control inputs. This makes control design challenging.
- I've implemented the dynamics in canonical form in order to compute the CBF constraints. The implementation supports batched inputs for parallel environments.
