## Reinforcement learning approach to safe 3D autonomous navigation of a quadrotor

This is a small applied research project to learn, apply, and demonstrate the use of reinforcement learning to create a controller for safe navigation of a simulated quadrotor in a 3D environment. This project uses the [Mujoco](https://mujoco.org/) physics engine to simulate the quadrotor, and uses Proximal Policy Optimization [(PPO)](https://arxiv.org/pdf/1707.06347), based on the [cleanRL implementation](https://github.com/vwxyzjn/cleanrl).

Before we get into the details, let's take a look at some of the results. Currently, model is trained on a fixed goal start and goal location, and the quadrotor is able to navigate to the goal, and hover over it. 

![Demo: Successful Target Hover with Sparse Obstacles](demo-videos/successful-target-hover-sparse-obstacles.gif)
![Demo: RL Episode 5200 High Quality](demo-videos/rl-video-episode-5200-hq.gif)

### Key features:
1. Custom Gymnasium environment based on the Mujoco [Skydio X2 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2)
2. Parallel environments using AsyncVectorEnv for more efficient data collection
3. Train using [PPO](https://github.com/vwxyzjn/cleanrl)
4. Dynamic environment generation during episode rollouts using MjSpec to support generalizability
5. (In progress) Obstacle avoidance with variable number of obstacles using top-k 
6. (Next) Start and goal location generalization
7. Added Euler-Lagrange [quadrotor dynamics](https://vnav.mit.edu/material/06-Control1-notes.pdf) in canonical form $\dot x = f(x) + g(x)u$ with neural-net based residual model to account for differences in analytical and Mujoco forward dynamics, as well as incorporating Mujoco [inertia-based aerodynamics modelling](https://mujoco.readthedocs.io/en/latest/computation/fluid.html#flinertia) of drag forces and torques into the analytical equations: $\dot x = f(x) + f_{residual}(x) + g(x, \tau_{d}, f_d)u$. This supports future work on model based RL.


### Details
#### Formulation
- The navigation problem is formulated as a finite horizon (episodic), continuous time MDP
- The quadrotor state is defined as $x = [p, q, v, w, p-p_{obs1}, ..., p-p_{obsK}, p-p_{goal}]$ where $p$ is position, $v$ is the velocity, $q$ is the quaternion, and $w$ is the angular velocity.

#### Custom Gymnasium environment
- Reward shaping with dense rewards to guide learning
- Curriculum learning to gradually require the agent to reach closer to the target
- Termination conditions tuned to support efficient data collection
- Wandb hyperparameter sweeps to determine the optimal reward function parameters
- Start location randomization in each parallel env to improve generalization i.e. reach the target from any start location. This will enable future work in which the drone is routed through arbitrary paths
- Dynamic generation of obstacles using the MjSpec API. Can generate new obstacle configurations for each episode during training. 
- Reset noise to improve generalization
- Wrapped the Gym observation normalization wrapper to save normalization stats learned during training. During evaluation on a new environment, it is important to load these normalization stats and not calculate new ones.

#### Obstacle Avoidance (in progress)
- Reinforcement learning algorithms estimate the solution to an unconstrained optimization problem, and as such, do not provide mathematical guarantees for safety. 
I explored a Control Barrier Function (CBF) approach based on this [paper](https://arxiv.org/pdf/2110.05415) that calculates the control input that satisfies the CBF constraints by solving a Quadratic Program (QP) at each time step. However, since the safety layer acts as a compensator to the potentially unsafe RL control input, it can modify the RL control. Since the policy and value networks are updated using these unsafe actions, the modified controls can make the learning process difficult. It also raises questions about the advantage and
returns calculations that are based on the potentially unsafe policy.
- I decided to use a simpler approach which does not provide mathematical guarantees but has been shown to work well in practice. I simply incorporate the top-k closest obstacles into the state space and learn the policy to avoid them. This is a work in progress.