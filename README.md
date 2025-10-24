## Reinforcement learning approach to safe 3D autonomous navigation of a quadrotor

This is a small applied research/engineering project to create a reinforcement-learning-based controller for safe navigation of a Skydio X2 quadrotor in a simulated 3D environment. This project uses the [Mujoco](https://mujoco.org/) physics engine to simulate the quadrotor, and uses Proximal Policy Optimization [(PPO)](https://arxiv.org/pdf/1707.06347), based on the [cleanRL implementation](https://github.com/vwxyzjn/cleanrl).

Below are a couple of rollouts from the policy. Currently, the model is trained on a fixed start and goal location with obstacles in the path between start and goal, and the quadrotor is able to avoid the obstacles, navigate to the goal, and hover near it. 

![Demo: 1](demo-videos/rl-video-episode-1600-300x300.gif)
<!-- ![Demo: 2](demo-videos/rl-video-episode-5200-300x300-20sec.gif) -->
![Demo: 3](demo-videos/rl-video-episode-2400-300x300.gif)

### Key features:
1. Custom Gymnasium environment based on the Mujoco [Skydio X2 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2)
2. Parallel environments using AsyncVectorEnv for more efficient data collection
3. Train using lightly modified [PPO](https://github.com/vwxyzjn/cleanrl) implementation from CleanRL
4. Dynamic environment generation during episode rollouts using MjSpec to support generalizability
6. (In progress) Obstacle avoidance with variable number of randomly positioned obstacles and start locations. Experimenting with curriculum learning to facilitate generalizability
8. Added Euler-Lagrange [quadrotor dynamics](https://vnav.mit.edu/material/06-Control1-notes.pdf) in canonical form $\dot x = f(x) + g(x)u$ with neural-net based residual model to account for differences in analytical and Mujoco forward dynamics, as well as incorporating Mujoco [inertia-based aerodynamics modelling](https://mujoco.readthedocs.io/en/latest/computation/fluid.html#flinertia) of drag forces and torques into the analytical equations: $\dot x = f(x) + f_{residual}(x) + g(x, \tau_{d}, f_d)u$. This supports future work on model based RL.


### Details
#### Formulation
- The navigation problem is formulated as a finite horizon (episodic), continuous time MDP
- The quadrotor state is defined as $x = [p, q, v, w, p-p_{obs1}, ..., p-p_{obsK}, p-p_{goal}]$ where $p$ is position, $v$ is the velocity, $q$ is the quaternion, and $w$ is the angular velocity.

#### Custom Gymnasium environment
- Reward shaping with dense rewards to guide learning
- Curriculum learning to gradually require the agent to reach closer to the target
- Termination conditions tuned to support efficient data collection
- Wandb hyperparameter sweeps to determine the optimal reward function parameters
- Start location and obstacle number/position randomization in each parallel env to improve generalization i.e. reach the target from any start location. This will enable future work in which the drone is routed through arbitrary paths
- Dynamic generation of obstacles using the MjSpec API. Can generate new obstacle configurations for each episode during training.
- Reset noise to improve generalization
- Wrapped the Gym observation normalization wrapper to save normalization stats learned during training. During evaluation on a new environment, it is important to load these normalization stats and not calculate new ones.

#### Obstacle Avoidance (in progress)
- Reinforcement learning algorithms estimate the solution to an unconstrained optimization problem, and as such, do not provide mathematical guarantees for safety. 
I initially explored a Control Barrier Function (CBF) approach based on this [paper](https://arxiv.org/pdf/2110.05415) which calculates the control input that satisfies the CBF constraints by solving a Quadratic Program (QP) at each time step. However, since the safety layer acts as a compensator to the potentially unsafe RL control input, it can modify the RL control. Since the policy and value networks are updated using these unsafe actions, the modified controls can affect the learning process. It also raises questions about the advantage and returns calculations that are based on the potentially unsafe policy.
- I decided to use a simpler approach which does not provide mathematical guarantees but has been shown to work well in practice. I simply incorporate the relative position vector to the top-k closest obstacles into the state space and learn the policy to avoid them. If there are fewer than k obstacles, i pad with a large dummy value.
- Obstacles are intentionally placed around the straight line between the start and goal to force the agent to avoid them. I use a Gaussian offset at each sampled location on the line to randomize the exact position and ensure all obstacles aren't in a straight line.
