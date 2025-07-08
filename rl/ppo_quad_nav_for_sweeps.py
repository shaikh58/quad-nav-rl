"""this file wraps the contents of ppo_cont_action.py within a function to pass to hyperparam sweeps etc.
"""

import os
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from utils.env_utils import make_env
import envs
import wandb


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def set_global_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # Initialize final layer bias to output hover thrust
        with torch.no_grad():
            # no action scaling
            self.actor_mean[-1].bias.fill_(envs.envs[0].unwrapped.model.keyframe('hover').ctrl[0])
            # with action scaling - makes it worse
            # hover_thrust = envs.envs[0].unwrapped.model.keyframe('hover').ctrl[0]
            # # scale to [-1,1]
            # scaled_hover_thrust = -1 + 2 * (hover_thrust - -1) / 2
            # self.actor_mean[-1].bias.fill_(scaled_hover_thrust)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train(config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init(config=config):

        config = wandb.config
        set_global_seed(config.seed)
        
        batch_size = int(config.num_envs * config.num_steps)
        minibatch_size = int(batch_size // config.num_minibatches)
        num_iterations = config.total_timesteps // batch_size

        start_location = np.array([config.start_x, config.start_y, config.start_z])
        target_location = np.array([config.target_x, config.target_y, config.target_z])
        env_kwargs = {
            "env_radius_lb": config.env_radius_lb,
            "env_radius_ub": config.env_radius_ub,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "progress_weight": config.progress_weight,
            "progress_type": config.progress_type,
            "body_rate_weight": config.body_rate_weight,
            "collision_ground_weight": config.collision_ground_weight,
            "collision_obstacles_weight": config.collision_obstacles_weight,
            "out_of_bounds_weight": config.out_of_bounds_weight,
            "success_weight": config.success_weight,
            "goal_threshold": config.goal_threshold,
            "min_height": config.min_height,
            "target_location": target_location,
            "start_location": start_location,
            "max_steps": config.max_steps,
            "reset_noise_scale": config.reset_noise_scale,
            # "save_path": f"runs/{config.run_name}",
            "mode": "train"
        }
        # NOTE: the seed is the same for all envs to ensure same start/goal locations chosen by randomizer
        # the seed will only be seed+i for goal conditioned RL
        envs = gym.vector.SyncVectorEnv(
            [make_env(config.env_id, i, False, None, config.gamma, config.seed, use_planner=config.use_planner, planner_type=config.planner_type, **env_kwargs) for i in range(config.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
        rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
        dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
        values = torch.zeros((config.num_steps, config.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=[config.seed+i for i in range(config.num_envs)])
        for i, env in enumerate(envs.envs): 
            print(f"Env: {i}", "Start: ", env.unwrapped.init_qpos[:3], "Target: ", env.unwrapped._target_location, 
            "Env radius: ", env.unwrapped.model.stat.extent, 
            "Distance from center: ", np.linalg.norm(env.unwrapped.init_qpos[:3] - env.unwrapped.model.stat.center),
            "Target distance from center: ", np.linalg.norm(env.unwrapped._target_location - env.unwrapped.model.stat.center))
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(config.num_envs).to(device)

        for iteration in range(1, num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * config.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, config.num_steps):
                global_step += config.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
                # upon episode completion, vectorEnv handles resets automatically
                for k in range(config.num_envs):
                    if truncations[k] or terminations[k]:
                        print(f"global_step={global_step}")
                        if "episode" in infos:
                            wandb.log({"episodic_return": infos["episode"]["r"][k], "episodic_length": infos["episode"]["l"][k]})
                            ep_ret = infos["episode"]["r"][k]
                            ep_len = infos["episode"]["l"][k]

            if config.adaptive_goal_threshold:
                for k in range(config.num_envs):
                    frac = 1.0 - (iteration - 1.0) / num_iterations
                    envs.envs[k].unwrapped._goal_threshold = frac * config.goal_threshold
                    
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                        # 1-step TD error for GAE
                    delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                    # recursive update of GAE (exponentially weighted average of k-step advantage estimates)
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    # mb_inds are the minibatch indices i.e. batch split into equal sized chunks
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    # compute the surrogate policy ratio pi/pi_old - in log space for numerical stability
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    # for keeping track of KL divergence
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -config.clip_coef,
                            config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # negative of (L_clip - L_value + L_entropy)
                    loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()

                if config.target_kl is not None and approx_kl > config.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y