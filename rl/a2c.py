import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import torch.nn.functional as F
import tyro
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import time
import random

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    # total_timesteps: int = 500000
    # """total timesteps of the experiments"""
    num_epochs: int = 20
    """number of epochs to train the policy for"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 5000
    """number of env rollout steps in a single gradient optimization step"""
    actor_loss_w: float = 1.0
    """weight for the actor loss component of total loss"""
    critic_loss_w: float = 0.5
    """weight for the critic loss component of total loss"""
    max_grad_norm: float = 0.5
    """scale factor for clipping gradients"""
    clip_grad: bool = False
    """if toggled, gradients will be clipped"""
    shared_network: bool = False
    """if toggled, the actor and critic share the same network"""


def env_maker(env_id, seed, idx, capture_video, run_name):
    def make():
        if idx == 0 and capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return make


def kl_divergence(net, prev_net, obs):
    with torch.no_grad():
        net_out = net(obs)
        prev_net_out = prev_net(obs)
        probs = F.softmax(net_out, dim=1)
        prev_probs = F.softmax(prev_net_out, dim=1)
        # can also use the KL div approximation from http://joschu.net/blog/kl-approx.html
        kl_div = F.kl_div(prev_probs.log(), probs, reduction='batchmean') # KL(q,p)
        var = torch.var(probs, dim=1).mean()
        return kl_div.item(), var.item()


def print_grad_graph(tensor, name=""):
    print(f"\nGradient graph for {name}:")
    print("Grad function: ", tensor.grad_fn)
    print("Next functions: ", tensor.grad_fn.next_functions)


class NetworkShared(nn.Module):
    def __init__(self, hidden_sizes, action_space, activation):
        """Initialize the network with shared layers and separate actor/critic heads.

        Args:
            hidden_sizes (list[int]): List of layer sizes from input to last hidden layer.
                Does not include the output layer size.
            action_space: Size of the environment's action space.
            activation: The activation function to use between layers.
        """
        super().__init__()
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), activation()]
        self.shared = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_sizes[-1], action_space)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)
    
    def forward(self, state):
        shared_features = self.shared(state)
        actor_output = self.actor_head(shared_features)
        critic_output = self.critic_head(shared_features)
        return actor_output, critic_output


class Network(nn.Module):
    """General network that can be used for both actor and critic."""
    def __init__(self, hidden_sizes, output_space, activation):
        super().__init__()
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), activation()]
        self.layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_sizes[-1], output_space)

    def forward(self, state):
        features = self.layers(state)
        output = self.output_head(features)
        return output


def compute_actor_loss(actions, logits, advantages):
    """Computes loss for policy gradient based on action taken in state, over the entire batch."""
    log_probs = Categorical(logits=logits).log_prob(actions) # action is index of the action to be taken
    assert log_probs.shape == advantages.shape, "In actor loss, log_probs and advantages must have the same shape"
    # recall policy fcn J = E[d\dw(log policy) * advantage]
    return -(log_probs * advantages).mean() # computed over the batch so use mean for expectation


def compute_critic_loss(batch_rewards, batch_values, batch_ep_dones):
    """Computes the value function loss using rewards-to-go and value estimates.
    Handles episodes of different lengths within the batch buffer.
    """
    returns = []
    for i in reversed(range(len(batch_rewards))):
        if batch_ep_dones[i]:
            # use value function estimate for episode terminal states
            R = batch_values[i]
        else:
            R = batch_rewards[i] + args.gamma * R
        returns.append(R)
    returns = torch.as_tensor(returns)
    assert returns.shape == batch_values.shape, "In critic loss, returns and batch_values must have the same shape"
    return F.mse_loss(returns, batch_values)


def compute_advantages(ep_rewards, ep_values):
    """Advantage uses full episodic reward-to-go rather than an estimate."""
    advs = []
    for i in reversed(range(len(ep_rewards))):
        if i == len(ep_rewards) - 1:
            # R = ep_values[i] # to use value function estimate for terminal state
            R = ep_rewards[i] # to use actual return
        else:
            # R = ep_rewards[i] + args.gamma * ep_values[i+1] - ep_values[i] # 1-step TD error
            R = ep_rewards[i] + args.gamma * R # official algorithm - sum of discounted rewards-to-go
        advs.append(R - ep_values[i]) # detach value estimates if actor/critic don't share the same network
    return advs


if __name__ == "__main__":
    # Add the project root directory to Python path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, 
            sync_tensorboard=True,
            config=vars(args), 
            name=run_name, 
            monitor_gym=True, 
            save_code=True,
        )
    # writes to Tensorboard - can view it by running tensoorboard --logdir /path/to/runs and open the server on a browser
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # setup envs; IMPORTANT: increment the seed for each environment! otherwise it will sample the same actions
    envs = gym.vector.SyncVectorEnv(
        [env_maker(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])

    obs_space = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space.n
    hidden_sizes = [64, 32]
    sizes = [obs_space] + hidden_sizes

    # setup the actor, critic
    # NOTE: in the official algorithm the actor and critic share params
    if args.shared_network:
        policy = NetworkShared(sizes, action_space, activation=nn.Tanh).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    else:
        policy = Network(sizes, action_space, activation=nn.Tanh).to(device)
        critic = Network(sizes, 1, activation=nn.ReLU).to(device)
        actor_optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)

    # start the simulation
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    for epoch in range(args.num_epochs):
        # TODO: implement early termination bad-state penalizer
        # single 'batch' per epoch, but each batch is multiple episodes
        print(f"Epoch {epoch + 1}/{args.num_epochs} started.")
        batch_done = False # batch ends at batch_size episodes
        batch_states = [] # (B,4)
        batch_rewards = [] # (B,)
        batch_actions = [] # (B,)
        batch_action_logits = [] # (B,2)
        batch_advantages = [] # (B,)
        batch_entropy = [] # (B,)
        batch_values = []
        batch_ep_dones = []
        ep_rewards = [] # (T,)
        ep_values = [] # (T,)
        while not batch_done:
            if global_step % 1000 == 0:
                print(f"Epoch {epoch + 1}, Global Step: {global_step}")
            # note: critic evaluated at s_t (obs), NOT next_obs (s_t+1)
            if args.shared_network:
                actions_logits, v = policy(torch.Tensor(obs).to(device))
            else:
                actions_logits = policy(torch.Tensor(obs).to(device))
                v = critic(torch.Tensor(obs).to(device))
            # if global_step % 1000 == 0: print_grad_graph(v, "v")
            # sample an action; forward pass
            actions_dist = Categorical(logits=actions_logits)
            # stochastic action selection
            action = actions_dist.sample() # keep action on device until appended to batch_actions to track gradient properly
            entropy = actions_dist.entropy().mean().item()
            # take a simulation step; currently only supports a single thread
            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # early termination penalty
            if terminations.item() or truncations.item():
                if infos['episode']['l'].item() < 20:
                    early_termination_reward_offset = infos['episode']['l'].item()
                    rewards = rewards.item() - early_termination_reward_offset * 0.5
                else: rewards = rewards.item()
            else: rewards = rewards.item()
            global_step += 1

            # update episode and batch info
            ep_rewards.append(rewards)
            ep_values.append(v)
            batch_states.append(obs.copy())
            batch_actions.append(action)
            batch_action_logits.append(actions_logits)
            batch_values.append(v)
            batch_rewards.append(rewards)
            batch_entropy.append(entropy)

            # important step, easy to forget - increment the current obs
            obs = next_obs

            # at episode end, compute advantages starting from terminal state, and reset the env
            if terminations.item() or truncations.item():
                writer.add_scalar("metrics/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("metrics/episodic_length", infos["episode"]["l"], global_step)
                if args.shared_network:
                    ep_advantages = compute_advantages(ep_rewards, ep_values)
                else:
                    # if separate critic, detach for adv to avoid gradient flowing into critic twice
                    ep_advantages = compute_advantages(ep_rewards, [v.detach() for v in ep_values])
                batch_advantages.extend(ep_advantages)
                batch_ep_dones.append(1)
                obs, _ = envs.reset()
                ep_rewards, ep_values = [], []
                if len(batch_states) > args.batch_size:
                    break
            else:
                batch_ep_dones.append(0)

        # prepare batch data
        actions = torch.as_tensor(batch_actions).to(device)
        logits = torch.stack(batch_action_logits).squeeze().to(device)
        advantages = torch.as_tensor(batch_advantages).to(device)
        # normalize advantages over the entire batch - can also normalize by episode
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        b_rewards = torch.as_tensor(batch_rewards).to(device)
        states = torch.as_tensor(batch_states)
        # print_grad_graph(batch_values[0], "batch_values")
        values = torch.stack(batch_values).squeeze().to(device) # use stack to maintain gradients
        dones = np.array(batch_ep_dones)
        entropy = torch.as_tensor(batch_entropy).mean().item()

        # logging: find highest and lowest value states
        max_idx = torch.argmax(values)
        min_idx = torch.argmin(values)
        state_max_val = states[max_idx].squeeze()
        state_min_val = states[min_idx].squeeze()
        if args.track:
            wandb.log({
                "analysis/max_value_state": {"cart_position": state_max_val[0].item(),"cart_velocity": state_max_val[1].item(),
                "pole_angle": state_max_val[2].item(), "pole_velocity": state_max_val[3].item()
                },
                "analysis/min_value_state": {"cart_position": state_min_val[0].item(),"cart_velocity": state_min_val[1].item(),
                    "pole_angle": state_min_val[2].item(),"pole_velocity": state_min_val[3].item()
                }}, step=global_step)

        # update actor, critic
        actor_loss = compute_actor_loss(actions, logits, advantages)
        critic_loss = compute_critic_loss(b_rewards, values, dones)
        if args.shared_network:
            loss = args.actor_loss_w * actor_loss + args.critic_loss_w * critic_loss
            optimizer.zero_grad()
            loss.backward()
            # clip gradients to prevent large updates
            if args.clip_grad:
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            writer.add_scalar("losses/total_loss", loss, global_step)
        else:
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # clip gradients to prevent large updates
            if args.clip_grad:
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            actor_optimizer.step()
            critic_optimizer.step()

        writer.add_scalar("losses/actor_loss", actor_loss, global_step)
        writer.add_scalar("losses/critic_loss", critic_loss, global_step)
        if args.shared_network:
            writer.add_scalar("losses/total_loss", loss, global_step)
        writer.add_scalar("losses/entropy", entropy, global_step)

        # track gradients
        for name, param in policy.named_parameters():
            if param.grad is not None:
                writer.add_scalar(f'gradients/{name}/norm', param.grad.norm(), epoch)
                writer.add_histogram(f'gradients/{name}/histogram', param.grad, epoch)

    print(f"Total training time: {time.time() - start_time}")