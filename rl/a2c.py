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
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 128
    """number of env rollout steps in a single gradient optimization step"""
    actor_loss_w: float = 0.5
    """weight for the actor loss component of total loss"""
    max_grad_norm: float = 0.5
    """scale factor for clipping gradients"""


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


class Network(nn.Module):
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
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
        self.shared = nn.Sequential(*layers)
        self.actor_head = nn.Linear(sizes[-1], action_space)
        self.critic_head = nn.Linear(sizes[-1], 1)
    
    def forward(self, state):
        shared_features = self.shared(state)
        actor_output = self.actor_head(shared_features)
        critic_output = self.critic_head(shared_features)
        return actor_output, critic_output


def compute_actor_loss(actions, logits, advantages):
    """Computes loss for policy gradient based on action taken in state, over the entire batch."""
    log_probs = Categorical(logits=logits).log_prob(actions) # action is index of the action to be taken
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
    returns = torch.as_tensor(returns).to(device)
    return F.mse_loss(returns, batch_values)


def compute_advantages(ep_rewards, ep_values):
    """Advantage uses full episodic reward-to-go rather than an estimate."""
    advs = []
    for i in reversed(range(len(ep_rewards))):
        if i == len(ep_rewards) - 1:
            # use value function estimate for terminal state
            R = ep_values[i]
            # R = ep_rewards[i] # to use actual return
        else:
            R = ep_rewards[i] + args.gamma * R
        # to use TD error, use R + ep_values[i+1] - ep_values[i]
        advs.append(R - ep_values[i]) # don't detach values since actor/critic share the same network
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

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device="mps"

    # setup envs; IMPORTANT: increment the seed for each environment! otherwise it will sample the same actions
    envs = gym.vector.SyncVectorEnv(
        [env_maker(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])

    obs_space = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space.n
    hidden_sizes = [32]
    sizes = [obs_space] + hidden_sizes

    # setup the actor, critic
    # NOTE: in the official algorithm the actor and critic share params
    policy = Network(sizes, action_space, activation=nn.Tanh).to(device)
    # track the kl div across updates
    prev_actor = Network(sizes, action_space, activation=nn.Tanh).to(device) 
    # TODO: add logic to update prev actor and compute KL div
    prev_actor.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    # start the simulation
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    for epoch in range(args.num_epochs):
        batch_done = False # batch ends at batch_size episodes
        batch_states = [] # (B,4)
        batch_rewards = [] # (B,)
        batch_actions = [] # (B,)
        batch_action_logits = [] # (B,2)
        batch_advantages = [] # (B,)
        batch_values = []
        batch_ep_dones = []
        ep_rewards = [] # (T,)
        ep_values = [] # (T,)
        while not batch_done:
            # note: critic evaluated at s_t (obs), NOT next_obs (s_t+1)
            actions_logits, v = policy(torch.Tensor(obs).to(device))
            # sample an action; forward pass
            actions_dist = Categorical(logits=actions_logits)
            # stochastic action selection
            action = actions_dist.sample() # keep action on device until appended to batch_actions to track gradient properly

            # take a simulation step; currently only supports a single thread
            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
            global_step += 1

            # update episode and batch info
            ep_rewards.append(rewards.item())
            ep_values.append(v)
            batch_states.append(obs.copy())
            batch_actions.append(action)
            batch_action_logits.append(actions_logits)
            batch_values.append(v)
            batch_rewards.append(rewards.item())

            # important step, easy to forget - increment the current obs
            obs = next_obs

            # at episode end, compute advantages starting from terminal state, and reset the env
            if terminations.item() or truncations.item():
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

                ep_advantages = compute_advantages(ep_rewards, ep_values)
                batch_advantages.extend(ep_advantages)
                batch_ep_dones.append(1)
                obs, _ = envs.reset()
                ep_rewards, ep_values = [], []
                if len(batch_states) > args.batch_size:
                    break
            else:
                batch_ep_dones.append(0)

        # update actor, critic
        actions = torch.as_tensor(batch_actions).to(device)
        logits = torch.stack(batch_action_logits).squeeze()
        advantages = torch.as_tensor(batch_advantages).to(device)
        b_rewards = torch.as_tensor(batch_rewards).to(device)
        values = torch.as_tensor(batch_values).to(device)
        dones = np.array(batch_ep_dones)
            
        actor_loss = compute_actor_loss(actions, logits, advantages)
        critic_loss = compute_critic_loss(b_rewards, values, dones)
        loss = args.actor_loss_w * actor_loss + (1 - args.actor_loss_w) * critic_loss
        optimizer.zero_grad() # alter this to implement gradient accumulation
        loss.backward()
        if global_step % 100 == 0:
            writer.add_scalar("losses/actor_loss", actor_loss, global_step)
            writer.add_scalar("losses/critic_loss", critic_loss, global_step)
            writer.add_scalar("losses/total_loss", loss, global_step)

        # track gradients
        for name, param in policy.named_parameters():
            if param.grad is not None:
                writer.add_scalar(f'gradients/{name}/norm', param.grad.norm(), epoch)
                writer.add_histogram(f'gradients/{name}/histogram', param.grad, epoch)

        # clip gradients to prevent large updates
        nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()