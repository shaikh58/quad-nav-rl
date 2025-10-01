# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from utils.env_utils import make_env
import envs
from qpth.qp import QPFunction
from utils import utils

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
    wandb_project_name: str = "quad_nav_rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "envs/QuadNav-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # environment specific arguments
    use_planner: bool = True
    """whether to use a planner"""
    planner_type: str = "straight_line"
    """the type of planner to use"""
    env_radius: float = 15
    """the environment radius"""
    goal_threshold: float = 0.5
    """distance to goal for success"""
    adaptive_goal_threshold: bool = False
    """whether to adapt the goal threshold based on the training stage"""
    progress_type: str = "euclidean"
    """the type of progress to use"""
    min_height: float = 0.1
    """minimum height above ground before ground collision"""
    ctrl_cost_weight: float = 0.1
    """control cost penalty for reward"""
    progress_weight: float = 0.1
    """weight for progress term in reward"""
    body_rate_weight: float = 0.1
    """weight for body rate penalty in reward"""
    collision_ground_weight: float = 1
    """weight for ground collision penalty in reward"""
    collision_obstacles_weight: float = 1
    """weight for obstacle collision penalty in reward"""
    out_of_bounds_weight: float = 1
    """weight for out of bounds penalty in reward"""
    success_weight: float = 10
    """weight for success term in reward"""
    start_location: str = None
    """the start location"""
    target_location: str = None
    """the target location"""
    max_steps: int = 2000
    """the maximum number of steps per episode before truncation"""
    reset_noise_scale: float = 1e-1
    """the noise scale for the reset"""
    use_obstacles: bool = False
    """whether to use obstacles"""
    regen_obstacles: bool = False
    """whether to regenerate obstacles at the end of each episode (w.p. obs_regen_eps)"""
    obs_regen_eps: float = 0.5
    """the probability with which to regenerate obstacles"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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

class QP(nn.Module):
    def __init__(self, dim, nineq, neq, eps=1e-4, envs=None):
        """More details available at https://locuslab.github.io/qpth/"""
        super().__init__()
        self.envs = envs
        self.nineq = nineq
        self.neq = neq
        self.eps = eps
        self.dim = dim
        self.M = torch.tril(torch.ones(dim, dim))
        self.L = torch.nn.Parameter(torch.tril(torch.rand(dim, dim)))
        # G (nineq, dim) is the CBF constraint so should not be learned
        # self.G = torch.Tensor(nineq,dim, requires_grad=False)
        self.z0 = torch.nn.Parameter(torch.zeros(dim)) # TODO: possibly make this a learnable parameter
        self.s0 = torch.nn.Parameter(torch.ones(nineq)) # learnable slack variable
        self.alpha = 1 # CBF tuning parameter; extended class K function
        self.inertia, self.inv_inertia = utils.get_inertia(envs)


    def forward(self, x, u_rl, sdfs, grads):
        """Solve QP: argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b
        where z is the compensation control to be determined.
        sdf, grad_sdf is evaluated at x for EACH environment i.e. x is 
        batch of states. But NOTE: the state is not used in in the QP as its indirectly 
        used to compute the sdf and grad_sdf.
        x: (n_envs, state_dim)
        u_rl: (n_envs, action_dim)
        sdfs: (n_envs, 1)
        grads: (n_envs, state_dim)
        """
        # f is batched (n_envs, state_dim)
        f = utils.f(x, gravity=np.abs(self.envs.envs[0].unwrapped.model.opt.gravity[-1]), 
                    inertia=self.inertia, inv_inertia=self.inv_inertia).to(x.device)
        g = utils.g(x).to(x.device) # batched (n_envs, state_dim, state_dim)
        self.G = -torch.mm(grads, g) # sign flipped to get CBF into QP canonical form
        self.G = torch.as_tensor(self.G, requires_grad=False) # again, don't learn G
        L = self.M*self.L # enforce lower triangular
        Q = L.mm(L.t()) + self.eps*torch.eye(self.dim)
        # h = self.G.mv(self.z0)+self.s0 # TODO: qpth recommends this initialization
        # CBF formulation RHS is expanded to isolate the RL control from the optimization variable u
        # and get the inequality in canonical form Gz <= h
        # NOTE: sign is +ve since QP canonical form uses less than for inequality constraints
        # h is batched (n_envs, 1)
        # TODO: fix the dimensions of the args of h esp the last term
        h = self.s0 + self.alpha * sdfs + torch.mm(grads, f) + torch.mm(grads, torch.bmm(g, u_rl))
        e = torch.Tensor() # empty placeholder since no equality constraints
        p = torch.zeros(self.dim, requires_grad=False)
        x = QPFunction(verbose=False)(Q, p, self.G, h, e, e)
        return x


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

        self.cbf_qp = QP(dim=np.array(envs.single_observation_space.shape).prod(),
                         nineq=1, neq=0, eps=1e-4, envs=envs)

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
            self.actor_final_layer.bias.fill_(envs.envs[0].unwrapped.model.keyframe('hover').ctrl[0])
            # with action scaling - makes it worse
            # hover_thrust = envs.envs[0].unwrapped.model.keyframe('hover').ctrl[0]
            # # scale to [-1,1]
            # scaled_hover_thrust = -1 + 2 * (hover_thrust - -1) / 2
            # self.actor_mean[-1].bias.fill_(scaled_hover_thrust)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, sdfs, grads, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # once the RL control is computed, we pass it through the safety compensator
        u_safe = self.cbf_qp(x, action, sdfs, grads)
        u_out = action_mean + u_safe
        # TODO: figure out how the log_prob changes because this is used to compute the policy loss! and returns?
        return u_out, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y%m%d_%H%M%S')}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            # entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    target_location = [float(y) for x in args.target_location.split(",") for y in x if y.isdigit()] if args.target_location is not None else None
    start_location = [float(y) for x in args.start_location.split(",") for y in x if y.isdigit()] if args.start_location is not None else None

    env_kwargs = {
        "env_radius": args.env_radius,
        "ctrl_cost_weight": args.ctrl_cost_weight,
        "progress_weight": args.progress_weight,
        "progress_type": args.progress_type,
        "body_rate_weight": args.body_rate_weight,
        "collision_ground_weight": args.collision_ground_weight,
        "collision_obstacles_weight": args.collision_obstacles_weight,
        "out_of_bounds_weight": args.out_of_bounds_weight,
        "success_weight": args.success_weight,
        "goal_threshold": args.goal_threshold,
        "min_height": args.min_height,
        "target_location": target_location,
        "start_location": start_location,
        "max_steps": args.max_steps,
        "reset_noise_scale": args.reset_noise_scale,
        "save_path": f"runs/{run_name}",
        "mode": "train",
        "use_obstacles": args.use_obstacles,
        "regen_obstacles": args.regen_obstacles,
        "obs_regen_eps": args.obs_regen_eps,
    } # note the target and start location are set in the env randomizer but can be overridden by the user
    # for negative rewards (w./only positive at goal), we dont want to discount the reward
    if args.progress_type == "negative": 
        args.gamma = 1
    # NOTE: the seed is the same for all envs to ensure same start/goal locations chosen by randomizer
    # the seed will only be args.seed+i for having e.g. different start locations in each env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args.seed, use_planner=args.use_planner, planner_type=args.planner_type, **env_kwargs) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=[args.seed+i for i in range(args.num_envs)])
    for i, env in enumerate(envs.envs): 
        print(f"Env: {i}", "Start: ", env.unwrapped.init_qpos[:3], "Target: ", env.unwrapped._target_location, 
        "Env radius: ", env.unwrapped.model.stat.extent, 
        "Distance from center: ", np.linalg.norm(env.unwrapped.init_qpos[:3] - env.unwrapped.model.stat.center),
        "Target distance from center: ", np.linalg.norm(env.unwrapped._target_location - env.unwrapped.model.stat.center))
    next_obs = torch.Tensor(next_obs).to(device)
    sdfs = torch.Tensor(infos["sdf"]).to(device)
    grads = torch.Tensor(infos["grad"]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    try:
        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs, sdfs, grads)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                sdfs = torch.Tensor(infos["sdf"]).to(device)
                grads = torch.Tensor(infos["grad"]).to(device)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                # upon episode completion, vectorEnv handles resets automatically
                for k in range(args.num_envs):
                    if truncations[k] or terminations[k]:
                        if infos["termination_msg"][k] == "collision_obstacles":
                            print("Collision with obstacles")
                        if infos["termination_msg"][k] == "success":
                            print("Success!!!", "End agent pos: ", infos["pos"][k], "Target pos: ", envs.envs[k].unwrapped._target_location)
                        # print(f"global_step={global_step}, episodic_return={mean_return}, episodic_length={mean_length}")
                        if "episode" in infos:
                            writer.add_scalar("charts/episodic_return", infos["episode"]["r"][k], global_step)
                            writer.add_scalar("charts/episodic_length", infos["episode"]["l"][k], global_step)
            
            if args.adaptive_goal_threshold:
                for k in range(args.num_envs):
                    frac = 1.0 - (iteration - 1.0) / args.num_iterations
                    envs.envs[k].unwrapped._goal_threshold = frac * args.goal_threshold
                    
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                        # 1-step TD error for GAE
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    # recursive update of GAE (exponentially weighted average of k-step advantage estimates)
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_sdfs = sdfs.reshape(-1)
            b_grads = grads.reshape((-1,) + envs.single_observation_space.shape) # grads are of shape (1,state_dim)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    # mb_inds are the minibatch indices i.e. batch split into equal sized chunks
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], b_sdfs[mb_inds], b_grads[mb_inds])
                    # compute the surrogate policy ratio pi/pi_old - in log space for numerical stability
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    # for keeping track of KL divergence
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # negative of (L_clip - L_value + L_entropy)
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)), "global_step:", global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            # track gradients
            for name, param in agent.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f'gradients/{name}/norm', param.grad.norm(), global_step)
                    writer.add_histogram(f'gradients/{name}/histogram', param.grad, global_step)
        print("Total training time:", time.time() - start_time)
        # save some videos (create an env, rollout, and capture the video)

    except KeyboardInterrupt:
        print("Training interrupted by user - saving model and running eval")
        
    if args.save_model:
        import imageio

        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        print("Beginning eval...") 
        video_path=f"videos/{run_name}-eval"
        # get the actual start/goal/extent used for training (after randomization)
        target_location = envs.envs[0].unwrapped._target_location
        radius = envs.envs[0].unwrapped.model.stat.extent
        # override the env kwargs with the actual start/goal/extent used for training
        # don't pass in start location as it's randomized during training and rollout by the env_config_generator
        env_kwargs["start_location"] = None
        env_kwargs["target_location"] = target_location
        env_kwargs["radius"] = radius
        env_kwargs["goal_threshold"] = args.goal_threshold
        env_kwargs["mode"] = "eval"

        n_episodes = 5

        # custom eval using exact same env as env of rank 0 (this is the env that has video capture during training)
        envs = gym.vector.SyncVectorEnv(# NOTE: we don't randomize the env during inference, except start position
        [make_env(args.env_id, 0, False, 
        run_name, args.gamma, args.seed, use_planner=args.use_planner, planner_type=args.planner_type,  
         **env_kwargs
        ) for i in range(1)]
        )
        obs, infos = envs.reset(seed=args.seed) # need to call with seed first time
        sdfs = torch.Tensor(infos["sdf"]).to(device)
        grads = torch.Tensor(infos["grad"]).to(device)
        for i in range(n_episodes):
            video = []
            over = False
            iters = 0
            while not over and iters < 1000: # manual termination to prevent infinite episode length
                with torch.no_grad():
                    u_rl, *_ = agent.actor_mean(torch.tensor(obs).to(device))
                    u_safe = agent.cbf_qp(obs, u_rl, sdfs, grads)
                    action = u_rl + u_safe
                    # agent.get_action_and_value(torch.tensor(obs).to(device), sdfs, grads) # deterministic action during inference
                obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                sdfs = torch.Tensor(infos["sdf"]).to(device)
                grads = torch.Tensor(infos["grad"]).to(device)
                rgb_array = envs.render()
                video.append(rgb_array)
                iters += 1
                over = terminated or truncated
            os.makedirs(video_path, exist_ok=True)
            imageio.mimsave(f"{video_path}/ep_{i}.mp4", np.array(video).squeeze(), fps=30)
            obs, infos = envs.reset() # don't call with seed again as it changes rng state
            sdfs = torch.Tensor(infos["sdf"]).to(device)
            grads = torch.Tensor(infos["grad"]).to(device)
        envs.close()

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()