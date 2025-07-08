import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import numpy as np
from rl.ppo_quad_nav import train

wandb.login()

sweep_config = {
    'method': 'random'
    }
metric = {
    'name': 'ep_ret',
    'goal': 'maximize'   
    }
parameters_dict = {
    'seed': {
        'value': 1
        },
    'ctrl_cost_weight': {
        'value': 0.1
        },
    'progress_weight': {
        'value': 100
        },
    'body_rate_weight': {
        'value': 0.5
        },
    'total_timesteps': {
        'value': 600000
        },
    'num_envs': {
        'value': 1
        },
    'env_id': {
        'value': "envs/QuadNav-v0"
        },
    'learning_rate': {
        'value': 3e-4
        },
    'num_steps': {
        'value': 2048
        },
    'anneal_lr': {
        'value': True
        },
    'gamma': {
        'value': 0.99
        },
    'gae_lambda': {
        'value': 0.95
        },
    'num_minibatches': {
        'value': 32
        },
    'update_epochs': {
        'value': 10
        },
    'norm_adv': {
        'value': True
        },
    'clip_coef': {
        'value': 0.2
        },
    'clip_vloss': {
        'value': True
        },
    'ent_coef': {
        'value': 0.0
        },
    'vf_coef': {
        'value': 0.5
        },
    'max_grad_norm': {
        'value': 0.5
        },
    'target_kl': {
        'value': None
        },
    'use_planner': {
        'value': True
        },
    'planner_type': {
        'value': "straight_line"
        },
    'env_radius_lb': {
        'value': 5
        },
    'env_radius_ub': {
        'value': 20
        },
    'goal_threshold': {
        'value': 0.5
        },
    'adaptive_goal_threshold': {
        'value': True
        },
    'progress_type': {
        'value': "euclidean"
        },
    'min_height': {
        'value': 0.1
        },
    'collision_ground_weight': {
        'value': 10
        },
    'collision_obstacles_weight': {
        'value': 10
        },
    'out_of_bounds_weight': {
        'value': 20
        },
    'success_weight': {
        'value': 100
        },
    'max_steps': {
        'value': 1000
        },
    'reset_noise_scale': {
        'value': 1e-1
        },
    'mode': {
        'value': "train"
        },
    'start_x': {
        'value': -2.739671654742875
        },
    'start_y': {
        'value': 8.51618754426344
        },
    'start_z': {
        'value': 2.0611356976708777
        },
    'target_x': {
        'value': 5.83362735338293
        },
    'target_y': {
        'value': -2.387708107620822
        },
    'target_z': {
        'value': 4.942356173982961
        },
}

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="quad_nav_rl")
wandb.agent(sweep_id, train, count=1)