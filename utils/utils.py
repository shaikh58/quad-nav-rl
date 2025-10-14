"""Deprecated - no longer using CBF QP"""


import numpy as np
import torch


def softmin(inputs, alpha=100):
    return -(1/alpha) * np.log(np.sum(np.exp(-alpha * inputs)))

def get_inertia(envs):
    l_inertia = []
    l_inv_inertia = []
    for env in envs.envs:
        inertia_diag = env.unwrapped.model.body_inertia[env.unwrapped.model.body('x2').id]
        inertia = torch.diag(torch.as_tensor(inertia_diag))
        inv_inertia = torch.diag(1.0 / torch.as_tensor(inertia_diag))
        l_inertia.append(inertia)
        l_inv_inertia.append(inv_inertia)
    return torch.stack(l_inertia), torch.stack(l_inv_inertia)

def f(x, gravity, inertia, inv_inertia):
    """First part of quadrotor dynamics in form x_dot = f(x) + g(x) u.
    Requires x to be batched input (n_envs, state_dim).
    """
    if x.dim() == 1: x = x.unsqueeze(0) # ensure x has at least 2 dims
    assert x.shape[-1] == 13, "State dimension must be 13"
    F = torch.zeros(*x.shape, device=x.device, dtype=x.dtype)
    F[:, :3] = x[:, 3:6]
    # 1/2 * S(w) * q; derivative of quaternion
    F[:, 3:6] = 0.5 * torch.bmm(S(x[:, 10:]), x[:, 3:7].unsqueeze(-1)).squeeze(-1)
    b_e3 = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
    b_e3[:, 2] = 1
    F[:, 7:10] = -gravity * b_e3
    # J^-1 (-w x J w)
    b_inertia = inertia.unsqueeze(0).expand(x.shape[0], -1, -1)
    b_inv_inertia = inv_inertia.unsqueeze(0).expand(x.shape[0], -1, -1)
    Jw = torch.bmm(b_inertia, x[:, 10:].unsqueeze(-1)).squeeze(-1)
    F[:, 10:] = torch.bmm(b_inv_inertia, torch.linalg.cross(-x[:, 10:], Jw, dim=-1))
    return F

def g(x, mass, inv_inertia):
    """Second part of quadrotor dynamics in form x_dot = f(x) + g(x) u.
    Requires x to be batched input (n_envs, state_dim).
    G is of shape (n_envs, state_dim, state_dim).
    """
    assert x.shape[-1] == 13, "State dimension must be 13"
    G = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1]), device=x.device, dtype=x.dtype)
    return

def S(w):
    """Quaternion derivative in matrix form. w is angular velocity.
    Assumes Hamilton convention: q = [w, x, y, z] where w is the scalar part.
    Supports batched input (n_envs, 3) for w.
    Returns tensor of shape (n_envs, 4, 4)
    """
    batch_size = w.shape[0]
    S_matrices = torch.zeros(batch_size, 4, 4, device=w.device, dtype=w.dtype)
    
    # Fill in the matrix elements for each batch
    S_matrices[:, 0, 1] = -w[:, 0]  # -w_x
    S_matrices[:, 0, 2] = -w[:, 1]  # -w_y  
    S_matrices[:, 0, 3] = -w[:, 2]  # -w_z
    
    S_matrices[:, 1, 0] = w[:, 0]   # w_x
    S_matrices[:, 1, 2] = w[:, 2]   # w_z
    S_matrices[:, 1, 3] = -w[:, 1]  # -w_y
    
    S_matrices[:, 2, 0] = w[:, 1]   # w_y
    S_matrices[:, 2, 1] = -w[:, 2]  # -w_z
    S_matrices[:, 2, 3] = w[:, 0]   # w_x
    
    S_matrices[:, 3, 0] = w[:, 2]   # w_z
    S_matrices[:, 3, 1] = w[:, 1]   # w_y
    S_matrices[:, 3, 2] = -w[:, 0]  # -w_x
    
    return S_matrices


def grad_sdf(state, closest_obs_pos):
    """Since spherical obstacles, use analytical gradient wrt state.
    Assume grad of SDF wrt orientation and angular velocity = 0.
    Doesn't need to be batched as gym VectorEnv handles batching.
    """
    grad = np.zeros_like(state)
    # normalize the gradient for more stability when further away from obstacle
    dir_vec = (state[:3] - closest_obs_pos) / np.linalg.norm(state[:3] - closest_obs_pos)
    grad[:3] = dir_vec
    return grad


def grad_sdf_softmin(grad_dim, dists_to_obstacles, alpha):
    """Gradient of softmin wrt state."""
    grad = np.zeros(grad_dim)
    # grad of SDF wrt orientation and angular velocity = 0
    weights = np.exp(-alpha * dists_to_obstacles)/np.sum(np.exp(-alpha * dists_to_obstacles))
    # TODO: finish this implementation
    # grad_sdf = np.sum(weights * )
    # grad[:3] = grad_sdf
    return grad