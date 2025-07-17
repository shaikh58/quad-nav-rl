import numpy as np

def softmin(inputs, alpha=100):
    return -(1/alpha) * np.log(np.sum(np.exp(-alpha * inputs)))

def cbf(x):

def grad_sdf_softmin(grad_dim, dists_to_obstacles, alpha):
    """Gradient of softmin wrt state."""
    grad = np.zeros(grad_dim)
    # grad of SDF wrt orientation and angular velocity = 0
    weights = np.exp(- alpha * dists_to_obstacles)/np.sum(np.exp(-alpha * dists_to_obstacles))
    # TODO: finish this implementation
    # grad_sdf = np.sum(weights * )
    grad[:3] = grad_sdf
    return grad

def grad_sdf(state, closest_obs_pos):
    """Since spherical obstacles, use analytical gradient wrt state."""
    grad = np.zeros_like(state)
    # grad of SDF wrt orientation and angular velocity = 0
    # normalize the gradient for more stability when further away from obstacle
    dir_vec = (state[:3] - closest_obs_pos) / np.linalg.norm(state[:3] - closest_obs_pos)
    grad[:3] = dir_vec
    return grad
