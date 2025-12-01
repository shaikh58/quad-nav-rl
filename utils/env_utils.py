import numpy as np
import os
from typing import List, Dict, Tuple
import gymnasium as gym
from utils.planner import StraightLinePlanner
from gymnasium.wrappers.utils import RunningMeanStd
from gymnasium.core import ActType, ObsType, WrapperObsType
from utils.env_config_generator import EnvironmentConfigGenerator
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLANNER_TYPES = {
    "straight_line": StraightLinePlanner,
}


def make_env(env_id, idx, capture_video, gamma, seed, **kwargs):
    def thunk():
        config_generator = EnvironmentConfigGenerator(seed=seed, **kwargs)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", 
                config_generator=config_generator, **kwargs)
            video_path = kwargs.get("video_path", f"videos/{kwargs.get('run_name', None)}")
            env = gym.wrappers.RecordVideo(env, f"{video_path}", episode_trigger=lambda x: x % 200 == 0)
        else:
            env = gym.make(env_id, config_generator=config_generator, **kwargs)
        # TODO: move this to config generator
        if kwargs.get("use_planner", False):
            planner = PLANNER_TYPES[kwargs.get("planner_type", "straight_line")]()
            trajectory, info = planner.plan_trajectory(env.unwrapped.init_qpos[:3], env.unwrapped._target_location)
            env.unwrapped.set_trajectory(trajectory, info)

        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.RescaleAction(env, env.action_space.low, env.action_space.high)
        save_path = kwargs.get("save_path", None)
        mode = kwargs.get("mode", None)
        env = NormalizeObservation(env, mode=mode, save_path=save_path)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # no need to set truncations in env.step() as TimeLimit wrapper handles it
        max_steps = kwargs.get("max_steps", 500)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        return env

    return thunk


def multiply_quaternions(q0, q1):
    """
    Multiplies two quaternions.
    Quaternions are represented as lists/tuples [w, x, y, z].
    """
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1

    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    x = w0*x1 + x0*w1 + y0*z1 - z0*y1
    y = w0*y1 - x0*z1 + y0*w1 + z0*x1
    z = w0*z1 + x0*y1 - y0*x1 + z0*w1

    return np.array([w, x, y, z])

def spherical_to_cartesian(r, az, el):
    x = r * np.cos(az) * np.cos(el)
    y = r * np.sin(az) * np.cos(el)
    z = r * np.sin(el)
    return np.array([x, y, z])

def create_lidar_scan_frame(lidar_scan, goal_vec_body_frame, list_obs_vec_body_frame, lidar_scan_range,
    lidar_min_elevation, lidar_max_elevation, lidar_min_fov, lidar_max_fov, lidar_elevation_bins, lidar_scan_bins):
    """Simple 3D visualization of voxel grid using matplotlib in body frame."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=45)

    # Add blue marker at the origin (0,0,0) for the agent position in body frame
    ax.scatter(0, 0, 0, c='b', marker='o', s=200, alpha=1.0, label='Agent')
    # obstacle positions
    # for obs_vec in list_obs_vec_body_frame:
    #     ax.scatter(obs_vec[0], obs_vec[1], obs_vec[2], 
    #                 c='r',marker='s',s=100,alpha=0.6)
    # goal position
    ax.scatter(goal_vec_body_frame[0], goal_vec_body_frame[1], goal_vec_body_frame[2], 
                c='g', marker='s', s=100, alpha=0.6, label=f'Goal: {goal_vec_body_frame}')
    # lidar scan
    # convert scan back into body frame
    occupied = np.where(lidar_scan < lidar_scan_range)
    for elev_idx, scan_idx in zip(*occupied):
        elevation_angle = lidar_min_elevation + elev_idx * (lidar_max_elevation - lidar_min_elevation) / (lidar_elevation_bins - 1)
        scan_angle = np.pi/2 - (lidar_min_fov + scan_idx * (lidar_max_fov - lidar_min_fov) / (lidar_scan_bins - 1))
        r = lidar_scan[elev_idx, scan_idx]
        x, y, z = spherical_to_cartesian(r, scan_angle, elevation_angle)
        ax.scatter(x, y, z, c='k', marker='.', s=10, alpha=0.8)
    ax.set_xlabel('X (body frame)')
    ax.set_ylabel('Y (body frame)')
    ax.set_zlabel('Z (body frame)')
    ax.set_xlim(-lidar_scan_range, lidar_scan_range)
    ax.set_ylim(-lidar_scan_range, lidar_scan_range)
    ax.set_zlim(-lidar_scan_range, lidar_scan_range)
    ax.legend()
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = buf.reshape((height, width, 4))
    plt.close(fig)
    return image[:, :, :3]

def save_lidar_scan_video(lidar_scan_frames, episode_count, output_dir, fps=10):
    """Save accumulated frames as video."""
    if len(lidar_scan_frames) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"lidar_scan_episode_{episode_count}.mp4")
    imageio.mimsave(filename, lidar_scan_frames, fps=fps)


class NormalizeObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """REIMPLEMENTED TO SUPPORT SAVING/LOADING RUNNING STATS FOR EVALUATION
    Taken from https://gymnasium.farama.org/_modules/gymnasium/wrappers/stateful_observation/#NormalizeObservation
    Normalizes observations to be centered at the mean with unit variance.
    """

    def __init__(self, env: gym.Env[ObsType, ActType], mode: str, save_path: str = None, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.ObservationWrapper.__init__(self, env)

        assert env.observation_space.shape is not None
        self.obs_rms = RunningMeanStd(
            shape=env.observation_space.shape, dtype=env.observation_space.dtype
        )
        self.epsilon = epsilon
        self._update_running_mean = mode == "train"
        self.save_path = save_path
        if mode == "eval":
            self.load_obs_rms()

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting
    
    def load_obs_rms(self) -> None:
        if self.save_path is not None:
            data = np.load(os.path.join(self.save_path, "obs_rms.npz"))
            self.obs_rms.mean = data["mean"]
            self.obs_rms.var = data["var"]
            self.epsilon = data["epsilon"]

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Normalises the observation using the running mean and variance of the observations."""
        if self._update_running_mean:
            self.obs_rms.update(np.array([observation]))
            if self.save_path is not None:
                np.savez(os.path.join(self.save_path, "obs_rms.npz"), mean=self.obs_rms.mean, var=self.obs_rms.var, epsilon=self.epsilon)
        
        return np.float32(
            (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        )


def dda_voxel_traversal_to_goal(start_grid, goal_position_grid, max_steps, grid_size):
    """
    3D DDA voxel grid traversal from start toward goal.
    Stops when reaching the goal voxel OR leaving the grid.
    
    Args:
        start_grid: Starting position in grid coordinates (z, y, x)
        goal_position_grid: Goal position in grid coordinates (z, y, x) - ABSOLUTE position, not relative
        max_steps: Maximum number of voxels to traverse
    
    Returns:
        Final voxel position (z, y, x) where ray stopped
    """
    # Direction from start to goal
    direction_grid = goal_position_grid - start_grid
    goal_voxel = np.floor(goal_position_grid).astype(int)
    current_voxel = np.floor(start_grid).astype(int)
    direction_normalized = direction_grid / np.linalg.norm(direction_grid)
    # Step direction for each axis (+1 or -1)
    step = np.sign(direction_normalized).astype(int)
    
    # tDelta: how far along the ray we must move (in units of t) to cross one voxel boundary
    tDelta = np.zeros(3)
    for i in range(3):
        if direction_normalized[i] != 0:
            tDelta[i] = abs(1.0 / direction_normalized[i])
        else:
            tDelta[i] = float('inf')
    
    # tMax: how far along the ray we must move to reach the next voxel boundary on each axis
    tMax = np.zeros(3)
    for i in range(3):
        if direction_normalized[i] > 0:
            tMax[i] = (np.ceil(start_grid[i]) - start_grid[i]) * tDelta[i]
        elif direction_normalized[i] < 0:
            tMax[i] = (start_grid[i] - np.floor(start_grid[i])) * tDelta[i]
        else:
            tMax[i] = float('inf')
    
    # Handle case where we start exactly on a boundary i.e. at the grid center
    for i in range(3):
        if tMax[i] == 0:
            tMax[i] = tDelta[i]
    
    for _ in range(max_steps):
        # Check if current voxel is out of bounds
        if (current_voxel < 0).any() or (current_voxel >= grid_size).any():
            # if oob, we want to project goal onto closest grid face anyway
            return np.clip(current_voxel, 0, grid_size - 1)
        # Check if we've reached the goal voxel; kept separate for easier debugging
        if np.array_equal(current_voxel, goal_voxel):
            # print("Reached goal voxel: ", current_voxel)
            return current_voxel
        
        # Find which axis to step on (the one with smallest tMax)
        axis = np.argmin(tMax)
        
        # Step to next voxel on that axis
        current_voxel[axis] += step[axis]
        tMax[axis] += tDelta[axis]
    
    # Max steps reached, return current position
    return np.clip(current_voxel, 0, grid_size - 1)

def fill_2d_gaussian_blob(voxel_grid, voxel, axis, grid_size, grid_res, goal_blob_std):
    """
    Fill in a 2D Gaussian blob around the given voxel.
    """
    zc, yc, xc = voxel
    if axis == "z":
        X, Y = np.meshgrid(np.arange(0, grid_size), np.arange(0, grid_size))
        dx = (X - xc) * grid_res
        dy = (Y - yc) * grid_res
        kernel = np.exp(-(dx**2 + dy**2) / (2 * goal_blob_std**2))
        voxel_grid[1, zc, :, :] = kernel
    elif axis == "y":
        X, Z = np.meshgrid(np.arange(0, grid_size), np.arange(0, grid_size))
        dx = (X - xc) * grid_res
        dz = (Z - zc) * grid_res
        kernel = np.exp(-(dx**2 + dz**2) / (2 * goal_blob_std**2))
        voxel_grid[1, :, yc, :] = kernel
    elif axis == "x":
        Y, Z = np.meshgrid(np.arange(0, grid_size), np.arange(0, grid_size))
        dy = (Y - yc) * grid_res
        dz = (Z - zc) * grid_res
        kernel = np.exp(-(dy**2 + dz**2) / (2 * goal_blob_std**2))
        voxel_grid[1, :, :, xc] = kernel
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    return voxel_grid

def fill_3d_gaussian_blob(voxel_grid, voxel, fill_channel, grid_size, grid_res, goal_blob_std):
    """
    Fill in a 3D Gaussian blob around the given voxel.
    """
    zc, yc, xc = voxel
    Z, Y, X = np.meshgrid(np.arange(0, grid_size), np.arange(0, grid_size), np.arange(0, grid_size))
    dx = (X - xc) * grid_res
    dy = (Y - yc) * grid_res
    dz = (Z - zc) * grid_res
    kernel = np.exp(-(dx**2 + dy**2 + dz**2) / (2 * goal_blob_std**2))
    voxel_grid[fill_channel, :, :, :] = kernel
    return voxel_grid

def fill_gaussian_blob(voxel_grid, voxel, fill_channel, grid_size, grid_res, goal_blob_std):
    """
    Fill in a Gaussian blob around the given voxel. If the goal is within the grid, fill in a 3d gaussian blob.
    Otherwise, fill in a 2d gaussian blob on the grid face.
    """
    zc, yc, xc = voxel
    if zc == 0 or zc == grid_size - 1 or yc == 0 or yc == grid_size - 1 or xc == 0 or xc == grid_size - 1:
        # 2d gaussian on the grid face
        if zc == 0 or zc == grid_size - 1: # top or bottom face
            voxel_grid = fill_2d_gaussian_blob(voxel_grid, voxel, "z", grid_size, grid_res, goal_blob_std)
        elif yc == 0 or yc == grid_size - 1: # front or back face
            voxel_grid = fill_2d_gaussian_blob(voxel_grid, voxel, "y", grid_size, grid_res, goal_blob_std)
        elif xc == 0 or xc == grid_size - 1: # left or right face
            voxel_grid = fill_2d_gaussian_blob(voxel_grid, voxel, "x", grid_size, grid_res, goal_blob_std)
    else:
        # 3d gaussian in the grid
        # print("Goal in grid: ", voxel)
        voxel_grid = fill_3d_gaussian_blob(voxel_grid, voxel, fill_channel, grid_size, grid_res, goal_blob_std)
    return voxel_grid