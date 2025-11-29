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

def create_voxel_grid_frame(voxel_grid, grid_size, grid_res, goal_vec_body_frame, list_obs_vec_body_frame, threshold=0.3):
    """Simple 3D visualization of voxel grid using matplotlib in body frame."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Grid center in indices
    grid_center_idx = np.array([grid_size // 2, grid_size // 2, grid_size // 2])
    grid_half_size = (grid_size // 2) * grid_res
    
    # Plot obstacles (red) and goal (green)
    obstacle_mask = voxel_grid[0] > threshold
    goal_mask = voxel_grid[1] > threshold
    # Add blue marker at the origin (0,0,0) for the agent position in body frame
    ax.scatter(0, 0, 0, c='b', marker='o', s=200, alpha=1.0, label='Agent')
    # Convert grid frame indices to body frame positions
    if np.any(obstacle_mask):
        z_grid, y_grid, x_grid = np.where(obstacle_mask)
        # Convert indices to grid frame positions (meters, centered at origin)
        pos_grid_frame = np.array([
            (x_grid - grid_center_idx[2]) * grid_res,  # x in grid frame
            (y_grid - grid_center_idx[1]) * grid_res,  # y in grid frame
            (z_grid - grid_center_idx[0]) * grid_res   # z in grid frame
        ]).T
        # Transform to body frame: body = [grid[2], -grid[1], -grid[0]]
        pos_body_frame = np.array([
            pos_grid_frame[:, 2],   # body x = grid z
            -pos_grid_frame[:, 1],  # body y = -grid y
            -pos_grid_frame[:, 0]   # body z = -grid x
        ]).T
        
        ax.scatter(pos_body_frame[:, 0], pos_body_frame[:, 1], pos_body_frame[:, 2], 
                    c='r',marker='s',s=100,alpha=0.6,
                    label='Obstacles:\n' + '\n'.join([f'{entry}' for entry in list_obs_vec_body_frame]))
    
    if np.any(goal_mask):
        z_grid, y_grid, x_grid = np.where(goal_mask)
        # Convert indices to grid frame positions (meters, centered at origin)
        pos_grid_frame = np.array([
            (x_grid - grid_center_idx[2]) * grid_res,  # x in grid frame
            (y_grid - grid_center_idx[1]) * grid_res,  # y in grid frame
            (z_grid - grid_center_idx[0]) * grid_res   # z in grid frame
        ]).T
        # Transform to body frame: body = [grid[2], -grid[1], -grid[0]]
        pos_body_frame = np.array([
            pos_grid_frame[:, 2],   # body x = grid z
            -pos_grid_frame[:, 1],  # body y = -grid y
            -pos_grid_frame[:, 0]   # body z = -grid x
        ]).T
        ax.scatter(pos_body_frame[:, 0], pos_body_frame[:, 1], pos_body_frame[:, 2], 
                    c='g', marker='s', s=100, alpha=0.6, label=f'Goal: {goal_vec_body_frame}')
    
    ax.set_xlabel('X (body frame)')
    ax.set_ylabel('Y (body frame)')
    ax.set_zlabel('Z (body frame)')
    ax.set_xlim(-grid_half_size, grid_half_size)
    ax.set_ylim(-grid_half_size, grid_half_size)
    ax.set_zlim(-grid_half_size, grid_half_size)
    ax.legend()
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = buf.reshape((height, width, 4))
    plt.close(fig)
    return image[:, :, :3]

def save_voxel_grid_video(voxel_grid_frames, episode_count, output_dir, fps=10):
    """Save accumulated frames as video."""
    if len(voxel_grid_frames) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"voxel_grid_episode_{episode_count}.mp4")
    imageio.mimsave(filename, voxel_grid_frames, fps=fps)


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