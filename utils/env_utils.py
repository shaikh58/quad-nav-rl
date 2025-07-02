import numpy as np
from typing import List, Dict, Tuple
import gymnasium as gym
from utils.planner import StraightLinePlanner
import random

PLANNER_TYPES = {
    "straight_line": StraightLinePlanner,
}


def make_env(env_id, idx, capture_video, run_name, gamma, seed, use_planner=None, planner_type=None, randomize_env=False, **kwargs):
    def thunk():
        random.seed(seed)
        np.random.seed(seed)

        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **kwargs)
            video_path = kwargs.get("video_path", f"videos/{run_name}")
            env = gym.wrappers.RecordVideo(env, f"{video_path}/{run_name}")
        else:
            env = gym.make(env_id, **kwargs)
        
        # randomly modify the environment; create the object inside for lazy env creation for vectorized envs
        if randomize_env:
            env_randomizer = EnvironmentRandomizer(env, seed=seed, **kwargs)
            env_randomizer.generate_env()

        if use_planner:
            planner = PLANNER_TYPES[planner_type]()
            trajectory, info = planner.plan_trajectory(env.unwrapped.init_qpos[:3], env.unwrapped._target_location)
            env.unwrapped.set_trajectory(trajectory, info)

        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # no need to set truncations in env.step() as TimeLimit wrapper handles it
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        return env

    return thunk


class EnvironmentRandomizer:
    """
    Random environment generator for MuJoCo-based gym environments.
    
    This class provides functionality to randomize:
    - Start and target positions
    - Obstacle positions, sizes, and types
    """
    
    def __init__(self, env, seed, **kwargs):
        self.env = env
        self.kwargs = kwargs
        self.obstacles = []
        self.env_radius_lb = self.kwargs.get("env_radius_lb", 3)
        self.env_radius_ub = self.kwargs.get("env_radius_ub", 15)
        self.start_orientation = np.array([1, 0, 0, 0])
        self.start_vel = np.array([0, 0, 0, 0, 0, 0])
        
        self.rng = np.random.default_rng(seed)

    def set_env_bounds(self) -> None:
        self.env.unwrapped.model.stat.extent = self.rng.uniform(self.env_radius_lb, self.env_radius_ub)
        
    def sample_sphere(self) -> np.ndarray:
        # sample x,y inside a spherical region; center, extent from the model file
        azimuth = self.rng.uniform(0, 2*np.pi) 
        elevation = self.rng.uniform(0, np.pi/2)
        # extent is radius of the sphere
        x = self.env.unwrapped.model.stat.extent * np.cos(azimuth) * np.sin(elevation)
        y = self.env.unwrapped.model.stat.extent * np.sin(azimuth) * np.sin(elevation)
        # constrain z so that the euclidean dist to center is less than extent
        x_c, y_c, z_c = self.env.unwrapped.model.stat.center
        z_max = np.sqrt(self.env.unwrapped.model.stat.extent**2 - (x-x_c)**2 - (y-y_c)**2) + z_c
        z = self.rng.uniform(z_c, z_max)
        return np.array([x, y, z])
        
    def set_initial_state(self) -> None:
        """
        Set initial state of the environment.
        """
        # allow user override
        if self.kwargs.get("start_location") is not None:
            self.env.unwrapped.init_qpos[:3] = np.array(self.kwargs.get("start_location"))
        else:
            self.start_pos = self.sample_sphere()
            # init_qpos is the state that the env initializes to when reset() is called
            self.env.unwrapped.init_qpos[:3] = self.start_pos

        # start upright
        self.env.unwrapped.init_qpos[3:7] = self.start_orientation
        # start stationary
        self.env.unwrapped.init_qvel[:] = self.start_vel

    def set_goal_state(self, min_distance: float = 1.0) -> None:
        """
        Set the goal state of the environment.
        """
        # allow user override
        if self.kwargs.get("target_location") is not None:
            self.env.unwrapped._target_location = np.array(self.kwargs.get("target_location"))
            return
        
        # Ensure target is at least min_distance away from start
        max_attempts = 300
        for _ in range(max_attempts):
            target_pos = self.sample_sphere()
            if np.linalg.norm(target_pos - self.start_pos) >= min_distance:
                break

        self.env.unwrapped._target_location = target_pos
                
    def add_obstacles(self, 
                    num_obstacles: int = 3,
                    bounds: Tuple[np.ndarray, np.ndarray] = None,
                    size_bounds: Tuple[np.ndarray, np.ndarray] = None,
                    obstacle_types: List[str] = None):
        """
        Add random obstacles to the environment.
        
        Args:
            num_obstacles: Number of obstacles to add
            bounds: (low, high) bounds for obstacle positions
            size_bounds: (low, high) bounds for obstacle sizes
            obstacle_types: List of possible obstacle types
        """
        pass
        # if bounds is None:
        #     bounds = (np.array([-3, -3, 0]), np.array([3, 3, 2]))
        # if size_bounds is None:
        #     size_bounds = (np.array([0.2, 0.2, 0.2]), np.array([1.0, 1.0, 1.0]))
        # if obstacle_types is None:
        #     obstacle_types = ["box"]
            
        # for _ in range(num_obstacles):
        #     position = self.env.np_random.uniform(
        #         low=bounds[0], high=bounds[1]
        #     )
        #     size = self.env.np_random.uniform(
        #         low=size_bounds[0], high=size_bounds[1]
        #     )
        #     obstacle_type = self.env.np_random.choice(obstacle_types)
            
        #     obstacle = {
        #     'position': position,
        #     'size': size,
        #     'type': obstacle_type
        #     }
        #     self.obstacles.append(obstacle)

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles = []

    def get_obstacles(self) -> List[Dict]:
        """Get list of current obstacles."""
        return self.obstacles.copy()

    def generate_env(self) -> None:
        """Generate a randomized environment."""
        if self.kwargs.get("start_location") is not None or self.kwargs.get("target_location") is not None:
            print("Start/goal locations overriden by user - environment bounds will be adjusted accordingly")
            goal_dist_center = np.linalg.norm(np.array(self.kwargs.get("target_location")) - self.env.unwrapped.model.stat.center)
            self.env_radius_ub = goal_dist_center + 1.0
        self.set_env_bounds()
        self.set_initial_state()
        self.set_goal_state()
        self.add_obstacles()
