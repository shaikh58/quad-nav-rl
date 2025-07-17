import numpy as np
from typing import List, Dict, Tuple


class EnvironmentConfigGenerator:
    """
    Configuration generator for MuJoCo-based gym environments.
    
    This class provides functionality to generate configuration for:
    - Start and target positions
    - Environment radius
    - Obstacle positions, sizes, and types
    """
    
    def __init__(self, seed, **kwargs):
        self.kwargs = kwargs
        self.env_radius_lb = self.kwargs.get("env_radius_lb", 3)
        self.env_radius_ub = self.kwargs.get("env_radius_ub", 15)
        self.start_orientation = np.array([1, 0, 0, 0])
        self.start_vel = np.array([0, 0, 0, 0, 0, 0])
        # if user passes in start/goal locations, use them
        self.start_location = self.kwargs.get("start_location", None)
        self.target_location = self.kwargs.get("target_location", None)
        # min distance between start and target
        self.min_distance = self.kwargs.get("min_distance", 3.0)
        
        self.rng = np.random.default_rng(seed)
        
        # Default values for center when calculating bounds
        self.default_center = np.array([0, 0, 0.1])

    def sample_sphere(self, extent: float) -> np.ndarray:
        """Sample x,y,z inside a spherical region."""
        azimuth = self.rng.uniform(0, 2*np.pi) 
        elevation = self.rng.uniform(0, np.pi/2)
        # don't let the start/goal be too close to the bounds of the env
        radius = self.rng.uniform(0, extent - 3.0)
        # extent is radius of the sphere
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = radius * np.cos(azimuth) * np.cos(elevation)
        z = radius * np.sin(elevation)
        return np.array([x, y, z])

    def set_env_bounds(self) -> float:
        """Set environment bounds based on user overrides or random sampling."""
        if self.start_location is not None or self.target_location is not None:
            goal_dist_center = np.linalg.norm(np.array(self.target_location) - self.default_center)
            start_dist_center = np.linalg.norm(np.array(self.start_location) - self.default_center)
            return max(goal_dist_center, start_dist_center) + 5.0
        else:
            return self.rng.uniform(self.env_radius_lb, self.env_radius_ub)

    def set_initial_state(self, env_radius: float) -> np.ndarray:
        """Generate initial state configuration."""
        if self.start_location is not None:
            return np.array(self.start_location)
        else:
            return self.sample_sphere(env_radius)

    def set_goal_state(self, env_radius: float, start_location: np.ndarray) -> np.ndarray:
        """Generate goal state configuration."""
        if self.target_location is not None:
            return np.array(self.target_location)
        
        # Ensure target is at least min_distance away from start
        max_attempts = 100
        for _ in range(max_attempts):
            target_pos = self.sample_sphere(env_radius)
            if np.linalg.norm(target_pos - start_location) >= self.min_distance:
                break
        return target_pos

    def sample_obstacle_params(self) -> Dict:
        """Sample obstacle parameters from uniform distributions."""
        params = {}
        params["num_obstacles"] = self.rng.integers(
            self.kwargs.get("obstacle_count_lb", 1),
            self.kwargs.get("obstacle_count_ub", 3) + 1
        )
        params["min_radius"] = 0.05
        params["max_radius"] = 0.1
        return params

    def add_obstacles(self, start_location: np.ndarray, target_location: np.ndarray) -> List[Dict]:
        """Generate obstacles configuration."""
        obstacles = []
        params = self.sample_obstacle_params()
        for i in range(params["num_obstacles"]):
            # azimuth = self.rng.uniform(0, 2*np.pi)
            # elevation = self.rng.uniform(0, np.pi/2)  # Keep obstacles in upper hemisphere
            # place the obstacle between start and target
            dir_vec = target_location - start_location
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            distance = self.rng.uniform(0,1) * np.linalg.norm(target_location - start_location)
            # project the obstacle onto the line between start and target
            obstacle_pos = start_location + distance * dir_vec
            # x = distance * np.cos(elevation) * np.sin(azimuth)
            # y = distance * np.cos(azimuth) * np.cos(elevation)
            # z = distance * np.sin(elevation)
            radius = self.rng.uniform(params["min_radius"], params["max_radius"])
            
            obstacle_config = {
                "name": f"obstacle_{i}",
                "type": "sphere",
                "position": [obstacle_pos[0], obstacle_pos[1], obstacle_pos[2]],
                "radius": radius
            }
            
            obstacles.append(obstacle_config)
        
        return obstacles

    def generate_env_config(self) -> Dict:
        """Generate a configuration dictionary for environment setup."""
        config = {}

        if self.start_location is not None or self.target_location is not None:
            print("Start/goal locations overriden by user - environment bounds will be adjusted accordingly")

        env_radius = self.set_env_bounds()
        config["radius"] = env_radius
        
        start_location = self.set_initial_state(env_radius)
        config["start_location"] = start_location
        
        target_location = self.set_goal_state(env_radius, start_location)
        config["target_location"] = target_location
        # obstacles are dynamically added in the enivronment code, using the method add_obstacles()
        
        return config 