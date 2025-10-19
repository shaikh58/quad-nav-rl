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
        print(self.kwargs)
        self.env_radius = self.kwargs.get("env_radius", 20)
        self.start_orientation = np.array([1, 0, 0, 0])
        self.start_vel = np.array([0, 0, 0, 0, 0, 0])
        # if user passes in start/goal locations, use them
        self.start_location = self.kwargs.get("start_location", None)
        self.target_location = self.kwargs.get("target_location", None)
        # min distance between start and target
        self.min_distance = self.kwargs.get("min_distance", 5.0)
        
        self.rng = np.random.default_rng(seed)
        # hardcode the goal seed so it always chooses the same goal
        self.goal_rng = np.random.default_rng(seed=529012)
        self.obs_rng = np.random.default_rng(seed)
        
        # Default values for center when calculating bounds
        self.default_center = np.array([0, 0, 0.1])

    def sample_sphere(self, extent: float, type: str = "start") -> np.ndarray:
        """Sample x,y,z inside a spherical region."""
        if type == "start":
            rng = self.rng
        elif type == "goal":
            rng = self.goal_rng
        else:
            raise ValueError(f"Invalid type: {type}")
        azimuth = rng.uniform(0, 2*np.pi) 
        elevation = rng.uniform(0, np.pi/2)
        # don't let the start/goal be too close to the bounds of the env
        radius = rng.uniform(0, extent - 3.0)
        # extent is radius of the sphere
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = radius * np.cos(azimuth) * np.cos(elevation)
        z = radius * np.sin(elevation)
        return np.array([x, y, z])

    def set_env_bounds(self) -> float:
        """Set environment radius bound."""
        return self.env_radius

    def set_initial_state(self, env_radius: float) -> np.ndarray:
        """Generate initial state configuration."""
        if self.start_location is not None:
            return np.array(self.start_location)
        else:
            # ensure start is at least min_distance away from target
            max_attempts = 100
            for _ in range(max_attempts):
                start_pos = self.sample_sphere(env_radius)
                if np.linalg.norm(start_pos - self.target_location) >= self.min_distance:
                    break
        self.start_location = start_pos
        return start_pos

    def set_goal_state(self, env_radius: float) -> np.ndarray:
        """Generate goal state configuration."""
        if self.target_location is not None:
            return np.array(self.target_location)
        
        target_pos = self.sample_sphere(env_radius, type="goal")
        self.target_location = target_pos
        # reset the generator state (only necessary if setting goal location again)
        self.goal_rng = np.random.default_rng(seed=529012)

        return target_pos

    def sample_obstacle_params(self) -> Dict:

        # TODO: fix this similar to the parallel env seeding - ensure the same obstacles are sampled every time in a given env
        
        """Sample obstacle parameters from uniform distributions."""
        params = {}
        params["num_obstacles"] = self.obs_rng.integers( # obstacles can use the same rng as start position as we want it different in each env
            self.kwargs.get("obstacle_count_lb", 3),
            self.kwargs.get("obstacle_count_ub", 5) + 1
        )
        params["min_radius"] = 0.1
        params["max_radius"] = 0.2
        return params

    def add_obstacles(self, start_location: np.ndarray, target_location: np.ndarray) -> List[Dict]:
        
        # TODO: fix this function too 
        
        """Generate obstacles configuration."""
        obstacles = []
        params = self.sample_obstacle_params()
        for i in range(params["num_obstacles"]):
            # place the obstacle between start and target
            dir_vec = target_location - start_location
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            # ensure obstacles are not too close to the start or target
            distance = self.obs_rng.uniform(0.25,0.75) * np.linalg.norm(target_location - start_location)
            # project the obstacle onto the line between start and target
            obstacle_pos = start_location + distance * dir_vec
            # sample the obstacle with a gaussian around the position on the straight line between start and target
            noise = self.obs_rng.normal(0, 0.8, 3)
            obstacle_pos = obstacle_pos + noise # 68% chance of being within scale units of the line
            radius = self.obs_rng.uniform(params["min_radius"], params["max_radius"])
            
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

        target_location = self.set_goal_state(env_radius)
        config["target_location"] = target_location
        
        start_location = self.set_initial_state(env_radius)
        config["start_location"] = start_location
        # obstacles are dynamically added in the enivronment code, using the method add_obstacles()
        
        return config 