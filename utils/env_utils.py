import numpy as np
from typing import List, Dict, Tuple


class EnvironmentRandomizer:
    """
    Random environment generator for MuJoCo-based gym environments.
    
    This class provides functionality to randomize:
    - Start and target positions
    - Obstacle positions, sizes, and types
    """
    
    def __init__(self, env, **kwargs):
        self.env = env
        self.kwargs = kwargs
        self.obstacles = []
        self.env_radius_lb = 3
        self.env_radius_ub = 15
        self.start_orientation = np.array([1, 0, 0, 0])
        self.start_vel = np.array([0, 0, 0, 0, 0, 0])

    def set_env_bounds(self):
        self.env.unwrapped.model.stat.extent = self.env.unwrapped.np_random.uniform(self.env_radius_lb, self.env_radius_ub)
        
    def sample_sphere(self):
        # sample x,y inside a spherical region; center, extent from the model file
        azimuth = self.env.unwrapped.np_random.uniform(0, 2*np.pi) 
        elevation = self.env.unwrapped.np_random.uniform(0, np.pi/2)
        # extent is radius of the sphere
        x = self.env.unwrapped.model.stat.extent * np.cos(azimuth) * np.sin(elevation)
        y = self.env.unwrapped.model.stat.extent * np.sin(azimuth) * np.sin(elevation)
        # constrain z so that the euclidean dist to center is less than extent
        x_c, y_c, z_c = self.env.unwrapped.model.stat.center
        z_max = np.sqrt(self.env.unwrapped.model.stat.extent**2 - (x-x_c)**2 - (y-y_c)**2) + z_c
        z = self.env.unwrapped.np_random.uniform(z_c, z_max)
        return np.array([x, y, z])
        
    def set_initial_state(self):
        """
        Set initial state of the environment.
        """
        self.start_pos = self.sample_sphere()
        # init_qpos is the state that the env initializes to when reset() is called
        self.env.unwrapped.init_qpos[:3] = self.start_pos
        # start upright
        self.env.unwrapped.init_qpos[3:7] = self.start_orientation
        # start stationary
        self.env.unwrapped.init_qvel[:] = self.start_vel

    def set_goal_state(self, min_distance: float = 1.0):
        """
        Set the goal state of the environment.
        """
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

    def clear_obstacles(self):
        """Clear all obstacles."""
        self.obstacles = []

    def get_obstacles(self) -> List[Dict]:
        """Get list of current obstacles."""
        return self.obstacles.copy()

    def generate_env(self):
        """Generate a randomized environment."""
        self.set_env_bounds()
        self.set_initial_state()
        self.set_goal_state()
        self.add_obstacles()
