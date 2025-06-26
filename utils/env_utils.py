import numpy as np
from typing import List, Dict, Tuple


class RandomEnvGenerator:
    """
    Random environment generator for MuJoCo-based gym environments.
    
    This class provides functionality to randomize:
    - Start and target positions
    - Obstacle positions, sizes, and types
    """
    
    def __init__(self, env, **kwargs):
        """
        Initialize the environment randomizer.
        
        Args:
            env: Gym environment object with MuJoCo model
            **kwargs: Additional configuration parameters
        """
        self.env = env
        self.kwargs = kwargs
        self.obstacles = []
        
    def sample_sphere(self):
        # sample x,y inside a spherical region; center, extent from the model file
        azimuth = self.env.np_random.uniform(0, 2*np.pi) 
        elevation = self.env.np_random.uniform(0, np.pi/2)
        # extent is radius of the sphere
        x = self.env.env_extent * np.cos(azimuth) * np.sin(elevation)
        y = self.env.env_extent * np.sin(azimuth) * np.sin(elevation)
        z = self.env.env_center[2] + self.env.env_extent * np.sin(elevation)

        return np.array([x, y, z])
        
        
    def set_initial_state(self):
        """
        Set initial state of the environment.
        """
        self.start_pos = self.sample_sphere()
        # init_qpos is the state that the env initializes to when reset() is called
        self.env.init_qpos[:3] = self.start_pos
        # start upright
        self.env.init_qpos[3:7] = np.array([1, 0, 0, 0])
        # start stationary
        self.env.init_qvel[:] = np.array([0, 0, 0, 0, 0, 0])
    

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

        self.env._target_location = target_pos
                
        
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
        self.set_initial_state()
        self.set_goal_state()
        self.add_obstacles()
