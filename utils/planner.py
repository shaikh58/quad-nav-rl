"""
Kinematic planner module for quadrotor navigation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class BasePlanner(ABC):
    """
    A planner takes the current state, goal state, and environment information
    to generate a reference trajectory for the controller to track.
    """
    
    def __init__(self, env, **kwargs):
        """
        Initialize the planner.
        
        Args:
            env: The environment object containing obstacle and world information
            **kwargs: Additional planner-specific parameters
        """
        self.env = env
        self._setup_planner(**kwargs)
    
    @abstractmethod
    def _setup_planner(self, **kwargs):
        """Setup planner-specific parameters and configurations."""
        pass
    
    @abstractmethod
    def plan_trajectory(
        self, 
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Plan a trajectory from current state to goal state."""
        pass
    
    def get_obstacle_info(self) -> Dict[str, Any]:
        """
        Extract obstacle information from the environment.
        
        Returns:
            Dictionary containing obstacle positions, dimensions, and types
        """
        obstacle_info = {}
        
        # Access Mujoco model data for obstacle information
        if hasattr(self.env, 'model') and hasattr(self.env.unwrapped.model, 'geom_name2id'):
            for geom_name, geom_id in self.env.unwrapped.model.geom_name2id.items():
                if 'obstacle' in geom_name.lower() or 'wall' in geom_name.lower():
                    # Get obstacle position and size
                    pos = self.env.unwrapped.model.geom_pos[geom_id]
                    size = self.env.unwrapped.model.geom_size[geom_id]
                    geom_type = self.env.unwrapped.model.geom_type[geom_id]
                    
                    obstacle_info[geom_name] = {
                        'position': pos,
                        'size': size,
                        'type': geom_type,
                        'id': geom_id
                    }
        
        return obstacle_info


class StraightLinePlanner(BasePlanner):
    """
    Straight-line trajectory planner.
    """
    
    def _setup_planner(self, **kwargs):
        """Setup straight-line planner parameters."""
        self.step_size = kwargs.get('step_size', 0.1)  # m
        self.min_points = kwargs.get('min_points', 10)
        
    def plan_trajectory(
        self,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Plan a straight-line trajectory with a fixed step size.

        Returns:
            trajectory: Array of shape (n_points, 3) with [x, y, z]
            info: Planning information
        """
        # Extract positions
        current_pos = self.env.unwrapped.data.qpos[:3]
        goal_pos = self.env.unwrapped._target_location
        
        # Calculate distance and direction
        displacement = goal_pos - current_pos
        distance = np.linalg.norm(displacement)
        direction = displacement / (distance + 1e-8) 
        
        # Calculate number of points
        n_points = max(self.min_points, int(distance / self.step_size))
        
        # Only plan position trajectory so as not to overconstrain the controller
        trajectory = np.zeros((n_points, 3)) 

        # Generate position trajectory (linear interpolation)
        for i in range(n_points):
            # Position: linear interpolation
            trajectory[i] = current_pos + self.step_size * direction
            current_pos = trajectory[i]
        
        info = {
            'dist_start_to_goal': distance,
            'n_points': n_points,
        }
        
        return trajectory, info

