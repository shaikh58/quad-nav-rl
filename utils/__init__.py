# Import and register the wrapper
from .env_utils import EnvironmentRandomizer
import gymnasium as gym

# Register with gymnasium
gym.wrappers.EnvironmentRandomizer = EnvironmentRandomizer

# Export for easy access
__all__ = ['EnvironmentRandomizer'] 