import numpy as np 
import gymnasium as gym 

class QuadNavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.info_dict = {}
        
    def _get_obs(self):
        return
    
    def _get_info(self):
        return self.info_dict

    def _compute_reward(self):
        return 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
    def step(self, action):
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        # TODO: use mj_step with the model and data
        reward = self._compute_reward()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
