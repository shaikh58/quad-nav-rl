import numpy as np
import os
from typing import List, Dict, Tuple
import gymnasium as gym
from utils.planner import StraightLinePlanner
from gymnasium.wrappers.utils import RunningMeanStd
from gymnasium.core import ActType, ObsType, WrapperObsType
from utils.env_config_generator import EnvironmentConfigGenerator

PLANNER_TYPES = {
    "straight_line": StraightLinePlanner,
}


def make_env(env_id, idx, capture_video, run_name, gamma, seed, use_planner=None, planner_type=None, **kwargs):
    def thunk():
        config_generator = EnvironmentConfigGenerator(seed=seed, **kwargs)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", 
                config_generator=config_generator, **kwargs)
            video_path = kwargs.get("video_path", f"videos/{run_name}")
            env = gym.wrappers.RecordVideo(env, f"{video_path}", episode_trigger=lambda x: x % 300 == 0)
        else:
            env = gym.make(env_id, config_generator=config_generator, **kwargs)
        # TODO: move this to config generator
        if use_planner:
            planner = PLANNER_TYPES[planner_type]()
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
        max_steps = kwargs.get("max_steps", None)
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
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

        self.obs_rms = RunningMeanStd(
            shape=self.observation_space.shape, dtype=self.observation_space.dtype
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