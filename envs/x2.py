import numpy as np 
import gymnasium as gym 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import os

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class QuadNavEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 20,
    }
    def __init__(
        self, 
        xml_file: str = None, 
        frame_skip: int = 5, 
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        goal_threshold: float = 0.1,
        min_height: float = 0.1,
        ctrl_cost_weight: float = 0.1,
        progress_weight: float = 0.1,
        body_rate_weight: float = 0.1,
        collision_ground_weight: float = 1,
        collision_obstacles_weight: float = 1,
        success_weight: float = 10,
        target_location: np.ndarray = None, # set by randomizer; user overrides also handled by randomizer
        start_location: np.ndarray = None, # set by randomizer; user overrides also handled by randomizer
        **kwargs,
    ):
        self.render_mode = kwargs.get("render_mode", "rgb_array")

        if xml_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            xml_file = os.path.join(current_dir, "..", "models", "skydio_x2", "scene.xml")

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            ctrl_cost_weight,
            progress_weight,
            body_rate_weight,
            collision_ground_weight,
            collision_obstacles_weight,
            success_weight,
            **kwargs,
        )

        super().__init__(
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            render_mode=self.render_mode,
        )
        # distance threshold to target location for success
        self._goal_threshold = goal_threshold
        # minimum required height above ground
        self._min_height = min_height

        # user defined parameters
        self._target_location = target_location
        # NOTE: the initial state is set to a default value env.init_qpos
        self._ctrl_cost_weight = ctrl_cost_weight
        self._progress_weight = progress_weight
        self._body_rate_weight = body_rate_weight
        self._collision_ground_weight = collision_ground_weight
        self._collision_obstacles_weight = collision_obstacles_weight
        self._success_weight = success_weight

        self.mass = self.model.body_mass.sum()
        self.g = self.model.opt.gravity[2].item()
        self.hover_thrust = self.model.keyframe('hover').ctrl.copy()

        # qpos is (7,) (x, y, z, qw, qx, qy, qz)
        # qvel is (6,) (vx, vy, vz, wx, wy, wz)
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        # initialize previous pos to current pos
        self.prev_pos = self.data.qpos[:3].copy()
        self.trajectory = None


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
        

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        observation = np.concatenate((position, velocity))
        return observation

    def set_trajectory(self, trajectory, info):
        self.trajectory = trajectory
        self.trajectory_info = info
    
    def progress(self, trajectory):
        # for straight line planner, trajectory is just the endpts of planner output
        pt1 = trajectory[0]
        pt2 = trajectory[-1]
        unit_vector = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
        progress = np.dot(self.data.qpos[:3] - pt1, unit_vector)
        progress_prev = np.dot(self.prev_pos - pt1, unit_vector)
        return progress - progress_prev


    def _compute_reward(self, terminated, msg):
        reward_components = {}
        # progress along planned trajectory; note init_qpos was sampled by RandomEnvGenerator
        curr_progress = self.progress(self.trajectory)
        reward_components["progress"] = self._progress_weight * curr_progress
        # ground collision penalty
        if terminated and msg == "collision_ground":
            reward_components["collision_ground"] = -self._collision_ground_weight
        # obstacle collision penalty; not implemented yet
        if terminated and msg == "collision_obstacles":
            reward_components["collision_obstacles"] = -self._collision_obstacles_weight
        # success reward
        if terminated and msg == "success":
            reward_components["success"] = self._success_weight
        # body rate penalty
        body_rate = np.linalg.norm(self.data.qvel[3:6])
        reward_components["body_rate"] = -self._body_rate_weight * body_rate**2
        # print("reward_components: ", reward_components)
        return reward_components


    def reset_model(self):
        """Resets the state of the environment and returns an initial observation."""
        # self.init_qpos comes from parent class 
        qpos = self.init_qpos
        qvel = self.init_qvel # initialize at hover
        # set action to hover thrust
        self.data.ctrl = self.hover_thrust

        # parent method from MujocoEnv; copies qpos, qvel to data.qpos, 
        # data.qvel to avoid pointer issues in underlying c++
        self.set_state(qpos, qvel) 
        observation = self._get_obs()
        return observation

    
    def _get_reset_info(self):
        """Initialized pos/vel values after environment reset"""
        return {
            "pos": self.data.qpos[0:3].flatten(),
            "quat": self.data.qpos[3:7].flatten(),
            "vel": self.data.qvel[0:3].flatten(),
            "ang_vel": self.data.qvel[3:6].flatten(),
        }
    

    def _get_info(self):
        """Current pos/vel/reward info after environment step"""
        info = {}
        info["pos"] = self.data.qpos[0:3].flatten()
        info["quat"] = self.data.qpos[3:7].flatten()
        info["vel"] = self.data.qvel[0:3].flatten()
        info["ang_vel"] = self.data.qvel[3:6].flatten()

        return info

    def step(self, action):
        """
        Action is (4,) (thrust1, thrust2, thrust3, thrust4)
        Trajectory is from planner (n_points, 3) (x, y, z)
        """
        # set prev pos before taking env step
        self.prev_pos = self.data.qpos[:3].copy()
        # takes env step. Sets data.ctrl to action
        self.do_simulation(action, self.frame_skip)
        terminated, msg = self._is_terminated()
        fwd_reward = 0
        reward_components = self._compute_reward(terminated, msg)
        for k,v in reward_components.items():
            fwd_reward += v
        ctrl_reward = self.control_cost(action)
        reward = fwd_reward - ctrl_reward
        observation = self._get_obs()
        info = self._get_info()
        # add reward and episode length to info as its only available in step()
        info["fwd_reward"] = fwd_reward
        info["ctrl_reward"] = ctrl_reward
        info["reward"] = reward
        info["reward_components"] = reward_components
        info["termination_msg"] = msg
        # episode length is automatically added by RecordEpisodeStatistics wrapper

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    
    def _is_terminated(self):
        """
        Termination conditions
        """
        def _check_success():
            """Check if quadrotor has reached the goal."""
            current_pos = self.data.qpos[:3]  # x, y, z position
            goal_pos = self._target_location
            
            # Distance-based success
            distance = np.linalg.norm(current_pos - goal_pos)
            return distance < self._goal_threshold  # e.g., 0.1 meters

        def _check_collision_obstacles():
            """Check if quadrotor has collided with an obstacle."""
            # currently not implemented until CBF is implemented
            return False
        
        def _check_collision_ground():
            """Check if quadrotor has collided with the ground."""
            return self.data.qpos[2] < self._min_height
        
        def _check_oob():
            """Check if quadrotor is out of bounds."""
            return np.linalg.norm(self.data.qpos[:3] - self.model.stat.center) > self.model.stat.extent


        if _check_success():
            return True, "success"
        if _check_collision_obstacles():
            return True, "collision_obstacles"
        if _check_collision_ground():
            return True, "collision_ground"
        if _check_oob():
            return True, "out_of_bounds"

        return False, "not_terminated"