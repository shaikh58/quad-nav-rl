import numpy as np 
import gymnasium as gym 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import os

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
GOAL_THRESHOLD = 0.1
MIN_HEIGHT = 0.1
MAX_BODY_RATE = 1.0
MAX_TILT_ANGLE = np.pi/4

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
        ctrl_cost_weight: float = 0.1, 
        reset_noise_scale: float = 0.1,
        target_location: np.ndarray = np.array([0, 0, 0]),
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
            reset_noise_scale,
            target_location,
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
        self._goal_threshold = GOAL_THRESHOLD
        # minimum required height above ground
        self._min_height = MIN_HEIGHT
        # maximum allowed roll/pitch/yaw rate (rad/s)
        self._max_body_rate = MAX_BODY_RATE
        # maximum allowed tilt angle (roll/pitch) (radians)
        self._max_tilt_angle = MAX_TILT_ANGLE

        # user defined parameters
        self._target_location = target_location
        # NOTE: the initial state is set to a default value env.init_qpos
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

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


    def _compute_reward(self, action, terminated, msg):
        reward_components = {}
        # progress along planned trajectory; note init_qpos was sampled by RandomEnvGenerator
        curr_progress = self.progress(self.trajectory)
        reward_components["progress"] = curr_progress
        # ground collision penalty
        if terminated and msg == "collision_ground":
            reward_components["collision_ground"] = -10
        # obstacle collision penalty; not implemented yet
        if terminated and msg == "collision_obstacles":
            reward_components["collision_obstacles"] = -10
        # success reward
        if terminated and msg == "success":
            reward_components["success"] = 10
        # body rate penalty
        body_rate = np.linalg.norm(self.data.qvel[3:6])
        reward_components["body_rate"] = body_rate**2
        # print("reward_components: ", reward_components)
        return reward_components


    def reset_model(self):
        """Resets the state of the environment and returns an initial observation."""
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale
        # self.init_qpos comes from parent class 
        qpos = self.init_qpos #+ self.np_random.uniform(
            # low=noise_low, high=noise_high, size=self.model.nq
        # )
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
        reward_components = self._compute_reward(action, terminated, msg)
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