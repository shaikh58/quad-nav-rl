import numpy as np 
import gymnasium as gym 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class QuadNavEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self, 
        xml_file: str = "./models/skydio_x2/scene.xml", 
        frame_skip: int = 5, 
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.1, 
        reset_noise_scale: float = 0.1,
        target_location: np.ndarray = np.array([0, 0, 0]),
        **kwargs,
    ):
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
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        # distance threshold to target locationfor success
        self._goal_threshold = 0.1
        # minimum required height above ground
        self._min_height = 0.1
        # maximum allowed angular velocity
        self._max_angular_velocity = 5.0  # rad/s
        # maximum allowed tilt angle (roll/pitch) (radians)
        self._max_tilt_angle = np.pi/4
        # maximum allowed roll/pitch rate (rad/s)
        self._max_roll_pitch_rate = 1.0
        # maximum allowed yaw rate (rad/s)
        self._max_yaw_rate = 1.0

        # user defined parameters
        self.env_center = self.model.stat.center.copy()
        # radius of environment
        self.env_extent = self.model.stat.extent.copy()
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


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
        

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        observation = np.concatenate((position, velocity))
        return observation


    def _compute_reward(self, action):
        # TODO: implement reward function with intermediate rewards
        return 0


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
        """
        # takes env step. Sets data.ctrl to action
        self.do_simulation(action, self.frame_skip)
        terminated, msg = self._is_terminated()
        fwd_reward = self._compute_reward(action)
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
            # TODO: implement collision with ground detection
            return False
        
        def _check_oob():
            """Check if quadrotor is out of bounds."""
            # TODO: implement out of bounds detection
            return False
        
        def _check_unstable_flight():
            """Check if quadrotor is unstable."""
            # TODO: implement unstable flight detection
            # angular velocity (norm of wx, wy, wz)
            # roll/pitch rate (wx, wy)
            # yaw rate (wz)
            
            return False

        if _check_success():
            return True, "success"
        if _check_collision_obstacles():
            return True, "collision_obstacles"
        if _check_collision_ground():
            return True, "collision_ground"
        if _check_oob():
            return True, "out_of_bounds"
        if _check_unstable_flight():
            return True, "unstable_flight"

        return False, "not_terminated"