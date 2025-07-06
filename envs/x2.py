import numpy as np 
import gymnasium as gym 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import xml.etree.ElementTree as ET
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
        out_of_bounds_weight: float = 1,
        success_weight: float = 10,
        target_location: np.ndarray = None, # set by randomizer; user overrides also handled by randomizer
        start_location: np.ndarray = None, # set by randomizer; user overrides also handled by randomizer
        radius: float = None, # set by randomizer; user overrides also handled by randomizer
        progress_type: str = "straight_line", # "straight_line" or "euclidean"
        **kwargs,
    ):
        self.render_mode = kwargs.get("render_mode", "rgb_array")

        if xml_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            xml_file = os.path.join(current_dir, "..", "models", "skydio_x2", "scene.xml")
            model_xml_file = os.path.join(current_dir, "..", "models", "skydio_x2", "x2.xml")
            self.model_xml_file = model_xml_file

        # add geoms to indicate start/goal locations
        if start_location is not None and target_location is not None and radius is not None:
            self.set_env_indicators(start_location, target_location)

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
            out_of_bounds_weight,
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
        self._start_location = start_location
        self._radius = radius
        # NOTE: the initial state is set to a default value env.init_qpos
        self._ctrl_cost_weight = ctrl_cost_weight
        self._progress_weight = progress_weight
        self._body_rate_weight = body_rate_weight
        self._collision_ground_weight = collision_ground_weight
        self._collision_obstacles_weight = collision_obstacles_weight
        self._out_of_bounds_weight = out_of_bounds_weight
        self._success_weight = success_weight
        self._progress_type = progress_type
        self.start_orientation = np.array([1, 0, 0, 0])
        self.start_vel = np.array([0, 0, 0, 0, 0, 0])

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

        # if user has passed in env params, set them
        if self._start_location is not None and self._target_location is not None and self._radius is not None:
            self.set_start_location(self._start_location, self.start_orientation, self.start_vel)
            self.set_target_location(self._target_location) 
            self.set_env_radius(self._radius)

    def set_start_location(self, start_location, start_orientation, start_vel):
        # init_qpos is the state that the env initializes to when reset() is called
        self.init_qpos[:3] = start_location
        # start upright
        self.init_qpos[3:7] = start_orientation
        # start stationary
        self.init_qvel[:] = start_vel
    
    def set_target_location(self, target_location):
        self._target_location = target_location
    
    def set_env_radius(self, radius):
        self.model.stat.extent = radius
    
    def set_env_indicators(self, start_location, target_location):
        # Parse the XML file
        tree = ET.parse(self.model_xml_file)
        root = tree.getroot()
        
        # Find and update the spheres
        for geom in root.iter('geom'):
            if geom.get('name') == 'green_sphere':
                geom.set('pos', f"{start_location[0]} {start_location[1]} {start_location[2]}")
            elif geom.get('name') == 'blue_sphere':
                geom.set('pos', f"{target_location[0]} {target_location[1]} {target_location[2]}")
        
        # Write the modified XML back to disk
        tree.write(self.model_xml_file, encoding='utf-8', xml_declaration=True)
    
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
    
    def progress(self):
        if self._progress_type == "straight_line":
            # for straight line planner, trajectory is just the endpts of planner output
            pt1 = self.trajectory[0]
            pt2 = self.trajectory[-1]
            unit_vector = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
            progress = np.dot(self.data.qpos[:3] - pt1, unit_vector)
            progress_prev = np.dot(self.prev_pos - pt1, unit_vector)
            return progress - progress_prev
        elif self._progress_type == "euclidean":
            return np.linalg.norm(self.data.qpos[:3] - self._target_location)
        else:
            raise ValueError(f"Invalid progress type: {self._progress_type}")


    def _compute_reward(self, terminated, msg):
        reward_components = {}
        # progress along planned trajectory; note init_qpos was sampled by RandomEnvGenerator
        curr_progress = self.progress()
        if self._progress_type == "straight_line":
            reward_components["progress"] = self._progress_weight * curr_progress
        elif self._progress_type == "euclidean":
            # R * (1 - dist_remaining/total_dist); this qty is 0 at start and R at goal
            reward_components["progress"] = self._progress_weight * (1 - curr_progress / np.linalg.norm(self.init_qpos[:3] - self._target_location))
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
        # out of bounds penalty
        if terminated and msg == "out_of_bounds":
            reward_components["out_of_bounds"] = -self._out_of_bounds_weight
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
        self.info = info # set info to be used in render() - called by RecordVideo wrapper
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

    def render(self):
        """
        Override the parent class render method to add custom logic before rendering.
        This allows intercepting the render() call from RecordVideo wrapper.
        """
        ego_coord = np.round(self.data.qpos[:3], 1)
        target_coord = np.round(self._target_location, 1)
        # add current coordinate as an overlay
        if self.mujoco_renderer.viewer is not None:
            # set the cam distance again as it was set before env randomizer changes env bounds
            self.mujoco_renderer.viewer.cam.distance = self.model.stat.extent + 4.0
            self.mujoco_renderer.viewer._overlays.clear()
            if self.info['termination_msg'] != "not_terminated":
                self.mujoco_renderer.viewer.add_overlay(0,f"Agent: {ego_coord}", f"Target: {target_coord}")
                self.mujoco_renderer.viewer.add_overlay(2, f"{self.info['termination_msg']}","")
            else:
                self.mujoco_renderer.viewer.add_overlay(0,f"Agent: {ego_coord}", f"Target: {target_coord}")
            self.mujoco_renderer.viewer.add_overlay(3, f"Distance to target: {np.round(np.linalg.norm(ego_coord - target_coord), 2)}","") 
        # Call the parent class's render method
        return super().render()