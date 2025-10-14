import numpy as np 
import gymnasium as gym 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import xml.etree.ElementTree as ET
import os
from utils.env_utils import multiply_quaternions
from utils.utils import softmin, grad_sdf_softmin, grad_sdf
import mujoco
from utils.env_config_generator import EnvironmentConfigGenerator
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}
DEFAULT_SIZE = 480

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
        progress_type: str = "euclidean", # "straight_line", "euclidean", "negative"
        reset_noise_scale: float = 0.1,
        mode: str = None,
        config_generator: EnvironmentConfigGenerator = None,
        use_obstacles: bool = False,
        regen_obstacles: bool = False,
        obs_regen_eps: float = 0.8, # prob with which to resample obstacles after each episode
        **kwargs,
    ):
        self.render_mode = kwargs.get("render_mode", "rgb_array")

        if xml_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.xml_file = os.path.join(current_dir, "..", "models", "skydio_x2", "scene.xml")
            self.model_xml_file = os.path.join(current_dir, "..", "models", "skydio_x2", "x2.xml")

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
        # generate env configs
        self._config_generator = config_generator
        env_config = self._config_generator.generate_env_config()
        self._start_location = env_config["start_location"]
        self._target_location = env_config["target_location"]
        self._radius = env_config["radius"]

        # setup Mujoco model; mjSpec is the new API for procedurally modifying a model
        self.width = DEFAULT_SIZE
        self.height = DEFAULT_SIZE
        self.mj_spec = mujoco.MjSpec.from_file(self.xml_file)
        self.x2_body_ix = [i for i, body in enumerate(self.mj_spec.bodies) if body.name == 'x2'][0]
        self.model = self.mj_spec.compile()
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        self._use_obstacles = use_obstacles
        if self._use_obstacles:
            # generate obstacles at the beginning; possibly regenerate after each episode
            self.obstacle_metadata = self._config_generator.add_obstacles(self._start_location, self._target_location)
            self.generate_obstacle_geoms()
            self.model, self.data = self.mj_spec.recompile(self.model, self.data)

        # the default start state is overwritten in reset_model()
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.frame_skip = frame_skip
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self._set_action_space()
        # renderer args
        self.camera_name = None
        self.camera_id = None
        self.max_geom=1000
        self.visual_options={}
        self.mujoco_renderer = MujocoRenderer(
                self.model,self.data,DEFAULT_CAMERA_CONFIG,self.width,self.height,
                self.max_geom,self.camera_id,self.camera_name,self.visual_options,
            )
        # distance threshold to target location for success
        self._goal_threshold = goal_threshold
        # minimum required height above ground
        self._min_height = min_height

        # user defined parameters
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
        self._reset_noise_scale = reset_noise_scale
        self._mode = mode
        self._obs_regen_eps = obs_regen_eps
        self._regen_obstacles = regen_obstacles

        self.mass = self.model.body_mass.sum()
        self.g = self.model.opt.gravity[2].item()
        self.hover_thrust = self.model.keyframe('hover').ctrl.copy()
        # qpos is (7,) (x, y, z, qw, qx, qy, qz)
        # qvel is (6,) (vx, vy, vz, wx, wy, wz)
        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.trajectory = None
        
        self.set_start_location() 
        self.set_start_goal_geoms()
        self.set_env_radius()

    def clear_obstacle_geoms(self):
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name.startswith("obstacle_"):
                self.model.geom_type[i] = mujoco.mjtGeom.mjGEOM_NONE

    def generate_obstacle_geoms(self):
        for i, obstacle in enumerate(self.obstacle_metadata):
            name = obstacle["name"]
            pos = obstacle["position"]
            radius = obstacle["radius"]
            obs_type = obstacle["type"]
            if obs_type == "sphere":
                obs_type = mujoco.mjtGeom.mjGEOM_SPHERE
            else:
                raise ValueError(f"Invalid obstacle type: {obs_type}")
            self.mj_spec.worldbody.add_geom(name=name,
                            type=obs_type,
                            rgba=[0, 0, 1, 1], # all obstacles are blue
                            pos=pos,
                            size=[radius, radius, 0])
        # print(f"Added {len(self.obstacle_metadata)} obstacles")

    def set_start_location(self):
        # init_qpos is the state that the env initializes to when reset() is called
        self.init_qpos[:3] = self._start_location
        # start upright
        self.init_qpos[3:7] = self.start_orientation
        # start stationary
        self.init_qvel[:] = self.start_vel
    
    def set_env_radius(self):
        self.model.stat.extent = self._radius
    
    def set_start_goal_geoms(self):
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name == "start_sphere":
                self.model.geom_pos[i] = self._start_location
            elif geom_name == "target_sphere":
                self.model.geom_pos[i] = self._target_location 
    
    def sdf_obstacle(self):
        """This implementation uses min_{all obstacles} SDF(x,obstacle set), i.e. 
        the distance to the nearest obstacle; however, this introduces a min function
        which is not differentiable. We could represent the obstacle set O as a union of 
        all obstacles, and computing SDF and gradient wrt all obstacles. However, this leads to 
        a vector valued SDF, which violates the SDF requirement of a scalar function. We can
        use a smooth approximation of the min function, i.e. the log sum exp function.
        """
        dists_to_obstacles = []
        if not self._use_obstacles:
            return np.inf, np.inf
        for obstacle in self.obstacle_metadata:
            # check distance of each rotor to the obstacle, accounting for rotor radius
            min_dist = np.inf
            for i in range(4):
                rotor_pos = self.model.geom(f'rotor{i+1}').pos
                rotor_radius = self.model.geom(f'rotor{i+1}').size[0]
                # distance to boundary of obstacle, from edge of rotor
                dist = np.linalg.norm(rotor_pos - np.array(obstacle["position"])) - obstacle['radius'] - rotor_radius
                # NOTE: we can use this non differentiable approximation of the dist of quadrotor
                # to account for rotors. Assuming rotor dists won't be too different from main body dist.
                min_dist = min(min_dist, dist)
            dists_to_obstacles.append(min_dist)
        # TODO: uncomment this line to use softmin, else keep regular min
        # sdf = softmin(dists_to_obstacles)
        closest_obs_ix = np.argmin(dists_to_obstacles)
        sdf = dists_to_obstacles[closest_obs_ix]
        # compute sdf and gradient together to avoid relying on data in env state
        state = np.concatenate((self.data.qpos, self.data.qvel))
        grad = grad_sdf(state, self.obstacle_metadata[closest_obs_ix]["position"])
        return sdf, grad
    
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
        elif self._progress_type in ["euclidean", "negative"]:
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
        elif self._progress_type == "negative":
            reward_components["progress"] = -self._progress_weight * curr_progress
            if self._check_collision_ground():
                reward_components["collision_ground"] = -self._collision_ground_weight
            if self._check_collision_obstacles():
                reward_components["collision_obstacles"] = -self._collision_obstacles_weight
            if self._check_oob():
                reward_components["out_of_bounds"] = -self._out_of_bounds_weight

        # for non-negative rewards, we wait till termination of episodes to penalize
        # ground collision penalty
        if terminated and msg == "collision_ground":
            reward_components["collision_ground"] = -self._collision_ground_weight
        # obstacle collision penalty; not implemented yet
        if terminated and msg == "collision_obstacles":
            reward_components["collision_obstacles"] = -self._collision_obstacles_weight
        # success reward
        if terminated and msg == "success":
            q = self.data.qpos[3:7]
            q_conj = np.array([1,0,0,0])
            q_product = multiply_quaternions(q, q_conj)
            angle_diff = 2 * np.arccos(np.clip(q_product[0], -1, 1))
            # print(f"Angle diff at success: {angle_diff}")
            reward_components["success"] = self._success_weight * np.abs(2*np.pi - angle_diff.item())
        # body rate penalty
        body_rate = np.linalg.norm(self.data.qvel[3:6])
        reward_components["body_rate"] = -self._body_rate_weight * body_rate**2
        # out of bounds penalty
        if terminated and msg == "out_of_bounds":
            reward_components["out_of_bounds"] = -self._out_of_bounds_weight
        return reward_components

    def reset(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,):
        # w.p. obs_regen_eps, regenerate obstacles; 
        # create a new spec, model, data, and recompile the spec
        
        if self._use_obstacles and self._regen_obstacles and \
            self.np_random.uniform() < self._obs_regen_eps:
            print("Regenerating obstacles")
            self.mj_spec = mujoco.MjSpec.from_file(self.xml_file)
            # generate obstacles at the beginning; possibly regenerate after each episode
            self.obstacle_metadata = self._config_generator.add_obstacles(self._start_location, self._target_location)
            self.generate_obstacle_geoms()
            self.model, self.data = self.mj_spec.recompile(self.model, self.data)
            self.init_qpos = self.data.qpos.ravel().copy()
            self.init_qvel = self.data.qvel.ravel().copy()
            self.mujoco_renderer = MujocoRenderer(
                self.model,self.data,DEFAULT_CAMERA_CONFIG,self.width,self.height,
                self.max_geom,self.camera_id,self.camera_name,self.visual_options,
            )
            self.set_start_location() 
            self.set_start_goal_geoms()
            self.set_env_radius()

        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        info = self._get_reset_info()
        if self.render_mode == "human":
            self.render()
        # compute sdf and gradient wrt state and output for agent forward pass
        # sdf, grad = self.sdf_obstacle()
        # info["sdf"] = sdf
        # info["grad"] = grad

        return ob, info

    def reset_model(self):
        """Resets the state of the environment and returns an initial observation."""

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        # Add noise to position and orientation with different scales
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        if self._mode == "train":
            qpos[:3] += self.np_random.uniform(
                low=noise_low, high=noise_high, size=3
            )
            qpos[3:7] += self.np_random.uniform(
                low=noise_low/10, high=noise_high/10, size=4
            )
            qvel = self.init_qvel + self.np_random.uniform(
                low=noise_low/10, high=noise_high/10, size=self.model.nv
            )
        # set action to hover thrust
        self.data.ctrl = self.hover_thrust
        # parent method from MujocoEnv; copies qpos, qvel to data.qpos, 
        # data.qvel to avoid pointer issues in underlying c++
        self.set_state(qpos, qvel) 
        observation = self._get_obs()
        # initialize previous pos to current pos
        self.prev_pos = qpos[:3].copy()

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
        # sdf, grad = self.sdf_obstacle()
        # info["sdf"] = sdf
        # info["grad"] = grad
        self.info = info # set info to be used in render() - called by RecordVideo wrapper
        # episode length is automatically added by RecordEpisodeStatistics wrapper
        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _check_success(self):
        """Check if quadrotor has reached the goal."""
        current_pos = self.data.qpos[:3]  # x, y, z position
        goal_pos = self._target_location
        
        # Distance-based success
        distance = np.linalg.norm(current_pos - goal_pos)
        return distance < self._goal_threshold  # e.g., 0.1 meters

    def _check_collision_obstacles(self):
        """Check if quadrotor has collided with an obstacle."""
        # refresh geoms ids since they can change due to dynamic obstacle generation
        ngeoms = self.model.ngeom
        obstacle_geoms = []
        rotor_geoms = []
        for i in range(ngeoms):
            if "obstacle" in self.model.geom(i).name:
                obstacle_geoms.append(i)
            if "rotor" in self.model.geom(i).name:
                rotor_geoms.append(i)
        for contact in self.data.contact:
            # if any rotor is in contact with any obstacle, there is a collision
            if (contact.geom1 in rotor_geoms or contact.geom2 in rotor_geoms)\
                and (contact.geom1 in obstacle_geoms or contact.geom2 in obstacle_geoms):
                return True
        return False
    
    def _check_collision_ground(self):
        """Check if quadrotor has collided with the ground."""
        return self.data.qpos[2] < self._min_height
    
    def _check_oob(self):
        """Check if quadrotor is out of bounds."""
        return np.linalg.norm(self.data.qpos[:3] - self.model.stat.center) > self.model.stat.extent

    def _is_terminated(self):
        """
        Termination conditions
        """
        if self._check_success():
            return True, "success"
        if self._progress_type != "negative":
            # for negative only rewards, terminating early can encourage the agent to end episodes early
            # to prevent accumulating larger negative rewards
            if self._check_collision_obstacles():
                return True, "collision_obstacles"
            if self._check_collision_ground():
                return True, "collision_ground"
            if self._check_oob():
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