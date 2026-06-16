import numpy as np
import copy
import os
import pickle
from d3il.environments.d3il.d3il_sim.utils.sim_path import d3il_path

from d3il.environments.d3il.d3il_sim.core import Scene
from d3il.environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from d3il.environments.d3il.d3il_sim.core.logger import ObjectLogger, CamLogger
from d3il.environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from d3il.environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from d3il.environments.d3il.d3il_sim.sims import MjCamera

from gym import spaces

from .objects.avoiding_objects import get_obj_list, init_end_eff_pos, get_obj_xy_list

obj_list = get_obj_list()


class BPCageCam(MjCamera):

    def __init__(self, width: int = 1024, height: int = 1024, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width,
            height,
            init_pos=[1.05, 0, 1.2],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],
            *args,
            **kwargs,
        )


class ObstacleAvoidanceManager:
    def __init__(self):
        self.index = 0
        pass

    def start(self):
        pass


class ObstacleAvoidanceEnv(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 35,
        max_steps_per_episode: int = 250,
        debug: bool = False,
        render: bool = False,
    ):
        self.name = "d3il-avoiding"

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        robot = MjRobot(
            scene, xml_path=d3il_path("models/mj/robot/panda_rod_invisible.xml")
        )
        controller = robot.cartesianPosQuatTrackingController

        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        # gym compatibility
        self._max_episode_steps = max_steps_per_episode

        self.manager = ObstacleAvoidanceManager()

        self.bp_cam = BPCageCam()

        self.scene.add_object(self.bp_cam)

        self.log_dict = {}
        self.cam_dict = {"bp-cam": CamLogger(scene, self.bp_cam)}

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.obj_xy_list = get_obj_xy_list()

        self.target_min_dist = 0.06

        level_distance = 0.18
        obstacle_offset = 0.075
        self.l1_ypos = -0.1
        self.l2_ypos = -0.1 + level_distance
        self.l3_ypos = -0.1 + 2 * level_distance
        self.goal_ypos = -0.1 + 2.5 * level_distance
        self.l1_xpos = 0.5
        self.l2_top_xpos = 0.5 - obstacle_offset
        self.l2_bottom_xpos = 0.5 + obstacle_offset
        self.l3_top_xpos = 0.5 - 2 * obstacle_offset
        self.l3_mid_xpos = 0.5
        self.l3_bottom_xpos = 0.5 + 2 * obstacle_offset

        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False

        self.mode_encoding = np.zeros(2 + 3 + 4)

        self.success = False

        self._obstacle_geom_ids = None

    def set_seed(self, seed):
        np.random.seed(seed)

    def get_observation(self) -> np.ndarray:
        robot_c_pos = self.robot_state()[:2]
        return robot_c_pos.astype(np.float32)

    def start(self):
        self.scene.start()

        try:
            self.scene.viewer.cam.elevation = -60
            self.scene.viewer.cam.distance = 2.5
            self.scene.viewer.cam.lookat[0] += 0.2
            self.scene.viewer.cam.lookat[2] -= 0.2
        except:
            pass

        initial_cart_position = copy.deepcopy(init_end_eff_pos)
        self.robot.gotoCartPosQuatController.setDesiredPos(
            [
                initial_cart_position[0],
                initial_cart_position[1],
                initial_cart_position[2],
                0,
                1,
                0,
                0,
            ]
        )
        self.robot.gotoCartPosQuatController.initController(self.robot, 1)

        self.robot.init_qpos = self.robot.gotoCartPosQuatController.trajectory[
            -1
        ].copy()
        self.robot.init_tcp_pos = initial_cart_position
        self.robot.init_tcp_quat = [0, 1, 0, 0]

        self.robot.beam_to_joint_pos(
            self.robot.gotoCartPosQuatController.trajectory[-1]
        )

        self.robot.gotoCartPositionAndQuat(
            desiredPos=initial_cart_position, desiredQuat=[0, 1, 0, 0], duration=0.5
        )

    def step(self, action, gripper_width=None):
        observation, _, done, _ = super().step(action, gripper_width)
        self.check_mode()
        reward = self.check_success()
        return observation, reward, done, (self.mode_encoding, self.success)

    def get_reward(self):
        return self.success

    def check_mode(self):
        r_x_pos = self.robot.current_c_pos[0]
        r_y_pos = self.robot.current_c_pos[1]
        if r_y_pos - 0.03 <= self.l1_ypos <= r_y_pos + 0.03 and (not self.l1_passed):
            if r_x_pos < self.l1_xpos:
                self.mode_encoding[0] = 1
            elif r_x_pos > self.l1_xpos:
                self.mode_encoding[1] = 1
            self.l1_passed = True

        if r_y_pos - 0.03 <= self.l2_ypos <= r_y_pos + 0.03 and (not self.l2_passed):
            if r_x_pos < self.l2_top_xpos:
                self.mode_encoding[2] = 1
            elif self.l2_top_xpos < r_x_pos < self.l2_bottom_xpos:
                self.mode_encoding[3] = 1
            elif r_x_pos > self.l2_bottom_xpos:
                self.mode_encoding[4] = 1
            self.l2_passed = True

        if r_y_pos >= self.l3_ypos and (not self.l3_passed):
            if r_x_pos < self.l3_top_xpos:
                self.mode_encoding[5] = 1
            if self.l3_top_xpos < r_x_pos < self.l3_mid_xpos:
                self.mode_encoding[6] = 1
            elif self.l3_mid_xpos < r_x_pos < self.l3_bottom_xpos:
                self.mode_encoding[7] = 1
            elif r_x_pos > self.l3_top_xpos:
                self.mode_encoding[8] = 1
            self.l3_passed = True

    def check_failure(self):
        import mujoco

        if self._obstacle_geom_ids is None:
            obstacles = [
                "l1_obs",
                "l2_top_obs",
                "l2_bottom_obs",
                "l3_top_obs",
                "l3_mid_obs",
                "l3_bottom_obs",
            ]
            self._obstacle_geom_ids = [
                mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle + ":geom"
                )
                for obstacle in obstacles
            ]

        for i in range(self.scene.data.ncon):
            contact = self.scene.data.contact[i]

            obstacle_involved = (
                contact.geom1 in self._obstacle_geom_ids
                or contact.geom2 in self._obstacle_geom_ids
            )

            if obstacle_involved:
                geom1_name = mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                )
                geom2_name = mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                )

                if (geom1_name and geom1_name.startswith("rod:")) or (
                    geom2_name and geom2_name.startswith("rod:")
                ):
                    return True

        return False

    def check_success(self):
        if self.robot.current_c_pos[1] > self.goal_ypos:
            return True
        else:
            return False

    def reset_mode_encoding(self):
        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False
        assert np.sum(self.mode_encoding) <= 3
        self.mode_encoding = np.zeros(2 + 3 + 4)

    def get_reward(self): ...

    def _check_early_termination(self) -> bool:
        success = self.check_success()
        failure = self.check_failure()

        if success or failure:
            if success:
                self.success = True
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self.reset_mode_encoding()
        self.success = False
        obs = self._reset_env(random=random, context=context)
        return obs

    def _reset_env(self, random=True, context=None):
        self.scene.reset()
        self.robot.beam_to_joint_pos(self.robot.init_qpos)
        self.scene.next_step()
        observation = self.get_observation()
        return observation

    def reward(self, x):
        def squared_exp_kernel(x, mean, scale, bandwidth):
            return scale * np.exp(
                np.square(np.linalg.norm(x - mean, axis=1)) / bandwidth
            )

        rewards = np.zeros(x.shape[0])
        for obs in self.obj_xy_list:
            rewards -= squared_exp_kernel(x, np.array(obs), 1, 1)

        rewards -= np.abs(x[:, 0] - 0.4)
        return rewards

    def mode_decoding(self, data):
        data_decimal = data.dot(1 << np.arange(data.shape[-1]))
        _, counts = np.unique(data_decimal, return_counts=True)
        mode_dist = counts / np.sum(counts)
        entropy = -np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))
        return counts, entropy

    def action_space(self): ...

    def get_dataset(self):
        data_directory = "../dataset/data/avoiding/data"
        data_dir = d3il_path(data_directory)
        state_files = os.listdir(data_dir)

        all_observations = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        all_timeouts = []

        for file in state_files:
            with open(os.path.join(data_dir, file), "rb") as f:
                env_state = pickle.load(f)

                robot_des_pos = env_state["robot"]["des_c_pos"][:, :2]
                robot_c_pos = env_state["robot"]["c_pos"][:, :2]

                input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

                vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
                valid_len = len(vel_state)

            all_observations.append(input_state[:-1])
            all_actions.append(vel_state)
            all_rewards.append(np.zeros(valid_len))
            all_terminals.append(
                np.concatenate((np.zeros(valid_len - 1), np.array([1])))
            )
            all_timeouts.append(np.zeros(valid_len))

        return {
            "observations": np.concatenate(all_observations, axis=0),
            "actions": np.concatenate(all_actions, axis=0),
            "rewards": np.concatenate(all_rewards, axis=0),
            "terminals": np.concatenate(all_terminals, axis=0),
            "timeouts": np.concatenate(all_timeouts, axis=0),
        }


class WrappedObstacleAvoidanceEnv(ObstacleAvoidanceEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "avoiding-v0"

        self.desired_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.fixed_z = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-0.0125, 0.0]),
            high=np.array([0.0125, 0.0125]),
            shape=(2,),
            dtype=np.float32,
        )

    def get_observation(self) -> np.ndarray:
        actual_pos = self.robot_state()[:2]
        observation = np.concatenate([self.desired_pos, actual_pos])
        return observation.astype(np.float32)

    def set_state(self, observation):

        # observation: [desired_x, desired_y, actual_x, actual_y]

        desired_pos = observation[:2]
        actual_pos = observation[2:]

        self.desired_pos = desired_pos.copy()

        if self.fixed_z is None:
            self.fixed_z = self.robot_state()[2:].copy()

        full_actual_pos = np.concatenate([actual_pos, self.fixed_z[:1]])
        full_actual_pos_with_quat = np.concatenate([full_actual_pos, [0, 1, 0, 0]])

        self.robot.gotoCartPosQuatController.setDesiredPos(full_actual_pos_with_quat)
        self.robot.gotoCartPosQuatController.initController(self.robot, 1)

        self.robot.beam_to_joint_pos(
            self.robot.gotoCartPosQuatController.trajectory[-1]
        )
        self.scene.next_step()

    def step(self, action, gripper_width=None):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        next_desired_pos = self.desired_pos + action
        self.desired_pos = next_desired_pos.copy()

        full_action = np.concatenate(
            (next_desired_pos, self.fixed_z, [0, 1, 0, 0]), axis=0
        )

        _, reward, done, info = super().step(full_action, gripper_width)

        observation = self.get_observation()
        return observation, reward, done, info

    def reset(self, random=True, context=None):
        _ = super().reset(random=random, context=context)
        self.desired_pos = self.robot_state()[:2].copy()
        self.fixed_z = self.robot_state()[2:].copy()

        observation = self.get_observation()
        assert np.array_equal(
            observation[:2], observation[2:]
        ), "Problematic initialization."
        return observation
