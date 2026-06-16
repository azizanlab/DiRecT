from abc import ABC, abstractmethod

import gym
from gym.utils import seeding
import numpy as np

from d3il.environments.d3il.d3il_sim.controllers.Controller import ControllerBase
from d3il.environments.d3il.d3il_sim.core import Scene


class GymEnvWrapper(gym.Env, ABC):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self,
        scene: Scene,
        controller: ControllerBase,
        max_steps_per_episode,
        n_substeps,
        debug: bool = False,
    ):
        self.scene = scene
        self.robot = scene.robots[0]

        self.controller = controller

        self.max_steps_per_episode = max_steps_per_episode
        self.n_substeps = n_substeps
        self.env_step_counter = 0

        self.episode = 0
        self.terminated = False
        self.debug = debug

    def start(self):
        self.scene.start()
        self.controller.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, gripper_width=None, desired_vel=None, desired_acc=None):

        if gripper_width is not None:
            self.robot.set_gripper_width = gripper_width

        self.robot.open_fingers()

        self.controller.setSetPoint(action)
        self.controller.executeControllerTimeSteps(
            self.robot, self.n_substeps, block=False
        )

        for i in range(self.n_substeps):
            self.scene.next_step()

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_finished()

        debug_info = {}
        if self.debug:
            debug_info = self.debug_msg()

        self.env_step_counter += 1
        return observation, reward, done, debug_info

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_reward(self):
        raise NotImplementedError

    @abstractmethod
    def _check_early_termination(self) -> bool:
        return self.terminated

    def is_finished(self):
        if (
            self.terminated
            or self._check_early_termination()
            or self.env_step_counter >= self.max_steps_per_episode - 1
        ):
            return True
        return False

    def debug_msg(self) -> dict:
        return {}

    @abstractmethod
    def _reset_env(self):
        raise NotImplementedError

    def reset(self):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        obs = self._reset_env()

        return obs

    def robot_state(self):
        self.robot.receiveState()

        joint_pos = self.robot.current_j_pos
        joint_vel = self.robot.current_j_vel

        gripper_vel = self.robot.current_fing_vel
        gripper_width = [self.robot.gripper_width]

        tcp_pos = self.robot.current_c_pos
        tcp_vel = self.robot.current_c_vel
        tcp_quad = self.robot.current_c_quat

        return tcp_pos
