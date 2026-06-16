import threading
from abc import abstractmethod

import numpy as np

import d3il.environments.d3il.d3il_sim.controllers.GainsInterface as gains


class ControllerBase:

    def __init__(self):
        self.paramsLock = threading.Lock()
        self.last_control_timestamp = np.nan
        self._max_duration = None
        self._max_timesteps = None
        self._controller_timer = None

    def isFinished(self, robot):

        if self._max_duration is not None:
            return robot.time_stamp - self._controller_timer >= self._max_duration

        if self._max_timesteps is not None:
            return robot.step_count - self._controller_timer >= self._max_timesteps
        return False

    def initController(self, robot, maxDuration):
        return

    def getControl(self, robot):
        self.last_control_timestamp = robot.time_stamp
        return 0

    def is_used(self, robot):
        return (
            not np.isnan(self.last_control_timestamp)
            and robot.time_stamp - self.last_control_timestamp < 0.03
        )

    def setAction(self, action):
        return 0

    def run(self, robot, log=True):

        while not self.isFinished(robot):
            robot.nextStep(log)

    def executeController(self, robot, maxDuration=10, block=True, log=True):

        self._max_duration = maxDuration
        self._max_timesteps = None
        self._controller_timer = robot.time_stamp

        self.initController(robot, maxDuration)
        robot.activeController = self

        if block:
            self.run(robot, log=log)

    def executeControllerTimeSteps(self, robot, timeSteps=10, block=True, log=True):

        self._max_duration = None
        self._max_timesteps = timeSteps
        self._controller_timer = robot.step_count

        self.initController(robot, timeSteps * robot.dt)
        robot.activeController = self

        if block:
            self.run(robot, log)

    @abstractmethod
    def reset(self):
        pass


class TorqueController(ControllerBase):

    def __init__(self):
        ControllerBase.__init__(self)
        self.reset()

    def getControl(self, robot):
        super(TorqueController, self).getControl(robot)
        return self.torque

    def setAction(self, action):
        self.torque = action.copy()

    def reset(self):
        self.torque = []


class TrackingController(ControllerBase):

    def __init__(self, dimSetPoint):
        ControllerBase.__init__(self)
        self.dimSetPoint = dimSetPoint
        self.tracking_error = False

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        pass

    def getCurrentPos(self, robot):
        pass

    def getDesiredPos(self, robot):
        pass

    @abstractmethod
    def reset(self):
        pass


class JointPDController(TrackingController, gains.JointPDGains):

    def __init__(self):
        TrackingController.__init__(self, dimSetPoint=7)
        gains.JointPDGains.__init__(self)

        self.reset()

    def reset(self):
        self.desired_joint_pos = np.array([0, 0, 0, -1.562, 0, 1.914, 0])
        self.desired_joint_vel = np.zeros((7,))
        self.desired_joint_acc = np.zeros((7,))

    def getControl(self, robot):

        super(JointPDController, self).getControl(robot)
        self.paramsLock.acquire()
        qd_d = self.desired_joint_pos - robot.current_j_pos
        vd_d = self.desired_joint_vel - robot.current_j_vel

        target_j_acc = self.pgain * qd_d + self.dgain * vd_d

        robot.des_joint_pos = self.desired_joint_pos.copy()
        robot.des_joint_vel = self.desired_joint_vel.copy()
        robot.des_joint_acc = self.desired_joint_acc.copy()

        self.paramsLock.release()
        return target_j_acc

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):

        self.paramsLock.acquire()
        self.desired_joint_pos = desired_pos
        if desired_vel is not None:
            self.desired_joint_vel = desired_vel
        if desired_acc is not None:
            self.desired_joint_acc = desired_acc
        self.paramsLock.release()

    def getCurrentPos(self, robot):

        return robot.current_j_pos

    def getDesiredPos(self, robot):

        return robot.des_joint_pos


class ModelBasedFeedforwardController(JointPDController):

    def __init__(self):
        JointPDController.__init__(self)

    def getControl(self, robot):

        super(ModelBasedFeedforwardController, self).getControl(robot)
        self.paramsLock.acquire()
        qd_d = self.desired_joint_pos - robot.current_j_pos
        vd_d = self.desired_joint_vel - robot.current_j_vel

        target_j_acc = self.pgain * qd_d + self.dgain * vd_d
        uff = robot.get_mass_matrix(self.desired_joint_pos).dot(
            self.desired_joint_acc
        ) + robot.get_coriolis(self.desired_joint_pos, self.desired_joint_vel)

        robot.des_joint_pos = self.desired_joint_pos.copy()
        robot.des_joint_vel = self.desired_joint_vel.copy()
        robot.des_joint_acc = self.desired_joint_acc.copy()

        self.paramsLock.release()
        return target_j_acc + uff


class ModelBasedFeedbackController(JointPDController):

    def __init__(self):
        JointPDController.__init__(self)

    def getControl(self, robot):

        super(ModelBasedFeedbackController, self).getControl(robot)
        self.paramsLock.acquire()
        qd_d = self.desired_joint_pos - robot.current_j_pos
        vd_d = self.desired_joint_vel - robot.current_j_vel

        target_j_acc = self.pgain * qd_d + self.dgain * vd_d + self.desired_joint_acc
        uff = robot.get_mass_matrix(robot.current_j_pos).dot(
            target_j_acc
        ) + robot.get_coriolis(robot.current_j_pos, robot.current_j_vel)

        robot.des_joint_pos = self.desired_joint_pos.copy()
        robot.des_joint_vel = self.desired_joint_vel.copy()
        robot.des_joint_acc = self.desired_joint_acc.copy()

        self.paramsLock.release()
        return uff


class JointPositionController(JointPDController):
    def setAction(self, action):
        self.desired_joint_pos = action


class JointVelocityController(JointPDController):
    def __init__(self):
        JointPDController.__init__(self)
        self.pgain = np.zeros((self.dimSetPoint,))

    def setAction(self, action):
        self.desired_joint_vel = action


class ZeroTorqueController(TrackingController):

    def __init__(self, dimSetPoint=7):
        TrackingController.__init__(self, dimSetPoint=dimSetPoint)

    def getControl(self, robot):
        super().getControl(robot)
        target_j_acc = np.zeros((self.dimSetPoint,))
        return target_j_acc

    def reset(self):
        pass


class DampingController(ControllerBase, gains.DampingGains):

    def __init__(self):
        ControllerBase.__init__(self)
        gains.DampingGains.__init__(self)

    def getControl(self, robot):

        super(DampingController, self).getControl(robot)
        self.paramsLock.acquire()
        target_j_acc = -self.dgain * robot.current_j_vel
        self.paramsLock.release()
        return target_j_acc
