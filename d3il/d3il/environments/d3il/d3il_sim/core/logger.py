import abc
import datetime
import os
from enum import Flag, auto

import numpy as np

try:
    import wandb
except ImportError:
    pass


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


class RobotPlotFlags(Flag):
    JOINT_POS = auto()
    JOINT_VEL = auto()
    JOINT_ACC = auto()

    CART_POS = auto()
    CART_VEL = auto()
    ORIENTATION = auto()
    ORIENTATION_VEL = auto()

    CART_POS_GLOBAL = auto()
    CART_VEL_GLOBAL = auto()
    ORIENTATION_GLOBAL = auto()
    ORIENTATION_VEL_GLOBAL = auto()

    COMMAND = auto()
    TORQUES = auto()
    GRAVITY = auto()
    LOADS = auto()
    TIME_STAMPS = auto()

    GRIPPER_POS = auto()
    GRIPPER_WIDTH = auto()
    GRIPPER_FORCE = auto()
    GRIPPER_VEL = auto()

    MISC = auto()

    GRIPPER = GRIPPER_WIDTH | GRIPPER_POS | GRIPPER_FORCE | GRIPPER_VEL
    JOINTS = JOINT_POS | JOINT_ACC | JOINT_VEL
    END_EFFECTOR = CART_POS | CART_VEL | ORIENTATION | ORIENTATION_VEL
    END_EFFECTOR_GLOBAL = (
        CART_POS_GLOBAL | CART_VEL_GLOBAL | ORIENTATION_GLOBAL | ORIENTATION_VEL_GLOBAL
    )

    FULL = GRIPPER | JOINTS | END_EFFECTOR | END_EFFECTOR_GLOBAL


class ObjectPlotFlags(Flag):
    POSITION = auto()
    ORIENTATION = auto()
    FULL = POSITION | ORIENTATION


class CamPlotFlags(Flag):
    IMAGE = auto()
    FULL = IMAGE


class LoggerBase(abc.ABC):
    def __init__(self):

        self.logged_time = 0
        self.log_interval = None
        self.log_duration = None
        self.logged_time_steps = 0
        self.max_time_steps = None
        self.last_log_time = None
        self.is_logging = False

    def start_logging(self, duration: float = 300.0, **kwargs):

        self.logged_time = 0
        self.logged_time_steps = 0
        self.log_duration = duration
        self._start(duration, **kwargs)
        self.is_logging = True

    def log_data(self):

        if self.is_logging and self._check_log_interval():
            self._log()
            self.logged_time_steps += 1
            if (
                self.max_time_steps is not None
                and self.logged_time_steps >= self.max_time_steps
            ) or self.logged_time >= self.log_duration:
                self.stop_logging()

    def stop_logging(self):

        if self.is_logging:
            self.is_logging = False
            if self.logged_time_steps > 0:
                self._stop()

    @abc.abstractmethod
    def _start(self, duration=600, **kwargs):
        pass

    @abc.abstractmethod
    def _log(self):
        pass

    @abc.abstractmethod
    def _stop(self):
        pass

    @abc.abstractmethod
    def _check_log_interval(self) -> bool:
        return True

    @abc.abstractmethod
    def reset(self):
        pass


class RobotLogger(LoggerBase):
    def __init__(self, robot):
        super().__init__()

        self.robot = robot
        self.plot_selection = RobotPlotFlags.FULL
        self.use_wandb = False

        self.reset()

    def reset(self):
        self.log_dict_full = None

    @property
    def log_dict(self):
        log_dict = dict()

        if RobotPlotFlags.JOINTS & self.plot_selection:
            log_dict.update(
                {
                    "joint_pos": self.robot.current_j_pos,
                    "joint_vel": self.robot.current_j_vel,
                    "des_joint_pos": self.robot.des_joint_pos,
                    "des_joint_vel": self.robot.des_joint_vel,
                    "des_joint_acc": self.robot.des_joint_acc,
                }
            )

        if RobotPlotFlags.GRIPPER & self.plot_selection:
            log_dict.update(
                {
                    "finger_pos": self.robot.current_fing_pos,
                    "finger_vel": self.robot.current_fing_vel,
                    "des_finger_pos": self.robot.set_gripper_width,
                    "gripper_width": self.robot.gripper_width,
                    "uff": self.robot.uff,
                }
            )

        if RobotPlotFlags.END_EFFECTOR & self.plot_selection:
            log_dict.update(
                {
                    "cart_pos": self.robot.current_c_pos,
                    "cart_vel": self.robot.current_c_vel,
                    "cart_quat": self.robot.current_c_quat,
                    "cart_quat_vel": self.robot.current_c_quat_vel,
                    "des_c_pos": self.robot.des_c_pos,
                    "des_c_vel": self.robot.des_c_vel,
                    "des_quat": self.robot.des_quat,
                    "des_quat_vel": self.robot.des_quat_vel,
                }
            )

        if RobotPlotFlags.END_EFFECTOR_GLOBAL & self.plot_selection:
            log_dict.update(
                {
                    "cart_pos_global": self.robot.current_c_pos_global,
                    "cart_vel_global": self.robot.current_c_vel_global,
                    "cart_quat_global": self.robot.current_c_quat_global,
                    "cart_quat_vel_global": self.robot.current_c_quat_vel_global,
                }
            )

        if RobotPlotFlags.COMMAND & self.plot_selection:
            log_dict.update(
                {
                    "uff": self.robot.uff,
                    "last_cmd": self.robot.last_cmd,
                    "command": self.robot.command,
                    "grav_terms": self.robot.grav_terms,
                    "load": self.robot.current_load,
                }
            )

        if RobotPlotFlags.MISC & self.plot_selection:
            log_dict.update({"misc": self.robot.misc_data})

        log_dict.update(
            {
                "time_stamp": self.robot.time_stamp,
                "step": self.robot.step_count,
                "wall_clock": self.robot.time_keeper.wall_clock,
            }
        )

        return log_dict

    def _start(
        self,
        duration=600,
        log_interval=None,
        plot_selection: RobotPlotFlags = RobotPlotFlags.FULL,
        use_wandb=False,
        **kwargs
    ):

        self.plot_selection = plot_selection
        self.use_wandb = use_wandb

        if log_interval is None:
            self.log_interval = 1
        else:
            self.log_interval = log_interval

        assert self.log_interval >= 1

        self.max_time_steps = int(duration / self.log_interval)

        if self.max_time_steps > 1800000:
            self.max_time_steps = 1800000

        self.log_dict_full = {
            k: [None] * self.max_time_steps for k, v in self.log_dict.items()
        }

        self.initial_time_stamp = self.robot.step_count

        if self.use_wandb:
            reset_wandb_env()
            now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            job_name = "robot_log_" + now
            self.run = wandb.init(
                project="sf",
                group="test_log",
                name=job_name,
                settings=wandb.Settings(_disable_stats=True),
            )

    def _log(self):
        log_properties = self.log_dict

        for k, v in log_properties.items():
            self.log_dict_full[k][self.logged_time_steps] = v

        self.last_log_time = log_properties["step"]

        self.logged_time = self.last_log_time - self.initial_time_stamp

        if self.use_wandb:
            flat_log_dict = {
                k + "_" + str(i): vv
                for k, v in log_properties.items()
                for i, vv in enumerate(np.atleast_1d(v))
            }
            self.run.log(flat_log_dict, step=self.logged_time_steps)

    def _stop(self):

        if self.logged_time_steps < self.max_time_steps:
            for v in self.log_dict_full.values():
                del v[self.logged_time_steps :]

        if self.use_wandb:
            if self.run is not None:
                self.run.finish()

        self.__dict__.update({k: np.array(v) for k, v in self.log_dict_full.items()})

    def _check_log_interval(self) -> bool:

        if (
            self.last_log_time is None
            or self.robot.step_count >= self.last_log_time + self.log_interval
        ):
            return True
        return False

    def plot(self, plot_selection=RobotPlotFlags.FULL, block=True):

        import matplotlib.pyplot as plt

        valid_plot_selections = plot_selection & self.plot_selection

        if self.is_logging:
            self.stop_logging()

        self.time_stamp = self.time_stamp - self.time_stamp[0]

        j1_limit_lower = -2.9671
        j1_limit_upper = 2.9671
        j2_limit_lower = -1.8326
        j2_limit_upper = 1.8326
        j3_limit_lower = -2.9671
        j3_limit_upper = 2.9671
        j4_limit_lower = -3.1416
        j4_limit_upper = 0.0873
        j5_limit_lower = -2.9671
        j5_limit_upper = 2.9671
        j6_limit_lower = -0.0873
        j6_limit_upper = 3.8223
        j7_limit_lower = -2.9671
        j7_limit_upper = 2.9671

        j1_limit_lower_safety = -2.8973
        j1_limit_upper_safety = 2.8973
        j2_limit_lower_safety = -1.7628
        j2_limit_upper_safety = 1.7628
        j3_limit_lower_safety = -2.8973
        j3_limit_upper_safety = 2.8973
        j4_limit_lower_safety = -3.0718
        j4_limit_upper_safety = 0.0175
        j5_limit_lower_safety = -2.8973
        j5_limit_upper_safety = 2.8973
        j6_limit_lower_safety = -0.0175
        j6_limit_upper_safety = 3.7525
        j7_limit_lower_safety = -2.8973
        j7_limit_upper_safety = 2.8973

        j1_vel_limit_lower = -2.3925
        j1_vel_limit_upper = 2.3925
        j2_vel_limit_lower = -2.3925
        j2_vel_limit_upper = 2.3925
        j3_vel_limit_lower = -2.3925
        j3_vel_limit_upper = 2.3925
        j4_vel_limit_lower = -2.3925
        j4_vel_limit_upper = 2.3925
        j5_vel_limit_lower = -2.8710
        j5_vel_limit_upper = 2.8710
        j6_vel_limit_lower = -2.8710
        j6_vel_limit_upper = 2.8710
        j7_vel_limit_lower = -2.8710
        j7_vel_limit_upper = 2.8710

        upper_limits = [
            j1_limit_upper,
            j2_limit_upper,
            j3_limit_upper,
            j4_limit_upper,
            j5_limit_upper,
            j6_limit_upper,
            j7_limit_upper,
        ]
        lower_limits = [
            j1_limit_lower,
            j2_limit_lower,
            j3_limit_lower,
            j4_limit_lower,
            j5_limit_lower,
            j6_limit_lower,
            j7_limit_lower,
        ]

        soft_upper_limits = [
            j1_limit_upper_safety,
            j2_limit_upper_safety,
            j3_limit_upper_safety,
            j4_limit_upper_safety,
            j5_limit_upper_safety,
            j6_limit_upper_safety,
            j7_limit_upper_safety,
        ]

        soft_lower_limits = [
            j1_limit_lower_safety,
            j2_limit_lower_safety,
            j3_limit_lower_safety,
            j4_limit_lower_safety,
            j5_limit_lower_safety,
            j6_limit_lower_safety,
            j7_limit_lower_safety,
        ]

        upper_limits_vel = [
            j1_vel_limit_upper,
            j2_vel_limit_upper,
            j3_vel_limit_upper,
            j4_vel_limit_upper,
            j5_vel_limit_upper,
            j6_vel_limit_upper,
            j7_vel_limit_upper,
        ]

        lower_limits_vel = [
            j1_vel_limit_lower,
            j2_vel_limit_lower,
            j3_vel_limit_lower,
            j4_vel_limit_lower,
            j5_vel_limit_lower,
            j6_vel_limit_lower,
            j7_vel_limit_lower,
        ]

        if RobotPlotFlags.JOINT_POS in valid_plot_selections:
            joint_pos_fig = plt.figure()
            for k in range(7):
                plt.figure(joint_pos_fig.number)
                plt.subplot(7, 1, k + 1)
                plt.plot(self.time_stamp, self.joint_pos[:, k])
                plt.plot(self.time_stamp, self.des_joint_pos[:, k], "r")
            plt.subplot(7, 1, 1)
            plt.title("joint positions ")

        if RobotPlotFlags.JOINT_VEL in valid_plot_selections:
            joint_vel_fig = plt.figure()
            for k in range(7):
                plt.figure(joint_vel_fig.number)
                plt.subplot(7, 1, k + 1)
                plt.plot(self.time_stamp, self.joint_vel[:, k])
                plt.plot(self.time_stamp, self.des_joint_vel[:, k], "r")
            plt.subplot(7, 1, 1)
            plt.title("joint velocities ")

        if RobotPlotFlags.JOINT_ACC in valid_plot_selections:
            robot_acceleration = np.diff(self.joint_vel, 1, axis=0) / self.robot.dt
            acceleration_fig = plt.figure()
            for k in range(7):
                plt.figure(acceleration_fig.number)
                plt.subplot(7, 1, k + 1)
                plt.plot(self.time_stamp[1:], robot_acceleration[:, k])
                plt.plot(self.time_stamp, self.des_joint_acc[:, k], "r")
            plt.subplot(7, 1, 1)
            plt.title("joint accelerations ")

        if RobotPlotFlags.COMMAND in valid_plot_selections:
            ctrl_torques_fig = plt.figure()
            for k in range(7):
                plt.figure(ctrl_torques_fig.number)
                plt.subplot(7, 1, k + 1)
                plt.plot(self.time_stamp, self.command[:, k])
            plt.subplot(7, 1, 1)
            plt.title(" control torques (without gravity comp.)")

        if RobotPlotFlags.TORQUES in valid_plot_selections:
            clipped_torques_fig = plt.figure()
            for k in range(7):
                plt.figure(clipped_torques_fig.number)
                plt.subplot(7, 1, k + 1)
                plt.plot(self.time_stamp, self.uff[:, k])
            plt.subplot(7, 1, 1)
            plt.title(" clipped torques (with gravity comp.)")

        if RobotPlotFlags.GRAVITY in valid_plot_selections:
            grav_terms_fig = plt.figure()
            for k in range(7):
                plt.figure(grav_terms_fig.number)
                plt.subplot(9, 1, k + 1)
                plt.plot(self.time_stamp, self.grav_terms[:, k])
            plt.subplot(7, 1, 1)
            plt.title(" gravity (+coriolis) ")

        if RobotPlotFlags.LOADS in valid_plot_selections:
            loads_fig = plt.figure()
            for k in range(7):
                plt.figure(loads_fig.number)
                plt.subplot(7, 1, k + 1)
                plt.plot(self.time_stamp, self.load[:, k])
            plt.subplot(7, 1, 1)
            plt.title(" loads - forces in joints")

        if RobotPlotFlags.GRIPPER_WIDTH in valid_plot_selections:
            gripper_fig = plt.figure()
            plt.figure(gripper_fig.number)
            plt.plot(self.time_stamp, self.gripper_width)
            plt.title(" gripper width ")

        if RobotPlotFlags.GRIPPER_FORCE in valid_plot_selections:
            gripper_force_fig = plt.figure()
            plt.figure(gripper_force_fig.number)
            plt.plot(self.time_stamp, self.uff[:, -2:])

            plt.title(" gripper force (without contact forces)")

        if RobotPlotFlags.GRIPPER_POS in valid_plot_selections:
            gripper_pos_fig = plt.figure()
            plt.figure(gripper_pos_fig.number)
            plt.plot(self.time_stamp, self.finger_pos[:, -2:], label="actual")
            plt.plot(self.time_stamp, self.des_finger_pos, label="desired")
            plt.legend()
            plt.title(" gripper pos ")

        if RobotPlotFlags.GRIPPER_VEL in valid_plot_selections:
            gripper_vel_fig = plt.figure()
            plt.figure(gripper_vel_fig.number)
            plt.plot(self.time_stamp, self.finger_vel[:, -2:])
            plt.title(" gripper vel ")

        if RobotPlotFlags.CART_POS in valid_plot_selections:
            cart_pos_fig = plt.figure()
            for j in range(3):
                plt.figure(cart_pos_fig.number)
                plt.subplot(3, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_pos[:, j])
                plt.plot(self.time_stamp, self.des_c_pos[:, j], "r")
            plt.subplot(3, 1, 1)
            plt.title(" endeffector pos ")

        if RobotPlotFlags.CART_VEL in valid_plot_selections:
            cart_vel_fig = plt.figure()
            for j in range(3):
                plt.figure(cart_vel_fig.number)
                plt.subplot(3, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_vel[:, j])
                plt.plot(self.time_stamp, self.des_c_vel[:, j], "r")
            plt.subplot(3, 1, 1)
            plt.title(" endeffector vel ")

        if RobotPlotFlags.ORIENTATION in valid_plot_selections:
            orientation_fig = plt.figure()
            for j in range(4):
                plt.figure(orientation_fig.number)
                plt.subplot(4, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_quat[:, j])
                plt.plot(self.time_stamp, self.des_quat[:, j], "r")
            plt.subplot(4, 1, 1)
            plt.title(" endeffector orientation ")

        if RobotPlotFlags.ORIENTATION_VEL in valid_plot_selections:
            orientation_vel_fig = plt.figure()
            for j in range(4):
                plt.figure(orientation_vel_fig.number)
                plt.subplot(4, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_quat_vel[:, j])
                plt.plot(self.time_stamp, self.des_quat_vel[:, j], "r")
            plt.subplot(4, 1, 1)
            plt.title(" endeffector angular velocity ")

        if RobotPlotFlags.CART_POS_GLOBAL in valid_plot_selections:
            cart_pos_fig = plt.figure()
            for j in range(3):
                plt.figure(cart_pos_fig.number)
                plt.subplot(3, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_pos_global[:, j])
            plt.subplot(3, 1, 1)
            plt.title(" endeffector pos global")

        if RobotPlotFlags.CART_VEL_GLOBAL in valid_plot_selections:
            cart_vel_fig = plt.figure()
            for j in range(3):
                plt.figure(cart_vel_fig.number)
                plt.subplot(3, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_vel_global[:, j])
            plt.subplot(3, 1, 1)
            plt.title(" endeffector vel global")

        if RobotPlotFlags.ORIENTATION_GLOBAL in valid_plot_selections:
            orientation_fig = plt.figure()
            for j in range(4):
                plt.figure(orientation_fig.number)
                plt.subplot(4, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_quat_global[:, j])
            plt.subplot(4, 1, 1)
            plt.title(" endeffector orientation global")

        if RobotPlotFlags.ORIENTATION_VEL_GLOBAL in valid_plot_selections:
            orientation_vel_fig = plt.figure()
            for j in range(4):
                plt.figure(orientation_vel_fig.number)
                plt.subplot(4, 1, j + 1)
                plt.plot(self.time_stamp, self.cart_quat_vel_global[:, j])
            plt.subplot(4, 1, 1)
            plt.title(" endeffector angular velocity global")

        if RobotPlotFlags.TIME_STAMPS in valid_plot_selections:
            timeStamp_fig = plt.figure()
            plt.figure(timeStamp_fig.number)
            plt.plot(np.diff(self.time_stamp))
            plt.title("time stamp difference ")

        if RobotPlotFlags.MISC in valid_plot_selections:
            misc_fig = plt.figure()
            numplots = self.misc.shape[1]
            for j in range(numplots):
                plt.figure(misc_fig.number)
                plt.subplot(numplots, 1, j + 1)
                plt.plot(self.misc[:, j])

            plt.subplot(numplots, 1, 1)
            plt.title("misc debug data ")
        plt.show(block=block)


class ObjectLogger(LoggerBase):

    def __init__(self, scene, sim_object=None, obj_name=None):
        super().__init__()
        self.scene = scene

        if obj_name is not None:
            sim_object = scene.get_object(name=obj_name)

        if sim_object is None:
            raise ValueError("No Simobject defined.")
        self.object = sim_object
        self.plot_selection = ObjectPlotFlags.FULL

    def reset(self):
        self.log_dict_full = None
        self.last_log_time = None

    @property
    def log_dict(self):
        log_dict = dict()

        if ObjectPlotFlags.POSITION in self.plot_selection:
            log_dict.update({"pos": self.scene.get_obj_pos(self.object).flatten()})

        if ObjectPlotFlags.ORIENTATION in self.plot_selection:
            log_dict.update({"orientation": self.scene.get_obj_quat(self.object)})

        log_dict.update({"time_stamp": self.scene.robots[0].time_stamp})
        log_dict.update({"step": self.scene.robots[0].step_count})

        return log_dict

    def _start(
        self,
        duration=600,
        log_interval=None,
        obj_plot_selection: ObjectPlotFlags = ObjectPlotFlags.FULL,
        use_wandb=False,
        **kwargs
    ):

        self.plot_selection = obj_plot_selection
        self.use_wandb = use_wandb

        if log_interval is None:
            self.log_interval = 1
        else:
            self.log_interval = log_interval

        assert self.log_interval >= 1

        self.max_time_steps = int(duration / self.log_interval)

        if self.max_time_steps > 1800000:
            self.max_time_steps = 1800000

        self.log_dict_full = {
            k: [None] * self.max_time_steps for k, v in self.log_dict.items()
        }

        self.initial_time_stamp = self.scene.robots[0].step_count

        if self.use_wandb:
            reset_wandb_env()
            now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            job_name = "robot_log_" + now
            self.run = wandb.init(
                project="sf",
                group="test_log",
                name=job_name,
                settings=wandb.Settings(_disable_stats=True),
            )

    def _log(self):
        log_properties = self.log_dict

        for k, v in log_properties.items():
            self.log_dict_full[k][self.logged_time_steps] = v

        self.last_log_time = log_properties["step"]

        self.logged_time = self.last_log_time - self.initial_time_stamp

        if self.use_wandb:
            flat_log_dict = {
                k + "_" + str(i): vv
                for k, v in log_properties.items()
                for i, vv in enumerate(np.atleast_1d(v))
            }
            self.run.log(flat_log_dict, step=self.logged_time_steps)

    def _stop(self):

        if self.logged_time_steps < self.max_time_steps:
            for v in self.log_dict_full.values():
                del v[self.logged_time_steps :]

        if self.use_wandb:
            if self.run is not None:
                self.run.finish()

        self.__dict__.update({k: np.array(v) for k, v in self.log_dict_full.items()})

    def _check_log_interval(self) -> bool:
        if (
            self.last_log_time is None
            or self.scene.robots[0].step_count >= self.last_log_time + self.log_interval
        ):
            return True
        return False

    def plot(self, plot_selection=ObjectPlotFlags.FULL, block=True):
        valid_plot_selections = plot_selection & self.plot_selection

        import matplotlib.pyplot as plt

        self.time_stamp = self.time_stamp - self.time_stamp[0]

        if ObjectPlotFlags.POSITION in valid_plot_selections:
            pos_fig = plt.figure()
            for k in range(3):
                plt.figure(pos_fig.number)
                plt.subplot(3, 1, k + 1)
                plt.plot(self.time_stamp, self.pos[:, k])
            plt.subplot(3, 1, 1)
            plt.title("object positions ")

        if ObjectPlotFlags.ORIENTATION in valid_plot_selections:
            orientation_fig = plt.figure()
            for k in range(4):
                plt.figure(orientation_fig.number)
                plt.subplot(4, 1, k + 1)
                plt.plot(self.time_stamp, self.orientation[:, k])
            plt.title(" object orientation ")
        plt.show(block=block)


class CamLogger(LoggerBase):

    def __init__(self, scene, camera):
        super().__init__()

        self.scene = scene
        self.cam = camera

        self.plot_selection = CamPlotFlags.FULL

    def reset(self):
        self.log_dict_full = None
        self.last_log_time = None
        self.cam.get_image(depth=False)

    @property
    def log_dict(self):
        log_dict = dict()

        if CamPlotFlags.IMAGE in self.plot_selection:

            log_dict.update({"color_image": self.cam.get_image(depth=False)})

        log_dict.update({"time_stamp": self.scene.robots[0].step_count})
        log_dict.update({"step": self.scene.robots[0].step_count})

        return log_dict

    def _start(
        self,
        duration=600,
        log_interval=None,
        obj_plot_selection: CamPlotFlags = CamPlotFlags.FULL,
        use_wandb=False,
        **kwargs
    ):

        self.plot_selection = obj_plot_selection
        self.use_wandb = use_wandb

        if log_interval is None:
            self.log_interval = 1
        else:
            self.log_interval = log_interval

        assert self.log_interval >= 1

        self.max_time_steps = int(duration / self.log_interval)

        if self.max_time_steps > 1800000:
            self.max_time_steps = 1800000

        self.log_dict_full = {
            k: [None] * self.max_time_steps for k, v in self.log_dict.items()
        }

        self.initial_time_stamp = self.scene.robots[0].step_count

        if self.use_wandb:
            reset_wandb_env()
            now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            job_name = "robot_log_" + now
            self.run = wandb.init(
                project="sf",
                group="test_log",
                name=job_name,
                settings=wandb.Settings(_disable_stats=True),
            )

    def _log(self):

        log_properties = self.log_dict

        for k, v in log_properties.items():
            self.log_dict_full[k][self.logged_time_steps] = v

        self.last_log_time = log_properties["step"]

        self.logged_time = self.last_log_time - self.initial_time_stamp

        if self.use_wandb:
            flat_log_dict = {
                k + "_" + str(i): vv
                for k, v in log_properties.items()
                for i, vv in enumerate(np.atleast_1d(v))
            }
            self.run.log(flat_log_dict, step=self.logged_time_steps)

    def _stop(self):

        if self.logged_time_steps < self.max_time_steps:
            for v in self.log_dict_full.values():
                del v[self.logged_time_steps :]

        if self.use_wandb:
            if self.run is not None:
                self.run.finish()

        self.__dict__.update({k: np.array(v) for k, v in self.log_dict_full.items()})

    def _check_log_interval(self) -> bool:

        if (
            self.last_log_time is None
            or self.scene.robots[0].step_count >= self.last_log_time + self.log_interval
        ):
            return True
        return False

    def plot(self, plot_selection=CamPlotFlags.FULL, block=True):
        valid_plot_selections = plot_selection & self.plot_selection

        import matplotlib.pyplot as plt

        self.time_stamp = self.time_stamp - self.time_stamp[0]

        if CamPlotFlags.IMAGE in valid_plot_selections:
            pos_fig = plt.figure()
            for k in range(3):
                plt.figure(pos_fig.number)
                plt.subplot(3, 1, k + 1)
                plt.plot(self.time_stamp, self.pos[:, k])
            plt.subplot(3, 1, 1)
            plt.title("camera images ")

        plt.show(block=block)
