import enum
from abc import ABC, abstractmethod
from typing import List

import numpy as np

import d3il.environments.d3il.d3il_sim.core.sim_object as sim_object
import d3il.environments.d3il.d3il_sim.core.time_keeper as time_keeper
import d3il.environments.d3il.d3il_sim.utils.geometric_transformation as geom_trans
from d3il.environments.d3il.d3il_sim.core import Camera, Robots


class Scene(ABC):

    class RenderMode(enum.Enum):
        BLIND = "blind"
        OFFSCREEN = "offscreen"
        HUMAN = "human"

    def __init__(
        self,
        object_list=None,
        dt=0.001,
        render: RenderMode = RenderMode.HUMAN,
        *args,
        **kwargs
    ):
        if object_list is None:
            object_list = []

        self.obj_repo = sim_object.SimObjectRepository(object_list)

        self.dt = dt
        self.render_mode = render

        self.setup_done = False

        self._robots: List[Robots.RobotBase] = []
        self.inhand_cam = None
        self.cage_cam = None

        self.step_callbacks = []
        self.additional_loggers = []
        self.time_keeper = time_keeper.TimeKeeper(self.dt)

    def add_robot(self, robot):
        self._robots.append(robot)

    @property
    def robots(self):
        return self._robots

    def register_callback(self, fn, **kwargs):

        self.step_callbacks.append((fn, kwargs))

    @property
    @abstractmethod
    def sim_name(self) -> str:

        return "NONE"

    @abstractmethod
    def _setup_scene(self):

        self._setup_objects(self.object_list)

    @abstractmethod
    def reset(self, obj_pos=None):

        raise NotImplementedError

    def start(self, robot_init_qpos: np.ndarray = None):

        self._setup_scene()
        self.load_robot_to_scene(robot_init_qpos=robot_init_qpos)
        self.setup_done = True
        self.reset()

    @abstractmethod
    def render(self):

        raise NotImplementedError

    def next_step(self, log=True):
        for rb in self.robots:
            rb.prepare_step()

        self._sim_step()

        self.time_keeper.tick()

        for rb in self.robots:
            rb.tick()
            rb.receiveState()

        for call_back, kwargs in self.step_callbacks:
            call_back(**kwargs)
        if log:
            self.log_data()

        self.render()

    @abstractmethod
    def _sim_step(self):
        raise NotImplementedError

    @abstractmethod
    def load_robot_to_scene(self, robot_init_qpos: np.ndarray = None):

        raise NotImplementedError

    @abstractmethod
    def _setup_objects(self, sim_objs: List[sim_object.SimObject]):

        raise NotImplementedError

    @abstractmethod
    def _rt_add_object(self, sim_obj: sim_object.SimObject):

        raise NotImplementedError

    def add_object(self, sim_obj: sim_object.SimObject):

        if not self.setup_done:
            self.obj_repo.add_object(sim_obj)
        else:
            self._rt_add_object(sim_obj)

    def list_objects(self):
        return list(self._objects.keys())

    def get_object(self, name: str = None, obj_id: int = None) -> sim_object.SimObject:

        return self.obj_repo.get_object(name, obj_id)

    def _query_pois(self, sim_obj: sim_object.SimObject, fn) -> np.ndarray:

        if len(sim_obj.get_poi()) > 1:
            return np.array([fn(poi, sim_obj) for poi in sim_obj.get_poi()])
        else:
            return fn(sim_obj.get_poi()[0], sim_obj)

    @abstractmethod
    def _get_obj_seg_id(self, obj_name: str) -> int:

        raise NotImplementedError

    def get_obj_seg_id(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> int:

        if obj_name is None:
            if sim_obj is None:
                sim_obj = self.get_object(obj_id=obj_id)
            obj_name = sim_obj.name

        return self._get_obj_seg_id(obj_name=obj_name)

    @abstractmethod
    def _get_obj_pos(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:

        raise NotImplementedError

    def get_obj_pos(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> np.ndarray:

        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)
        return self._query_pois(sim_obj, self._get_obj_pos)

    @abstractmethod
    def _get_obj_quat(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:

        raise NotImplementedError

    def get_obj_quat(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> np.ndarray:

        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)
        return self._query_pois(sim_obj, self._get_obj_quat)

    def _get_obj_rot_mat(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:

        quat = self._get_obj_quat(poi, sim_obj)
        return geom_trans.mat2quat(quat)

    def get_obj_rot_mat(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> np.ndarray:

        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)
        return self._query_pois(sim_obj, self._get_obj_rot_mat)

    @abstractmethod
    def _set_obj_pos(self, new_pos, sim_obj: sim_object.SimObject):

        raise NotImplementedError

    def set_obj_pos(
        self,
        new_pos,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):

        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._set_obj_pos(new_pos, sim_obj)

    @abstractmethod
    def _set_obj_quat(self, new_quat, sim_obj: sim_object.SimObject) -> np.ndarray:

        raise NotImplementedError

    def set_obj_quat(
        self,
        new_quat,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):

        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._set_obj_quat(new_quat, sim_obj)

    @abstractmethod
    def _set_obj_pos_and_quat(
        self, new_pos, new_quat, sim_obj: sim_object.SimObject
    ) -> np.ndarray:

        raise NotImplementedError

    def set_obj_pos_and_quat(
        self,
        new_pos,
        new_quat,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):

        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._set_obj_pos_and_quat(new_pos, new_quat, sim_obj)

    @abstractmethod
    def _remove_object(self, sim_obj: sim_object.SimObject):
        raise NotImplementedError

    def remove_object(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._remove_object(sim_obj)
        self.obj_repo.remove_object(sim_obj)

    def get_cage_cam(self) -> Camera.Camera:
        return self.cage_cam

    def start_logging(self, duration: float = 300.0, **kwargs):

        for rb in self.robots:
            rb.start_logging(duration, **kwargs)

        for logger in self.additional_loggers:
            logger.start_logging(duration, **kwargs)

    def stop_logging(self):

        for rb in self.robots:
            rb.stop_logging()

        for logger in self.additional_loggers:
            logger.stop_logging()

    def add_logger(self, logger):

        self.additional_loggers.append(logger)

    def log_data(self):
        for rb in self.robots:
            rb.log_data()
        for logger in self.additional_loggers:
            logger.log_data()

    @property
    def step_count(self):
        return self.time_keeper.step_count

    @property
    def time_stamp(self):
        return self.time_keeper.time_stamp
