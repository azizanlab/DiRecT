import abc
from typing import List


class SimObject(abc.ABC):

    GLOBAL_NAME_COUNTER = 0

    def __init__(self, name: str = None, init_pos=None, init_quat=None):

        if name is None:
            name = "SIM_OBJ_{}".format(SimObject.GLOBAL_NAME_COUNTER)

        self.name = name

        self.init_pos = init_pos
        self.init_quat = init_quat

        self.obj_id = None

        SimObject.GLOBAL_NAME_COUNTER += 1

    @abc.abstractmethod
    def get_poi(self) -> list:

        return [self.name]


class IntelligentSimObject(SimObject, abc.ABC):

    def __init__(self, name: str = None, init_pos=None, init_quat=None):

        super(IntelligentSimObject, self).__init__(name, init_pos, init_quat)

        self.sim = None
        self.sim_name = None

    def register_sim(self, sim, sim_name):

        self.sim = sim
        self.sim_name = sim_name


class DummyObject(SimObject):
    def __init__(self, pois: List[str]):
        self.pois = pois

    def get_poi(self) -> list:
        return self.pois
