from typing import List

from d3il.environments.d3il.d3il_sim.utils.unique_dict import UniqueDict

from .sim_object import SimObject


class SimObjectRepository:

    def __init__(self, obj_list=None) -> None:
        self._name2id_map = UniqueDict(err_msg="Duplicate object name:")
        self._id2name_map = UniqueDict(err_msg="Duplicate object id:")
        self._objects = UniqueDict(err_msg="Duplicate object name:")

        if obj_list is not None:
            for obj in obj_list:
                self.add_object(obj)

    def add_object(self, sim_obj: SimObject) -> None:
        self._objects[sim_obj.name] = sim_obj

    def remove_object(self, sim_obj: SimObject) -> None:
        del self._name2id_map[sim_obj.name]

        del self._objects[sim_obj.name]

    def register_obj_id(self, sim_obj: SimObject, obj_id: int):
        sim_obj.obj_id = obj_id
        self._name2id_map[sim_obj.name] = obj_id

    def get_obj_list(self) -> List[SimObject]:
        return list(self._objects.values())

    def get_object(self, name: str = None, obj_id: int = None) -> SimObject:

        if obj_id is not None:
            name = self.get_name_from_id(obj_id)
        return self._objects[name]

    def get_id_from_name(self, name: str) -> int:

        return self._name2id_map[name]

    def get_name_from_id(self, obj_id: int) -> str:

        return self._id2name_map[obj_id]
