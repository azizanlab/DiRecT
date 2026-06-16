import abc
from functools import wraps

from d3il.environments.d3il.d3il_sim.core import Camera, RobotBase, Scene
from d3il.environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import (
    PrimitiveObject,
)


def _register_function_on_class(cls, fn):

    @wraps(fn)
    def self_wrapper(self, *args, **kwargs):
        return fn(self, *args, **kwargs)

    setattr(cls, fn.__name__, self_wrapper)


class SimFactory(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        fn = self.prim_loading()
        if fn is not None:
            _register_function_on_class(PrimitiveObject, fn)

    @abc.abstractmethod
    def create_scene(
        self,
        gin_config=None,
        object_list: list = None,
        dt: float = 0.001,
        render: Scene.RenderMode = Scene.RenderMode.HUMAN,
        *args,
        **kwargs
    ) -> Scene:
        pass

    @abc.abstractmethod
    def create_robot(self, scene, *args, **kwargs) -> RobotBase:
        pass

    @abc.abstractmethod
    def create_camera(
        self,
        name: str,
        width: int = 1000,
        height: int = 1000,
        init_pos=None,
        init_quat=None,
        *args,
        **kwargs
    ) -> Camera.Camera:
        pass

    @abc.abstractmethod
    def prim_loading(self):
        return None

    @property
    def RenderMode(self):
        return Scene.RenderMode


class SimRepository:

    _repository = {}

    @classmethod
    def register(cls, factory: SimFactory, sim_name: str):
        if sim_name in cls._repository:
            return
        cls._repository[sim_name] = factory

    @classmethod
    def get_factory(cls, sim_name: str) -> SimFactory:
        return cls._repository[sim_name]

    @classmethod
    def list_all_sims(cls):
        return cls._repository.keys()
