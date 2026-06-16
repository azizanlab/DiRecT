import abc
import os
import xml.etree.ElementTree as Et
from typing import Tuple

from d3il.environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType


class MjLoadable(abc.ABC):

    @abc.abstractmethod
    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        raise NotImplementedError


class MjXmlLoadable(MjLoadable):

    def __init__(self, full_xml_path, asset_path=None) -> None:
        super().__init__()
        self._full_xml_path = full_xml_path
        self._asset_path = asset_path

    @property
    def loadable_dir(self):
        return os.path.split(self._full_xml_path)[0]

    @property
    def asset_path(self):
        if self._asset_path is not None:
            return self._asset_path
        return os.path.join(self.loadable_dir, "assets")

    @property
    def file_name(self):
        return os.path.split(self._full_xml_path)[1]

    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        et_include = Et.Element("include")
        et_include.set("file", self.file_name)
        assets = {}

        if os.path.isdir(self.asset_path):
            for dirpath, dirnames, files in os.walk(self.asset_path):
                for f in files:
                    with open(os.path.join(dirpath, f), "rb") as file:
                        assets[f] = file.read()

        with open(os.path.join(self._full_xml_path), "rb") as file:
            assets[self.file_name] = file.read()
        return et_include, assets, IncludeType.FILE_INCLUDE


class MjIncludeTemplate(MjXmlLoadable):
    def __init__(self, full_xml_path, asset_path=None) -> None:
        super().__init__(full_xml_path, asset_path)
        self._tmp_filled_xml = None

    @abc.abstractmethod
    def modify_template(self, et: Et.ElementTree) -> Et.ElementTree:
        pass

    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        inc, assets, include_type = super().mj_load()

        obj = Et.parse(self._full_xml_path)
        new_xml = self.modify_template(obj)

        self._tmp_filled_xml = new_xml

        et_include = Et.Element("include")
        et_include.set("file", new_xml)
        return et_include, assets, include_type

    def cleanup(self):
        if self._tmp_filled_xml is not None:
            os.remove(self._tmp_filled_xml)


class MjFreezable:
    @abc.abstractmethod
    def freeze(self, data, model):
        pass

    @abc.abstractmethod
    def unfreeze(self, data, model):
        pass
