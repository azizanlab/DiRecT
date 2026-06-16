import xml.etree.ElementTree as Et
from itertools import chain
from xml.etree.ElementTree import Element

import mujoco

from d3il.environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType
from d3il.environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import (
    MjLoadable,
    MjIncludeTemplate,
)
from d3il.environments.d3il.d3il_sim.utils.sim_path import d3il_path


class MjSceneParser:

    def __init__(self, main_xml_path=None):
        if main_xml_path is None:
            main_xml_path = "./models/mj/surroundings/base.xml"

        with open(d3il_path(main_xml_path), "r") as file:
            self.scene_xml = file.read()

        self.assets = {}
        self.mj_xml_string = ""

        self._tree = Et.ElementTree(Et.fromstring(self.scene_xml))
        self._root = self._tree.getroot()
        self._worldbody = self._root.find("worldbody")
        assert self._worldbody, "Error, xml file does contain a world body."

    def create_scene(self, mj_robots, mj_surrounding, object_list: list, dt):
        if object_list is None:
            object_list = []
        if not isinstance(object_list, list):
            object_list = [object_list]

        all_objects = chain([mj_surrounding], object_list, mj_robots)

        for obj in all_objects:
            self.load_mj_loadable(obj)

        self.set_dt(dt)

        model = mujoco.MjModel.from_xml_string(self.mj_xml_string, self.assets)
        data = mujoco.MjData(model)

        self.cleanup(all_objects)
        return model, data

    def load_mj_loadable(self, mj_loadable: MjLoadable):
        xml_element, assets, include_type = mj_loadable.mj_load()
        if include_type == IncludeType.FILE_INCLUDE:
            self._root.append(xml_element)
        elif include_type == IncludeType.WORLD_BODY:
            self._worldbody.append(xml_element)
        elif include_type == IncludeType.MJ_INCLUDE:
            for child in xml_element:
                self._root.append(child)
        elif include_type == IncludeType.VIRTUAL_INCLUDE:
            return
        else:
            raise ValueError("Unknown IncludeType")

        self.indent(self._root)

        self.mj_xml_string = Et.tostring(self._root, encoding="unicode", method="xml")
        self.assets.update(assets)

    def set_dt(self, dt=0.001):
        options = self._root.find("option")
        options.set("timestep", str(dt))
        self.indent(self._root)
        self.mj_xml_string = Et.tostring(self._root, encoding="unicode", method="xml")

    def indent(self, elem: Element, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def cleanup(self, objects: list):
        for obj in objects:
            if isinstance(obj, MjIncludeTemplate):
                obj.cleanup()
