from enum import Enum

import mujoco
import numpy as np


class IncludeType(Enum):
    FILE_INCLUDE = 0
    MJ_INCLUDE = 1
    WORLD_BODY = 2
    VIRTUAL_INCLUDE = 3


def change_domain(value, in_low, in_high, out_low, out_high):
    return (out_high - out_low) * ((value - in_low) / (in_high - in_low)) + out_low


def obj_position(data, obj_id: str):
    return data.get_body_xpos(obj_id)


def obj_distance(data, obj1: str, obj2: str):
    obj1_pos = obj_position(data, obj1)
    obj2_pos = obj_position(data, obj2)
    dist = np.linalg.norm(obj1_pos - obj2_pos)
    rel_dist = (obj1_pos - obj2_pos) / dist + 1e-8
    return dist, rel_dist


def reset_mocap2body_xpos(model, data):

    if model.eq_type is None or model.eq_obj1id is None or model.eq_obj2id is None:
        return
    for eq_type, obj1_id, obj2_id in zip(
        model.eq_type, model.eq_obj1id, model.eq_obj2id
    ):
        if eq_type != mujoco.const.EQ_WELD:
            continue

        mocap_id = model.body_mocapid[obj1_id]
        if mocap_id != -1:
            body_idx = obj2_id
        else:
            mocap_id = model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        data.mocap_pos[mocap_id][:] = data.body_xpos[body_idx]
        data.mocap_quat[mocap_id][:] = data.body_xquat[body_idx]


def get_body_jacr(model, data, name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, None, jacr, body=id)
    return jacr


def get_body_jacp(model, data, name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, body=id)
    return jacp


def get_body_xvelp(model, data, name):
    jacp = get_body_jacp(model, data, name).reshape((3, model.nv))
    xvelp = np.dot(jacp, data.qvel.copy())
    return xvelp


def get_body_xvelr(model, data, name):
    jacr = get_body_jacr(model, data, name).reshape((3, model.nv))
    xvelr = np.dot(jacr, data.qvel.copy())
    return xvelr
