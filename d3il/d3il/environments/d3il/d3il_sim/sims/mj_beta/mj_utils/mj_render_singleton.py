from mujoco import MjData, MjModel, mj_name2id, mjtObj

from d3il.environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_renderer import (
    RenderContext,
    RenderContextOffscreen,
)

__RENDER_CTX_MAP = {}


def reset_singleton():
    global __RENDER_CTX_MAP
    keys = list(__RENDER_CTX_MAP.keys())
    for key in keys:
        del __RENDER_CTX_MAP[key]
    __RENDER_CTX_MAP = {}


def get_renderer(
    name: str, width: int, height: int, model: MjModel, data: MjData
) -> RenderContextOffscreen:
    global __RENDER_CTX_MAP
    if name not in __RENDER_CTX_MAP:
        ctx = RenderContextOffscreen(width, height, model, data)
        __RENDER_CTX_MAP[name] = ctx
    ctx = __RENDER_CTX_MAP[name]
    return ctx


def render(
    cam_name: str,
    width: int,
    height: int,
    model: MjModel,
    data: MjData,
    depth: bool,
    segmentation: bool,
):

    ctx = get_renderer(cam_name, width, height, model, data)
    ctx.opengl_context.make_current()
    cam_id = mj_name2id(m=model, name=cam_name, type=mjtObj.mjOBJ_CAMERA)
    ctx.render(width, height, cam_id, segmentation=segmentation)
    return ctx.read_pixels(width, height, depth=depth, segmentation=segmentation)
