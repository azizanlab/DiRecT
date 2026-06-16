from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from d3il.environments.d3il.d3il_sim.core.sim_object.sim_object import (
    IntelligentSimObject,
)


class Camera(IntelligentSimObject, ABC):

    def __init__(
        self,
        name: str,
        width: int = 1000,
        height: int = 1000,
        init_pos=None,
        init_quat=None,
        near: float = 0.01,
        far: float = 10,
        fovy: int = 45,
        *args,
        **kwargs
    ):

        if init_pos is None:
            init_pos = [0, 0, 0]
        if init_quat is None:
            init_quat = [0, 1, 0, 0]

        super(Camera, self).__init__(name, init_pos, init_quat)
        self.width = width
        self.height = height

        self.near = near
        self.far = far
        self.fovy = fovy
        self.fovx = (
            2
            * np.arctan(
                self.width
                * 0.5
                / (self.height * 0.5 / np.tan(self.fovy * np.pi / 360 / 2))
            )
            / np.pi
            * 360
        )

        self.fx = (self.width / 2) / (np.tan(self.fovx * np.pi / 180 / 2))
        self.fy = (self.height / 2) / (np.tan(self.fovy * np.pi / 180 / 2))
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        )

    def set_cam_params(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        fovy: Optional[int] = None,
    ):

        self.width = width or self.width
        self.height = height or self.height

        self.near = near or self.near
        self.far = far or self.far
        self.fovy = fovy or self.fovy
        self.fovx = (
            2
            * np.arctan(
                self.width
                * 0.5
                / (self.height * 0.5 / np.tan(self.fovy * np.pi / 360 / 2))
            )
            / np.pi
            * 360
        )

        self.fx = (self.width / 2) / (np.tan(self.fovx * np.pi / 180 / 2))
        self.fy = (self.height / 2) / (np.tan(self.fovy * np.pi / 180 / 2))
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        )

    def get_segmentation(
        self, width: int = None, height: int = None, depth: bool = True
    ) -> np.ndarray:

        return self._get_img_data(
            width=width, height=height, depth=depth, segmentation=True
        )

    def get_image(
        self,
        width: int = None,
        height: int = None,
        depth: bool = True,
        denormalize_depth: bool = True,
    ) -> np.ndarray:

        return self._get_img_data(width, height, depth, denormalize_depth, False)

    def calc_point_cloud(
        self, width: int = None, height: int = None, denormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:

        rgb_img, depth_img = self.get_image(
            width, height, denormalize_depth=denormalize
        )

        return self.calc_point_cloud_from_images(rgb_img=rgb_img, depth_img=depth_img)

    def calc_point_cloud_from_images(
        self, rgb_img: np.ndarray, depth_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        true_width = rgb_img.shape[1]
        true_height = rgb_img.shape[0]
        if self.height == true_height and self.width == true_width:
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy
        else:
            fx = (true_width / 2) / (np.tan(self.fovx * np.pi / 180 / 2))
            fy = (true_height / 2) / (np.tan(self.fovy * np.pi / 180 / 2))
            cx = true_width / 2
            cy = true_height / 2

        z = depth_img
        u = np.arange(true_width) - cx
        v = np.arange(true_height) - cy

        x = (z * u) / fx
        y = (z.T * v).T / fy

        points = np.stack((x, y, z), axis=-1).reshape((true_width * true_height, 3))
        colors = rgb_img.reshape((true_width * true_height, 3)) / 255.0

        valid_points = ~np.isnan(points).any(axis=1)
        points = points[valid_points]
        colors = colors[valid_points]

        return points, colors

    def denormalize_depth(self, depth_img: np.ndarray) -> np.ndarray:

        z = self.near / (1 - depth_img * (1 - self.near / self.far))
        return z

    def apply_noise(self, depth_img: np.ndarray) -> np.ndarray:

        z = self.denormalize_depth(depth_img)

        depth_img = depth_img + (
            0.0001 * np.power(z - 0.5, 2) + 0.0004
        ) * np.random.rand(self.height, self.width)

        depth_img = ((40000 * depth_img).astype(int) / 40000.0).astype(np.float32)

        z = (
            2
            * self.far
            * self.near
            / (self.far + self.near - (self.far - self.near) * (2 * depth_img - 1))
        )

        z = cv2.bilateralFilter(z, 5, 0.1, 5)

        return z

    def get_poi(self) -> list:

        return [self.name]

    @abstractmethod
    def _get_img_data(
        self,
        width: int = None,
        height: int = None,
        depth: bool = True,
        denormalize_depth: bool = True,
        segmentation: bool = False,
    ) -> np.ndarray:

        pass

    @abstractmethod
    def get_cart_pos_quat(self) -> Tuple[np.ndarray, np.ndarray]:

        pass

    @property
    def fov(self):
        return self.fovy
