import numpy as np
import torch
# import scipy.interpolate as interpolate

from typing import Any, Sequence, Callable, Optional, Dict, List
from numpy.typing import NDArray

POINTMASS_KEYS = [
        "observations",
        "actions",
        "next_observations",
        "deltas"
]


class DatasetNormalizer:
    def __init__(
            self,
            dataset: Any,
            normalizer: Any,
            path_lengths: Optional[Sequence[int]] = None
        ):
        dataset = flatten(dataset, path_lengths)

        self.observation_dim = dataset["observations"].shape[1]
        self.action_dim = dataset["actions"].shape[1]
        self.normalizer = normalizer

        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except Exception as e:
                print(f"[ utils/normalization ] Skipping {key} | {normalizer} | {e}")

    def __repr__(self):
        string = ""
        for key, normalizer in self.normalizers.items():
            string += f"{key}: {normalizer}]\n"
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x: Any, key: str) -> Any:
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x: Any, key: str) -> Any:
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self) -> Dict[str, Any]:
        return self.normalizers


def flatten(dataset: Any, path_lengths: Optional[Sequence[int]]) -> Dict[str, NDArray]:
    assert path_lengths is not None
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate(
            [x[:length] for x, length in zip(xs, path_lengths)], axis=0
        )
    return flattened


class PointMassDatasetNormalizer(DatasetNormalizer):
    def __init__(
            self,
            preprocess_fns: Callable,
            dataset: Any,
            normalizer: Any,
            keys: List[str]=POINTMASS_KEYS
        ):
        del preprocess_fns
        reshaped = {}
        for key, val in dataset.items():
            dim = val.shape[-1]
            reshaped[key] = val.reshape(-1, dim)

        self.observation_dim = reshaped["observations"].shape[1]
        self.action_dim = reshaped["actions"].shape[1]

        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        self.normalizers = {key: normalizer(reshaped[key]) for key in keys}


class Normalizer:
    def __init__(self, X: NDArray):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self) -> str:
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def __call__(self, x: NDArray):
        return self.normalize(x)

    def normalize(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs) -> Any:
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    def normalize(self, x, *args, **kwargs) -> Any:
        return x

    def unnormalize(self, x, *args, **kwargs) -> Any:
        return x


class GaussianNormalizer(Normalizer):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device
        if self.device is None:
            self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.means = torch.tensor(self.X, device=self.device).mean(dim=0)
        self.stds = torch.tensor(self.X, device=self.device).std(dim=0)
        self.z = 1


    def __repr__(self) -> str:
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    """
            f"""means: {np.round(self.means, 2)}\n    """
            f"""stds: {np.round(self.z * self.stds, 2)}\n"""
        )

    # Refactored the method to be compatible with cpu
    def normalize(self, x: Any) -> NDArray:
        x = torch.tensor(x, device=self.device)
        result = (x - self.means.cpu()) / self.stds.cpu()
        result = result.cpu().numpy()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    # Refactored the method to be compatible with cpu
    def unnormalize(self, x: Any) -> NDArray:
        x = torch.tensor(x, device="cpu")
        result = (x * self.stds.cpu() + self.means.cpu()).cpu().numpy()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result


class LimitsNormalizer(Normalizer):
    def normalize(self, x: Any) -> Any:
        x = (x - self.mins) / (self.maxs - self.mins)
        x = 2 * x - 1
        return x

    def unnormalize(self, x: NDArray, eps: float=1e-4) -> Any:
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)
        x = (x + 1) / 2.0
        return x * (self.maxs - self.mins) + self.mins


