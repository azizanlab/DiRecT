from typing import Any, Dict, Optional
import numpy as np
from numpy.typing import NDArray

# Note: np.atleast_2d adds unitary leading dims
def atleast_2d(x: NDArray) -> NDArray:
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:
    def __init__(
        self, max_n_episodes: int, max_path_length: int, termination_penalty: Optional[float]
    ):
        self._dict = {
            "path_lengths": np.zeros(max_n_episodes, dtype=np.int32),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return "[ datasets/buffer ] Fields:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()  # type: ignore
        )

    def __getitem__(self, key: Any) -> Any:
        return self._dict[key]

    def __setitem__(self, key: Any, val: Any):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self) -> int:
        return self._count

    @property
    def n_steps(self) -> Any:
        return sum(self["path_lengths"])

    def _add_keys(self, path: Any):
        if hasattr(self, "keys"):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self) -> Dict[str, Any]:
        return {k: v for k, v in self._dict.items() if k != "path_lengths"}.items()  # type: ignore

    def _allocate(self, key: Any, array: NDArray):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32) #type: ignore

    def add_path(self, path: Any):
        path_length = len(path["observations"])
        assert path_length <= self.max_path_length

        self._add_keys(path)

        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict:
                self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        if path["terminals"].any() and self.termination_penalty is not None:
            assert not path["timeouts"].any(), "Penalized a timeout episode for early termination"
            self._dict["rewards"][self._count, path_length - 1] += self.termination_penalty

        self._dict["path_lengths"][self._count] = path_length

        self._count += 1

    def truncate_path(self, path_ind: int, step: int):
        old = self._dict["path_lengths"][path_ind]
        new = min(step, old)
        self._dict["path_lengths"][path_ind] = new

    def finalize(self):
        for key in self.keys + ["path_lengths"]:
            self._dict[key] = self._dict[key][: self._count]
        self._add_attributes()
        print(f"[ datasets/buffer ] Finalized replay buffer | {self._count} episodes")

