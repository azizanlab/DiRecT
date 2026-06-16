import os
from collections import namedtuple
import numpy as np
import torch
import tqdm

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

from typing import Any, Sequence, Dict, Tuple, Optional
from numpy.typing import NDArray

Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env: str="maze2d-large-v1",
        horizon: int=64,
        normalizer: str="LimitsNormalizer",
        preprocess_fns: Sequence[str]=[],
        max_path_length: int=1000,
        max_n_episodes: int=10000,
        termination_penalty: float=0.0,
        seed: Optional[int]=None,
        cheap_load: bool = False,
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.env = env = load_environment(env)
        # self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length

        normalizer_cache_path = os.path.join(
            "logs", "normalizer", f"normalizer_H{horizon}.pth"
        )

        # Normalizer
        build_from_scratch = True
        if os.path.exists(normalizer_cache_path):
            normalizer_obj = torch.load(normalizer_cache_path, weights_only=False)
            if normalizer_obj.normalizer == normalizer:
                self.normalizer=normalizer_obj
                build_from_scratch = False

        if build_from_scratch or not cheap_load:

            itr = sequence_dataset(env, self.preprocess_fn)

            fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
            for _, episode in enumerate(itr):
                fields.add_path(episode)
            fields.finalize()

            print(f"[ datasets/sequence ] Calculating normalizer...")
            self.normalizer = DatasetNormalizer(
                fields, normalizer, path_lengths=fields["path_lengths"]
            )
            os.makedirs(os.path.dirname(normalizer_cache_path), exist_ok=True)
            torch.save(self.normalizer, normalizer_cache_path)
            print(f"[ datasets/sequence ] Normalizer saved at {normalizer_cache_path}")

            print("[ datasets/sequence ] Normalizer acquired")
            self.indices = self.make_indices(fields.path_lengths, horizon)      #type: ignore
            print("[ datasets/sequence ] Indices made")

            self.observation_dim = fields.observations.shape[-1] #type: ignore
            self.action_dim = fields.actions.shape[-1]           #type: ignore
            self.fields = fields
            self.n_episodes = fields.n_episodes
            self.path_lengths = fields.path_lengths             #type: ignore
            self.normalize()
            print("[ datasets/sequence ] Normalized")
            print(fields)

    def normalize(self, keys: Sequence[str]=["observations", "actions"]):
        print("Normalizing...")
        for key in tqdm.tqdm(keys):
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )

    def make_indices(self, path_lengths: Sequence[int], horizon: int) -> NDArray:
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations: Any) -> Dict[int, Any]:
        return {0: observations[0]}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int, eps: float=1e-4):
        del eps

        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end] #type: ignore
        actions = self.fields.normed_actions[path_ind, start:end]           #type: ignore

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):
    def get_conditions(self, observations: Any) -> Dict[int, Any]:
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    def __init__(self,
                 *args,
                 discount: float=0.99,
                 normed: bool=False,
                 inf_horizon: bool=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.inf_horizon = inf_horizon
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self) -> Tuple[float,float]:
        vmin = np.inf
        vmax = -np.inf
        for i in tqdm.tqdm(range(len(self.indices))):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        return vmin, vmax

    def normalize_value(self, value: float) -> float:
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx: int) -> ValueBatch:          #type: ignore
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        if self.inf_horizon:
            rewards = self.fields["rewards"][path_ind, start:]
        else:
            rewards = self.fields["rewards"][
                path_ind, start:end
            ]  # only rewards in the planning horizon is predicted
        discounts = self.discounts[: len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch


