from collections import namedtuple
import os
import numpy as np
import torch
import tqdm

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple("Batch", "trajectories conditions")


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="avoiding-v0",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        seed=None,
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        normalizer_cache_path = os.path.join(
            "logs", self.env_name, "normalizer", f"normalizer_H{horizon}.pth"
        )
        print(f"[ datasets/sequence ] Calculating normalizer...")
        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        os.makedirs(os.path.dirname(normalizer_cache_path), exist_ok=True)
        torch.save(self.normalizer, normalizer_cache_path)
        print(f"[ datasets/sequence ] Normalizer saved at {normalizer_cache_path}")

        print("[ datasets/sequence ] Normalizer acquired")
        self.indices = self.make_indices(fields.path_lengths, horizon)
        print("[ datasets/sequence ] Indices made")

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        print("[ datasets/sequence ] Normalized")

        print(fields)

    def normalize(self, keys=["observations", "actions"]):
        print("Normalizing...")
        for key in tqdm.tqdm(keys):
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )

    def make_indices(self, path_lengths, horizon):
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):

        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
