import os
import collections
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
from typing import Any, Callable, Dict, Generator, Union
from numpy.typing import NDArray

import numpy as np
import gymnasium as gym

@contextmanager
def suppress_output(enabled: bool = False):
    if not enabled:
        yield None
        return
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def load_environment(name: Union[str,gym.Env]) -> gym.Env:
    if not isinstance(name, str):
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def sequence_dataset(env: gym.Env, preprocess_fn: Callable) -> Generator[Dict[str, NDArray], None, None]:
    dataset = env.get_dataset()
    dataset = preprocess_fn(dataset)

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        for k in dataset:
            if "metadata" in k:
                continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if "maze2d" in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1

def process_maze2d_episode(episode: Dict[str, Any]) -> Any:
    assert "next_observations" not in episode
    length = len(episode["observations"])
    next_observations = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode


