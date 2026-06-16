import numpy as np

from .d4rl import load_environment


def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn


def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)


def get_policy_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)


def arctanh_actions(*args, **kwargs):
    epsilon = 1e-4

    def _fn(dataset):
        actions = dataset["actions"]
        assert (
            actions.min() >= -1 and actions.max() <= 1
        ), f"applying arctanh to actions in range [{actions.min()}, {actions.max()}]"
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset["actions"] = np.arctanh(actions)
        return dataset

    return _fn


def add_deltas(env):

    def _fn(dataset):
        deltas = dataset["next_observations"] - dataset["observations"]
        dataset["deltas"] = deltas
        return dataset

    return _fn


def maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset["observations"][:, :2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset["timeouts"])

        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f"[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | "
            f"min length: {path_lengths.min()} | max length: {path_lengths.max()}"
        )

        dataset["timeouts"] = timeouts
        return dataset

    return _fn
