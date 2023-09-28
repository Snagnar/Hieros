import collections
from tqdm import tqdm
import numpy as np

from .basics import convert


class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._acts = {
            k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
            for k, v in self._env.act_space.items()
        }
        self._acts["reset"] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        iter_val = steps or episodes
        pBar = tqdm(total=iter_val, miniters=20)
        last_val = 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)
            if steps:
                pBar.update(step - last_val)
                last_val = step
            else:
                pBar.update(episode - last_val)
                last_val = episode

    def _step(self, policy, step, episode):
        # print_current_usage("1")
        assert all(len(x) == len(self._env) for x in self._acts.values())
        acts = {k: v for k, v in self._acts.items() if not k.startswith("log_")}
        # print("input actions min:", acts["action"].min(), "max:", acts["action"].max())
        obs = self._env.step(acts)
        # print_current_usage("2")
        obs = {k: convert(v) for k, v in obs.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs
        acts, self._state = policy(obs, self._state, **self._kwargs)
        # print_current_usage("3")
        acts = {k: convert(v) for k, v in acts.items()}
        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()
        self._acts = acts
        trns = {**obs, **acts}
        # print_current_usage("4")
        if obs["is_first"].any():
            for i, first in enumerate(obs["is_first"]):
                if first:
                    self._eps[i].clear()
        # print_current_usage("5")
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1

        # print_current_usage("6")
        if obs["is_last"].any():
            for i, done in enumerate(obs["is_last"]):
                if done:
                    # print("ep done!")
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        # print_current_usage("7")
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value


# import torch
# def print_current_usage(prefix="here"):
#     usage = torch.cuda.memory_reserved(0)
#     print(f"{prefix} memory reserved: {usage/1024**3:.5f} GB")
