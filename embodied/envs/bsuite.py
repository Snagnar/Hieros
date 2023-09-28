import functools
import os

import embodied
import numpy as np


class BSuite(embodied.Env):
    # DEFAULT_CAMERAS = dict(
    #     locom_rodent=1,
    #     quadruped=2,
    # )

    def __init__(self, env, repeat=1, render=True, size=(64, 64), camera=-1):
        # TODO: This env variable is meant for headless GPU machines but may fail
        # on CPU-only machines.
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if isinstance(env, str):
            # domain, task = env.split("/", 1)
            import bsuite

            print("loading bsuite env:", env)
            env = bsuite.load_from_id(env)
            # if camera == -1:
            #     camera = self.DEFAULT_CAMERAS.get(domain, 0)
            # if domain == "cup":  # Only domain with multiple words.
            #     domain = "ball_in_cup"
            # if domain == "manip":
            #     from dm_control import manipulation

            #     env = manipulation.load(task + "_vision")
            # elif domain == "locom":
            #     from dm_control.locomotion.examples import basic_rodent_2020

            #     env = getattr(basic_rodent_2020, task)()
            # else:
            #     from dm_control import suite

            #     env = suite.load(domain, task)
        self._bsenv = env
        from . import from_dm

        self._env = from_dm.FromDM(self._bsenv)
        self._env = embodied.wrappers.ExpandScalars(self._env)
        self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
        self._env = embodied.wrappers.FlattenTwoDimObs(self._env)
        self._render = render
        self._size = tuple(
            size for size in self._bsenv.observation_spec().shape if size > 1
        )
        self._camera = camera

    @functools.cached_property
    def obs_space(self):
        spaces = self._env.obs_space.copy()
        # if self._render:
        #     spaces["image"] = embodied.Space(np.uint8, self._size + (3,))
        # if "observation" in spaces:
        #     spaces["observation"] = embodied.Space(
        #         spaces["observation"].dtype, (*self._size, 1)
        #     )
        # spaces[] = embodied.Space(spaces[field].dtype, int(np.prod(spaces[field].shape)))
        # for field in spaces:
        #     if field not in ["reward", "is_terminal"] and hasattr(spaces[field], "shape") and len(spaces[field].shape) < 3:
        #         spaces[field] = embodied.Space(spaces[field].dtype, int(np.prod(spaces[field].shape)))
        # print("spaces:", spaces)
        # spaces["observation"] = embodied.Space(spaces["observation"].dtype, int(np.prod(spaces["observation"].shape)))
        # print("spaces after:", spaces)
        return spaces

    @functools.cached_property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])
        obs = self._env.step(action)
        obs["observation"] = obs["observation"].squeeze()
        # obs["observation"] = obs["observation"].reshape(-1)
        # for field, value in obs.items():
        #     print("field", field, "value", value.shape if hasattr(value, "shape") else value)
        # for field in obs:
        #     if field not in ["reward", "is_terminal"] and hasattr(obs[field], "shape") and len(obs[field].shape) < 3:
        #         obs[field] = obs[field].reshape(-1)
        # if "observation" in obs:
        #     obs["observation"] = np.expand_dims(obs["observation"], 2)
        # if self._render:
        #     obs["image"] = np.repeat(np.expand_dims(obs["observation"], 2), 3, axis=2)
        return obs
