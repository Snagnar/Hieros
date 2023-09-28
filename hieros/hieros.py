import wandb
import datetime
import argparse
import functools
import os
import pathlib
import copy
import sys

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


import exploration as expl
import models
import tools

import importlib

import embodied
from embodied import wrappers
import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()
from tqdm import tqdm


class Hieros(nn.Module):
    def __init__(self, obs_space, act_space, config, prefilled_replay):
        super(Hieros, self).__init__()
        self._config = config
        self._should_log = tools.Every(config.log_every)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._should_add_hierarchy = tools.VariableEvery(
            config.add_hierarchy_every, lambda x: int(x * config.add_hierarchy_every)
        )
        self._metrics = {}
        self._step = count_steps(config.traindir)
        self._update_count = 0
        self._subgoal_shape = config.subgoal_shape
        # Schedules.
        if config.max_hierarchy == 1:
            config.use_subgoal = False
        new_config = copy.deepcopy(config)
        if config.only_subgoal_reward:
            new_config.reward_weight = 0
        if config.novelty_only_higher_level:
            new_config.novelty_reward_weight = 0
        new_config.action_shape = None
        self._subactors = nn.ModuleList(
            [
                SubActor(
                    "Subactor-0",
                    obs_space,
                    act_space,
                    self._subgoal_shape,
                    new_config,
                    # logger,
                    prefilled_replay,
                    compute_subgoal=config.max_hierarchy > 1,
                )
            ]
        )

        initial_state = self._subactors[-1]._wm.dynamics.initial(1)
        initial_feature = self._subactors[-1]._wm.dynamics.get_feat(initial_state)
        initial_feature_shape = initial_feature.shape
        self._should_update_subactor = [tools.Every(1)]
        self._should_train_subactor = [tools.Every(1)]
        self._subgoal_tensor_shape = self._initial_subgoal().shape
        self._environment_time_steps = 0
        self._training_steps = 0
        self.sync()
        self._image_in_obs = "image" in obs_space
        if "image" in obs_space:
            self._subgoal_cache = torch.zeros(
                (
                    config.subgoal_cache_size,
                    config.max_hierarchy - 1,
                    config.envs["amount"],
                    *config.subgoal_shape,
                ),
                device=config.device,
            )
            self._det_cache = [[] for _ in range(config.subgoal_cache_size)]
            self._stoch_cache = [[] for _ in range(config.subgoal_cache_size)]
            self._img_cache = np.zeros(
                (
                    config.subgoal_cache_size,
                    config.envs["amount"],
                    *obs_space["image"].shape,
                )
            )
        self._subgoal_cache_idx = 0
        self._policy_metrics = {}

    def dataset(self, generator):
        batcher = embodied.Batcher(
            sources=[generator] * self._config.batch_size,
            workers=self._config.data_loaders,
            prefetch_source=4,
            prefetch_batch=self._config.prefetch_batches,
        )
        return batcher()

    def set_training(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def set_eval(self):
        self.set_training(False)

    def train(self, data=None, state=None):
        metrics = {}
        time_record = tools.TimeRecording()
        for subactor, should_train in zip(self._subactors, self._should_train_subactor):
            if not should_train(self._training_steps):
                continue
            subactor.train()
            if len(subactor._replay) >= self._config.batch_size:
                if (
                    not self._config.hierarchical_world_models
                    and subactor._name != "Subactor-0"
                ):
                    data = self._subactors[0]._last_start
                else:
                    data = next(subactor._dataset)
                with time_record:
                    subactor_metrics = subactor._train(data)
                metrics[f"{subactor._name}/train_time"] = time_record.elapsed_time
                metrics.update(
                    {
                        f"{subactor._name}/{key}": value
                        for key, value in subactor_metrics.items()
                    }
                )
            else:
                print(
                    "not enough data in replay buffer for",
                    subactor._name,
                    ": ",
                    len(subactor._replay),
                    "/",
                    self._config.batch_size,
                )
        if not self._config.hierarchical_world_models and self._config.higher_level_wm:
            metrics.update(self._joint_behaviour_training())
        self.set_eval()
        for subactor in self._subactors:
            subactor.eval()
        self._training_steps += 1
        return {}, state, metrics

    def _joint_imagination(self):
        start = self._subactors[0]._last_start

        dynamics = self._subactors[0]._wm.dynamics
        wm = self._subactors[0]._wm
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        subgoal_shape = (
            self._subgoal_shape
            if not self._config.decompress_subgoal_for_input
            else self._subactors[0]._decoded_subgoal_shape[1:]
        )
        start["subgoals"] = torch.zeros(
            (len(self._subactors), start["stoch"].shape[0], *subgoal_shape),
            device=self._config.device,
        )
        updates = [
            self._config.subactor_update_every**i for i in range(len(self._subactors))
        ]

        def step(prev, step_idx):
            state, _, _ = prev
            feat = wm.get_feat(state)
            inp = (
                feat.detach()
                if self._subactors[0]._task_behavior._stop_grad_actor
                else feat
            )
            actions = []
            for i in range(len(self._subactors) - 1, -1, -1):
                if step_idx % updates[i] != 0:
                    continue
                action = (
                    self._subactors[i]
                    ._task_behavior.actor(inp, state["subgoals"][i])
                    .sample()
                )
                actions.append(action)
                if i > 0:
                    if self._config.decompress_subgoal_for_input:
                        state["subgoals"][i - 1] = self._subactors[
                            i - 1
                        ].decode_subgoal(
                            action.reshape((action.shape[0], *self._subgoal_shape)),
                            isfirst=False,
                        )
                    else:
                        state["subgoals"][i - 1] = action
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            succ["subgoals"] = state["subgoals"]
            if "state" in succ:
                succ["state"] = succ["state"].detach()
            all_actions = {
                i: single_action for i, single_action in enumerate(actions[::-1])
            }
            return succ, feat, all_actions

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(self._config.imag_horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _joint_behaviour_training(self):
        imag_features, imag_states, imag_actions = self._joint_imagination()
        updates = [
            self._config.subactor_update_every**i for i in range(len(self._subactors))
        ]
        metrics = {}
        for i, subactor in enumerate(self._subactors):
            own_imag_states = {k: v[:: updates[i]] for k, v in imag_states.items()}
            own_imag_features = imag_features[:: updates[i]]
            own_imag_actions = imag_actions[i]
            subgoal = imag_states["subgoals"][:: updates[i], i]
            reward_fn = lambda f, s, a: {
                "extrinsic": subactor._wm.heads["reward"](
                    subactor._wm.dynamics.get_feat(s)
                ).mode(),
                "subgoal": subactor._subgoal_reward(s, subgoal),
                "novelty": subactor._novelty_reward(s),
            }
            train_metrics = subactor._task_behavior._train_wo_imag(
                own_imag_features, own_imag_states, own_imag_actions, subgoal, reward_fn
            )[-1]
            metrics.update(
                {
                    f"{subactor._name}/{key}": value
                    for key, value in train_metrics.items()
                }
            )
        return metrics

    def policy(self, obs, state=None, mode="train"):
        if (
            self._should_add_hierarchy(self._environment_time_steps)
            and (
                self._environment_time_steps > 0
                or self._config.add_hierarchy_every == 0
            )
            and len(self._subactors) < self._config.max_hierarchy
        ):
            print("Adding a new hierarchy layer")
            self._create_subactor()
        subactor_updates = [
            should_update(self._environment_time_steps)
            for should_update in self._should_update_subactor
        ]
        step_reward = obs["reward"]
        time_record = tools.TimeRecording()
        with time_record:
            for subactor, update in zip(self._subactors, [1, *subactor_updates[:-1]]):
                if update:
                    subactor.set_obs(obs)
                    if self._config.hierarchical_world_models:
                        obs = subactor.encode_obs()
        self._metrics["encoding_time"] = time_record.elapsed_time
        self._metrics["step_reward"] = step_reward.max().item()
        subgoal = self._initial_subgoal()
        if state is None:
            state = [None for _ in range(len(self._subactors))]
        if len(state) < len(self._subactors):
            state.extend([None for _ in range(len(self._subactors) - len(state))])
        cached_subgoals = None
        try:
            for idx, (update, subactor) in enumerate(
                zip(reversed(subactor_updates), reversed(self._subactors))
            ):
                with time_record:
                    subgoal, subactor_state = subactor(
                        state[len(self._subactors) - 1 - idx],
                        subactor.decode_subgoal(subgoal, isfirst=idx == 0)
                        if update
                        else None,
                        obs["is_last"] | obs["is_terminal"],
                        step_reward
                        / self._should_update_subactor[
                            len(self._subactors) - 1 - idx
                        ]._every,
                        mode in ["train", "explore"],
                        update,
                        subgoal.reshape((*subgoal.shape[:-2], -1))
                        if not self._config.decompress_subgoal_for_input
                        else None,
                        outside_features=state[0][0] if state[0] is not None else None,
                    )
                state[len(self._subactors) - 1 - idx] = subactor_state
                self._metrics[f"{subactor._name}_time"] = time_record.elapsed_time
                if idx < len(self._subactors) - 1:
                    subgoal = subgoal["action"].reshape(self._subgoal_tensor_shape)
                    if cached_subgoals is None:
                        cached_subgoals = torch.zeros(
                            (len(self._subactors) - 1, *subgoal.shape),
                            device=self._config.device,
                        )
                    cached_subgoals[idx] = subgoal.detach()
            if (
                self._config.subgoal_visualization
                and mode in ["train", "explore"]
                and self._image_in_obs
                and cached_subgoals is not None
                and not any(subactor_state is None for subactor_state in state)
            ):
                cached_subgoals = torch.cat(
                    (
                        torch.zeros(
                            (
                                self._config.max_hierarchy - len(cached_subgoals) - 1,
                                self._config.envs["amount"],
                                *self._config.subgoal_shape,
                            ),
                            device=self._config.device,
                        ),
                        cached_subgoals.detach(),
                    )
                )
                self._subgoal_cache[self._subgoal_cache_idx] = cached_subgoals
                self._det_cache[self._subgoal_cache_idx] = [
                    s[0]["deter"].detach().unsqueeze(0) for s in reversed(state[:-1])
                ]
                self._stoch_cache[self._subgoal_cache_idx] = [
                    s[0]["stoch"].detach().unsqueeze(0) for s in reversed(state[:-1])
                ]
                self._img_cache[self._subgoal_cache_idx] = self._subactors[0]._obs[
                    "image"
                ]
                self._subgoal_cache_idx = (
                    self._subgoal_cache_idx + 1
                ) % self._config.subgoal_cache_size
        except Exception as e:
            print("Exception in hieros policy:", e)
            if "exception_count" not in self._metrics:
                self._metrics["exception_count"] = 1
            else:
                self._metrics["exception_count"] += 1
        policy_output = subgoal

        self._environment_time_steps += 1
        if mode in ["train", "explore"]:
            if self._environment_time_steps % 100 == 0:
                print("environment time steps: ", self._environment_time_steps)
        return policy_output, state

    def _create_subactor(self):
        new_config = copy.deepcopy(self._config)
        if self._config.subactor_encoding_architecture == "mlp":
            new_config.encoder["mlp_keys"] = ".*"
            new_config.decoder["mlp_keys"] = ".*"
            new_config.encoder["cnn_keys"] = "$^"
            new_config.decoder["cnn_keys"] = "$^"
        elif self._config.subactor_encoding_architecture == "cnn":
            new_config.encoder["mlp_keys"] = "$^"
            new_config.decoder["mlp_keys"] = "$^"
            new_config.encoder["cnn_keys"] = ".*"
            new_config.decoder["cnn_keys"] = ".*"
        new_config.pretrain = 0
        new_config.num_actions = np.prod(self._subgoal_shape)
        new_config.video_pred_log = False
        if len(self._subactors) == (self._config.max_hierarchy - 1):
            new_config.use_subgoal = False
        if new_config.only_subgoal_reward and len(self._subactors) < (
            self._config.max_hierarchy - 1
        ):
            new_config.reward_weight = 0
        new_config.actor_dist = "onehot_categorical"
        new_config.action_shape = self._subgoal_shape

        use_world_model = (
            new_config.higher_level_wm and new_config.hierarchical_world_models
        )
        if new_config.hierarchy_decrease_sizes["enabled"]:
            print("DECREASE SIZE ENABLED")
            for field_name in new_config.hierarchy_decrease_sizes["sizes"]:
                if hasattr(new_config, field_name):
                    setattr(
                        new_config,
                        field_name,
                        int(
                            max(
                                getattr(new_config, field_name)
                                // new_config.hierarchy_decrease_sizes["factor"],
                                new_config.hierarchy_decrease_sizes["min"],
                            )
                        ),
                    )
                    print(
                        "decreased", field_name, "to", getattr(new_config, field_name)
                    )
                else:
                    print("Warning: field", field_name, "not found in config")
            if self._config.dynamics_model == "s5":
                new_config.s5["model_dim"] = int(
                    max(
                        new_config.s5["model_dim"]
                        // new_config.hierarchy_decrease_sizes["factor"],
                        new_config.hierarchy_decrease_sizes["min"],
                    )
                )
                new_config.s5["state_dim"] = int(
                    max(
                        new_config.s5["state_dim"]
                        // new_config.hierarchy_decrease_sizes["factor"],
                        new_config.hierarchy_decrease_sizes["min"],
                    )
                )
        else:
            print("DECREASE SIZE DISABLED")
        self._subactors.append(
            SubActor(
                f"Subactor-{len(self._subactors)}",
                self._subactors[-1].encoded_obs_space(),
                {"action": embodied.Space(np.float32, self._subgoal_shape)},
                self._subgoal_shape,
                new_config,
                make_replay(
                    self._config,
                    self._config.traindir / f"replay-{len(self._subactors)}",
                ),
                buffer_obs=self._config.subactor_encode_intermediate,
                buffer_obs_keys=list(self._subactors[-1].encoded_obs_space().keys()),
                use_world_model=use_world_model,
                other_world_model=self._subactors[0]._wm
                if not use_world_model
                else None,
            )
        )
        self._should_update_subactor.append(
            tools.Every(
                self._should_update_subactor[-1]._every
                * self._config.subactor_update_every,
            )
        )
        self._should_train_subactor.append(
            tools.Every(
                self._should_train_subactor[-1]._every
                * self._config.subactor_train_every,
            )
        )
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"##### Number of trainable parameters: {num_parameters:_}")
        if self._config.wandb_logging:
            self._config.num_trainable_params = num_parameters
            wandb.config.update(
                {"num_trainable_params": num_parameters}, allow_val_change=True
            )
        self._subactors[-1].eval()

    def _initial_subgoal(self):
        if self._config.dyn_discrete:
            return torch.zeros(
                self._config.envs["amount"],
                *self._config.subgoal_shape,
                device=self._config.device,
            )
        return torch.zeros(
            self._config.envs["amount"],
            self._config.subgoal_shape[0],
            device=self._config.device,
        )

    def report(self, data):
        timer = tools.TimeRecording()
        report = {}
        if (
            self._config.video_pred_log
            and self._image_in_obs
            and len(self._subactors[0]._replay) > self._config.batch_size
        ):
            with timer:
                real_vid = (
                    self._subactors[0]
                    ._wm.video_pred(next(self._subactors[0]._dataset))
                    .detach()
                    .cpu()
                    .numpy()
                )
                report = {"video": self.format_video(real_vid)}
                # check if the buffer has been filled completely and if so, generate a video
                if (
                    self._config.subgoal_visualization
                    and len(self._det_cache[self._subgoal_cache_idx]) > 0
                ):
                    report["subgoal_visualization"] = self._visualize_subgoals()
            report["video_generation_time"] = timer.elapsed_time
        report.update(self._metrics)
        report["num_subactors"] = len(self._subactors)
        return report

    def format_video(self, value):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 2, 0, 3, 4).reshape((T, H, B * W, C))
        return value

    def _visualize_subgoals(self):
        print("creating video!")
        # create video from subgoal cache
        frame_num, num_subactors, num_envs, *subgoal_shape = self._subgoal_cache.shape
        lowest_subactor = self._subactors[0]
        frames = list(range(self._subgoal_cache_idx + 1, frame_num)) + list(
            range(self._subgoal_cache_idx + 1)
        )
        image_shape = list(lowest_subactor._obs["image"].shape)

        image_shape[1] *= len(self._subactors)
        frame_list = np.zeros((len(frames), *image_shape))
        for idx, frame in enumerate(frames):
            # todo: do with itertools
            subgoals = []
            for subgoal, subactor, deter, stoch in zip(
                self._subgoal_cache[frame],
                reversed(self._subactors[:-1]),
                self._det_cache[frame],
                self._stoch_cache[frame],
            ):
                subgoals.append(
                    subactor.decode_subgoal(subgoal, isfirst=False).unsqueeze(0)
                )
                subgoals = [
                    subactor._wm.decode_state(
                        {
                            "stoch": subgoal
                            if self._config.subgoal["use_stoch"]
                            else stoch,
                            "deter": subgoal
                            if self._config.subgoal["use_deter"]
                            else deter,
                        }
                    )
                    for subgoal in subgoals
                ]
                subgoals = [
                    subgoal[
                        "image"
                        if "image" in subgoal
                        else ("stoch" if self._config.subgoal["use_stoch"] else "deter")
                    ]
                    for subgoal in subgoals
                ]
            subgoals = [subgoal.detach().squeeze() for subgoal in subgoals]
            subgoals = [
                (subgoal - subgoal.min()) / (subgoal.max() - subgoal.min())
                for subgoal in subgoals
            ]
            subgoals = [subgoal.cpu().numpy() for subgoal in subgoals]
            if len(subgoals[0].shape) < 4:
                subgoals = [subgoal[None] for subgoal in subgoals]
            subgoals.append(self._img_cache[frame] / 255.0)
            full_frame = np.concatenate(subgoals, axis=1)
            frame_list[idx] = full_frame
        video = np.array(frame_list)
        video = np.swapaxes(video, 0, 1)
        return self.format_video(video)

    def save(self):
        data = {
            "subactor_states": [subactor.state_dict() for subactor in self._subactors]
        }
        return data

    def load(self, data):
        for subactor, subactor_data in zip(self._subactors, data["subactor_states"]):
            subactor.load_state_dict(subactor_data)

    def sync(self):
        self._subactors = nn.ModuleList(
            [subactor.to(self._config.device) for subactor in self._subactors]
        )


from collections import deque


class SubActor(nn.Module):
    def __init__(
        self,
        name,
        obs_space,
        act_space,
        subgoal_shape,
        config,
        replay,
        compute_subgoal=True,
        buffer_obs=False,
        buffer_obs_keys=None,
        use_world_model=True,
        other_world_model=None,
    ):
        super(SubActor, self).__init__()
        self._config = config
        self._act_space = act_space
        self._obs_space = obs_space
        self._name = name
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once() if config.pretrain else lambda: False
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = count_steps(config.traindir)
        self._update_count = 0
        self._buffer_obs = buffer_obs
        self._buffer_obs_keys = buffer_obs_keys
        self._subgoal_shape = subgoal_shape
        self._compute_subgoal = config.use_subgoal and compute_subgoal
        self._compute_novelty = config.novelty_reward_weight > 0

        if buffer_obs and not use_world_model:
            raise ValueError(
                "buffering observations is only supported when using a world model"
            )
        # Schedules
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
            x, self._step
        )
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        )
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
            x, self._step
        )
        self._replay = replay
        self._dataset = make_dataset(self._replay.dataset, config)
        self._last_action = {
            key: torch.zeros((config.batch_size, *act_space[key].shape))
            for key in act_space.keys()
        }
        if use_world_model:
            self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        else:
            self._wm = other_world_model

        self._decoded_subgoal_shape = self._initial_subgoal().shape
        self._task_behavior = models.ImagBehavior(
            config,
            self._wm,
            self._decoded_subgoal_shape[-1]
            if config.decompress_subgoal_for_input
            else np.prod(config.subgoal_shape),
            config.behavior_stop_grad,
            use_subgoal=config.use_subgoal,
            use_novelty=config.novelty_reward_weight > 0,
        )

        self.subgoal_autoencoder = models.SubgoalAutoencoder(
            input_shape=self._decoded_subgoal_shape,
            layers=config.subgoal_compression["layers"],
            bottleneck_shape=config.subgoal_shape,
            activation=config.subgoal_compression["act"],
            norm=config.subgoal_compression["norm"],
            encoding_symlog=config.subgoal_compression["encoding_symlog"],
            decoding_symlog=config.subgoal_compression["decoding_symlog"],
            kl_scale=config.subgoal_compression["kl_scale"],
            unimix_ratio=config.unimix_ratio,
            config=config,
        )
        self._state = None
        if config.compile and hasattr(torch, "compile"):
            print("compiling models....")
            try:
                self._wm = torch.compile(self._wm)
                self._task_behavior = torch.compile(self._task_behavior)
                self.subgoal_autoencoder = torch.compile(self.subgoal_autoencoder)
            except Exception as e:
                print("model compilation failed: ", e)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
        self._last_subgoal = None

        self._last_policy_output = {
            key: torch.zeros(
                (config.envs["amount"], *act_space[key].shape), device=config.device
            )
            for key in act_space.keys()
        }
        self._trains = 0
        self._obs = None
        self._summed_reward = np.zeros(self._config.envs["amount"])
        self._obs_buffer = {key: deque() for key in obs_space.keys()}
        self._last_start = {}

    def set_obs(self, obs):
        if obs is None:
            raise ValueError("obs is expected to be not None")
        if self._buffer_obs:
            for key in self._buffer_obs_keys:
                self._obs_buffer[key].append(obs[key])
                if len(self._obs_buffer[key]) > self._config.subactor_update_every:
                    self._obs_buffer[key].popleft()
        self._obs = obs

    def _initial_subgoal(self):
        return torch.zeros_like(
            self.get_subgoal(self._wm.dynamics.initial(self._config.envs["amount"])),
            device=self._config.device,
        )

    def get_subgoal(self, latent):
        if self._config.subgoal["use_stoch"]:
            stoch = latent["stoch"]
            if self._config.dyn_discrete:
                shape = list(stoch.shape[:-2]) + [
                    self._config.dyn_stoch * self._config.dyn_discrete
                ]
                stoch = stoch.reshape(shape)
            subgoal = stoch
        if self._config.subgoal["use_deter"]:
            if not self._config.subgoal["use_stoch"]:
                subgoal = latent["deter"]
            else:
                subgoal = torch.cat([subgoal, latent["deter"]], dim=-1)
        return subgoal

    def encode_obs(self):
        # encodes observation and adds latent information for input of next hierarchy layer (subactor)
        if self._state is None:
            latent = self._wm.dynamics.initial(self._config.envs["amount"])
        else:
            latent, _ = self._state
        latent = {k: v.detach() for k, v in latent.items()}

        stoch = latent["stoch"]
        if self._config.dyn_discrete:
            shape = list(stoch.shape[:-2]) + [
                self._config.dyn_stoch * self._config.dyn_discrete
            ]
            stoch = stoch.reshape(shape)

        carry_obs = {
            k: self._obs[k] for k in ("is_first", "is_last", "is_terminal", "reward")
        }
        if self._config.symlog_subactor_obs:
            encoded_obs = {"stoch": stoch, "deter": tools.symlog(latent["deter"])}
        else:
            encoded_obs = {"stoch": stoch, "deter": latent["deter"]}
        return {**carry_obs, **encoded_obs}

    def decode_subgoal(self, subgoal, isfirst=False):
        result = self.subgoal_autoencoder.decode(subgoal)
        if isfirst:
            return torch.zeros_like(result)
        return result

    def encoded_obs_space(self):
        initial_obs = self._wm.dynamics.initial(1)
        buffer_dim = (
            self._config.subactor_update_every
            if self._config.subactor_encode_intermediate
            else 1
        )
        if self._config.subactor_encoding_architecture == "mlp":
            return {
                "stoch": embodied.Space(
                    np.float32, (buffer_dim * np.prod(initial_obs["stoch"].shape),)
                ),
                "deter": embodied.Space(
                    np.float32, (buffer_dim * initial_obs["deter"].shape[1],)
                ),
            }
        elif self._config.subactor_encoding_architecture == "cnn":
            return {
                "stoch": embodied.Space(
                    np.float32, (buffer_dim, np.prod(initial_obs["stoch"].shape))
                ),
                "deter": embodied.Space(
                    np.float32, (buffer_dim, initial_obs["deter"].shape[1])
                ),
            }

    def __call__(
        self,
        state,
        subgoal,
        reset,
        reward,
        training=True,
        should_update=False,
        encoded_subgoal=None,
        outside_features=None,
    ):
        step = self._step
        obs = self._obs
        if self._should_reset(step):
            state = None
            self._last_policy_output = {
                key: torch.zeros(
                    (self._config.envs["amount"], *self._act_space[key].shape),
                    device=self._config.device,
                )
                for key in self._act_space.keys()
            }
            self._last_subgoal = self._initial_subgoal()
            self._summed_reward = np.zeros(self._config.envs["amount"])
        if training and obs is not None:
            self._summed_reward += (
                reward.cpu().numpy() if hasattr(reward, "cpu") else reward
            )

        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
            if self._last_policy_output is not None:
                for key in self._last_policy_output.keys():
                    for i in range(self._last_policy_output[key].shape[0]):
                        self._last_policy_output[key][i] *= mask[i]
            if self._last_subgoal is not None:
                for i in range(len(self._last_subgoal)):
                    self._last_subgoal[i] *= mask[i]
            self._summed_reward = np.zeros(self._config.envs["amount"])

        if not should_update:
            return self._last_policy_output, state

        if self._buffer_obs:
            if (
                len(self._obs_buffer[self._buffer_obs_keys[0]])
                < self._config.subactor_update_every
            ):
                return self._last_policy_output, state
            obs.update(
                {
                    k: torch.stack(list(self._obs_buffer[k]), dim=1)
                    for k in self._obs_buffer.keys()
                }
            )
            if self._config.subactor_encoding_architecture == "mlp":
                obs.update(
                    {
                        k: obs[k].reshape(obs[k].shape[0], -1)
                        for k in self._obs_buffer.keys()
                    }
                )
        if obs is None:
            return self._last_policy_output, state

        if not self._config.hierarchical_world_models and self._name != "Subactor-0":
            obs_to_use = {**obs, **outside_features}
        else:
            obs_to_use = obs
        policy_output, state = self._policy(
            obs_to_use,
            state,
            subgoal if self._config.decompress_subgoal_for_input else encoded_subgoal,
            training,
        )
        acts = {k: embodied.convert(v) for k, v in policy_output.items()}
        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {k: v * expand(mask, len(v.shape)) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()

        if (
            training
            and policy_output is not None
            and not self._config.fix_dataset
            and (self._config.hierarchical_world_models or self._name == "Subactor-0")
        ):
            obs_to_store = obs.copy()
            obs_to_store["reward"] = torch.tensor(
                self._summed_reward, dtype=torch.float32, device=self._config.device
            )
            tools.add_step_to_replay(
                self._replay,
                obs_to_store,
                acts,
                subgoal,
                self._config,
                True,
                encoded_subgoal,
            )
            self._summed_reward = np.zeros(self._config.envs["amount"])
        if training:
            self._step += len(reset)
        self._state = state
        self._last_policy_output = policy_output
        self._last_subgoal = subgoal if subgoal is not None else self._initial_subgoal()
        return policy_output, state

    def _policy(self, obs, state, subgoal, training):
        if state is None:
            print("##### state is none in policy, creating new ####")
            latent = self._wm.dynamics.initial(self._config.envs["amount"])
            action = torch.zeros(
                (self._config.envs["amount"], self._config.num_actions),
                device=self._config.device,
            )
        else:
            latent, action = state

        latent = {k: torch.as_tensor(v).detach() for k, v in latent.items()}
        if self._config.hierarchical_world_models or self._name == "Subactor-0":
            obs = self._wm.preprocess(obs)

            embed = self._wm.encoder(obs)
            latent, _ = self._wm.dynamics.obs_step(
                latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
            )
        else:
            latent = obs

        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat, subgoal)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat, subgoal)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat, subgoal)
            action = actor.sample()
        logprob = actor.entropy().detach()
        latent = {k: torch.as_tensor(v).detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        if training:
            policy_output = {"action": action, "log_entropy": logprob}
        else:
            policy_output = {"action": action}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)

    def _train(self, data):
        self._trains += 1
        metrics = {}
        timer = tools.TimeRecording()
        if self._config.hierarchical_world_models or self._name == "Subactor-0":
            self._wm.train()
            with timer:
                post, context, mets = self._wm._train(data)
            self._wm.eval()
            metrics["wm_traintime"] = timer.elapsed_time
            metrics.update(mets)
        else:
            post = data
        start = post
        reward = lambda f, s, a: {
            "extrinsic": self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode(),
            "subgoal": self._subgoal_reward(s, start["subgoal"]),
            "novelty": self._novelty_reward(s),
        }
        self._task_behavior.train()
        if self._config.hierarchical_world_models or self._name == "Subactor-0":
            with timer:
                train_result = self._task_behavior._train(start, reward)
                metrics.update(train_result[-1])
        self._task_behavior.eval()
        self.subgoal_autoencoder.train()

        with timer:
            if self._config.subgoal_autoencoder_imag_training:
                mets = self.subgoal_autoencoder._train(
                    self.get_subgoal(train_result[1])
                )
            else:
                mets = self.subgoal_autoencoder._train(self.get_subgoal(start))
        self.subgoal_autoencoder.eval()
        metrics["subgoal_traintime"] = timer.elapsed_time
        metrics.update(mets)
        metrics["task_traintime"] = timer.elapsed_time

        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        if self._config.autoregressive_evalution:
            metrics["autoregression_loss_curve"] = autoregression_loss_curve(self, data)
            metrics.update(
                tools.tensorstats(
                    metrics["autoregression_loss_curve"],
                    "autoregression_loss_curve_stat",
                )
            )
            # make stats for last quartile of loss curve
            metrics.update(
                tools.tensorstats(
                    metrics["autoregression_loss_curve"][
                        -metrics["autoregression_loss_curve"].shape[0] // 4 :
                    ],
                    "autoregression_loss_curve_last_quartile_stat",
                )
            )
            metrics["len_loss_curve"] = len(metrics["autoregression_loss_curve"])
        self._last_start = {k: v.detach() for k, v in start.items()}
        return metrics

    def _subgoal_reward(self, state, subgoal):
        # computes the cosine max reward
        if not self._compute_subgoal:
            return None
        state_representation = self.get_subgoal(state)
        if self._config.subgoal_compression["encoding_symlog"]:
            state_representation = tools.symlog(state_representation)
        reshaped_subgoal = subgoal.reshape(
            [subgoal.shape[0] * subgoal.shape[1]] + list(subgoal.shape[2:])
        ).expand(state_representation.shape)
        dims_to_sum = list(range(len(state_representation.shape)))[2:]
        gnorm = torch.norm(reshaped_subgoal, dim=dims_to_sum) + 1e-12
        fnorm = torch.norm(state_representation, dim=dims_to_sum) + 1e-12
        norm = torch.max(gnorm, fnorm)
        cos = torch.sum(reshaped_subgoal * state_representation, dim=dims_to_sum) / (
            norm * norm
        )
        subgoal_reward = torch.clamp(cos.unsqueeze(-1), min=0)

        if self._config.subgoal_reward_symlog:
            return tools.symlog(subgoal_reward)
        return subgoal_reward

    def _novelty_reward(self, state):
        # computes the novelty reward
        features = self.get_subgoal(state)
        if not self._compute_novelty:
            return None
        with torch.no_grad():
            reconstruction = self.subgoal_autoencoder(features)
        novelty_reward = torch.mean((features - reconstruction) ** 2, dim=-1).unsqueeze(
            -1
        )
        if self._config.novelty_reward_symlog:
            return tools.symlog(novelty_reward)
        return novelty_reward


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(generator, config):
    batcher = embodied.Batcher(
        sources=[generator] * config.batch_size,
        workers=config.data_loaders,
        prefetch_source=4,
        prefetch_batch=config.prefetch_batches,
    )
    return batcher()


def expand(value, dims):
    while len(value.shape) < dims:
        value = value[..., None]
    return value


def autoregression_loss_curve(model, data):
    data = model._wm.preprocess(data)
    embed = model._wm.encoder(data)
    observe_portion = embed.shape[1] // 2

    states, _ = model._wm.dynamics.observe(
        embed[:, :observe_portion],
        data["action"][:, :observe_portion],
        data["is_first"][:, :observe_portion],
    )
    init = {k: v[:, -1] for k, v in states.items()}
    num_timesteps = observe_portion
    errors = []
    for i in range(1, num_timesteps):
        prediction = model._wm.dynamics.img_step(
            init, data["action"][:, observe_portion + i]
        )
        decoded = model._wm.heads["decoder"](
            model._wm.dynamics.get_feat(prediction).unsqueeze(1)
        )
        decoded = {k: v.mode() for k, v in decoded.items()}
        diffs = [
            torch.mean((decoded[k] - data[k][:, observe_portion + i]) ** 2).item()
            for k in decoded.keys()
        ]
        errors.append(sum(diffs) / len(diffs))
        init = prediction
    return np.array(errors)


def make_replay(config, directory=None, is_eval=False, rate_limit=False, **kwargs):
    assert config.replay == "uniform" or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == "uniform" or is_eval:
        kw = {"online": config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw["samples_per_insert"] = config.run.train_ratio / config.batch_length
            kw["tolerance"] = 10 * config.batch_size
            kw["min_size"] = config.batch_size
        print("using size", size)
        replay = embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == "reverb":
        replay = embodied.replay.Reverb(length, size, directory)
    elif config.replay == "chunks":
        replay = embodied.replay.NaiveChunks(length, size, directory)
    elif config.replay == "timebalanced":
        replay = embodied.replay.TimeBalanced(
            length, size, directory, bias_factor=config.replay_bias_factor
        )
    elif config.replay == "timebalancednaive":
        replay = embodied.replay.TimeBalancedNaive(
            length, size, directory, bias_factor=config.replay_bias_factor
        )
    elif config.replay == "efficienttimebalanced":
        replay = embodied.replay.EfficientTimeBalanced(
            length,
            size,
            directory,
            temperature=config.replay_temperature,
        )
    else:
        raise NotImplementedError(config.replay)
    return replay
