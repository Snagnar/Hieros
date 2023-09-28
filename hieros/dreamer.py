import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "egl"

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


import exploration as expl
import tools
import models_dreamer


import embodied
import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, replay=None):
        super(Dreamer, self).__init__()
        self._config = config
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = count_steps(config.traindir)
        self._update_count = 0

        self._replay = replay
        self._dataset = self.dataset(self._replay.dataset)
        # Schedules.
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
            x, self._step
        )
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        )
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
            x, self._step
        )
        self._wm = models_dreamer.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models_dreamer.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if config.compile and hasattr(torch, "compile"):
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
        self._last_policy_output = None

    def policy(self, obs, state=None, mode="train"):
        training = mode in ["train", "explore"]
        reset = obs["is_last"] | obs["is_terminal"]
        step = self._step
        if self._should_reset(step):
            state = None
            self._last_policy_output = None
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

        policy_output, state = self._policy(obs, state, training)

        acts = {k: embodied.convert(v) for k, v in policy_output.items()}
        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {k: v * expand(mask, len(v.shape)) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()

        # if training and self._manual_add and policy_output is not None and not self._config.fix_dataset:
        if training and policy_output is not None and not self._config.fix_dataset:
            tools.add_step_to_replay(
                self._replay,
                obs,
                acts,
                None,
                self._config,
            )
        self._last_policy_output = policy_output

        if training:
            self._step += len(reset)
        return policy_output, state

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value

    def train(self, data=None, state=None):
        mets = self._train(next(self._dataset))
        self._update_count += 1
        mets["update_count"] = self._update_count
        to_cpu = (
            lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        )
        metrics = {f"Subactor-0/{k}": to_cpu(v) for k, v in mets.items()}
        return {}, state, metrics

    def dataset(self, generator):
        batcher = embodied.Batcher(
            sources=[generator] * self._config.batch_size,
            workers=self._config.data_loaders,
            # postprocess=lambda x: torch.Tensor(x).to(self._config.device),
            prefetch_source=4,
            prefetch_batch=1,
        )
        return batcher()

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.entropy()
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "log_entropy": logprob}
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
        metrics = {}
        if self._replay is not None:
            metrics["dataset_size"] = len(self._replay)
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        return metrics

    def report(self, data):
        report = {}
        return report

    def save(self):
        data = {}
        return data

    def load(self, data):
        ...

    def sync(self):
        ...


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def expand(value, dims):
    while len(value.shape) < dims:
        value = value[..., None]
    return value
