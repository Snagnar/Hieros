from torch import distributions as torchd
import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import wandb

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


def create_dynamics_model(config, embed_size, act_space, device):
    if config.dynamics_model == "rssm":
        return networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            embed_size,
            config.device,
        ).to(device)
    elif config.dynamics_model == "s5":
        return networks.Seq2SeqDynamics(
            seq_model="s5",
            stoch=config.dyn_stoch,
            deter=config.dyn_deter,
            hidden=config.dyn_hidden,
            layers_input=config.dyn_input_layers,
            layers_output=config.dyn_output_layers,
            rec_depth=config.dyn_rec_depth,
            shared=config.dyn_shared,
            discrete=config.dyn_discrete,
            act=config.act,
            norm=config.norm,
            mean_act=config.dyn_mean_act,
            std_act=config.dyn_std_act,
            temp_post=config.dyn_temp_post,
            min_std=config.dyn_min_std,
            unimix_ratio=config.unimix_ratio,
            initial=config.initial,
            num_actions=config.num_actions,
            embed=embed_size,
            stochastic=config.dyn_stochasticity,
            device=config.device,
            config=config,
        ).to(device)
    elif config.dynamics_model == "transformer":
        raise ValueError("transformer is not supported yet")


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self._reward_actor_input = config.reward_actor_input
        self._cont_actor_input = config.cont_actor_input
        self._stochasticity_actor_input = config.stochasticity_actor_input

        self.dynamics = create_dynamics_model(
            config, self.embed_size, act_space, config.device
        )
        if (
            config.dynamics_model != "s5"
            and config.max_hierarchy == 1
            and config.wandb_logging
        ):
            wandb.watch(self.dynamics, log="gradients")
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        if config.reward_head == "symlog_disc":
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                (255,),
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                [],
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )

        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            lr_scheduler=config.lr_scheduler,
            warmup_steps=config.warmup_steps,
            max_steps=config.steps,
            use_amp=self._use_amp,
        )
        self._scales = dict(
            reward=config.reward_scale,
            cont=config.cont_scale,
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        metrics = {}
        data = self.preprocess(data)
        timer = tools.TimeRecording()
        self.train()
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = tools.schedule(self._config.kl_free, self._step)
                dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                rep_scale = tools.schedule(self._config.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    like = pred.log_prob(data[name])
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                model_loss = sum(losses.values()) + kl_loss
            metrics.update(self._model_opt(model_loss, self.parameters()))
        metrics["wm_train_update"] = timer.elapsed_time
        with timer:
            metrics.update(
                {f"{name}_loss": loss.detach() for name, loss in losses.items()}
            )
            metrics["kl_free"] = kl_free
            metrics["dyn_scale"] = dyn_scale
            metrics["rep_scale"] = rep_scale
            metrics["dyn_loss"] = dyn_loss.detach()
            metrics["rep_loss"] = rep_loss.detach()
            metrics["kl"] = torch.mean(kl_value.detach())
            with torch.cuda.amp.autocast(self._use_amp):
                metrics["prior_ent"] = torch.mean(
                    self.dynamics.get_dist(prior).entropy().detach()
                )
                metrics["post_ent"] = torch.mean(
                    self.dynamics.get_dist(post).entropy().detach()
                )
                context = dict(
                    embed=embed,
                    feat=self.dynamics.get_feat(post),
                    kl=kl_value,
                    postent=self.dynamics.get_dist(post).entropy(),
                )
            post = {k: v.detach() for k, v in post.items()}
            post["subgoal"] = data["subgoal"]
            if "encoded_subgoal" in data:
                post["encoded_subgoal"] = data["encoded_subgoal"]
        metrics["wm_train_metrics"] = timer.elapsed_time
        self.eval()
        return post, context, metrics

    def preprocess(self, obs):
        obs = {
            k: (
                v.to(self._config.device).float()
                if torch.is_tensor(v)
                else torch.from_numpy(v).to(self._config.device).float()
            )
            for k, v in obs.items()
        }

        if "image" in obs:
            obs["image"] = obs["image"] / 255.0 - 0.5
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        obs["reward"] = obs["reward"].unsqueeze(-1)
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        if "is_terminal" in obs:
            # this label is necessary to train cont_head
            obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        else:
            raise ValueError('"is_terminal" was not found in observation.')
        if "subgoal" in obs:
            obs["subgoal"] = obs["subgoal"]
        return obs

    def get_feat(self, state, rewards=None, conts=None):
        dynamics_features = self.dynamics.get_feat(state)
        additional_features = []
        if self._reward_actor_input:
            additional_features.append(
                self.heads["reward"](dynamics_features).mode()
                if rewards is None
                else rewards
            )
        if self._cont_actor_input:
            additional_features.append(
                self.heads["cont"](dynamics_features).mean if conts is None else conts
            )
        if (
            self._stochasticity_actor_input
            and self._config.dyn_stochasticity
            and "entropy" in state
        ):
            additional_features.append(state["entropy"].unsqueeze(-1))
        if len(additional_features) == 0:
            return dynamics_features
        additional_features = torch.cat(additional_features, -1)
        if self._config.additional_features_symlog:
            additional_features = tools.symlog(additional_features)
        return torch.cat([dynamics_features, additional_features], -1)

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0
        progress = (
            torch.arange(0, model.shape[1])
            / (model.shape[1] - 1)
            * torch.max(model).item()
        )
        progress = progress[None, :, None, None, None].expand_as(model).to(model.device)
        progress = progress[:, :, : progress.shape[2] // 4]

        return torch.cat([truth, model, error, progress], 2).squeeze()

    def decode_state(self, state):
        if isinstance(state, dict):
            decoded = self.heads["decoder"](self.dynamics.get_feat(state))
        else:
            decoded = self.heads["decoder"](state)
        if "image" in decoded:
            return {"image": decoded["image"].mode()}
        stoch = decoded["stoch"].mode()
        deter = decoded["deter"].mode()
        if self._config.subactor_encode_intermediate:
            if self._config.subactor_encoding_architecture == "mlp":
                stoch_slice_size = int(
                    stoch.shape[-1] // self._config.subactor_update_every
                )
                stoch = stoch[..., -stoch_slice_size:]
                deter_slice_size = int(
                    deter.shape[-1] // self._config.subactor_update_every
                )
                deter = deter[..., -deter_slice_size:]
            else:
                stoch = stoch.select(-2, -1)
                deter = deter.select(-2, -1)
        if self.dynamics._discrete:
            shape = list(stoch.shape[:-1]) + [
                self.dynamics._stoch,
                self.dynamics._discrete,
            ]
            stoch = stoch.reshape(shape)
        return {"stoch": stoch, "deter": deter}


class ImagBehavior(nn.Module):
    def __init__(
        self,
        config,
        world_model,
        subgoal_shape,
        stop_grad_actor=True,
        reward=None,
        use_subgoal=False,
        use_novelty=False,
    ):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._use_subgoal = use_subgoal
        self._reward = reward
        if not use_subgoal:
            print("not using subgoal, setting shape to 0")
            subgoal_shape = 0
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        feat_size += subgoal_shape
        if config.reward_actor_input:
            feat_size += 1
        if config.cont_actor_input:
            feat_size += 1
        if config.stochasticity_actor_input and config.dyn_stochasticity:
            feat_size += 1
        print("feat_size", feat_size)
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
            action_shape=config.action_shape,
            use_subgoal=self._use_subgoal,
        )
        self._value_heads = nn.ModuleDict()
        self._slow_values = nn.ModuleDict()
        heads = ["extrinsic"]
        if use_subgoal:
            heads.append("subgoal")
        if use_novelty:
            heads.append("novelty")
        for head in heads:
            if config.value_head == "symlog_disc":
                self._value_heads[head] = networks.MLP(
                    feat_size,
                    (255,),
                    config.value_layers,
                    config.units,
                    config.act,
                    config.norm,
                    config.value_head,
                    outscale=0.0,
                    device=config.device,
                )
            else:
                self._value_heads[head] = networks.MLP(
                    feat_size,
                    [],
                    config.value_layers,
                    config.units,
                    config.act,
                    config.norm,
                    config.value_head,
                    outscale=0.0,
                    device=config.device,
                )
            if config.slow_value_target:
                self._slow_values[head] = copy.deepcopy(self._value_heads[head])
        self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opts = {
            head: tools.Optimizer(
                f"value_{head}",
                self._value_heads[head].parameters(),
                config.value_lr,
                config.ac_opt_eps,
                config.value_grad_clip,
                **kw,
            )
            for head in self._value_heads
        }
        self._value_weights = {
            "extrinsic": config.extrinsic_reward_weight,
            "subgoal": config.subgoal_reward_weight,
            "novelty": config.novelty_reward_weight,
        }
        if self._config.reward_EMA:
            self.reward_ema = {
                head: RewardEMA(device=self._config.device)
                for head in self._value_heads
            }

    def _value_input(self, feature, subgoal):
        if not self._use_subgoal:
            return feature
        if len(subgoal.shape) > len(feature.shape):
            subgoal_shape = list(subgoal.shape[:-2]) + [
                subgoal.shape[-2] * subgoal.shape[-1]
            ]
            return torch.cat([feature, subgoal.reshape(subgoal_shape)], -1)
        return torch.cat([feature, subgoal], -1)

    def _reshape_subgoal(self, subgoal, target_shape):
        expand_size = [target_shape[0]] + [-1] * (len(subgoal.shape) - 1)
        return subgoal.reshape(
            [subgoal.shape[0] * subgoal.shape[1]] + list(subgoal.shape[2:])
        ).expand(*expand_size)

    def _train(
        self,
        start,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        self.train()
        self._world_model.eval()
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}
        subgoal = (
            start["subgoal"]
            if self._config.decompress_subgoal_for_input
            else start["encoded_subgoal"]
        )

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats
                )
                subgoal = self._reshape_subgoal(subgoal, imag_feat.shape)
                reward = objective(imag_feat, imag_state, imag_action)

                actor_ent = self.actor(imag_feat, subgoal).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                targets, weights, base = self._compute_target(
                    self._value_input(imag_feat, subgoal),
                    imag_state,
                    reward,
                    actor_ent,
                    state_ent,
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    subgoal,
                    imag_state,
                    imag_action,
                    targets,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat

        for head, value_network in self._value_heads.items():
            with tools.RequiresGrad(value_network):
                with torch.cuda.amp.autocast(self._use_amp):
                    value = value_network(
                        self._value_input(
                            value_input[:-1].detach(), subgoal[:-1].detach()
                        )
                    )
                    target = torch.stack(targets[head], dim=1)
                    # (time, batch, 1), (time, batch, 1) -> (time, batch)
                    value_loss = -value.log_prob(target.detach())
                    slow_target = self._slow_values[head](
                        self._value_input(
                            value_input[:-1].detach(), subgoal[:-1].detach()
                        )
                    )
                    if self._config.slow_value_target:
                        value_loss = value_loss - value.log_prob(
                            slow_target.mode().detach()
                        )
                    if self._config.value_decay:
                        value_loss += self._config.value_decay * value.mode()
                    # (time, batch, 1), (time, batch, 1) -> (1,)
                    value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
                    metrics.update(
                        self._value_opts[head](value_loss, value_network.parameters())
                    )
                    metrics.update(tools.tensorstats(value.mode(), f"value_{head}"))
                    metrics.update(tools.tensorstats(target, f"target_{head}"))

        for key, value in reward.items():
            if value is not None:
                metrics.update(tools.tensorstats(value, f"imag_{key}_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = torch.mean(actor_ent.detach())
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
        self.eval()
        return imag_feat, imag_state, imag_action, weights, metrics

    def _train_wo_imag(
        self,
        imag_feat,
        imag_state,
        imag_action,
        subgoal,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        self.train()
        self._world_model.eval()
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                subgoal = self._reshape_subgoal(subgoal, imag_feat.shape)
                reward = objective(imag_feat, imag_state, imag_action)

                actor_ent = self.actor(imag_feat, subgoal).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                targets, weights, base = self._compute_target(
                    self._value_input(imag_feat, subgoal),
                    imag_state,
                    reward,
                    actor_ent,
                    state_ent,
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    subgoal,
                    imag_state,
                    imag_action,
                    targets,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat

        for head, value_network in self._value_heads.items():
            with tools.RequiresGrad(value_network):
                with torch.cuda.amp.autocast(self._use_amp):
                    value = value_network(
                        self._value_input(
                            value_input[:-1].detach(), subgoal[:-1].detach()
                        )
                    )
                    target = torch.stack(targets[head], dim=1)
                    # (time, batch, 1), (time, batch, 1) -> (time, batch)
                    value_loss = -value.log_prob(target.detach())
                    slow_target = self._slow_values[head](
                        self._value_input(
                            value_input[:-1].detach(), subgoal[:-1].detach()
                        )
                    )
                    if self._config.slow_value_target:
                        value_loss = value_loss - value.log_prob(
                            slow_target.mode().detach()
                        )
                    if self._config.value_decay:
                        value_loss += self._config.value_decay * value.mode()
                    # (time, batch, 1), (time, batch, 1) -> (1,)
                    value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
                    metrics.update(
                        self._value_opts[head](value_loss, value_network.parameters())
                    )
                    metrics.update(tools.tensorstats(value.mode(), f"value_{head}"))
                    metrics.update(tools.tensorstats(target, f"target_{head}"))

        metrics.update(tools.tensorstats(value_input, "value_input"))
        for key, value in reward.items():
            if value is not None:
                metrics.update(tools.tensorstats(value, f"imag_{key}_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = torch.mean(actor_ent.detach())
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
        self.eval()
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        wm = self._world_model
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        if (
            self._config.dynamics_model == "s5"
            and self._config.s5["context_fraction"] > 0
        ):
            context_len = int(
                start["state"].shape[1] * self._config.s5["context_fraction"]
            )
            start = {k: v[:, context_len:] for k, v in start.items()}
            horizon += context_len
        start = {k: flatten(v) for k, v in start.items()}
        subgoal = (
            start["subgoal"]
            if self._config.decompress_subgoal_for_input
            else start["encoded_subgoal"]
        )

        def step(prev, _):
            state, _, _ = prev
            feat = wm.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat

            action = policy(inp, subgoal).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            if "state" in succ:
                succ["state"] = succ["state"].detach()
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(self, value_input, imag_state, reward, actor_ent, state_ent):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent

        targets, bases = {}, {}
        for head, value_network in self._value_heads.items():
            value = value_network(value_input).mode()
            targets[head] = tools.lambda_return(
                reward[head][:-1],
                value[:-1],
                discount[:-1],
                bootstrap=value[-1],
                lambda_=self._config.discount_lambda,
                axis=0,
            )
            bases[head] = value[:-1]
        # value(15, 960, ch)
        # action(15, 960, ch)
        # discount(15, 960, ch)
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return targets, weights, bases

    def _compute_actor_loss(
        self,
        imag_feat,
        subgoal,
        imag_state,
        imag_action,
        targets,
        actor_ent,
        state_ent,
        weights,
        bases,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp, subgoal)
        actor_ent = policy.entropy()
        advantage = None
        # Q-val for actor is not transformed using symlog
        for head, target in targets.items():
            target = torch.stack(target, dim=1)
            offset, scale = self.reward_ema[head](target)
            normed_target = (target - offset) / scale
            normed_base = (bases[head] - offset) / scale
            adv = normed_target - normed_base
            metrics[f"{head}_adv"] = torch.mean(adv.detach())
            values = self.reward_ema[head].values
            metrics[f"EMA_005_{head}"] = values[0].detach()
            metrics[f"EMA_095_{head}"] = values[1].detach()
            if advantage is None:
                advantage = adv * self._value_weights[head]
            else:
                advantage += adv * self._value_weights[head]
        metrics["final_adv"] = torch.mean(advantage.detach())
        if self._config.imag_gradient == "dynamics":
            actor_target = advantage
        elif self._config.imag_gradient == "reinforce":
            value_input = self._value_input(imag_feat[:-1], subgoal[:-1])
            mult = sum(
                [
                    (
                        torch.stack(target, dim=1)
                        - self._value_heads[head](value_input).mode()
                    ).detach()
                    * self._value_weights[head]
                    for head, target in targets.items()
                ]
            ) / sum([self._value_weights[head] for head in targets])
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * mult
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None] * advantage.detach()
            )
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = torch.mean(state_entropy).detach()
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for head in self._slow_values:
                    for s, d in zip(
                        self._value_heads[head].parameters(),
                        self._slow_values[head].parameters(),
                    ):
                        d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1


class SubgoalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape=[32],
        discrete_size=False,
        layers=[],
        bottleneck_shape=[8, 8],
        activation="SiLU",
        norm="LayerNorm",
        encoding_symlog=False,
        decoding_symlog=False,
        kl_scale=1.0,
        unimix_ratio=0.01,
        config={},
    ):
        super().__init__()
        self._input_shape = (
            [input_shape] if isinstance(input_shape, int) else input_shape
        )
        self._unimix_ratio = unimix_ratio
        self._use_amp = True if config.precision <= 16 else False
        self._encoding_symlog = encoding_symlog

        self._encoder = networks.MLP(
            input_shape[-1],
            None,
            layers,
            np.prod(bottleneck_shape),
            activation,
            norm,
            symlog_inputs=encoding_symlog,
        ).to(config.device)
        self._decoder = networks.MLP(
            np.prod(bottleneck_shape),
            None,
            layers,
            input_shape[-1],
            activation,
            norm,
            symlog_inputs=decoding_symlog,
        ).to(config.device)
        self.kl_scale = kl_scale

        self.prior = tools.OneHotDist(
            torch.zeros([1, *bottleneck_shape], device=config.device),
            unimix_ratio=0.0,
        )
        self._model_opt = tools.Optimizer(
            "subgoal_autoencoder",
            self.parameters(),
            config.subgoal_compression["lr"],
            config.subgoal_compression["opt_eps"],
            config.subgoal_compression["grad_clip"],
            config.subgoal_compression["weight_decay"],
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._bottleneck_shape = bottleneck_shape

    def forward(self, inputs):
        return self.decode(self.encode(inputs))

    def encode(self, inputs):
        encoded = self._encoder(inputs).view(
            *inputs.shape[:-1], *self._bottleneck_shape
        )
        return self.onehot_dist(encoded)[0]

    def decode(self, inputs):
        return self._decoder(inputs.view(*inputs.shape[:-2], -1))

    def loss(self, inputs):
        encoded = self._encoder(inputs).view(
            *inputs.shape[:-1], *self._bottleneck_shape
        )
        encoded, dist = self.onehot_dist(encoded)
        decoded = self._decoder(encoded.view(*inputs.shape[:-1], -1))
        reconstruction_loss = (
            torch.mean((tools.symlog(inputs) - decoded) ** 2)
            if self._encoding_symlog
            else torch.mean((inputs - decoded) ** 2)
        )
        ...
        return (
            reconstruction_loss,
            torchd.kl.kl_divergence(dist, self.prior).mean(),
        )

    def _train(self, data):
        reconstruction_loss, kl_loss = self.loss(data)
        loss = reconstruction_loss + self.kl_scale * kl_loss
        metrics = self._model_opt(loss, self.parameters())
        metrics.update(
            {
                "subgoal_autoencoder_reconstruction_loss": reconstruction_loss.detach(),
                "subgoal_autoencoder_kl_loss": kl_loss.detach(),
                "subgoal_autoencoder_loss": loss.detach(),
            }
        )
        return metrics

    def onehot_dist(self, logits):
        dist = tools.OneHotDist(logits, unimix_ratio=self._unimix_ratio)
        return dist.mode(), dist
