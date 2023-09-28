import numpy as np
import datetime
import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "egl"

import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


import tools

# import envs.wrappers as wrappers
import importlib

import embodied
from embodied import wrappers
import torch
from torch import distributions as torchd

from hieros import Hieros, make_replay
from dreamer import Dreamer

to_np = lambda x: x.detach().cpu().numpy()
from tqdm import tqdm


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_logger(logdir, step, config):
    multiplier = config.action_repeat
    outputs = [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
        embodied.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score"),
        # embodied.logger.MLFlowOutput(logdir.name),
    ]
    if config.tensorboard_logging:
        outputs.append(embodied.logger.TensorBoardOutput(logdir))
    if config.wandb_logging:
        outputs.append(embodied.logger.WandBOutput(logdir, config.__dict__))
    logger = embodied.Logger(
        step,
        outputs,
        multiplier,
    )
    return logger


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config):
    # convert all lists in config to tuples
    def convert_to_tuple(config):
        for key in config:
            if isinstance(config[key], dict):
                convert_to_tuple(config[key])
            elif isinstance(config[key], list):
                config[key] = tuple(config[key])

    convert_to_tuple(config.__dict__)
    if "logdir" not in config or config.logdir is None:
        config.logdir = f"logs/{config.task}-{datetime.datetime.now():%Y%m%d-%H%M%S}"

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    step = embodied.Counter()
    logger = make_logger(logdir, step, config)

    print("Create envs.")
    train_envs = make_envs(config)
    eval_envs = make_envs(config)
    acts = train_envs.act_space
    config.num_actions = acts["action"].shape
    if hasattr(config.num_actions, "__len__") and len(config.num_actions) == 1:
        config.num_actions = config.num_actions[0]

    config._replay_dir = config.traindir / "replay"

    train_replay = make_replay(config, config.traindir / "replay")
    if not config.offline_traindir and config.enable_prefill:
        prefill = max(
            0,
            config.prefill - count_steps(config.traindir),
            config.batch_length * config.batch_size,
        )
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts["action"], "discrete") and acts["action"].discrete:
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs["amount"], 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts["action"].low).repeat(config.envs["amount"], 1),
                    torch.Tensor(acts["action"].high).repeat(config.envs["amount"], 1),
                ),
                1,
            )

        def random_agent(o, d, r):
            action = random_actor.sample()
            logprob = random_actor.entropy()
            random_action = {
                key: value.sample() for key, value in train_envs.act_space.items()
            }
            random_action.update(
                {
                    "action": action,
                    "log_entropy": logprob,
                    "reset": torch.zeros(config.envs["amount"], dtype=torch.bool),
                }
            )
            logger.add(
                tools.tensorstats(random_action["action"], "action"),
                prefix="random_agent",
            )
            logger.add(
                tools.tensorstats(random_action["log_entropy"], "log_entropy"),
                prefix="random_agent",
            )
            acts = {k: embodied.convert(v) for k, v in random_action.items()}
            if o["is_last"].any():
                mask = 1 - o["is_last"]
                acts = {
                    k: v * expand(mask, len(v.shape)) for k, v in random_action.items()
                }
            acts["reset"] = o["is_last"].copy()

            tools.add_step_to_replay(
                train_replay,
                o,
                acts,
                None,
                config,
                config.model_name != "dreamer",
            )

            return acts

        if prefill > 0:
            tools.simulate(
                random_agent, train_envs, prefill, replay=train_replay, config=config
            )

    print("Simulate agent.")
    if config.model_name == "dreamer":
        agent = Dreamer(
            train_envs.obs_space,
            train_envs.act_space,
            config,
            train_replay,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
    else:
        agent = Hieros(
            train_envs.obs_space,
            train_envs.act_space,
            config,
            train_replay,
        ).to(config.device)
        agent.requires_grad_()
        agent.set_training()

    num_parameters = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"##### Number of trainable parameters: {num_parameters:_}")
    if config.wandb_logging:
        import wandb

        config.num_trainable_params = num_parameters
        wandb.config.update(
            {k: v for k, v in config.__dict__.items() if k[0] != "_"},
            allow_val_change=True,
        )
    if (logdir / "latest_model.pt").exists():
        agent.load_state_dict(torch.load(logdir / "latest_model.pt"))
        agent._should_pretrain._once = False
    if config.model_name == "dreamer":
        agent.eval()
    else:
        agent.set_eval()
    if config.pretrain:
        print("pretraining for", config.pretrain, "steps...")
        for _ in tqdm(range(config.pretrain)):
            agent.train()
    config.batch_steps = config.batch_size * config.batch_length

    embodied.run.train_eval(agent, train_envs, eval_envs, None, None, logger, config)

    if config.autoregressive_evalution:
        if config.model_name == "hieros":
            data = next(agent._subactors[0]._dataset)
            errors = autoregression_loss_curve(agent._subactors[0], data)
            logger.add({"autoregression_loss": errors}, prefix="autoregression")
            logger.write()
        else:
            raise NotImplementedError

    for env in [train_envs]:
        try:
            env.close()
        except Exception:
            pass


def expand(value, dims):
    while len(value.shape) < dims:
        value = value[..., None]
    return value


def make_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    ctors = []
    for index in range(config.envs["amount"]):
        ctor = lambda: make_env(config, **overrides)
        if config.envs["parallel"] != "none":
            ctor = functools.partial(embodied.Parallel, ctor, config.envs["parallel"])
        if config.envs["restart"]:
            ctor = functools.partial(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs["parallel"] != "none"))


def make_env(config, **overrides):
    # You can add custom environments by creating and returning the environment
    # instance here. Environments with different interfaces can be converted
    # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
    print("entering make_env")
    suite, task = config.task.split("_", 1)
    ctor = {
        "dummy": "embodied.envs.dummy:Dummy",
        "gym": "embodied.envs.from_gym:FromGym",
        "dm": "embodied.envs.from_dmenv:FromDM",
        "crafter": "embodied.envs.crafter:Crafter",
        "dmc": "embodied.envs.dmc:DMC",
        "atari": "embodied.envs.atari:Atari",
        "dmlab": "embodied.envs.dmlab:DMLab",
        "minecraft": "embodied.envs.minecraft:Minecraft",
        "loconav": "embodied.envs.loconav:LocoNav",
        "pinpad": "embodied.envs.pinpad:PinPad",
        "bsuite": "embodied.envs.bsuite:BSuite",
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(":")
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args["discretize"]:
            env = wrappers.DiscretizeAction(env, name, args["discretize"])
        else:
            env = wrappers.NormalizeAction(env, name)
    env = wrappers.ExpandScalars(env)
    if args["length"]:
        env = wrappers.TimeLimit(env, args["length"], args["reset"])
    if args["checks"]:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env


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
    for i in range(num_timesteps):
        prediction = model._wm.dynamics.img_step(
            init, data["action"][:, observe_portion + i]
        )
        decoded = model._wm.heads["decoder"](
            model._wm.dynamics.get_feat(prediction).unsqueeze(1)
        )["image"].mode()
        errors.append(
            torch.mean((decoded - data["image"][:, observe_portion + i]) ** 2).item()
        )
        print(errors[-1])
        init = prediction
    return np.array(errors)


if __name__ == "__main__":
    import lovely_tensors as lt

    lt.monkey_patch()

    try:
        import lovely_numpy as ln

        ln.monkey_patch()
    except:
        pass
    # torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--additional_configs", nargs="*")
    parser.add_argument("--subgoal_shape_configs", nargs="*")
    parser.add_argument("--s5_configs", nargs="*")
    parser.add_argument("--s5_loss_clip", nargs="*")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    if args.additional_configs:
        name_list += args.additional_configs
    if args.subgoal_shape_configs:
        name_list += args.subgoal_shape_configs
    if args.s5_configs:
        name_list += args.s5_configs
    if args.s5_loss_clip:
        name_list += args.s5_loss_clip
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    args = parser.parse_args(remaining)
    main(args)
