# Hieros
Implemenation of the HIERarchical imagionation On Structured State Space Sequence Models (HIEROS) paper in pytorch. This repository is based on the [DreamerV3](https://github.com/danijar/dreamerv3), [DreamerV3 in pytorch](https://github.com/NM512/dreamerv3-torch) and [S5 in pytorch](https://github.com/i404788/s5-pytorch) repositories.

# Installation

Install pip dependencies:
```
pip install -r requirements.txt
```

Install required tools:
```
sudo apt update && sudo apt install -y wget unrar
```

Install atari roms:
```
bash embodied/scripts/install-atari.sh
```

# Usage

To train a model on a atari game, run:
```
python hieros/train.py --configs atari100k --task=atari_alien
```
You can specify the task to train on with the `--task` flag. The available tasks are:
```
atari_alien, atari_amidar, atari_assault, atari_asterix, atari_bank_heist, atari_battle_zone, atari_boxing, atari_breakout, atari_chopper_command, atari_crazy_climber, atari_demon_attack, atari_freeway, atari_frostbite, atari_gopher, atari_hero, atari_jamesbond, atari_kangaroo, atari_krull, atari_kung_fu_master, atari_ms_pacman, atari_pong, atari_private_eye, atari_qbert, atari_road_runner, atari_seaquest
```

We also support a wide range of other benchmarks. For this, please reference the `hieros/config.yml` to find different configurations. For example, to train on the `dmc_vision` task, run:
```
python hieros/train.py --configs dmc_vision --task=dmc_cheetah_run
```

All flags available in `hieros/config.yml` are configurable as command line arguments. For example, to train on the `atari_alien` task with a different number of layers, run:
```
python hieros/train.py --configs atari100k --task=atari_alien --max_hierarchy=2
```

We also include an implementation of the original [DreamerV3](https://github.com/NM512/dreamerv3-torch) model, which is accessible with `--model_name=dreamer`.

The metrics are logged to tensorboard by default. To visualize the training progress, run:
```
tensorboard --logdir=logs
```
With these training statistics, you can also reproduce the plots in the paper.

# Repository

The repository is structured as follows:
- `hieros/` contains the implementation and training code of the HIEROS model.
- `embodied/` contains the implementation of some basic tools like logging, replay buffers, environments, etc. This is largely copied from [here](https://github.com/danijar/dreamerv3)
- `resettable_s5/` contains our implementation of the resettable S5 model used for the S5WM. This is based on the [pytorch s5 implementation](https://github.com/i404788/s5-pytorch)
- `experiments/` contains wandb sweep configurations for the experiments in the paper.
- `sampler_visualization.py` contains code to visualize the sampling methods used in the paper (ETBS and the standard uniform sampling).