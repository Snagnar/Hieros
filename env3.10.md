# Hieros on Python 3.10

1. Environment Setup
```
conda create -n hieros python=3.10 -y
conda activate hieros
conda install -c conda-forge wget unrar cmake zlib -y
```
(since s5_model.py requires match syntax from python>=3.10) \
(atari-py requires python<3.10 so requires cmake to be installed) \
(conda-forge channel, which is a community-driven channel, is needed to install unrar. Should use conda-forge for all packages to avoid conflicts.) \

In repository root folder, run: 
```
pip install "pip<24.0"
pip install -r requirements.txt
bash embodied/scripts/install-atari.sh
```

1.5. How to use w&b
login to your wandb account:
```
wandb login
```
Change the "wandb_name", "wandb_prefix" in hieros/config.yml to your desired names. \



2. Minimal test (small model size, fewer steps)
```
python hieros/train.py --configs atari100k small_model_size_old --task=atari_pong --steps=400 --eval_every=100 --eval_eps 1 --batch_size=4 --batch_length=16
```

3. Pilot run
```
python hieros/train.py --configs atari100k small_model_size_old --task=atari_pong --steps=10000 --eval_every=2000 --eval_eps 5 --batch_size=8 --batch_length=32
```

4. Full run
```
python hieros/train.py --configs atari100k --task=atari_pong
```





5. Notes
RTX 5070 (16GB):
- Tested up to batch_size=16, batch_length=64 in small_model_size_old config
- Tested up to batch_size=4, batch_length=64 in default config (batch_size=8 causes OOM)
- Tested up to batch_size=8, batch_length=32 in default config (batch_size=16 causes OOM)

