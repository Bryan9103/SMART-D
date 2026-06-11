# DRL Final Project: SMART-D

## 1. Setup

From the project root, execute:

```bash
# 1. Clone the Overcooked-AI env into ./overcooked_ai/
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git

# 2. Create a virtualenv at the project root and activate it.
python3.10 -m venv .venv
source .venv/bin/activate          # bash/zsh
# .venv\Scripts\activate           # PowerShell

# or use conda:
# conda create -y --name .venv python=3.10
# conda activate .venv

# 3. Install the env
pip install -e ./overcooked_ai

# If you want a CUDA build of PyTorch, install it from the official index, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install extra dependencies
pip install -r requirements.txt
```

Optional but recommended — `uv` is much faster than pip:
```bash
pip install uv
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ./overcooked_ai
uv pip install -r requirements.txt
```

### Verify the install

```bash
python -c "from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld; \
           import torch; \
           print('overcooked_ai OK, torch', torch.__version__)"
```

If both print without error, you're good. From here on, all commands in this README assume the venv is activated — i.e. `python` resolves to `.venv/bin/python`.

### Adding your own deps (LLM SDKs, etc.)
When you wire in an LLM, add its SDK to `requirements.txt` and reinstall:

```bash
echo "anthropic>=0.40" >> requirements.txt   # or openai, etc.
pip install -r requirements.txt
```

---

## 3. Training

### Standard PPO
The trainer for Standard PPO is `train_ppo.py`

```bash
python train_ppo.py \
    --layout cramped_room \
    --total_steps 2000000 \
    --device cuda
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--layout` | `cramped_room` | Layout name: `cramped_room`, `coordination_ring`, `forced_coordination`
| `--total_steps` | `1000000` | Total env steps |
| `--horizon` | `400` | Steps per episode before truncation |
| `--n_rollouts_per_update` | `8` | Episodes collected per PPO update |
| `--save_interval` | `50` | Save checkpoint every N updates |
| `--run_name` | auto | Subdir of `runs/` to save into |
| `--device` | `cpu` | `cuda` if available |

Checkpoints is stored in `runs/<run_name>/ckpt_<N>.pt` and `runs/<run_name>/final.pt`.


### SMART-D

Initialize the vLLM server:
```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "Qwen/Qwen2.5-1.5B-Instruct" \
    --host 0.0.0.0 \
    --port 8020
```

Wait for the line:
```
INFO:     Application startup complete.
```

For WandB (optional), execute: `wandb login`

Execute training:
```
# training on cramped_room
python -u -m src.trainer configs/SMART-D_cramped_room.yaml --seed 0

# Override seed without editing the config
python -u -m src.trainer configs/SMART-D_cramped_room.yaml --seed 42

# Resume an interrupted run
python -u -m src.trainer configs/SMART-D_cramped_room.yaml --seed 0 --resume runs/<run_dir_name>
```

Console output every 10 iterations:
```
[ 100/1000] steps=320,000  reward=12.34  λ_sd=0.0123  loss=0.0456  ent=0.123
  [eval] return=180.00  soups=9.00  gap=12.50
```
- `reward` — mean shaped reward during rollout
- `λ_sd` — current self-distillation coefficient (starting from second half of training)
- `gap` — eval return with LLM minus eval return without LLM (shrinks down as distillation succeeds)

---

## 4. Evaluating a checkpoint with `eval.py`

`eval.py` loads a `.pt` saved by the trainer, rolls out N episodes, and prints per-episode + aggregate stats. The only positional argument is the checkpoint path; everything else is a flag.

### Usage

```bash
python eval.py <checkpoint.pt> [flags]
```

| Flag | Default | Description |
|---|---|---|
| `checkpoint` (positional) | — | Path to a `.pt` file (e.g. `runs/<run>/final.pt` or `ckpt_N.pt`) |
| `--n_episodes` | `10` | How many episodes to roll out |
| `--layout` | from ckpt | Override the layout choice |
| `--horizon` | from ckpt | Override episode length |
| `--deterministic` | off | Use `argmax` over action logits instead of sampling |
| `--seed` | `0` | NumPy + torch seed |
| `--device` | `cpu` | `cuda` |

### Examples

```bash
# Headline number: self-play, stochastic policy, 20 episodes.
python eval.py \ runs/<run_name>/final.pt --seed 42
```