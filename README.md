# RLHF for Open-Ended Instruction Following

**Authors:** Vrushank Prakash, Sarveshan Saravanan, Krish Yadav  
**Institution:** UC Berkeley (CS 185/285A)  
**Paper:** [Final Report (PDF)](RLHF_Final_Report.pdf)

---

## Overview

We implement and evaluate a suite of RLHF methods for aligning a small open-weight LLM
(Qwen2.5-1.5B-Instruct) on the WildChat-5k benchmark — an open-ended instruction-following
task where models must produce helpful, stylistically consistent responses across diverse user prompts.

The project covers both offline preference optimization and online policy gradient methods,
culminating in a novel **Rank-based Leave-One-Out (LOO) advantage baseline** for GRPO that
mitigates reward model hacking and achieves an **82.4% win rate** against the frozen base model.

---

## Key Results

| Method | Win Rate vs. Base |
|---|---|
| DPO | 78.2% |
| IPO | 64.7% |
| AOT | 72.2% |
| GRPO | 78.7% |
| DrGRPO | 73.1% |
| GSPO | 65.4% |
| **Rank-LOO (Ours)** | **82.4%** |
| Standard LOO | 77.8% |

**Evaluation:** GPT-5.4 as judge on 128-prompt held-out test set (head-to-head win rate vs. frozen base model).  
**Reward Model:** Bradley-Terry objective, ~85% held-out pair accuracy at step 445.

---

## Methods

### Offline Preference Optimization

All offline methods optimize a reference-corrected preference margin:

```
Δθ(x, y+, y−) = [log πθ(y+|x) − log πθ(y−|x)] − [log πref(y+|x) − log πref(y−|x)]
```

**DPO** — Logistic loss on the preference margin. Achieved best offline win rate (78.2%) by
capturing the structural and stylistic nuances of reference responses.

**IPO** — Squared loss matching the margin to a fixed target, preventing indefinite margin growth
and reducing reward over-optimization.

**AOT** — Distributional alignment across the minibatch: sorts chosen and rejected reward
distributions and aligns their quantiles, encouraging the entire chosen distribution to shift upward
rather than optimizing pair-by-pair.

### Reward Model

Trained with Bradley-Terry objective on chosen/rejected preference pairs:

```
L_RM = −log σ(r_ϕ(x, y+) − r_ϕ(x, y−))
```

Reached 85% held-out pair accuracy. Mean score margin between chosen and rejected responses
grew steadily throughout training, confirming reliable preference separation.

### Online Methods

**GRPO** — PPO-style clipped surrogate at the token level with group-relative advantage normalization.
Best baseline online method (78.7%).

**DrGRPO** — Removes per-group std normalization and per-sequence length normalization from
GRPO, useful as a comparison point for evaluating length bias.

**GSPO** — Clips at the sequence level rather than token level, treating the full response as a single
structured action.

### Rank-based LOO Baseline (Novel Contribution)

Standard GRPO suffered from reward hacking: the reward model favored long, sycophantic responses,
and absolute reward scores had high variance from outliers that destabilized policy gradients.

Our fix: replace absolute reward scores with normalized intra-group ranks before computing advantages.

For a group of G samples, we normalize each reward to rank:

```
r_norm,i = (2 · rank_i / (G − 1)) − 1          # maps ranks to [−1, 1]
```

Then compute the LOO baseline (average of all other ranks) and the advantage:

```
b_i = Σ_{j≠i} r_norm,j / (G − 1)
A_i = (r_norm,i − b_i) / (σ_norm + ε)
```

**Why it works:**
- Rank normalization bounds advantage values, preventing gradient explosions from outlier rewards
- LOO structure ensures no sample pulls its own baseline down, reducing bias
- Together they provide a stable, robust advantage signal that resists reward model overconfidence

**Result:** 82.4% win rate — outperforming GRPO (78.7%) and all other baselines.

**Why Standard LOO failed:** Using LOO structure on raw reward scores (without rank normalization)
still exposes the model to absolute reward scale outliers. Win rate dropped to 77.8%, *worse* than
base GRPO. This confirms rank normalization, not just the LOO structure, drives the improvement.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-rlhf.git
cd llm-rlhf
uv run modal token new
uvx wandb login
export OPENAI_API_KEY=...   # for GPT-5.4 judge evaluation
```

---

## Training

**Reward Model**
```bash
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --lr 3e-5 --num_train_epochs 3
```

**Offline (DPO)**
```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo dpo --beta 0.1 --lr 5e-5 --num_train_epochs 3
```

**Online (GRPO with Rank-LOO)**
```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo rank_loo_grpo --lr 1e-5 --steps 100 \
  --group_size 4 --clip_eps 0.2 --kl_coef 0.01
```

---

## Hyperparameters

| Category | Key Parameters |
|---|---|
| Offline | lr=5e-5, batch=4, epochs=3, β=0.1 (DPO/IPO), β=0.2 (AOT) |
| Reward Model | lr=3e-5, batch=8, epochs=3 |
| Online | lr=1e-5, batch=16, group_size=4, ε=0.2, λ_KL=0.01, steps=25–100 |

---

## Dataset

`wildchat_min4_judged_5k_v1`: 4,744 training / 256 test preference pairs from filtered WildChat
prompts with LLM-ranked model-generated responses.

- `train_prefs`: chosen/rejected pairs for offline optimization and reward model training
- `train_gen`: prompt-only split for online rollouts
- `test_gen`: held-out prompts for GPT-5.4 judge evaluation

---

## Citation

```bibtex
@misc{prakash2026rlhf,
  title={RLHF for Open-Ended Instruction Following},
  author={Prakash, Vrushank and Saravanan, Sarveshan and Yadav, Krish},
  year={2026},
  institution={UC Berkeley}
}
```
