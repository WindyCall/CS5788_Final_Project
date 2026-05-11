# CS5788 Final Project — Toxicity Mitigation via SFT, DPO, and ORPO

**Team:** Zijing Wu (zw795) · Leyan Wang (lw838) · Yan Xiao (yx689)

## Overview

This project compares three alignment techniques for reducing toxic output in language models under resource-constrained conditions:

- **SFT** (Supervised Fine-Tuning) — baseline, trains only on preferred responses
- **DPO** (Direct Preference Optimization) — trains a policy against a frozen reference model using chosen/rejected pairs
- **ORPO** (Odds-Ratio Preference Optimization) — reference-free; combines SFT loss with an odds-ratio penalty in a single pass

All three methods are applied to **GPT-2 Small (124 M parameters)** trained on the Anthropic HH-RLHF `harmless-base` subset, and evaluated on **RealToxicityPrompts** using `unitary/toxic-bert` as the toxicity classifier.

---

## Results

Evaluated on 1 199 `challenging=True` prompts from RealToxicityPrompts.

| Model | Avg Toxicity Score | P(toxic > 0.5) | Perplexity | Size (MB) | Throughput (tok/s) |
|-------|--------------------|----------------|------------|-----------|-------------------|
| GPT-2 base | 0.5297 | 54.2% | 16.75 | 474.7 | 221.3 |
| SFT | 0.4433 | 45.0% | **13.44** | 237.4 | 5 507.6 |
| **DPO** | **0.2075** | **20.8%** | 16.23 | 237.4 | 5 454.5 |
| ORPO | 0.4529 | 46.2% | 13.26 | 237.4 | 5 471.4 |

**Key findings:**
- DPO reduces average toxicity by **61%** relative to the base model and halves the probability of a toxic output.
- SFT and ORPO achieve similar, modest reductions (~16 pp lower toxicity probability vs. base).
- ORPO did not outperform SFT at 1 epoch, suggesting that its odds-ratio signal may need more training steps or a higher β.
- All fine-tuned checkpoints are stored in fp16, halving disk size relative to the fp32 base (237.4 MB vs. 474.7 MB).

---

## Project Structure

```text
CS5788_final/
├── data/
│   ├── raw/
│   │   └── huggingface/          # HuggingFace download cache
│   └── processed/
│       ├── training/
│       │   └── hh_rlhf/
│       │       ├── hh_rlhf_train.jsonl
│       │       └── hh_rlhf_test.jsonl
│       └── evaluation/
│           └── realtoxicity/
│               ├── realtoxicity_all.jsonl
│               └── realtoxicity_challenging.jsonl
├── models/
│   ├── sft/                      # SFT fine-tuned checkpoint
│   ├── dpo/                      # DPO fine-tuned checkpoint (from SFT)
│   └── orpo/                     # ORPO fine-tuned checkpoint
├── results/
│   ├── summary.json              # Side-by-side metrics for all models
│   ├── base.json                 # Per-sample evaluation for base GPT-2
│   ├── sft.json                  # Per-sample evaluation for SFT
│   ├── dpo.json                  # Per-sample evaluation for DPO
│   ├── orpo.json                 # Per-sample evaluation for ORPO
│   └── training_logs/
│       ├── trainer_state_sft.json
│       ├── trainer_state_dpo.json
│       └── trainer_state_orpo.json
├── scripts/
│   ├── preprocess_hh_rlhf.py
│   ├── preprocess_realtoxicity.py
│   ├── train_sft_raw.py
│   ├── train_dpo_raw.py
│   ├── train_orpo.py
│   └── evaluate_toxicity.py
├── requirements.txt
└── CS5788_Project_Proposal.pdf
```

---

## Dataset Layout

### HH-RLHF (training / validation)

Each line is one JSON object:

```json
{
  "prompt":   "Human: ...\nAssistant:",
  "chosen":   "preferred assistant response",
  "rejected": "dispreferred assistant response"
}
```

- **Training file:** `data/processed/training/hh_rlhf/hh_rlhf_train.jsonl`
- **Validation file:** `data/processed/training/hh_rlhf/hh_rlhf_test.jsonl`
- Used by SFT (prompt + chosen), DPO, and ORPO (prompt + chosen + rejected).

### RealToxicityPrompts (evaluation only)

```json
{
  "filename": "...",
  "begin": 0, "end": 0,
  "challenging": true,
  "prompt": "evaluation prompt text",
  "continuation": "original dataset continuation",
  "prompt_toxicity": 0.0,
  "continuation_toxicity": 0.0
}
```

- Prefer `realtoxicity_challenging.jsonl` (`challenging=True` prompts) for final evaluation.

---

## Installation

```bash
pip install -r requirements.txt
```

Install PyTorch separately with the correct CUDA version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Execution Order

### 1. Preprocess HH-RLHF

Quick sanity check first:

```bash
python scripts/preprocess_hh_rlhf.py --max-train-samples 100 --max-test-samples 20
```

Then run the full job (uses `harmless-base` subset by default, caches to `data/raw/huggingface`):

```bash
python scripts/preprocess_hh_rlhf.py
```

Optional flags:

```bash
python scripts/preprocess_hh_rlhf.py --data-dir helpful-base   # different subset
python scripts/preprocess_hh_rlhf.py --cache-dir /path/to/cache
```

### 2. Preprocess RealToxicityPrompts

Quick sanity check:

```bash
python scripts/preprocess_realtoxicity.py --max-samples 100
```

Full run:

```bash
python scripts/preprocess_realtoxicity.py
```

### 3. Train SFT

Custom PyTorch loop (no TRL). Outputs to `models/sft`.

```bash
python scripts/train_sft_raw.py \
  --model-name gpt2 \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-5 \
  --max-length 512 \
  --max-prompt-length 256
```

Add `--fp16` if running on GPU with fp16 support.

Key defaults:

| Argument | Default |
|----------|---------|
| `--model-name` | `gpt2` |
| `--train-file` | `data/processed/training/hh_rlhf/hh_rlhf_train.jsonl` |
| `--output-dir` | `models/sft` |
| `--epochs` | 1 |
| `--lr` | 2e-5 |
| `--batch-size` | 4 |
| `--grad-accum` | 4 |

### 4. Train DPO

Custom PyTorch loop with a frozen reference model. Starts from the SFT checkpoint.

```bash
python scripts/train_dpo_raw.py \
  --model-name models/sft \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-5 \
  --beta 0.1
```

Key defaults:

| Argument | Default |
|----------|---------|
| `--model-name` | `models/sft` |
| `--ref-model-name` | same as `--model-name` |
| `--output-dir` | `models/dpo` |
| `--beta` | 0.1 |
| `--lr` | 1e-5 |

### 5. Train ORPO

Uses HuggingFace `Trainer`. Starts from base GPT-2 (no reference model needed).

```bash
python scripts/train_orpo.py \
  --model-name gpt2 \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 1e-5 \
  --beta 0.1
```

Key defaults:

| Argument | Default |
|----------|---------|
| `--model-name` | `gpt2` |
| `--output-dir` | `models/orpo` |
| `--beta` | 0.1 (odds-ratio penalty weight λ) |
| `--lr` | 1e-5 |

### 6. Evaluate Toxicity

Run once per model. Results accumulate in `results/summary.json`.

```bash
# Base model
python scripts/evaluate_toxicity.py \
  --model-path gpt2 \
  --model-label base

# SFT
python scripts/evaluate_toxicity.py \
  --model-path models/sft \
  --model-label sft

# DPO
python scripts/evaluate_toxicity.py \
  --model-path models/dpo \
  --model-label dpo

# ORPO
python scripts/evaluate_toxicity.py \
  --model-path models/orpo \
  --model-label orpo
```

Key flags:

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompts-file` | `realtoxicity_challenging.jsonl` | Evaluation prompts |
| `--max-new-tokens` | 50 | Continuation length |
| `--batch-size` | 16 | Generation and scoring batch size |
| `--no-perplexity` | off | Skip HH-RLHF perplexity computation |
| `--max-samples` | None | Cap prompts (for quick debugging) |
| `--fp16` | off | Load model in fp16 |

Each run saves per-sample results to `results/<model-label>.json` and prints a side-by-side comparison table once the summary file has multiple entries.

---

## Methods

### SFT

Standard teacher-forcing on the preferred (`chosen`) response. Prompt tokens are masked out (label = −100) so loss is computed only on the response. Uses a cosine learning rate schedule with a 0.1× floor.

### DPO

Trains a policy π against a frozen reference π_ref. The loss is:

```
L_DPO = -E[log σ(β · ((log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))))]
```

where y_w is the chosen response and y_l the rejected response. β = 0.1 controls the penalty strength. Both policy and reference share the same starting checkpoint (SFT).

### ORPO

Reference-free. Combines SFT cross-entropy with an odds-ratio penalty in a single forward pass:

```
L_ORPO = L_SFT + λ · L_OR
L_OR   = -E[log σ(log_odds(π(y_w|x)) - log_odds(π(y_l|x)))]
```

where log_odds(p) = log(p / (1 − p)). No frozen reference model is loaded, halving GPU memory for preference learning. λ = β = 0.1.

---

## Training Details

All runs: 1 epoch · 2 656 optimizer steps · cosine LR schedule · AdamW (β₁=0.9, β₂=0.999) · gradient clipping at 1.0

| Setting | SFT | DPO | ORPO |
|---------|-----|-----|------|
| Starting model | GPT-2 base | SFT checkpoint | GPT-2 base |
| Learning rate | 2e-5 | 1e-5 | 1e-5 |
| Batch size / device | 4 | 2 | 4 |
| Gradient accumulation | 4 | 8 | 4 |
| Effective batch | 16 | 16 | 16 |
| Max sequence length | 512 | 512 | 512 |
| Eval loss (end of epoch) | 2.164 | 0.588 | 2.383 |

---

## References

- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
- Hong et al. (2024). *ORPO: Monolithic Preference Optimization without Reference Model.* arXiv:2403.07691.
- Bai et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* (HH-RLHF dataset)
- Gehman et al. (2020). *RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models.* EMNLP Findings.
- Detoxify / `unitary/toxic-bert` — toxicity classifier used for evaluation.
