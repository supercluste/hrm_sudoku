# Recreating the Hierarchical Reasoning Model (HRM) for Sudoku

An implementation project exploring the **Hierarchical Reasoning Model (HRM)** and its application to structured reasoning through Sudoku.

The objective is to understand and recreate the important components of HRM, prepare a suitable Sudoku dataset, and investigate how hierarchical recurrent computation (`H_cycles` and `L_cycles`) and adaptive computation time (ACT) affect learning, runtime, and puzzle-solving accuracy.

> **Project status:** Core architecture, dataset pipeline, configurable training/resume, ACT-verified evaluation, controlled cycle ablations, and a larger-batch scaling experiment are complete. All experimental results, limitations, and deviations from the official setup are documented below.

---

## 1. Introduction

Traditional neural networks typically process information through a sequence of layers. However, complex reasoning problems may require multiple rounds of computation and refinement.

The Hierarchical Reasoning Model (HRM) explores a different approach.

HRM uses two interacting recurrent modules:

- **High-Level Module (H):** Maintains higher-level representations and coordinates slower, abstract reasoning.
- **Low-Level Module (L):** Performs detailed computations and refines representations at a faster timescale.

These modules interact repeatedly across recurrent cycles (`H_cycles` and `L_cycles`), using a 1-step gradient approximation where earlier iterations run under `torch.no_grad()` and only the final `L -> H` transition backpropagates gradients.

HRM also incorporates **Adaptive Computation Time (ACT)** via a Q-learning head (`q_halt_logits`, `q_continue_logits`), allowing the model to learn when to halt during training (with exploration `halt_exploration_prob = 0.1`), while running all `halt_max_steps = 16` iterations during evaluation for batch uniformity.

## 2. Why Sudoku?

Sudoku provides a useful environment for investigating structured reasoning.

A standard Sudoku puzzle consists of a 9 × 9 grid with three fundamental constraints:

1. Every row must contain digits 1–9 exactly once.
2. Every column must contain digits 1–9 exactly once.
3. Every 3 × 3 subgrid must contain digits 1–9 exactly once.

The model receives an incomplete Sudoku puzzle and predicts the completed solution. Satisfying all constraints simultaneously requires learning global relationships across rows, columns, and 3 × 3 subgrids.

## 3. Architecture

The active model is implemented in `models/hrm/hrm_act_v1.py`.

### Hierarchical reasoning

| Component | Baseline Configuration |
|---|---:|
| High-level cycles (`H_cycles`) | 2 |
| Low-level cycles (`L_cycles`) | 2 |
| High-level transformer layers (`H_layers`) | 4 |
| Low-level transformer layers (`L_layers`) | 4 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Expansion factor (SwiGLU) | 4 |
| Positional encoding | RoPE |
| Forward precision | `bfloat16` |

### Adaptive Computation Time

- Maximum reasoning steps (`halt_max_steps`): `16`
- Halt exploration probability (`halt_exploration_prob`): `0.1`
- Training loss (`models/losses.py`): `ACTLossHead` combining `stablemax_cross_entropy` on all 81 cells with binary cross-entropy Q-halt and Q-continue bootstrapping losses.

### Input and output representation

Each Sudoku puzzle is represented as a sequence of 81 tokens (`vocab_size = 11`):

| Token | Meaning |
|---|---|
| 0 | Padding / ignored label (`ignore_label_id`) |
| 1 | Empty input cell |
| 2–10 | Sudoku digits 1–9 |

## 4. Dataset

The project uses the [Sudoku Extreme dataset](https://huggingface.co/datasets/sapientinc/sudoku-extreme) from Hugging Face, preprocessed using NumPy memory-mapped arrays (`datasets/preprocess.py`) and verified with `datasets/validate.py`.

| Property | Value |
|---|---:|
| Original training puzzles selected | 100,000 |
| Augmentations per puzzle (`--num-aug`) | 2 |
| Examples per puzzle group | 3 |
| Total training examples | 300,000 |
| Test puzzles | 422,786 |
| Train clue ratio (first 3,200) | 30.99% |
| Test clue ratio (first 3,200) | 31.16% |

The active loader (`puzzle_dataset.py`) samples one variant from each puzzle group per iteration during training, and iterates sequentially in `test_set_mode=True` during evaluation.

## 5. Implementation & Deviations from Original Source

Important components include:

- HRM architecture and recurrent computation (`models/hrm/hrm_act_v1.py`)
- Transformer blocks, RoPE, RMSNorm, and SwiGLU (`models/layers.py`)
- Sparse puzzle embeddings and `CastedSparseEmbeddingSignSGD_Distributed` (`models/sparse_embedding.py`)
- `ACTLossHead` with `stablemax_cross_entropy` (`models/losses.py`)
- Official-style `PuzzleDataset` loader (`puzzle_dataset.py`)
- Memory-mapped preprocessing and Sudoku validator (`datasets/preprocess.py`, `datasets/validate.py`)

### Documented deviations from the original paper / source

| Component | Original Paper / Official Source | This Reimplementation |
|---|---|---|
| Attention | FlashAttention (`flash_attn`) | PyTorch `F.scaled_dot_product_attention` (SDPA) |
| Main optimizer | `AdamATan2` | `torch.optim.AdamW` (`lr=1e-4`, `betas=(0.9, 0.95)`, `wd=0.1`) |
| Dataset scale | Full dataset | 100,000 subsampled training puzzles (300,000 with `num_aug=2`) |
| Training scale | Large batch size (e.g., 768) & many epochs | Controlled `batch_size=4` (5,000 steps) + `batch_size=128` (1,500 steps) on 12 GB GPU |
| Warmup schedule | `warmup_steps=2000` | Configurable via `--warmup-steps` (`200` in experiments) |

## 6. Repository Structure

```text
HRM_Reimplementation/
│
├── data/sudoku/                 # Preprocessed train/test splits (.npy + dataset.json)
├── dataset/                     # Metadata schema & dihedral helpers
├── datasets/                    # Preprocessing & Sudoku validation scripts
├── docs/                        # Paper PDF and project handoff document
├── models/
│   ├── common.py
│   ├── layers.py
│   ├── losses.py
│   ├── sparse_embedding.py
│   └── hrm/hrm_act_v1.py        # Active HRM + ACT implementation
├── results/                     # Saved JSON & CSV experimental results
├── checkpoints/                 # Saved model checkpoints (.pt, gitignored)
├── puzzle_dataset.py
├── train.py
├── evaluate.py
├── requirements.txt
└── test_*.py
```

## 7. Environment

| Component | Version / Specification |
|---|---|
| Operating system | Linux |
| Python | 3.12.16 |
| PyTorch | 2.15.0+cu132 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM) |
| CUDA availability | Enabled |
| Precision | BF16 |

## 8. Getting Started

### Clone & setup environment

```bash
git clone https://github.com/supercluste/hrm_sudoku.git
cd hrm_sudoku
python3.12 -m venv .venv
source .venv/bin/activate
# Install CUDA-compatible PyTorch first, then:
pip install -r requirements.txt
```

### Prepare & validate dataset

```bash
python datasets/preprocess.py --subsample-size 100000 --num-aug 2
python datasets/validate.py
```

### Verify components

```bash
python test_official_hrm.py
python test_official_dataset.py
```

### Train

`train.py` supports configurable recurrent cycles, warmup, seed, run naming, and checkpoint resumption:

```bash
# Baseline (H_cycles=2, L_cycles=2)
python train.py --steps 5000 --batch-size 4 --save-every 5000 --warmup-steps 200 --h-cycles 2 --l-cycles 2 --seed 42 --run-name baseline

# Reduced H-cycles (H_cycles=1, L_cycles=2)
python train.py --steps 5000 --batch-size 4 --save-every 5000 --warmup-steps 200 --h-cycles 1 --l-cycles 2 --seed 42 --run-name reduced_h

# Reduced L-cycles (H_cycles=2, L_cycles=1)
python train.py --steps 5000 --batch-size 4 --save-every 5000 --warmup-steps 200 --h-cycles 2 --l-cycles 1 --seed 42 --run-name reduced_l

# Larger batch size (H_cycles=2, L_cycles=2, batch_size=128)
python train.py --steps 1500 --batch-size 128 --save-every 500 --warmup-steps 200 --h-cycles 2 --l-cycles 2 --seed 42 --run-name baseline_bs128

# Resume from a checkpoint
python train.py --resume checkpoints/baseline_step_5000.pt --steps 1000 --save-every 500 --warmup-steps 200 --run-name baseline
```

### Evaluate

`evaluate.py` loads the architecture configuration directly from the checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/baseline_step_5000.pt --batches 100 --batch-size 32 --output results/baseline.json
python evaluate.py --checkpoint checkpoints/reduced_h_step_5000.pt --batches 100 --batch-size 32 --output results/reduced_h.json
python evaluate.py --checkpoint checkpoints/reduced_l_step_5000.pt --batches 100 --batch-size 32 --output results/reduced_l.json
python evaluate.py --checkpoint checkpoints/baseline_bs128_step_1500.pt --batches 100 --batch-size 32 --output results/baseline_bs128.json
```

## 9. Experimental Results

All evaluations below were run on the same deterministic subset of **3,200 unseen test puzzles** (`--batches 100 --batch-size 32`) with `warmup_steps=200` and `seed=42`. Raw outputs are stored in [`results/comparison.csv`](results/comparison.csv) and `results/*.json`.

### Part A: Controlled Cycle Ablations (`batch_size=4`, `steps=5000`, `417.3 MB` VRAM)

| Experiment | H cycles | L cycles | Train Time | Train Speed | Test Cell Acc | Exact Puzzle Acc | Avg LM Loss | Avg Eval Steps | Eval Time (3,200) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | 2 | 2 | 105.7s | 47.34 steps/s | 35.33% | 0.00% | 1.9866 | 16.00 | 35.04s |
| **Reduced H** | 1 | 2 | 79.6s | 62.90 steps/s | 39.42% | 0.00% | 1.9171 | 16.00 | 18.43s |
| **Reduced L** | 2 | 1 | 88.6s | 56.49 steps/s | 38.79% | 0.00% | 1.9904 | 16.00 | 23.96s |

### Part B: Batch-Size Scaling Progression (`H=2, L=2`, `batch_size=128`, `445.9 MB` VRAM)

| Checkpoint | Training Steps | Train Time | Test Cell Acc | Exact Puzzle Acc | Avg LM Loss | Avg Eval Steps |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_bs128_step_500.pt` | 500 | — | 47.56% | 0.00% | 1.2572 | 16.00 |
| `baseline_bs128_step_1000.pt` | 1000 | — | 49.97% | 0.00% | 1.1769 | 16.00 |
| `baseline_bs128_step_1500.pt` | 1500 | 402.7s (6.7m) | **52.13%** | 0.00% | **1.1455** | 16.00 |

### Key Findings & Limitations

1. **Cycle count vs. compute speed:** Reducing `H_cycles` (`2 -> 1`) boosted training throughput by **+32.9%** and sped up evaluation by **1.90×** (`35.04s -> 18.43s`). Reducing `L_cycles` (`2 -> 1`) boosted training throughput by **+19.3%** and evaluation speed by **1.46×**.
2. **Batch size is critical for generalization:** At `batch_size=4`, active puzzles stay in the ACT carry for up to 16 consecutive optimizer steps, causing the model to memorize the 4 active training puzzles (reaching 88.9%–98.8% in-flight training cell accuracy) without generalizing to test puzzles (`35.33%–39.42%`, near the ~38.8% clue-copy + random-blank baseline). Increasing `batch_size` to `128` (using only `445.9 MB` VRAM thanks to 1-step grad + BF16) reduced test LM loss from `1.9866` to `1.1455` and raised test cell accuracy monotonically to `52.13%` in 1,500 steps.
3. **ACT evaluation behavior:** Verified in `models/hrm/hrm_act_v1.py`: in `eval()` mode, early halting is disabled (`halt_max_steps = 16` always executes). Diagnostic Q-head tracking showed `Q-halt accuracy = 100.00%` and `0.0%–0.06%` early-halt predictions, consistent with `0.00%` exact puzzle accuracy.
4. **Remaining gap:** Exact puzzle solving (`>0%` exact puzzle accuracy) was not yet reached within 1,500 steps at `batch_size=128` and requires longer training runs.

## 10. Current Progress

| Component | Status |
|---|---|
| Official HRM architecture & SDPA adaptation | Completed & verified |
| Dataset preprocessing (300K train / 422K test) & validation | Completed & verified |
| Configurable training (`--warmup-steps`, cycles, `--resume`) | Completed & verified |
| Checkpoint-driven ACT evaluation & JSON/CSV export | Completed & verified |
| Controlled cycle ablations (`H=2/L=2`, `H=1/L=2`, `H=2/L=1`) | Completed (3,200 test puzzles) |
| Larger-batch scaling experiment (`batch_size=128`, 1,500 steps) | Completed (52.13% cell acc) |
| Extended multi-epoch training for exact puzzle accuracy | Future work |
| `AdamATan2` optimizer parity | Future work |

## 11. References

- **Original HRM paper:** [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
- **Original implementation:** [Sapient AI Lab – HRM](https://github.com/sapientinc/HRM)
- **Dataset:** [Sudoku Extreme – Hugging Face](https://huggingface.co/datasets/sapientinc/sudoku-extreme)

---

## Disclaimer

This repository is an educational reimplementation and experimentation project based on the HRM architecture. It is not an official release of HRM, and contains documented implementation substitutions (`PyTorch SDPA` instead of FlashAttention, `AdamW` instead of `AdamATan2`, and subsampled training). Results should not be considered a reproduction of the original paper's reported performance.
