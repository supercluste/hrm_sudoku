# HRM Sudoku Reimplementation

An educational and experimental reimplementation of the **Hierarchical Reasoning Model (HRM)** applied to Sudoku puzzles, investigating how hierarchical recurrent computation (`H_cycles` and `L_cycles`) and Adaptive Computation Time (ACT) affect learning, runtime, and puzzle-solving accuracy.

---

## Architecture Overview

The model (`models/hrm/hrm_act_v1.py`) implements a two-level hierarchical recurrent Transformer architecture with Adaptive Computation Time (ACT):

- **Low-level module (`L_level`)**: 4 Transformer blocks (hidden size 512, 8 heads, SwiGLU, RoPE, post-RMSNorm) performing faster, detailed computation, updated `L_cycles` times per high-level step with additive injection `z_H + input_embeddings`.
- **High-level module (`H_level`)**: 4 Transformer blocks performing slower, abstract reasoning, updated once per `L_cycles` block (`H_cycles` times per ACT step) with additive injection `z_L`.
- **1-Step Gradient Approximation**: All recurrent `(H_cycles, L_cycles)` iterations except the final `L -> H` transition execute inside `torch.no_grad()`, reducing activation memory (~417 MB VRAM at `batch_size=4`).
- **Adaptive Computation Time (ACT)**: A Q-head (`q_halt_logits`, `q_continue_logits`) learns when to halt during training (up to `halt_max_steps = 16`, with `halt_exploration_prob = 0.1`). During evaluation (`model.eval()`), the model runs all `halt_max_steps` iterations to keep batch computation uniform.
- **Loss (`models/losses.py`)**: `ACTLossHead` combines `stablemax_cross_entropy` on all 81 Sudoku cells with binary cross-entropy Q-halting and Q-continue bootstrapping losses.

---

## Dataset

Prepared from Hugging Face `sapientinc/sudoku-extreme` using `datasets/preprocess.py` and verified with `datasets/validate.py`:

| Split | Original Puzzles | Examples | Sequence Length | Vocab Size | Clue Ratio (first 3,200) |
|---|---:|---:|---:|---:|---:|
| Train | 100,000 | 300,000 (`num_aug=2`) | 81 | 11 | 30.99% |
| Test | 422,786 | 422,786 | 81 | 11 | 31.16% |

Token encoding: `0` = padding / ignored label (`ignore_label_id`), `1` = blank input cell, `2–10` = Sudoku digits `1–9`.

---

## Installation & Environment

Tested on Linux / WSL with Python 3.12.16, PyTorch 2.15.0+cu132, and an NVIDIA RTX 5070 Ti Laptop GPU (12 GB VRAM).

```bash
python -m venv .venv
source .venv/bin/activate
# Install PyTorch matching your CUDA version, then:
pip install -r requirements.txt
```

To regenerate or validate the dataset:

```bash
python datasets/preprocess.py --subsample-size 100000 --num-aug 2
python datasets/validate.py
```

---

## Usage

### Training

`train.py` supports configurable cycle counts, warmup, seed, and checkpoint resumption:

```bash
# Baseline (H_cycles=2, L_cycles=2)
python train.py --steps 5000 --batch-size 4 --save-every 5000 --warmup-steps 200 --h-cycles 2 --l-cycles 2 --seed 42 --run-name baseline

# Reduced H-cycles (H_cycles=1, L_cycles=2)
python train.py --steps 5000 --batch-size 4 --save-every 5000 --warmup-steps 200 --h-cycles 1 --l-cycles 2 --seed 42 --run-name reduced_h

# Reduced L-cycles (H_cycles=2, L_cycles=1)
python train.py --steps 5000 --batch-size 4 --save-every 5000 --warmup-steps 200 --h-cycles 2 --l-cycles 1 --seed 42 --run-name reduced_l

# Resume from a checkpoint
python train.py --resume checkpoints/baseline_step_5000.pt --steps 1000 --save-every 500 --warmup-steps 200 --run-name baseline
```

### Evaluation

`evaluate.py` reads the model configuration directly from the checkpoint and evaluates on `data/sudoku` (`test` split):

```bash
python evaluate.py --checkpoint checkpoints/baseline_step_5000.pt --batches 100 --batch-size 32 --output results/baseline.json
python evaluate.py --checkpoint checkpoints/reduced_h_step_5000.pt --batches 100 --batch-size 32 --output results/reduced_h.json
python evaluate.py --checkpoint checkpoints/reduced_l_step_5000.pt --batches 100 --batch-size 32 --output results/reduced_l.json
```

---

## Experimental Configurations & Results

### Shared Settings (Constant Across Experiments)

- **Model**: `hidden_size=512`, `num_heads=8`, `expansion=4`, `H_layers=4`, `L_layers=4`, `pos_encodings="rope"`, `halt_max_steps=16`, `halt_exploration_prob=0.1`, `forward_dtype="bfloat16"`
- **Training**: `steps=5000`, `batch_size=4`, `warmup_steps=200` (linear warmup, then constant LR), `seed=42`
- **Optimizers**: `AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)` + `CastedSparseEmbeddingSignSGD_Distributed(lr=1e-2, weight_decay=0.1)`
- **Evaluation**: 100 deterministic test batches × batch size 32 = **3,200 test puzzles** per configuration

### Comparison Table (5,000 Training Steps)

Results are stored in [`results/comparison.csv`](results/comparison.csv) and individual JSON files under `results/`:

| Experiment | H cycles | L cycles | Train Time | Speed (steps/s) | Cell Accuracy | Exact Puzzle Accuracy | Avg LM Loss | Avg Eval Steps | Eval Time (3,200 puzzles) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | 2 | 2 | 105.7s | 47.34 | 35.33% | 0.00% | 1.9866 | 16.00 | 35.04s |
| **Reduced H** | 1 | 2 | 79.6s | 62.90 | 39.42% | 0.00% | 1.9171 | 16.00 | 18.43s |
| **Reduced L** | 2 | 1 | 88.6s | 56.49 | 38.79% | 0.00% | 1.9904 | 16.00 | 23.96s |

### Key Observations

1. **Compute scaling**: Reducing `H_cycles` from 2 to 1 gave the largest speedup (**+32.9% training throughput**, **1.90x faster evaluation**) because `H_cycles=1` skips the outer `no_grad` H-cycle and half of the L-cycles. Reducing `L_cycles` from 2 to 1 yielded **+19.3% training throughput** and **1.46x faster evaluation**.
2. **In-flight training vs. generalization**: During training with `batch_size=4`, halted sequences reached up to 88.9%–98.8% cell accuracy on the 4 active carry puzzles (which remain in the carry across up to 16 consecutive optimizer updates), showing the model can optimize individual Sudoku grids. However, on unseen test puzzles, all three 5,000-step runs remained at **35.33%–39.42% cell accuracy** and **0.00% exact puzzle accuracy** (note that ~31.16% of test cells are given clues, so copying clues plus uniform random guessing on blanks is ~38.8%).
3. **ACT halting behavior**: Verified in `models/hrm/hrm_act_v1.py`. During evaluation, `self.training` is `False`, so the Q-head early halting branch is disabled and all puzzles execute `halt_max_steps = 16` steps. Diagnostic tracking of `q_halt_logits > q_continue_logits` during eval showed `0.0%` early halts for Baseline and Reduced H and `0.06%` for Reduced L (`Q-halt accuracy = 100.00%`), correctly reflecting that none of the test puzzles were solved.

---

## Limitations

- **Severely under-trained regime**: At `batch_size=4` and `steps=5000` with ~16 ACT steps per puzzle, each run processes only ~1,250–2,000 puzzles total (<1% of the 300,000-example training split) with high-variance batch-4 updates. Full Sudoku generalization requires much larger batch sizes (e.g., 64–768) and substantially more training steps.
- **Exact puzzle solving not achieved**: Because exact puzzle accuracy is `0.00%` across all runs, these experiments demonstrate pipeline correctness, throughput trade-offs, and ACT mechanics rather than full Sudoku solving.
- **Single-seed runs**: Each configuration was run once (`seed=42`).

---

## Deviations from the Original Paper / Source

| Component | Original Paper / Official Source | This Reimplementation |
|---|---|---|
| Attention | FlashAttention (`flash_attn`) | PyTorch `F.scaled_dot_product_attention` (SDPA) |
| Main Optimizer | `AdamATan2` | `torch.optim.AdamW` |
| Dataset Scale | Full dataset | 100,000 subsampled training puzzles (300,000 with `num_aug=2`) |
| Training Scale | Large batch size (e.g., 768) & many epochs | `batch_size=4`, `steps=5000` on local 12 GB GPU |
| Warmup Schedule | `warmup_steps=2000` | Configurable `--warmup-steps` (`200` for 5,000-step runs) |
