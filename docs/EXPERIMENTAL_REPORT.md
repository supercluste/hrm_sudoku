# HRM Sudoku Reimplementation — Experimental Findings & Evaluation Report

**Date:** 2026-11-26  
**Repository:** `https://github.com/supercluste/hrm_sudoku`  
**Hardware:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM), Python 3.12.16, PyTorch 2.15.0+cu132 (`bfloat16`)  
**Dataset:** `sapientinc/sudoku-extreme` subsampled to 100,000 training puzzles (`300,000` examples with `num_aug=2`) and `422,786` test puzzles (`data/sudoku`)

---

## 1. Executive Summary

We implemented, debugged, benchmarked, and scaled a Hierarchical Reasoning Model (HRM) with Adaptive Computation Time (ACT) to solve 9×9 Sudoku puzzles. Across **42,000 training steps** (`baseline_bs128`, `H_cycles=2, L_cycles=2`, `batch_size=128`), the model progressed from random guessing to **solving 10.10% of unseen test puzzles completely** (**~1,012 out of 10,016 puzzles**; **69.46% cell accuracy**; **0.6858 average LM loss**), including **98.96% exact puzzle accuracy (95/96)** on puzzles with `>= 29` clues and **5.92%–8.91%** on harder `23–28` clue puzzles.

Key engineering and scientific findings:
1. **Batch-size scaling unlocked generalization:** At `batch_size=4` (`5,000` steps), the ACT carry retained active puzzles across up to 16 consecutive updates, causing 4-puzzle memorization (`88.9%–98.8%` in-flight training accuracy) with `0.00%` test exact accuracy (`35.33%–39.42%` cell accuracy). Scaling to `batch_size=128` (`441–446 MB` VRAM) broke the memorization bottleneck and enabled steady generalization.
2. **High-level cycles (`H_cycles`) drive exact puzzle solving:** In controlled 5,000-step ablations at `batch_size=128`, halving `H_cycles` (`2 → 1`) cut exact accuracy by more than half (`0.97% → 0.47%`) and worsened LM loss (`0.9462 → 1.4101`), whereas halving `L_cycles` (`2 → 1`) preserved solving (`1.25%`) at higher speed (`4.91` vs. `~3.6 steps/s`).
3. **Iterative ACT reasoning produces >100× exact-solving gains across steps `1 → 16`:** At step 42,000, exact puzzle accuracy grew monotonically from **0.09% at ACT Step 1** to **9.72% at ACT Step 16** on 3,200 unseen test puzzles (`Blank cell accuracy: 47.47% → 55.36%`).
4. **ACT Q-head learns calibrated halting confidence:** Mean `q_halt_logits - q_continue_logits` separated solved puzzles (**`+0.3442`**, predicting halt) from unsolved puzzles (**`-7.4012`**, predicting continue), reaching **99.57% Q-halt accuracy** on 10,016 test puzzles.

---

## 2. Pipeline Fixes, Official Code Comparison & Improvements

Comparing our codebase against the official paper repository (`HRM/`) revealed exact parity in the core model (`models/hrm/hrm_act_v1.py`), loss head (`models/losses.py`), sparse embeddings (`models/sparse_embedding.py`), and dataset loader (`puzzle_dataset.py`), alongside critical training-script gaps that we fixed:

1. **Configurable Warmup & Cycles (`train.py`):** Replaced hardcoded `warmup_steps=2000` (which prevented LR ramp-up in short runs) and hardcoded `H_cycles`/`L_cycles` with CLI flags `--warmup-steps`, `--h-cycles`, `--l-cycles`, `--halt-max-steps`, `--pos-encodings`, `--loss-type`, `--seed`, `--run-name`, and `--resume`.
2. **Checkpoint-Driven Evaluation (`evaluate.py`):** Replaced hardcoded baseline config in `evaluate.py` with automatic loading of `checkpoint["config"]` and `loss_type`, plus JSON export (`--output`), `q_halt_accuracy`, and diagnostic tracking of when `q_halt_logits > q_continue_logits` triggers during the 16-step eval loop.
3. **Fresh Data Ordering on `--resume`:** Identified that `PuzzleDataset._iter_train()` resets `_iters=0` on restart; used distinct `--seed` values (`42`, `1042`, `2042`, `3042`) across resumed segments so the model continued sampling fresh puzzles across the 100,000-puzzle dataset.
4. **Official Paper Optimizer & Hyperparameter Insights (`HRM/`):**
   - Implemented pure-PyTorch `AdamATan2` ([models/adam_atan2.py](file:///home/shadowspace/HRM_Reimplementation/models/adam_atan2.py)) with `--optimizer {adamw,adam_atan2}` and cosine LR scheduling (`--lr-min-ratio`).
   - Added `--compile` support (probed `torch.compile(dynamic=False)`; unavailable on this host due to missing `Python.h` / `python3.12-dev` headers).
   - Identified that `puzzle_emb_lr=1e-2` (ARC-AGI default) caused the single shared `puzzle_emb.weights[0]` vector to reach norm `4.295` at step 10,000, motivating configurable `--puzzle-emb-lr` (`1e-4` / `5e-5` in late fine-tuning).

### Documented Deviations from Official Paper

| Component | Official Paper / Source | This Reimplementation |
|---|---|---|
| Attention kernel | FlashAttention (`flash_attn`) | PyTorch `F.scaled_dot_product_attention` (SDPA) |
| Primary optimizer (main runs) | `AdamATan2` | `torch.optim.AdamW` (pure-PyTorch `AdamATan2` also added and probed) |
| Compilation | `torch.compile(model, dynamic=False)` | Eager PyTorch (missing `Python.h` on host) |
| Dataset preparation | 1,000 × 1,000 aug (laptop demo) or 3.83M full | 100,000 subsampled puzzles × 3 (`300,000` train examples) |
| Compute budget | ~10h (`bs=384`, 20k epochs) or 8× GPU cluster | Single RTX 5070 Ti Laptop GPU (`bs=4` and `bs=128`, up to 42,000 steps) |

---

## 3. Experimental Results

All evaluations below (except the final 10,016-puzzle confirmation) were measured on the identical deterministic subset of **3,200 unseen test puzzles** (`--batches 100 --batch-size 32`).

### 3.1 Batch-Size 4 vs. Batch-Size 128 Architectural Ablations (5,000 Steps)

| Experiment | Batch Size | `H` | `L` | Train Time | Speed (`steps/s`) | Cell Acc | Exact Acc | Avg LM Loss | Eval Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 4 | 2 | 2 | 105.7s | 47.34 | 35.33% | 0.00% | 1.9866 | 35.04s |
| `reduced_h` | 4 | 1 | 2 | 79.6s | 62.90 | 39.42% | 0.00% | 1.9171 | 18.43s |
| `reduced_l` | 4 | 2 | 1 | 88.6s | 56.49 | 38.79% | 0.00% | 1.9904 | 23.96s |
| **`baseline_bs128`** | 128 | 2 | 2 | 23.1m (`1383.4s`) | ~3.6 | **60.58%** | 0.97% | **0.9462** | 34.68s |
| **`reduced_h_bs128`** | 128 | 1 | 2 | 14.2m (`854.2s`) | 5.85 | 58.19% | 0.47% | 1.4101 | 17.65s |
| **`reduced_l_bs128`** | 128 | 2 | 1 | 17.0m (`1018.1s`) | 4.91 | 59.34% | **1.25%** | 1.3897 | 24.89s |

*(Note: In `data/sudoku/test`, `31.16%` of cells are given clues; copying clues plus uniform random guessing over blanks scores `~38.8%` cell accuracy.)*

**Observations:**
- At `batch_size=4`, 5,000 steps processed too few unique puzzles (~1,250–2,000) with high gradient noise, keeping all models near the random-blank baseline (`35.33%–39.42%`).
- At `batch_size=128`, reducing `H_cycles` (`2 → 1`) sped up training by ~38% (`1383.4s → 854.2s`) and evaluation by **1.96×** (`34.68s → 17.65s`), at the cost of halving exact accuracy (`0.97% → 0.47%`) and raising LM loss (`0.9462 → 1.4101`).
- Reducing `L_cycles` (`2 → 1`) preserved `H_cycles=2` and achieved `1.25%` exact accuracy at step 5,000 with 1.39× faster evaluation (`24.89s`), though with higher LM loss (`1.3897` vs. `0.9462`) and lower cell accuracy (`59.34%` vs. `60.58%`).

### 3.2 Alternative Paper Hyperparameter Probes (`batch_size=128`, 1,500–10,000 Steps)

| Run Name | Key Config Differences | Steps | Train Time | Cell Acc | Exact Acc | Avg LM Loss |
|---|---|---:|---:|---:|---:|---:|
| `baseline_bs128_pe1e4` | `puzzle_emb_lr=1e-4` (vs `1e-2`) | 5,000 | — | 57.23% | 0.16% | 1.2960 |
| `baseline_bs128_pe1e4` | `puzzle_emb_lr=1e-4` (vs `1e-2`) | 10,000 | 44.7m (`2683.2s`) | 60.67% | 2.09% | 1.1160 |
| `sudoku_hard_l8` | `L=8, halt=8, learned, softmax, lr=3e-4, adamw` | 1,500 | 12.6m (`756.9s`) | 40.77% | 0.00% | 1.4995 |
| `hrm_atan2_learned` | `L=4, halt=8, learned, stablemax, lr=1e-4, adam_atan2` | 1,500 | 7.9m (`471.5s`) | 40.96% | 0.00% | 1.5075 |

**Takeaway:** `lr=3e-4` with `AdamW` at `batch_size=128` (`sudoku_hard_l8`) collapsed around step 300–400 to the uniform-blank plateau (`1.50` LM loss), whereas `learned` positional embeddings (`hrm_atan2_learned`) learned significantly more slowly in the first 1,500 steps than `rope`, which provides immediate relative geometry from step 1.

### 3.3 `baseline_bs128` Learning Curve (Steps 500 → 42,000)

Schedule: `lr=1e-4` (`1–25,000`), `puzzle_emb_lr=1e-4` (`10,001–25,000`), `lr=5e-5, puzzle_emb_lr=5e-5` (`25,001–37,000`), cosine decay `lr_min_ratio=0.1` (`37,001–42,000`, `LR 6.57e-6 → 5.00e-6`). Peak GPU memory: `441.2–445.9 MB`.

| Step | Test Puzzles | Cell Acc | Exact Puzzle Acc | Avg LM Loss | Q-Halt Acc | Early Halt % |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 3,200 | 47.56% | 0.00% | 1.2572 | 100.00% | 0.0% |
| 1,000 | 3,200 | 49.97% | 0.00% | 1.1769 | 100.00% | 0.0% |
| 1,500 | 3,200 | 52.13% | 0.00% | 1.1455 | 100.00% | 0.0% |
| 2,000 | 3,200 | 55.84% | 0.00% | 1.0073 | 100.00% | 0.0% |
| 3,000 | 3,200 | 58.55% | 0.00% | 1.0655 | 100.00% | 0.0% |
| 4,000 | 3,200 | 59.28% | 1.19% | 1.3062 | 96.88% | 2.4% |
| 5,000 | 3,200 | 60.58% | 0.97% | 0.9462 | 99.06% | 0.0% |
| 7,000 | 3,200 | 61.35% | 1.06% | 0.8686 | 98.94% | 0.0% |
| 9,000 | 3,200 | 61.31% | 2.00% | 1.0688 | 98.59% | 0.6% |
| 10,000 | 3,200 | 61.61% | 2.41% | 0.9321 | 99.50% | 0.38% |
| 15,000 | 3,200 | 62.31% | 2.81% | 1.1037 | 95.16% | 2.7% |
| 20,000 | 3,200 | 63.66% | 2.66% | 0.8542 | 99.28% | 0.3% |
| 25,000 | 3,200 | 64.08% | 3.22% | 0.9074 | 98.66% | 3.2% |
| 30,000 | 3,200 | 66.04% | 3.12% | 0.7598 | 99.53% | 3.1% |
| 33,000 | 3,200 | 66.62% | 3.81% | 0.7442 | 99.53% | 1.6% |
| 36,000 | 3,200 | 66.76% | 3.09% | 0.7317 | 99.94% | 0.0% |
| 37,000 | 3,200 | 67.27% | 5.84% | 0.7586 | 99.06% | 3.0% |
| **42,000** | **3,200** | **69.27%** | **9.72%** (`311/3200`) | **0.6917** | **99.47%** | **9.06%** |
| **42,000** | **10,016** | **69.46%** | **10.10%** (~1,012 solved) | **0.6858** | **99.57%** | **9.27%** |

---

## 4. Deep Diagnostics on `baseline_bs128_step_42000.pt` (3,200 Puzzles)

### 4.1 Iterative Reasoning Trajectory Across ACT Steps `1 → 16`

Evaluated using [analyze_hrm.py](file:///home/shadowspace/HRM_Reimplementation/analyze_hrm.py) (`results/diagnostics_step42000.json`):

| ACT Step | All-Cell Acc | Blank-Cell Acc | Exact Puzzle Acc | Q-Head Would Halt % |
|---:|---:|---:|---:|---:|
| 1 | 63.84% | 47.47% | 0.09% | 0.00% |
| 2 | 67.09% | 52.20% | 2.56% | 0.41% |
| 4 | 68.40% | 54.10% | 2.97% | 2.94% |
| 6 | 68.90% | 54.82% | 4.12% | 3.03% |
| 8 | 69.09% | 55.11% | 6.16% | 4.28% |
| 10 | 69.21% | 55.28% | 7.31% | 6.12% |
| 12 | 69.27% | 55.37% | 8.66% | 7.34% |
| 14 | 69.27% | 55.36% | 9.28% | 8.50% |
| **16** | **69.27%** | **55.36%** | **9.72%** | **9.38%** |

- Iterative recurrent computation monotonically boosts exact puzzle solving from **0.09% (Step 1)** to **9.72% (Step 16)** (>100× increase), proving that the model uses multi-step recurrent refinement to resolve global board constraints.

### 4.2 Difficulty Breakdown by Given Clue Count

| Clue Count Bucket | Solved / Total | Exact Puzzle Accuracy |
|---|---:|---:|
| `<= 22` clues | 0 / 119 | 0.00% |
| `23–25` clues | 99 / 1,672 | 5.92% |
| `26–28` clues | 117 / 1,313 | 8.91% |
| `>= 29` clues | 95 / 96 | **98.96%** |

### 4.3 ACT Q-Head Calibration

- Mean `q_halt_logits - q_continue_logits` on **solved** puzzles (`n=311`): **`+0.3442`**
- Mean `q_halt_logits - q_continue_logits` on **unsolved** puzzles (`n=2,889`): **`-7.4012`**
- During evaluation (`model.eval()`), early halting is intentionally disabled for batching uniformity (`avg_steps = 16.00`), while the Q-head's hypothetical early-halt rate (`9.06%` on 3,200 puzzles; `9.27%` on 10,016 puzzles) closely tracks true exact puzzle accuracy (`9.72%` / `10.10%`).

---

## 5. Concrete Unseen Test Puzzle Solutions (`step_42000`)

### Example 1 — Hard 24-Clue Unseen Puzzle (Batch 0, Item 10; `q_margin = +0.4688`, Valid Sudoku: `True`)

**Input (24 clues):**
```text
. . . | . . 9 | . . .
8 . . | . . . | . . 3
9 5 . | 7 4 . | . . .
------+-------+------
1 . 7 | . 6 . | 4 . .
. . . | . . . | . 8 .
. . 6 | 2 . . | . . 5
------+-------+------
2 . . | . 9 . | . . .
. . 5 | . . 6 | 1 . 4
. . 8 | . 3 7 | . . .
```

**Model Prediction (100% Exact Match with Ground Truth):**
```text
7 6 2 | 3 8 9 | 5 4 1
8 1 4 | 6 5 2 | 7 9 3
9 5 3 | 7 4 1 | 8 2 6
------+-------+------
1 8 7 | 9 6 5 | 4 3 2
5 2 9 | 4 1 3 | 6 8 7
4 3 6 | 2 7 8 | 9 1 5
------+-------+------
2 7 1 | 5 9 4 | 3 6 8
3 9 5 | 8 2 6 | 1 7 4
6 4 8 | 1 3 7 | 2 5 9
```

### Example 2 — Hard 25-Clue Unseen Puzzle (Batch 1, Item 9; `q_margin = +0.1875`, Valid Sudoku: `True`)

**Input (25 clues):**
```text
. . . | . 7 3 | . . .
. . . | 1 9 . | . . 7
. 8 . | . . . | . . .
------+-------+------
. . 1 | 5 6 . | . 3 .
9 . . | . . . | 4 . .
8 . . | . . . | 9 6 .
------+-------+------
6 . . | . 8 2 | 5 1 .
. . . | . 1 . | 2 . .
. 4 . | 7 . . | 6 . .
```

**Model Prediction (100% Exact Match with Ground Truth):**
```text
2 1 9 | 6 7 3 | 8 4 5
5 6 4 | 1 9 8 | 3 2 7
7 8 3 | 2 5 4 | 1 9 6
------+-------+------
4 2 1 | 5 6 9 | 7 3 8
9 3 6 | 8 2 7 | 4 5 1
8 7 5 | 3 4 1 | 9 6 2
------+-------+------
6 9 7 | 4 8 2 | 5 1 3
3 5 8 | 9 1 6 | 2 7 4
1 4 2 | 7 3 5 | 6 8 9
```

---

## 6. Limitations & Gap to Paper Results

- **Why exact accuracy is 10.10% vs. paper's ~55–99%:** Our 42,000-step `batch_size=128` run processed ~`5.38M` forward passes (~336k puzzles, ~3.4 epochs over `100,000` original puzzles) without `torch.compile` or `FlashAttention`, whereas the official single-GPU laptop demo (`HRM/README.md`) runs **~20M forward passes** (`epochs=20000`, `global_batch_size=384`) for ~10 hours on `1,000` puzzles × `1,000` augmentations, and the multi-GPU `Full Sudoku-Hard` run uses `global_batch_size=2304` and `epochs=100` (`~23M` puzzle-epochs).
- **Clear upward trend:** Exact accuracy was still rising steeply at step 42,000 (`3.09%` at 36k → `5.84%` at 37k → `9.72%` at 42k on 3,200 puzzles; `10.10%` on 10,016 puzzles), indicating further training with `--resume` on `baseline_bs128_step_42000.pt` would continue improving exact solving.
