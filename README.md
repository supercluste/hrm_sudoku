# Recreating the Hierarchical Reasoning Model (HRM) for Sudoku

An implementation and experimental study of the **Hierarchical Reasoning Model (HRM)** applied to 9×9 Sudoku puzzles (`sapientinc/sudoku-extreme`), investigating hierarchical recurrent computation (`H_cycles`, `L_cycles`), batch-size scaling, optimizer/schedule choices, and Adaptive Computation Time (ACT).

> **Project status:** Complete baseline, architectural ablations, 42,000-step scaling run (**10.10% exact complete-puzzle accuracy** across 10,016 unseen test puzzles; **98.96%** on 29+ clue puzzles), and ACT step-by-step diagnostics finished. See the full report in [`docs/EXPERIMENTAL_REPORT.md`](docs/EXPERIMENTAL_REPORT.md).

---

## 1. Architecture Overview

The active model ([`models/hrm/hrm_act_v1.py`](models/hrm/hrm_act_v1.py)) implements two interacting recurrent Transformer modules:

- **High-Level Module (`H_level`):** 4 Transformer blocks operating at a slower timescale (`H_cycles = 2` baseline).
- **Low-Level Module (`L_level`):** 4 Transformer blocks performing faster, detailed computation (`L_cycles = 2` baseline) with additive input injection.
- **1-Step Gradient Approximation:** Earlier recurrent iterations run under `torch.no_grad()`, backpropagating only through the final `L -> H` transition (~441–446 MB VRAM at `batch_size=128`).
- **Adaptive Computation Time (ACT):** A Q-head (`q_halt_logits`, `q_continue_logits`) trained via PQN-style bootstrapping (`halt_max_steps = 16`, `halt_exploration_prob = 0.1`). During evaluation (`model.eval()`), all `halt_max_steps` iterations run for batch uniformity.

| Component | Baseline Configuration |
|---|---:|
| `H_cycles` / `L_cycles` | 2 / 2 |
| `H_layers` / `L_layers` | 4 / 4 |
| Hidden dimension / Heads / Expansion | 512 / 8 / 4 |
| Positional encoding | RoPE (`pos_encodings="rope"`) |
| Vocabulary / Sequence length | 11 (`0`=pad, `1`=blank, `2–10`=digits 1–9) / 81 |
| Precision | `bfloat16` |

---

## 2. Dataset

Prepared from [`sapientinc/sudoku-extreme`](https://huggingface.co/datasets/sapientinc/sudoku-extreme) via `datasets/preprocess.py` and verified with `datasets/validate.py`:

- **Train:** 100,000 original puzzles × 3 variants (`--num-aug 2`) = **300,000 examples** (30.99% given clues)
- **Test:** **422,786 unseen puzzles** (31.16% given clues)

---

## 3. Repository Structure

```text
HRM_Reimplementation/
├── data/sudoku/                  # Preprocessed train/test splits (.npy + dataset.json)
├── dataset/                      # Metadata & dihedral transforms
├── datasets/                     # Preprocessing & Sudoku validation scripts
├── docs/
│   ├── EXPERIMENTAL_REPORT.md    # Full findings, evaluations, ACT curve & examples
│   ├── handoff.md                # Project handoff context
│   └── 2506.21734v3.pdf          # HRM research paper
├── models/
│   ├── adam_atan2.py             # Pure-PyTorch Adam-atan2 optimizer
│   ├── common.py
│   ├── layers.py                 # RoPE, SwiGLU, RMSNorm, PyTorch SDPA Attention
│   ├── losses.py                 # ACTLossHead (stablemax / softmax cross-entropy)
│   ├── sparse_embedding.py       # Sparse puzzle embedding + SignSGD
│   └── hrm/hrm_act_v1.py         # Active HRM + ACT model
├── results/                      # JSON & CSV experimental metrics
├── train.py                      # Configurable training & --resume CLI
├── evaluate.py                   # Checkpoint-driven evaluation & JSON export
├── analyze_hrm.py                # ACT step 1..16 diagnostics & solved grid extractor
└── requirements.txt
```

---

## 4. Getting Started

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Install CUDA-compatible PyTorch first, then:
pip install -r requirements.txt

# Validate dataset
python datasets/validate.py

# Train (e.g. baseline at batch_size=128)
python train.py --steps 10000 --batch-size 128 --save-every 2500 --warmup-steps 200 --lr 1e-4 --puzzle-emb-lr 1e-4 --h-cycles 2 --l-cycles 2 --run-name baseline_bs128

# Resume from checkpoint
python train.py --resume checkpoints/baseline_bs128_step_10000.pt --steps 5000 --batch-size 128 --save-every 5000 --lr 5e-5 --puzzle-emb-lr 5e-5 --seed 1042 --run-name baseline_bs128

# Evaluate checkpoint (loads architecture config automatically)
python evaluate.py --checkpoint checkpoints/baseline_bs128_step_42000.pt --batches 100 --batch-size 32 --output results/baseline_bs128_step42000.json

# Run ACT step-by-step diagnostics & extract solved boards
python analyze_hrm.py --checkpoint checkpoints/baseline_bs128_step_42000.pt --batches 100 --batch-size 32 --output results/diagnostics_step42000.json
```

---

## 5. Key Experimental Results

Full results are in [`results/comparison.csv`](results/comparison.csv) and [`docs/EXPERIMENTAL_REPORT.md`](docs/EXPERIMENTAL_REPORT.md).

### Architectural Ablations (`batch_size=128`, 5,000 steps, 3,200 test puzzles)

| Experiment | `H_cycles` | `L_cycles` | Train Time | Speed | Cell Acc | Exact Puzzle Acc | Avg LM Loss | Eval Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | 2 | 2 | 23.1m | ~3.6 steps/s | **60.58%** | 0.97% | **0.9462** | 34.68s |
| **Reduced H** | 1 | 2 | **14.2m** | **5.85 steps/s** | 58.19% | 0.47% | 1.4101 | **17.65s** |
| **Reduced L** | 2 | 1 | 17.0m | 4.91 steps/s | 59.34% | **1.25%** | 1.3897 | 24.89s |

*(At `batch_size=4`, all three configurations scored `35.33%–39.42%` cell accuracy and `0.00%` exact accuracy due to 4-puzzle carry memorization.)*

### Extended Training Progression (`baseline_bs128`, `H=2, L=2`)

| Step | Test Puzzles | Cell Acc | Exact Puzzle Acc | Avg LM Loss | Q-Halt Acc | Early Halt % |
|---:|---:|---:|---:|---:|---:|---:|
| 1,500 | 3,200 | 52.13% | 0.00% | 1.1455 | 100.00% | 0.0% |
| 5,000 | 3,200 | 60.58% | 0.97% | 0.9462 | 99.06% | 0.0% |
| 10,000 | 3,200 | 61.61% | 2.41% | 0.9321 | 99.50% | 0.38% |
| 25,000 | 3,200 | 64.08% | 3.22% | 0.9074 | 98.66% | 3.22% |
| 37,000 | 3,200 | 67.27% | 5.84% | 0.7586 | 99.06% | 3.00% |
| **42,000** | **3,200** | **69.27%** | **9.72%** | **0.6917** | **99.47%** | **9.06%** |
| **42,000** | **10,016** | **69.46%** | **10.10%** | **0.6858** | **99.57%** | **9.27%** |

### ACT Step-by-Step Diagnostics (Step 42,000 on 3,200 Puzzles)

- **Iterative refinement:** Exact accuracy increased from **0.09% (ACT Step 1)** → **2.97% (Step 4)** → **6.16% (Step 8)** → **8.66% (Step 12)** → **9.72% (Step 16)**.
- **By clue count:** `<=22` clues: `0.00%` (`0/119`), `23–25` clues: `5.92%` (`99/1672`), `26–28` clues: `8.91%` (`117/1313`), `>=29` clues: **98.96% (`95/96`)**.
- **Q-head calibration:** Mean `q_halt - q_continue` was **`+0.3442`** on solved puzzles vs. **`-7.4012`** on unsolved puzzles.

---

## 6. Documented Deviations & Disclaimer

- **Attention:** PyTorch `F.scaled_dot_product_attention` (SDPA) instead of FlashAttention.
- **Optimizer:** `torch.optim.AdamW` for the main scaling runs (a pure-PyTorch `AdamATan2` in `models/adam_atan2.py` was also implemented and probed).
- **Scale:** 100,000 subsampled training puzzles and single laptop GPU (`batch_size=128`, up to 42,000 steps) rather than multi-GPU `global_batch_size=2304` or 10-hour `1k-aug-1000` runs.
- This is an educational reimplementation, not an official release of HRM.
