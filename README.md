# Recreating the Hierarchical Reasoning Model (HRM) for Sudoku

An implementation project exploring the **Hierarchical Reasoning Model (HRM)** and its application to structured reasoning through Sudoku.

The objective is to understand and recreate the important components of HRM, prepare a suitable Sudoku dataset, and investigate how hierarchical recurrent computation and adaptive computation time can help neural networks solve constraint-based problems.

> **Project status:** Work in progress. The model architecture, dataset pipeline, forward pass, and initial training pipeline have been implemented. Full training and evaluation are ongoing.

---

## 1. Introduction

Traditional neural networks typically process information through a sequence of layers. However, complex reasoning problems may require multiple rounds of computation and refinement.

The Hierarchical Reasoning Model (HRM) explores a different approach.

HRM uses two interacting recurrent modules:

- **High-Level Module (H):** Maintains higher-level representations and coordinates reasoning.
- **Low-Level Module (L):** Performs detailed computations and refines representations.

These modules interact repeatedly, allowing the network to perform hierarchical computation.

HRM also incorporates **Adaptive Computation Time (ACT)**, allowing the model to learn when to continue computation and when to halt.

In this project, I am implementing this architecture for Sudoku solving to study its reasoning behavior in a structured problem domain.

## 2. Why Sudoku?

Sudoku provides a useful environment for investigating structured reasoning.

A standard Sudoku puzzle consists of a 9 × 9 grid with three fundamental constraints:

1. Every row must contain digits 1–9 exactly once.
2. Every column must contain digits 1–9 exactly once.
3. Every 3 × 3 subgrid must contain digits 1–9 exactly once.

The model receives an incomplete Sudoku puzzle and predicts the completed solution.

Although the rules are simple, satisfying all constraints simultaneously requires the model to learn relationships across the grid.

This makes Sudoku a useful experimental task for studying hierarchical computation.

## 3. Architecture

The implementation follows the HRM architecture.

### Hierarchical reasoning

The model contains two recurrent modules:

| Component | Configuration |
|---|---:|
| High-level cycles (H) | 2 |
| Low-level cycles (L) | 2 |
| High-level transformer layers | 4 |
| Low-level transformer layers | 4 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Expansion factor | 4 |
| Positional encoding | RoPE |

The model uses recurrent hidden states to exchange information between the two modules.

### Adaptive Computation Time

HRM uses ACT to control the amount of computation performed for each input.

The model predicts whether it should continue reasoning or halt.

Current configuration:

- Maximum reasoning steps: 16
- Halt exploration probability: 0.1

### Input and output representation

Each Sudoku puzzle is represented as a sequence of 81 tokens.

The vocabulary contains 11 tokens:

| Token | Meaning |
|---|---|
| 0 | Padding / ignored label |
| 1 | Empty cell |
| 2–10 | Sudoku digits 1–9 |

The model produces predictions over the 11-token vocabulary for each of the 81 positions.

## 4. Dataset

The project uses the [Sudoku Extreme dataset](https://huggingface.co/datasets/sapientinc/sudoku-extreme) from Hugging Face.

The dataset contains millions of training puzzles and a separate test split.

### Preprocessing

A custom preprocessing pipeline converts the original Sudoku data into the format expected by the HRM dataset loader.

The pipeline includes:

- Conversion of Sudoku boards into numerical token representations.
- Consistent transformation of input puzzles and their solutions.
- Digit permutation.
- Valid row and column permutations.
- Sudoku-preserving transposition.
- Group indexing for augmented examples.

### Data augmentation

Each selected training puzzle is used to generate multiple equivalent examples.

The transformations preserve the Sudoku constraints and maintain the correspondence between the puzzle and its solution.

### Memory-efficient preprocessing

The original training dataset contains approximately 3.83 million puzzles.

To avoid loading the entire dataset into RAM, the preprocessing pipeline uses NumPy memory-mapped arrays.

This allows large datasets to be written incrementally.

**Current dataset configuration:**

| Property | Value |
|---|---:|
| Original training puzzles selected | 100,000 |
| Augmentations per puzzle | 2 |
| Examples per puzzle group | 3 |
| Total training examples | 300,000 |
| Test puzzles | 422,786 |
| Sequence length | 81 |
| Vocabulary size | 11 |

The dataset loader samples one example from each puzzle group during an iteration.

## 5. Implementation

The implementation is based on the official HRM codebase.

Important components include:

- HRM architecture and recurrent computation.
- Transformer layers and attention.
- Rotary positional embeddings.
- Sparse puzzle embeddings.
- StableMax cross-entropy loss.
- Adaptive computation time.
- Official puzzle dataset loading logic.
- Training and checkpoint management.
- Dataset preprocessing and validation.

### Hardware compatibility adjustments

Two implementation differences are currently present:

**1. Attention implementation**

The original implementation uses FlashAttention.

For compatibility with the current environment, the attention operation has been adapted to use PyTorch's `scaled_dot_product_attention`.

This retains the scaled dot-product attention computation but uses a different implementation and potentially a different kernel.

**2. Main optimizer**

The original training setup uses AdamATan2.

The current training script uses AdamW as a temporary substitute, with the configured learning rate, beta values, and weight decay.

The sparse puzzle embedding optimizer is retained.

These differences are documented because the current implementation is not yet an exact reproduction of the original training setup.

## 6. Repository Structure

```text
HRM_Reimplementation/
│
├── data/
│   └── sudoku/
│       ├── train/
│       └── test/
│
├── dataset/
│   ├── __init__.py
│   └── common.py
│
├── datasets/
│   ├── preprocess.py
│   ├── sudoku_dataset.py
│   └── validate.py
│
├── models/
│   ├── __init__.py
│   ├── common.py
│   ├── layers.py
│   ├── losses.py
│   ├── sparse_embedding.py
│   │
│   └── hrm/
│       ├── __init__.py
│       └── hrm_act_v1.py
│
├── checkpoints/
│
├── puzzle_dataset.py
├── train.py
├── evaluate.py
├── test_official_hrm.py
├── test_official_dataset.py
└── test_train_step.py
```

## 7. Environment

The current development environment:

| Component | Version / Specification |
|---|---|
| Operating system | Linux |
| Python | 3.12.16 |
| PyTorch | 2.15.0+cu132 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| GPU memory | 12 GB |
| CUDA availability | Enabled |
| Precision | BF16 |

A CUDA-enabled PyTorch installation and a compatible NVIDIA driver are required for GPU training.

Python 3.12 is the version used for development.

## 8. Getting Started

### Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd HRM_Reimplementation
```

### Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

Install a CUDA-compatible PyTorch build for your environment, along with the project's Python dependencies.

The exact dependency list and installation instructions will be maintained as the environment is finalized.

### Prepare the dataset

Download and preprocess the training data:

```bash
python datasets/preprocess.py \
    --subsample-size 100000 \
    --num-aug 2
```

The existing test split is preserved by the preprocessing script.

### Validate the dataset

```bash
python datasets/validate.py
```

The validator checks:

- Valid Sudoku rows.
- Valid Sudoku columns.
- Valid 3 × 3 subgrids.
- Preservation of original clues.

### Test the model

```bash
python test_official_hrm.py
```

### Test the dataset loader

```bash
python test_official_dataset.py
```

### Run a short training test

```bash
python train.py --steps 20 --batch-size 4 --save-every 10
```

This is a smoke test to verify the training pipeline, not a meaningful model-training run.

### Train the model

```bash
python train.py --steps 500 --batch-size 4 --save-every 100
```

Training checkpoints are stored in the `checkpoints/` directory.

### Evaluate

```bash
python evaluate.py --batches 10 --batch-size 4
```

Evaluation results should be interpreted in the context of the checkpoint's training duration and dataset configuration.

## 9. Current Progress

| Component | Status |
|---|---|
| Official HRM architecture integration | Implemented |
| Transformer and attention layers | Implemented with SDPA adaptation |
| Rotary positional embeddings | Integrated |
| Sparse puzzle embeddings | Integrated |
| ACT mechanism | Integrated |
| StableMax loss | Integrated |
| Sudoku dataset preprocessing | Implemented |
| Data augmentation | Implemented |
| Memory-efficient data generation | Implemented |
| Official dataset loader integration | Tested |
| Sudoku data validation | Implemented |
| GPU forward pass | Tested |
| Initial training pipeline | Smoke-tested |
| Extended model training | Pending |
| Full evaluation | Pending |
| Optimizer parity with original HRM | Pending |

## 10. Future Work

The next stages of this project are:

1. Complete dataset validation and verify augmentation integrity.
2. Resolve the optimizer difference and investigate compatibility with AdamATan2.
3. Establish a suitable training configuration for the available GPU hardware.
4. Train the model for a meaningful number of optimization steps.
5. Evaluate cell-level accuracy and complete-puzzle accuracy.
6. Investigate the model's adaptive reasoning steps.
7. Compare performance across different dataset sizes and training configurations.
8. Analyze failure cases to understand where the model struggles.

The long-term objective is to understand whether hierarchical recurrent computation can learn useful reasoning strategies for structured constraint-solving tasks.

## 11. References

- **Original HRM paper:** [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
- **Original implementation:** [Sapient AI Lab – HRM](https://github.com/sapientinc/HRM)
- **Dataset:** [Sudoku Extreme – Hugging Face](https://huggingface.co/datasets/sapientinc/sudoku-extreme)

---

## Disclaimer

This repository is an ongoing reimplementation and experimentation project based on the HRM architecture.

It is not an official release of HRM, and the current training pipeline contains documented implementation substitutions. Results should not be considered a reproduction of the original paper's reported performance.
