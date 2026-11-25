# HRM Sudoku Reimplementation — Project Handoff Document

**Purpose:** Transfer the complete project context to Antigravity so development can continue without repeating completed work or making incorrect assumptions.

**Project status:** Core architecture implemented; dataset prepared and validated; forward pass verified; training and evaluation pipeline implemented but requires final debugging, controlled experiments, and result validation.

**Environment:** Linux / WSL  
**GPU:** NVIDIA RTX 5070 Ti Laptop GPU (12 GB VRAM)  
**Python:** 3.12.16  
**PyTorch:** 2.15.0+cu132  
**Repository:** `https://github.com/supercluste/hrm_sudoku`  
**Local project directory:** `~/HRM_Reimplementation`

---

# 1. Project objective

## 1.1 Main goal

The goal is to implement and experimentally investigate a Hierarchical Reasoning Model (HRM) on Sudoku puzzles, based on the research paper that will be provided separately.

This project is **not intended to be an exact reproduction of every detail in the paper**.

The actual objective is:

> Demonstrate that a hierarchical recurrent reasoning architecture can learn to solve Sudoku puzzles, investigate how its hierarchical computation affects performance, and produce measurable experimental results.

The implementation should preserve the important ideas of HRM:

- Two interacting recurrent modules operating at different computational levels.
- A high-level module responsible for slower, abstract reasoning.
- A low-level module responsible for faster, detailed computation.
- Recurrent state updates across reasoning cycles.
- Adaptive computation / halting (ACT).
- Training on Sudoku input-output examples.
- Evaluation using cell-level accuracy and complete-puzzle accuracy.

The research paper should be treated as the primary reference for understanding the architecture and experimental motivation. However, implementation choices may differ when necessary because of hardware limitations, available time, or engineering constraints.

## 1.2 What we want to demonstrate

The project should answer the following experimental questions:

1. Can an HRM-style architecture learn Sudoku patterns from training data?
2. Does hierarchical computation improve or affect Sudoku-solving performance?
3. What happens when the number of high-level recurrent cycles is reduced?
4. What happens when the number of low-level recurrent cycles is reduced?
5. How does computation depth affect training time and inference performance?
6. Does adaptive computation behave meaningfully during evaluation?

The most important deliverable is a working model supported by honest, reproducible experiments.

## 1.3 Expected final deliverables

By the end of development, the repository should contain:

- A working HRM implementation.
- A reproducible Sudoku dataset preparation pipeline.
- A validated training pipeline.
- A validated evaluation pipeline.
- Saved model checkpoints.
- Experimental results for the baseline and architectural ablations.
- A results table comparing the experiments.
- Training and evaluation logs.
- Relevant plots, if time permits.
- A README documenting architecture, implementation decisions, deviations, and results.

Do not claim that the model successfully solves Sudoku unless complete-puzzle evaluation supports that claim.

---

# 2. Research and implementation approach

The research paper will be supplied separately to Antigravity.

Antigravity should read the paper before making substantial architectural changes.

The following principles should guide development:

1. Preserve the central HRM architecture.
2. Avoid unnecessary rewrites of already-working components.
3. Make experimental changes configurable rather than hardcoding them.
4. Keep the baseline implementation separate from experimental modifications where practical.
5. Record all deviations from the paper.
6. Prioritize reliable experiments over adding unnecessary features.
7. Do not fabricate results or describe unexecuted experiments as completed.

The implementation should be an independent educational and experimental reimplementation, not presented as the official HRM implementation.

---

# 3. Development environment

The project is being developed in Linux / WSL.

| Component | Current configuration |
|---|---|
| Operating environment | Linux / WSL |
| Python | 3.12.16 |
| Virtual environment | `.venv` |
| PyTorch | 2.15.0+cu132 |
| CUDA availability | Available |
| GPU | NVIDIA RTX 5070 Ti Laptop GPU |
| GPU VRAM | 12 GB |
| BF16 support | Verified |
| Project directory | `~/HRM_Reimplementation` |
| Official source clone | `~/HRM_Official` |

All commands should be executed from the project directory with the project's virtual environment activated:

```bash
cd ~/HRM_Reimplementation
source .venv/bin/activate
```

The machine has enough GPU memory to run the current model. The architecture's forward pass was successfully tested.

**Important:** Do not unnecessarily change Python, PyTorch, CUDA, or the virtual environment. The current environment is working.

---

# 4. Current project structure

The project currently has the following structure:

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
│   ├── hrm/
│   │   ├── __init__.py
│   │   └── hrm_act_v1.py
│   │
│   └── hrm_prototype_backup.py
│
├── puzzle_dataset.py
├── test_official_hrm.py
├── test_official_dataset.py
├── test_train_step.py
├── train.py
├── evaluate.py
│
├── checkpoints/
├── requirements.txt
└── README.md
```

Some files are adapted from the official HRM source repository. Others were written or modified for this reimplementation.

Do not assume that every file is currently part of the active training pipeline. In particular, `datasets/sudoku_dataset.py` is an older custom loader, while the active training script uses the official-style `PuzzleDataset` implementation in `puzzle_dataset.py`.

---

# 5. Dataset preparation

## 5.1 Dataset source

The dataset is:

**Hugging Face:** `sapientinc/sudoku-extreme`

The original training split contains approximately 3.83 million puzzles.

Because of the limited development time and local compute budget, the project uses a subsampled training dataset.

## 5.2 Current dataset configuration

Preprocessing was run with:

```bash
python datasets/preprocess.py --subsample-size 100000 --num-aug 2
```

This produced:

| Dataset property | Value |
|---|---:|
| Original training puzzles selected | 100,000 |
| Generated training examples | 300,000 |
| Test examples | 422,786 |
| Training examples per original puzzle | 3 |
| Sequence length | 81 |
| Vocabulary size | 11 |
| Puzzle identifiers | 1 |

The 300,000 training examples represent 100,000 original puzzle groups with three variants per group.

The test split was retained separately.

The preprocessing script was rewritten to use streaming and memory-mapped arrays to avoid loading the entire original dataset into RAM.

## 5.3 Token encoding

Each Sudoku puzzle is represented as a sequence of 81 tokens.

Current encoding:

| Token | Meaning |
|---|---|
| 0 | Padding / ignored label |
| 1 | Blank input cell |
| 2–10 | Sudoku digits 1–9 |

The model uses:

```python
vocab_size = 11
seq_len = 81
```

The label ignore ID is `0`.

## 5.4 Data loading

The active training pipeline uses the official-style `PuzzleDataset` loader.

The dataset is organized into groups using `group_indices`.

During training, one variant from each original puzzle group is sampled per iteration.

This avoids treating the three variants of one original puzzle as completely independent original puzzles.

The test loader uses:

```python
test_set_mode=True
```

and loads the test split.

## 5.5 Dataset validation

A validator was implemented at:

```text
datasets/validate.py
```

It checks Sudoku solution validity, including:

- Every row contains digits 1–9.
- Every column contains digits 1–9.
- Every 3×3 box contains digits 1–9.
- The solution preserves the original clues.

The validator successfully passed:

```text
train: 300000 valid Sudoku examples
test: 422786 valid Sudoku examples
```

**Status: Dataset preprocessing and validation are completed.**

Do not regenerate the dataset unless there is a concrete reason.

---

# 6. HRM architecture implementation

## 6.1 Architecture overview

The model follows the hierarchical reasoning concept from the HRM research.

It contains two recurrent components:

### High-level module (H)

- Operates at a slower reasoning timescale.
- Maintains a high-level latent state.
- Interacts with the low-level module.
- Performs higher-level iterative computation.

### Low-level module (L)

- Performs more frequent computation.
- Maintains its own latent state.
- Interacts with the high-level representation.
- Supports detailed iterative processing.

The two modules are implemented in the HRM model file:

```text
models/hrm/hrm_act_v1.py
```

The supporting neural-network components are in:

```text
models/layers.py
models/common.py
models/sparse_embedding.py
models/losses.py
```

## 6.2 Current baseline configuration

The intended baseline configuration is:

```python
H_cycles = 2
L_cycles = 2

H_layers = 4
L_layers = 4

hidden_size = 512
num_heads = 8
expansion = 4

pos_encodings = "rope"

halt_max_steps = 16
halt_exploration_prob = 0.1

puzzle_emb_ndim = 512

vocab_size = 11
seq_len = 81

forward_dtype = "bfloat16"
```

The model constructor expects the configuration as a dictionary:

```python
config.model_dump()
```

Do not change the baseline configuration without recording the change.

## 6.3 Attention implementation

The original source uses FlashAttention.

The current implementation replaces that attention operation with PyTorch's scaled dot-product attention (SDPA).

The relevant logic in `models/layers.py` is:

```python
query = query.transpose(1, 2)
key = key.transpose(1, 2)
value = value.transpose(1, 2)

if self.num_heads != self.num_key_value_heads:
    repeats = self.num_heads // self.num_key_value_heads
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)

attn_output = F.scaled_dot_product_attention(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=self.causal
)

attn_output = attn_output.transpose(1, 2).contiguous()
```

This preserves the general scaled dot-product attention operation while using PyTorch's implementation instead of the original FlashAttention dependency.

**Document this as an implementation deviation.**

Do not replace it with a different attention mechanism unless necessary.

## 6.4 Adaptive computation and halting

The architecture includes adaptive computation time (ACT).

The current configuration uses:

```python
halt_max_steps = 16
halt_exploration_prob = 0.1
```

The model produces:

- Language-model logits.
- Halting logits.
- Continue logits.

The model's ACT behavior is implemented in `models/hrm/hrm_act_v1.py`.

**Important unresolved issue:** The evaluation behavior of ACT has not yet been fully verified.

Before trusting evaluation metrics, inspect how the model handles:

- Training mode versus evaluation mode.
- The `halted` state.
- Maximum reasoning steps.
- Continue and halt decisions.
- Whether evaluation stops when the model predicts halting.
- Whether evaluation instead always executes the maximum number of steps.

The last known debugging task was to inspect:

```bash
grep -n -A100 -B20 "def forward" models/hrm/hrm_act_v1.py
```

and:

```bash
grep -n -A50 -B10 "halt_max_steps\|halted" models/hrm/hrm_act_v1.py
```

The output of these commands had not yet been reviewed.

**Antigravity must inspect this before modifying the evaluation pipeline.**

---

# 7. Architecture verification already completed

The official HRM forward pass was tested using:

```text
test_official_hrm.py
```

The test successfully produced:

```text
Logits shape: torch.Size([2, 81, 11])
Halt logits shape: torch.Size([2])
Continue logits shape: torch.Size([2])

GPU memory (MB): 190.26

OFFICIAL HRM FORWARD PASS: SUCCESS
```

This confirms that:

- The model can be instantiated.
- The forward pass executes.
- The output dimensions are correct.
- Halting and continuation logits are produced.
- The model runs on the available GPU.

The forward pass was also tested after the attention implementation changes.

**Status: Basic model forward-pass verification completed.**

This does not establish that the model has learned Sudoku or that the complete training pipeline is correct.

---

# 8. Training pipeline

The active training script is:

```text
train.py
```

## 8.1 Current training implementation

The script currently:

1. Parses training arguments.
2. Creates the HRM configuration.
3. Initializes the HRM model.
4. Initializes the ACT loss head.
5. Loads the training dataset using `PuzzleDataset`.
6. Initializes the recurrent carry.
7. Moves batches to CUDA.
8. Runs forward propagation.
9. Computes the loss.
10. Performs backpropagation.
11. Updates model parameters.
12. Updates puzzle embeddings using the sparse optimizer.
13. Logs training metrics.
14. Saves checkpoints periodically.

The training script currently supports:

```bash
python train.py --steps 500 --batch-size 4 --save-every 100
```

The default values are:

```python
steps = 500
batch_size = 4
save_every = 100
```

## 8.2 Optimizers

The original implementation includes a sparse optimizer for puzzle embeddings:

```python
CastedSparseEmbeddingSignSGD_Distributed
```

Its current settings are:

```python
lr = 0.01
weight_decay = 0.1
world_size = 1
```

The main model optimizer currently uses AdamW:

```python
AdamW(
    ...,
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)
```

This is an intentional temporary substitution for the optimizer used in the original implementation.

The original HRM implementation uses AdamATan2 for the main optimizer.

**Deviation:** AdamW is being used instead of AdamATan2 to simplify the local implementation and reduce setup overhead.

Do not silently describe the implementation as using AdamATan2.

## 8.3 Learning-rate schedule

The current script uses a linear warmup followed by a constant learning rate.

However, the current warmup setting is:

```python
warmup_steps = 2000
```

This is unsuitable for short experimental runs of only a few hundred steps.

An earlier 20-step smoke test showed:

```text
Step 1/20:
Loss: 10.2620
LR: 5e-08

Step 10/20:
Loss: 10.0033
LR: 5e-07

Step 20/20:
Loss: 10.2774
LR: 1e-06
```

The learning rate barely increased because the warmup was much longer than the run.

**Required improvement:** Make warmup configurable through a command-line argument and choose a suitable warmup for short experiments.

A possible initial setting is 100–200 warmup steps for a short experimental run, but this should be tested rather than treated as an established optimal value.

All experiments being compared must use the same learning-rate schedule.

## 8.4 Recurrent carry

The training pipeline initializes the recurrent carry and preserves it across training iterations.

The carry includes:

- `steps`
- `halted`
- `inner_carry.z_H`
- `inner_carry.z_L`

The model uses this carry to maintain its recurrent state.

The code manually handles the sparse puzzle embedding weights and their gradients.

Antigravity should inspect the carry handling and ensure that:

- Tensors are on the correct device.
- Recurrent states are detached appropriately between training iterations where required.
- No unintended computational graph is retained across batches.
- Carry behavior matches the intended official implementation.

Do not casually rewrite the carry logic without understanding the original model.

## 8.5 Checkpoint format

The current training script saves:

```python
{
    "step": step,
    "model": model.state_dict(),
    "main_optimizer": main_optimizer.state_dict(),
    "puzzle_optimizer": puzzle_optimizer.state_dict(),
    "config": config.model_dump(),
}
```

Checkpoints are saved under:

```text
checkpoints/
```

The training script does not yet have complete resume-from-checkpoint functionality.

**Required improvement:** Add a reliable `--resume` option that restores:

- Model weights.
- Main optimizer state.
- Sparse optimizer state.
- Current training step.
- Relevant configuration.

Also ensure that the script can save the final checkpoint even when the total step count is not divisible by `save_every`.

---

# 9. Evaluation pipeline

The active evaluation script is:

```text
evaluate.py
```

## 9.1 Current functionality

The script:

1. Loads a checkpoint.
2. Creates the HRM model.
3. Loads model weights.
4. Initializes the test dataset.
5. Runs inference.
6. Calculates evaluation metrics.

Current arguments:

```bash
python evaluate.py \
    --checkpoint checkpoints/hrm_step_500.pt \
    --batches 10 \
    --batch-size 4
```

## 9.2 Metrics

The script currently tracks:

| Metric | Meaning |
|---|---|
| Cell accuracy | Percentage of correctly predicted Sudoku cells |
| Exact puzzle accuracy | Percentage of puzzles solved completely |
| Average reasoning steps | Average number of reasoning steps |
| LM loss | Language-model loss |

Cell accuracy is useful for understanding partial learning.

However, exact puzzle accuracy is especially important because a Sudoku solution is only fully correct when every cell is correct.

A model with high cell accuracy can still have zero exact-puzzle accuracy.

## 9.3 Evaluation issues to fix

The current evaluator has several limitations:

**Issue 1 — Configuration is hardcoded**

The evaluator currently creates the baseline configuration independently.

It should instead load the configuration saved inside the checkpoint:

```python
checkpoint["config"]
```

This is necessary to evaluate ablation checkpoints correctly.

**Issue 2 — ACT behavior needs verification**

The current evaluation loop runs:

```python
for _ in range(config.halt_max_steps):
    ...
```

It is not yet established whether this correctly respects the model's evaluation-time halting behavior.

Inspect the model's ACT implementation before changing the loop.

**Issue 3 — Evaluation aggregation needs validation**

Check that:

- Cell accuracy is calculated over valid cells.
- Ignored labels are excluded.
- Exact accuracy is calculated per puzzle, not per cell.
- Metrics are aggregated across all evaluated batches correctly.
- The test subset is deterministic and reproducible.

**Issue 4 — Evaluation subset**

The current evaluator supports a limited number of batches.

For quick experiments, this is useful. For final reporting, evaluate a sufficiently large and clearly specified test subset, ideally the complete test split if time and compute permit.

Always report the number of evaluated puzzles.

---

# 10. Previous training results

An earlier experiment was conducted using a much smaller training subset.

It produced the following approximate training cell-accuracy values:

| Training step | Cell accuracy |
|---:|---:|
| 80 | 27.47% |
| 160 | 65.43% |
| 240 | 77.78% |
| 320 | 76.23% |
| 400 | 65.74% |
| 480 | 44.75% |

The subsequent evaluation on only 40 test puzzles produced:

| Metric | Result |
|---|---:|
| Cell accuracy | 41.76% |
| Exact puzzle accuracy | 0% |
| Average reasoning steps | 16 |
| LM loss | 1.5714 |

**These are preliminary results only.**

They were obtained using an earlier, much smaller dataset and training configuration. They must not be presented as the final results of the current 100,000-puzzle training setup.

An additional 20-step smoke test on the current pipeline produced losses around 10, but it was not a meaningful training experiment because of the excessively long warmup.

No reliable final baseline result has yet been established for the current dataset and configuration.

---

# 11. Planned experiments and ablations

The primary objective is to compare the baseline architecture against reduced-computation variants.

The experiments should differ only in the specified architectural setting.

## Experiment A — Baseline

```python
H_cycles = 2
L_cycles = 2
```

This is the reference configuration.

## Experiment B — Reduced high-level cycles

```python
H_cycles = 1
L_cycles = 2
```

Purpose:

Investigate the effect of reducing high-level recurrent computation while retaining the low-level configuration.

## Experiment C — Reduced low-level cycles

```python
H_cycles = 2
L_cycles = 1
```

Purpose:

Investigate the effect of reducing low-level recurrent computation while retaining the high-level configuration.

## Comparison table to complete

| Experiment | H cycles | L cycles | Training steps | Cell accuracy | Exact accuracy | Time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2 | 2 | TBD | TBD | TBD | TBD |
| Reduced H | 1 | 2 | TBD | TBD | TBD | TBD |
| Reduced L | 2 | 1 | TBD | TBD | TBD | TBD |

Do not fill in results until experiments have actually been run.

## Experimental fairness

Keep the following constant across experiments:

- Training dataset.
- Test dataset and evaluation subset.
- Random seed.
- Batch size.
- Number of optimizer updates.
- Optimizer.
- Learning rate.
- Warmup schedule.
- Loss function.
- Model hidden size.
- Number of layers.
- Evaluation procedure.

Only change the intended recurrent cycle parameter.

Because different cycle counts change computation cost, record training time and inference time as well as accuracy.

These experiments are exploratory. Avoid claiming statistical significance from a single run.

## Optional experiments

If time permits, consider:

- Maximum reasoning steps of 8 versus 16.
- Fixed computation versus adaptive computation.
- Different batch sizes.
- Additional training steps.

These are secondary. Do not let optional experiments delay a working baseline and the two main ablations.

---

# 12. Implementation deviations from the original paper/source

The following deviations are already known and must be documented honestly.

| Component | Original approach | Current implementation |
|---|---|---|
| Attention | FlashAttention | PyTorch SDPA |
| Main optimizer | AdamATan2 | AdamW |
| Training dataset | Large original dataset | 100,000 selected original puzzles |
| Training examples | Original dataset setup | 300,000 examples after augmentation |
| Compute environment | Original research environment | Local RTX 5070 Ti Laptop GPU |
| Experiment scope | Paper's experimental setup | Focused baseline and architectural ablations |

These deviations do not invalidate the educational experiment, but they must be disclosed.

Antigravity should read the paper and identify any other deviations that become apparent.

Do not claim exact reproduction of the paper's results.

---

# 13. GitHub and version control

Remote repository:

```text
https://github.com/supercluste/hrm_sudoku
```

The local repository has an initial commit:

```text
daa3f34 Initial commit: HRM Sudoku reimplementation
```

The remote also has an unrelated initial commit:

```text
f6d9e39 Initial commit
```

An initial push encountered an unrelated-history conflict.

The suggested resolution was:

```bash
git merge origin/main --allow-unrelated-histories
```

The merge/push completion has not been confirmed.

Before making substantial changes:

```bash
git status
git log --oneline --all --decorate -5
```

Do not overwrite remote work.

After confirming the repository state, commit meaningful changes in small logical commits.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc
.env
data/sudoku/*/*.npy
checkpoints/*.pt
wandb/
outputs/
```

Large datasets and model checkpoints should not be committed directly to GitHub.

---

# 14. Requirements

The current `requirements.txt` was intended to include:

```text
numpy>=1.26
einops>=0.8
pydantic>=2.8
tqdm>=4.66
huggingface_hub>=0.26
```

PyTorch is intentionally handled separately because the installation must match the CUDA environment.

Do not reinstall PyTorch without a specific reason.

---

# 15. Immediate unresolved tasks

These are the most important tasks remaining.

## Priority 1 — Verify ACT implementation

Inspect:

```text
models/hrm/hrm_act_v1.py
```

Understand the exact evaluation-time behavior of:

- Halting.
- Continuing.
- Maximum reasoning steps.
- Recurrent carry updates.
- Training/evaluation mode differences.

This is necessary before trusting evaluation results.

## Priority 2 — Make training configurable

Update `train.py` to support:

```text
--steps
--batch-size
--save-every
--warmup-steps
--h-cycles
--l-cycles
--seed
--resume
```

Do not change the default baseline configuration.

Ensure that the selected settings are stored in the checkpoint.

## Priority 3 — Fix evaluation configuration

Update `evaluate.py` to:

- Read configuration from the checkpoint.
- Evaluate baseline and ablation checkpoints.
- Correctly handle ACT.
- Exclude ignored cells from accuracy calculations.
- Calculate exact puzzle accuracy correctly.
- Report the number of evaluated puzzles.
- Use a reproducible test subset.

## Priority 4 — Run a short training benchmark

Before committing to long experiments:

1. Run a short baseline training job.
2. Confirm that loss decreases meaningfully.
3. Check whether cell accuracy improves.
4. Monitor GPU memory and training speed.
5. Confirm checkpoints load correctly.

Do not start three long experiments before establishing that the training pipeline works.

## Priority 5 — Run the main experiments

Run:

1. Baseline: H=2, L=2.
2. Reduced H: H=1, L=2.
3. Reduced L: H=2, L=1.

Use identical training and evaluation settings.

If the time budget is insufficient, prioritize a reliable baseline and at least one useful ablation over several incomplete runs.

## Priority 6 — Record results

Create a results file, for example:

```text
results/
├── baseline.json
├── reduced_h.json
├── reduced_l.json
└── comparison.csv
```

Record:

- Model configuration.
- Training steps.
- Batch size.
- Optimizer.
- Learning rate.
- Warmup steps.
- Random seed.
- Training duration.
- Final training loss.
- Test cell accuracy.
- Test exact-puzzle accuracy.
- Average reasoning steps.
- Number of test puzzles.

## Priority 7 — Final documentation

Update the README with:

- Architecture overview.
- Dataset information.
- Installation instructions.
- Training instructions.
- Evaluation instructions.
- Experimental configurations.
- Actual results.
- Limitations.
- Deviations from the original implementation.

Finally, commit and push the completed work to GitHub.

---

# 16. Instructions to Antigravity

You are taking over an existing project. Do not treat this as a blank repository.

Follow this order:

### Step 1 — Understand before changing

Read:

1. The supplied HRM research paper.
2. This handoff document.
3. `README.md`.
4. `train.py`.
5. `evaluate.py`.
6. `models/hrm/hrm_act_v1.py`.
7. `models/layers.py`.
8. `models/losses.py`.
9. `puzzle_dataset.py`.
10. `datasets/preprocess.py`.
11. `datasets/validate.py`.

Compare the implementation with the paper and identify any discrepancies.

### Step 2 — Inspect the existing environment

Check:

```bash
cd ~/HRM_Reimplementation
source .venv/bin/activate

python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Do not rebuild the environment if it already works.

### Step 3 — Inspect the current code

Run the existing tests and inspect the ACT implementation.

Do not blindly replace existing code.

Understand how the current architecture, carry, loss, optimizer, and dataset loader work.

### Step 4 — Fix the training and evaluation pipeline

Implement the missing configuration, resume, and evaluation features described above.

Make changes incrementally and run tests after each important change.

### Step 5 — Run experiments

Start with a short benchmark.

Only after the benchmark works should you run the baseline and ablations.

Prioritize results that are valid and reproducible over impressive-looking numbers.

### Step 6 — Produce final outputs

The final project should be runnable by another person using the README instructions.

Provide a concise report of:

- What was implemented.
- What was tested.
- Which experiments were completed.
- What the results show.
- What limitations remain.

---

# 17. Important constraints and working preferences

The project is being developed under a tight deadline.

Therefore:

- Be practical and prioritize completion.
- Avoid lengthy theoretical detours unless they are necessary to understand a code change.
- Do not rewrite working modules unnecessarily.
- Do not introduce dependencies without a clear reason.
- Prefer the current local GPU setup.
- Keep experiments small enough to finish.
- Avoid running multiple GPU-heavy experiments simultaneously.
- Do not fabricate metrics.
- Do not claim full paper reproduction.
- Preserve working checkpoints and datasets.
- Explain significant code changes and why they are needed.

The user wants to understand the implementation, not merely receive a large amount of generated code. Explain important architectural and experimental decisions clearly, but keep routine debugging concise.

---

# 18. Definition of completion

The project can be considered complete for its intended scope when:

- [ ] The HRM architecture runs successfully.
- [ ] The dataset is validated.
- [ ] Training runs reliably.
- [ ] Checkpoints can be saved and resumed.
- [ ] Evaluation correctly handles ACT.
- [ ] Cell accuracy and exact-puzzle accuracy are verified.
- [ ] A baseline experiment is completed.
- [ ] At least one architectural ablation is completed; ideally both planned ablations.
- [ ] Results are recorded with their experimental settings.
- [ ] Deviations from the research paper are documented.
- [ ] README is updated.
- [ ] GitHub contains the final code and documentation.

**The central success criterion is not matching the research paper's reported accuracy. It is demonstrating a functioning HRM-style Sudoku model through a sound implementation and transparent experiments.**

---

# Final handoff note

The most important thing to understand is that substantial groundwork has already been completed:

- The environment is configured.
- The model architecture has been implemented.
- The attention mechanism has been adapted.
- The dataset has been prepared and validated.
- The model forward pass has been tested.
- Training and evaluation scripts already exist.

The next stage is primarily about **debugging, making the existing scripts configurable, validating ACT evaluation, running controlled experiments, and documenting the results**.

Start by inspecting the supplied research paper and the existing code. Do not restart the project or regenerate the dataset without identifying a concrete problem.

**Current state: The implementation is not yet experimentally complete. The next reliable milestone is a verified baseline training and evaluation run.**