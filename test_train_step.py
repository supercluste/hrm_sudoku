import torch

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed

device = "cuda"
batch_size = 4

# Official architecture
config = HierarchicalReasoningModel_ACTV1Config(
    batch_size=batch_size,
    seq_len=81,
    puzzle_emb_ndim=512,
    num_puzzle_identifiers=1,
    vocab_size=11,
    H_cycles=2,
    L_cycles=2,
    H_layers=4,
    L_layers=4,
    hidden_size=512,
    expansion=4,
    num_heads=8,
    pos_encodings="rope",
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    forward_dtype="bfloat16",
)

base_model = HierarchicalReasoningModel_ACTV1(
    config.model_dump()
).to(device)

model = ACTLossHead(
    base_model,
    loss_type="stablemax_cross_entropy"
).to(device)

# Official dataset sampler
dataset_config = PuzzleDatasetConfig(
    seed=42,
    dataset_path="data/sudoku",
    global_batch_size=batch_size,
    test_set_mode=False,
    epochs_per_iter=1,
    rank=0,
    num_replicas=1,
)

dataset = PuzzleDataset(dataset_config, split="train")
_, batch, _ = next(iter(dataset))
batch = {k: v.to(device) for k, v in batch.items()}

# Initialize and move carry to GPU
carry = model.initial_carry(batch)

carry.steps = carry.steps.to(device)
carry.halted = carry.halted.to(device)
carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
carry.current_data = {
    k: v.to(device) for k, v in carry.current_data.items()
}

# Restore local embedding weights as a leaf tensor after CUDA transfer
base_model.puzzle_emb.local_weights = (
    base_model.puzzle_emb.local_weights.detach().requires_grad_(True)
)
# Separate official sparse embedding optimizer
puzzle_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
    base_model.puzzle_emb.buffers(),
    lr=1e-2,
    weight_decay=0.1,
    world_size=1,
)

# Temporary replacement for AdamATan2
main_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

model.train()
main_optimizer.zero_grad(set_to_none=True)
puzzle_optimizer.zero_grad(set_to_none=True)

carry, loss, metrics, _, halted = model(
    return_keys=[],
    carry=carry,
    batch=batch,
)

print("Loss:", loss.item())
print("Metrics:", {k: v.item() for k, v in metrics.items()})

loss.backward()

main_optimizer.step()
puzzle_optimizer.step()

print("GPU memory (MB):", round(torch.cuda.memory_allocated() / 1024**2, 2))
print("REAL SUDOKU TRAINING UPDATE: SUCCESS")
