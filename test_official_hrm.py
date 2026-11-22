import torch
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)

device = "cuda"

config = HierarchicalReasoningModel_ACTV1Config(
    batch_size=2,
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

model = HierarchicalReasoningModel_ACTV1(config.model_dump()).to(device)

batch = {
    "inputs": torch.randint(1, 11, (2, 81), device=device),
    "labels": torch.randint(2, 11, (2, 81), device=device),
    "puzzle_identifiers": torch.zeros(2, dtype=torch.int32, device=device),
}

carry = model.initial_carry(batch)

carry.steps = carry.steps.to(device)
carry.halted = carry.halted.to(device)

carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)

carry.current_data = {
    k: v.to(device) for k, v in carry.current_data.items()
}

carry, outputs = model(carry, batch)

print("Logits shape:", outputs["logits"].shape)
print("Halt logits shape:", outputs["q_halt_logits"].shape)
print("Continue logits shape:", outputs["q_continue_logits"].shape)
print("GPU memory (MB):", round(torch.cuda.memory_allocated() / 1024**2, 2))
print("OFFICIAL HRM FORWARD PASS: SUCCESS")
