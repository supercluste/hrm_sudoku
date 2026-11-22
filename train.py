
import argparse
import os

import torch

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


def move_carry_to_device(carry, device):
    carry.steps = carry.steps.to(device)
    carry.halted = carry.halted.to(device)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.current_data = {
        k: v.to(device) for k, v in carry.current_data.items()
    }
    return carry


def metric_value(value):
    if torch.is_tensor(value):
        return value.detach().float().mean().item()
    return value


def get_lr(step, base_lr, warmup_steps=2000):
    """
    Official schedule:
    - Linear warmup for 2000 steps.
    - Constant learning rate afterwards (lr_min_ratio = 1.0).
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    return base_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"
    batch_size = args.batch_size

    # --------------------------------------------------
    # Official HRM architecture configuration
    # --------------------------------------------------

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

    # Keep the sparse embedding's local gradient buffer a leaf tensor.
    base_model.puzzle_emb.local_weights = (
        base_model.puzzle_emb.local_weights.detach().requires_grad_(True)
    )

    model = ACTLossHead(
        base_model,
        loss_type="stablemax_cross_entropy",
    ).to(device)

    # --------------------------------------------------
    # Official Sudoku dataset loader
    # --------------------------------------------------

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
    iterator = iter(dataset)

    # --------------------------------------------------
    # Optimizers
    # --------------------------------------------------

    # Official sparse puzzle embedding optimizer.
    puzzle_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
        base_model.puzzle_emb.buffers(),
        lr=1e-2,
        weight_decay=0.1,
        world_size=1,
    )

    # TEMPORARY SUBSTITUTION:
    # Official implementation uses AdamATan2.
    # AdamW is being used here instead.
    main_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    warmup_steps = 2000

    model.train()
    carry = None

    os.makedirs("checkpoints", exist_ok=True)

    for step in range(1, args.steps + 1):

        # Get next training batch.
        try:
            _, batch, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            _, batch, _ = next(iterator)

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        # Initialize ACT carry only once.
        if carry is None:
            carry = move_carry_to_device(
                model.initial_carry(batch),
                device,
            )

        # Set learning rates BEFORE optimizer updates.
        main_lr = get_lr(step - 1, 1e-4, warmup_steps)
        puzzle_lr = get_lr(step - 1, 1e-2, warmup_steps)

        for group in main_optimizer.param_groups:
            group["lr"] = main_lr

        for group in puzzle_optimizer.param_groups:
            group["lr"] = puzzle_lr

        # Clear gradients.
        main_optimizer.zero_grad(set_to_none=True)
        puzzle_optimizer.zero_grad(set_to_none=True)

        # Forward pass.
        carry, loss, metrics, _, halted = model(
            return_keys=[],
            carry=carry,
            batch=batch,
        )

        # Official loss scaling.
        (loss / batch_size).backward()

        # Optimizer updates.
        main_optimizer.step()
        puzzle_optimizer.step()

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        if step == 1 or step % 10 == 0:

            values = {
                k: metric_value(v)
                for k, v in metrics.items()
            }

            print(
                f"Step {step}/{args.steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Metrics: {values} | "
                f"LR: {main_lr:.2e} | "
                f"Puzzle LR: {puzzle_lr:.2e}",
                flush=True,
            )

        # --------------------------------------------------
        # Checkpointing
        # --------------------------------------------------

        if step % args.save_every == 0 or step == args.steps:

            path = f"checkpoints/hrm_step_{step}.pt"

            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "main_optimizer": main_optimizer.state_dict(),
                    "puzzle_optimizer": puzzle_optimizer.state_dict(),
                    "config": config.model_dump(),
                },
                path,
            )

            print(
                f"Checkpoint saved: {path}",
                flush=True,
            )

    print("\nTRAINING FINISHED")

    print(
        "GPU memory (MB):",
        round(torch.cuda.memory_allocated() / 1024**2, 2),
    )


if __name__ == "__main__":
    main()
