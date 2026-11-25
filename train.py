
import argparse
import os
import time

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


def get_lr(step, base_lr, warmup_steps):
    """
    Linear warmup then constant learning rate.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    return base_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--h-cycles", type=int, default=2)
    parser.add_argument("--l-cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (used in checkpoint filenames)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"
    batch_size = args.batch_size

    # --------------------------------------------------
    # Determine configuration
    # --------------------------------------------------

    start_step = 0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        config = HierarchicalReasoningModel_ACTV1Config(**checkpoint["config"])
        start_step = checkpoint["step"]
        print(f"  Resuming from step {start_step}")
        print(f"  Config from checkpoint: H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
    else:
        config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=batch_size,
            seq_len=81,
            puzzle_emb_ndim=512,
            num_puzzle_identifiers=1,
            vocab_size=11,
            H_cycles=args.h_cycles,
            L_cycles=args.l_cycles,
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

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

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
    # Resume state
    # --------------------------------------------------

    if args.resume:
        model.load_state_dict(checkpoint["model"])
        main_optimizer.load_state_dict(checkpoint["main_optimizer"])
        puzzle_optimizer.load_state_dict(checkpoint["puzzle_optimizer"])
        print(f"  Model and optimizer states restored.")
        del checkpoint

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset_config = PuzzleDatasetConfig(
        seed=args.seed,
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
    # Training
    # --------------------------------------------------

    warmup_steps = args.warmup_steps

    model.train()
    carry = None

    os.makedirs("checkpoints", exist_ok=True)

    # Determine run name for checkpoint files.
    run_name = args.run_name
    if run_name is None:
        run_name = f"H{config.H_cycles}_L{config.L_cycles}"

    total_steps = start_step + args.steps

    print(f"\nTraining config:")
    print(f"  H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
    print(f"  batch_size={batch_size}, warmup={warmup_steps}")
    print(f"  steps={start_step + 1}..{total_steps}")
    print(f"  run_name={run_name}")
    print(f"  seed={args.seed}")
    print()

    t_start = time.time()

    for step in range(start_step + 1, total_steps + 1):

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

        if step == start_step + 1 or step % 10 == 0:

            values = {
                k: metric_value(v)
                for k, v in metrics.items()
            }

            elapsed = time.time() - t_start
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

            print(
                f"Step {step}/{total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Metrics: {values} | "
                f"LR: {main_lr:.2e} | "
                f"Speed: {steps_per_sec:.2f} steps/s",
                flush=True,
            )

        # --------------------------------------------------
        # Checkpointing
        # --------------------------------------------------

        if step % args.save_every == 0 or step == total_steps:

            path = f"checkpoints/{run_name}_step_{step}.pt"

            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "main_optimizer": main_optimizer.state_dict(),
                    "puzzle_optimizer": puzzle_optimizer.state_dict(),
                    "config": config.model_dump(),
                    "args": vars(args),
                },
                path,
            )

            print(
                f"Checkpoint saved: {path}",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"\nTRAINING FINISHED in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    print(
        "GPU memory (MB):",
        round(torch.cuda.memory_allocated() / 1024**2, 2),
    )


if __name__ == "__main__":
    main()
