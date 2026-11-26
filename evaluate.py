import argparse
import json
import os
import time
import torch

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)
from models.losses import ACTLossHead


def move_carry_to_device(carry, device):
    carry.steps = carry.steps.to(device)
    carry.halted = carry.halted.to(device)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.current_data = {
        k: v.to(device) for k, v in carry.current_data.items()
    }
    return carry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/baseline_step_5000.pt"
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON evaluation results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"
    batch_size = args.batch_size

    # --------------------------------------------------
    # Load config from checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    if "config" in checkpoint:
        config = HierarchicalReasoningModel_ACTV1Config(
            **checkpoint["config"]
        )
        print("Config loaded from checkpoint:")
        print(f"  H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
        print(f"  halt_max_steps={config.halt_max_steps}")
    else:
        print("WARNING: No config in checkpoint, using defaults.")
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

    # Override batch_size from CLI (safe in eval mode because sparse embedding
    # skips local_weights/local_ids buffers and indexes weights directly).
    config.batch_size = batch_size

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    loss_type = checkpoint.get("loss_type", "stablemax_cross_entropy")

    base_model = HierarchicalReasoningModel_ACTV1(
        config.model_dump()
    ).to(device)

    model = ACTLossHead(
        base_model,
        loss_type=loss_type
    ).to(device)

    state_dict = {
        k.removeprefix("_orig_mod."): v
        for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    step_info = checkpoint.get("step", "unknown")
    train_args = checkpoint.get("args", {})
    del checkpoint

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path="data/sudoku",
        global_batch_size=batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(dataset_config, split="test")
    iterator = iter(dataset)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    totals = {
        "count": 0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "steps": 0.0,
        "lm_loss": 0.0,
    }

    # Diagnostic: track when Q-head would predict halt (q_halt > q_continue)
    # even though eval mode runs all halt_max_steps for batching.
    predicted_halt_steps_sum = 0.0
    predicted_early_halt_count = 0

    evaluated = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_index in range(args.batches):
            try:
                _, batch, _ = next(iterator)
            except StopIteration:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            carry = move_carry_to_device(
                model.initial_carry(batch), device
            )

            first_halt_step = torch.full(
                (batch["inputs"].shape[0],),
                config.halt_max_steps,
                dtype=torch.int32,
                device=device,
            )
            already_would_halt = torch.zeros(
                batch["inputs"].shape[0],
                dtype=torch.bool,
                device=device,
            )

            # Evaluation: run all halt_max_steps reasoning steps.
            # In eval mode, the model always runs to max steps
            # (no early halting), so metrics are only populated
            # on the final step when halted=True for all items.
            for act_step in range(1, config.halt_max_steps + 1):
                carry, loss, metrics, detached, _ = model(
                    return_keys=["q_halt_logits", "q_continue_logits"],
                    carry=carry,
                    batch=batch,
                )
                would_halt = detached["q_halt_logits"] > detached["q_continue_logits"]
                newly_halted = would_halt & (~already_would_halt)
                first_halt_step = torch.where(
                    newly_halted, act_step, first_halt_step
                )
                already_would_halt = already_would_halt | would_halt

            for key in totals:
                value = metrics[key]
                totals[key] += (
                    value.item() if torch.is_tensor(value) else value
                )

            valid_mask = (batch["labels"] != -100).sum(-1) > 0
            predicted_halt_steps_sum += first_halt_step[valid_mask].float().sum().item()
            predicted_early_halt_count += (
                (first_halt_step[valid_mask] < config.halt_max_steps).sum().item()
            )

            evaluated += 1

            if evaluated == 1 or evaluated % 10 == 0 or evaluated == args.batches:
                print(
                    f"Batch {evaluated}/{args.batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Exact correct: {int(metrics['exact_accuracy'].item())}/"
                    f"{int(metrics['count'].item())}",
                    flush=True,
                )

    elapsed = time.time() - t_start
    count = totals["count"]

    if count == 0:
        print("No test examples were evaluated. Check the test dataset.")
        return

    cell_acc = 100 * totals["accuracy"] / count
    exact_acc = 100 * totals["exact_accuracy"] / count
    q_halt_acc = 100 * totals["q_halt_accuracy"] / count
    avg_steps = totals["steps"] / count
    avg_loss = totals["lm_loss"] / count
    avg_q_predicted_steps = predicted_halt_steps_sum / count
    early_halt_pct = 100 * predicted_early_halt_count / count

    print("\n========== TEST RESULTS ==========")
    print("Checkpoint:", args.checkpoint)
    print("Training step:", step_info)
    print(f"Config: H_cycles={config.H_cycles}, L_cycles={config.L_cycles}, pos_encodings={config.pos_encodings}, loss_type={loss_type}")
    print("Test puzzles:", int(count))
    print(f"Cell accuracy: {cell_acc:.2f}%")
    print(f"Exact puzzle accuracy: {exact_acc:.2f}%")
    print(f"Q-halt accuracy: {q_halt_acc:.2f}%")
    print(f"Average reasoning steps (eval): {avg_steps:.2f}")
    print(f"Avg Q-predicted halt step: {avg_q_predicted_steps:.2f} ({early_halt_pct:.1f}% would halt early)")
    print(f"Average LM loss: {avg_loss:.4f}")
    print(f"Evaluation time: {elapsed:.2f}s ({count / elapsed:.1f} puzzles/s)")

    results = {
        "checkpoint": args.checkpoint,
        "step": step_info,
        "h_cycles": config.H_cycles,
        "l_cycles": config.L_cycles,
        "h_layers": config.H_layers,
        "l_layers": config.L_layers,
        "hidden_size": config.hidden_size,
        "halt_max_steps": config.halt_max_steps,
        "pos_encodings": config.pos_encodings,
        "loss_type": loss_type,
        "train_args": train_args,
        "test_puzzles": int(count),
        "cell_accuracy": round(cell_acc, 2),
        "exact_accuracy": round(exact_acc, 2),
        "q_halt_accuracy": round(q_halt_acc, 2),
        "avg_steps": round(avg_steps, 2),
        "avg_q_predicted_steps": round(avg_q_predicted_steps, 2),
        "early_halt_pct": round(early_halt_pct, 2),
        "avg_lm_loss": round(avg_loss, 4),
        "eval_time_sec": round(elapsed, 2),
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
