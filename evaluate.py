import argparse
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
        default="checkpoints/hrm_step_500.pt"
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda"
    batch_size = args.batch_size

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

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

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

    totals = {
        "count": 0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "steps": 0.0,
        "lm_loss": 0.0,
    }

    evaluated = 0

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

            # Evaluation always uses all 16 reasoning steps.
            for _ in range(config.halt_max_steps):
                carry, loss, metrics, _, _ = model(
                    return_keys=[],
                    carry=carry,
                    batch=batch,
                )

            for key in totals:
                value = metrics[key]
                totals[key] += (
                    value.item() if torch.is_tensor(value) else value
                )

            evaluated += 1

            print(
                f"Batch {evaluated}/{args.batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Exact correct: {metrics['exact_accuracy'].item()}/"
                f"{metrics['count'].item()}",
                flush=True,
            )

    count = totals["count"]

    if count == 0:
        print("No test examples were evaluated. Check the test dataset.")
        return

    print("\n========== TEST RESULTS ==========")
    print("Checkpoint:", args.checkpoint)
    print("Test puzzles:", int(count))
    print(f"Cell accuracy: {100 * totals['accuracy'] / count:.2f}%")
    print(
        f"Exact puzzle accuracy: "
        f"{100 * totals['exact_accuracy'] / count:.2f}%"
    )
    print(
        f"Average reasoning steps: {totals['steps'] / count:.2f}"
    )
    print(
        f"Average LM loss: {totals['lm_loss'] / count:.4f}"
    )


if __name__ == "__main__":
    main()
