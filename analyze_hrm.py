import argparse
import json
import os
import numpy as np
import torch

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)
from models.losses import ACTLossHead, IGNORE_LABEL_ID


def move_carry_to_device(carry, device):
    carry.steps = carry.steps.to(device)
    carry.halted = carry.halted.to(device)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.current_data = {
        k: v.to(device) for k, v in carry.current_data.items()
    }
    return carry


def format_board(tokens):
    """Format an 81-token board (1=blank, 2..10=digits 1..9) into a 9x9 grid string."""
    digits = np.asarray(tokens, dtype=np.int64) - 1
    lines = []
    for r in range(9):
        row = []
        for c in range(9):
            v = int(digits[r * 9 + c])
            row.append("." if v == 0 else str(v))
            if c in (2, 5):
                row.append("|")
        lines.append(" ".join(row))
        if r in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)


def is_valid_sudoku(tokens):
    """Check if an 81-token prediction (2..10) satisfies all Sudoku rules."""
    arr = np.asarray(tokens, dtype=np.int64) - 1
    if arr.shape != (81,) or np.any(arr < 1) or np.any(arr > 9):
        return False
    grid = arr.reshape(9, 9)
    valid = set(range(1, 10))
    for i in range(9):
        if set(grid[i]) != valid or set(grid[:, i]) != valid:
            return False
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            if set(grid[r:r + 3, c:c + 3].flatten()) != valid:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=32,
                        help="Number of test batches (x batch-size) to analyze")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = HierarchicalReasoningModel_ACTV1Config(**checkpoint["config"])
    config.batch_size = args.batch_size
    loss_type = checkpoint.get("loss_type", "stablemax_cross_entropy")

    base_model = HierarchicalReasoningModel_ACTV1(config.model_dump()).to(device)
    model = ACTLossHead(base_model, loss_type=loss_type).to(device)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    step_info = checkpoint.get("step", "unknown")
    del checkpoint

    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path="data/sudoku",
        global_batch_size=args.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(dataset_config, split="test")
    iterator = iter(dataset)

    T = config.halt_max_steps
    step_cell_correct = np.zeros(T, dtype=np.float64)
    step_exact_correct = np.zeros(T, dtype=np.float64)
    step_blank_correct = np.zeros(T, dtype=np.float64)
    step_would_halt = np.zeros(T, dtype=np.float64)
    total_puzzles = 0
    total_cells = 0
    total_blanks = 0

    clue_bins = {
        "<=22 clues": [0, 0],
        "23-25 clues": [0, 0],
        "26-28 clues": [0, 0],
        ">=29 clues": [0, 0],
    }

    q_margins_solved = []
    q_margins_unsolved = []
    solved_examples = []

    with torch.no_grad():
        for b_idx in range(args.batches):
            try:
                _, batch, _ = next(iterator)
            except StopIteration:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch["inputs"]
            labels = batch["labels"]
            valid_item = (labels != IGNORE_LABEL_ID).sum(-1) > 0
            blank_mask = (inputs == 1) & (labels != IGNORE_LABEL_ID)

            carry = move_carry_to_device(model.initial_carry(batch), device)

            final_preds = None
            final_q_margin = None

            for act_idx in range(T):
                carry, _, _, detached, _ = model(
                    return_keys=["logits", "q_halt_logits", "q_continue_logits"],
                    carry=carry,
                    batch=batch,
                )
                preds = torch.argmax(detached["logits"], dim=-1)
                cell_ok = ((preds == labels) & (labels != IGNORE_LABEL_ID))
                exact_ok = (cell_ok.sum(-1) == (labels != IGNORE_LABEL_ID).sum(-1)) & valid_item
                blank_ok = ((preds == labels) & blank_mask)
                q_margin = detached["q_halt_logits"] - detached["q_continue_logits"]

                step_cell_correct[act_idx] += cell_ok[valid_item].sum().item()
                step_exact_correct[act_idx] += exact_ok.sum().item()
                step_blank_correct[act_idx] += blank_ok[valid_item].sum().item()
                step_would_halt[act_idx] += ((q_margin > 0) & valid_item).sum().item()

                if act_idx == T - 1:
                    final_preds = preds
                    final_q_margin = q_margin
                    final_exact_ok = exact_ok

            n_valid = int(valid_item.sum().item())
            total_puzzles += n_valid
            total_cells += int((labels[valid_item] != IGNORE_LABEL_ID).sum().item())
            total_blanks += int(blank_mask[valid_item].sum().item())

            clue_counts = ((inputs != 1) & (labels != IGNORE_LABEL_ID)).sum(-1)
            for idx in range(inputs.shape[0]):
                if not valid_item[idx]:
                    continue
                solved = bool(final_exact_ok[idx].item())
                margin = float(final_q_margin[idx].item())
                if solved:
                    q_margins_solved.append(margin)
                else:
                    q_margins_unsolved.append(margin)

                nc = int(clue_counts[idx].item())
                if nc <= 22:
                    key = "<=22 clues"
                elif nc <= 25:
                    key = "23-25 clues"
                elif nc <= 28:
                    key = "26-28 clues"
                else:
                    key = ">=29 clues"
                clue_bins[key][0] += int(solved)
                clue_bins[key][1] += 1

                if solved and len(solved_examples) < args.max_examples:
                    pred_np = final_preds[idx].cpu().numpy()
                    inp_np = inputs[idx].cpu().numpy()
                    lbl_np = labels[idx].cpu().numpy()
                    solved_examples.append({
                        "batch_index": b_idx,
                        "item_index": idx,
                        "clues": nc,
                        "q_margin": round(margin, 4),
                        "valid_sudoku": is_valid_sudoku(pred_np),
                        "input_grid": format_board(inp_np),
                        "pred_grid": format_board(pred_np),
                        "label_grid": format_board(lbl_np),
                    })

    trajectory = []
    for t in range(T):
        trajectory.append({
            "act_step": t + 1,
            "cell_acc_pct": round(100.0 * step_cell_correct[t] / max(1, total_cells), 2),
            "blank_acc_pct": round(100.0 * step_blank_correct[t] / max(1, total_blanks), 2),
            "exact_acc_pct": round(100.0 * step_exact_correct[t] / max(1, total_puzzles), 2),
            "would_halt_pct": round(100.0 * step_would_halt[t] / max(1, total_puzzles), 2),
        })

    difficulty_breakdown = {
        k: {
            "solved": v[0],
            "total": v[1],
            "exact_acc_pct": round(100.0 * v[0] / max(1, v[1]), 2),
        }
        for k, v in clue_bins.items()
    }

    summary = {
        "checkpoint": args.checkpoint,
        "step": step_info,
        "test_puzzles": total_puzzles,
        "act_trajectory": trajectory,
        "difficulty_breakdown": difficulty_breakdown,
        "q_margin_solved_mean": round(float(np.mean(q_margins_solved)), 4) if q_margins_solved else None,
        "q_margin_unsolved_mean": round(float(np.mean(q_margins_unsolved)), 4) if q_margins_unsolved else None,
        "solved_count": len(q_margins_solved),
        "unsolved_count": len(q_margins_unsolved),
        "solved_examples": solved_examples,
    }

    print(f"Analyzed {total_puzzles} puzzles from {args.checkpoint} (step {step_info})")
    print("\nACT Step-by-Step Trajectory:")
    for row in trajectory:
        print(
            f"  Step {row['act_step']:2d}: Cell={row['cell_acc_pct']:5.2f}% | "
            f"Blank={row['blank_acc_pct']:5.2f}% | Exact={row['exact_acc_pct']:5.2f}% | "
            f"Q-Halt={row['would_halt_pct']:5.2f}%"
        )

    print("\nDifficulty Breakdown (by clue count):")
    for k, v in difficulty_breakdown.items():
        print(f"  {k:12s}: {v['solved']}/{v['total']} ({v['exact_acc_pct']:.2f}%)")

    print(
        f"\nQ-margin (q_halt - q_continue): "
        f"solved={summary['q_margin_solved_mean']} (n={len(q_margins_solved)}), "
        f"unsolved={summary['q_margin_unsolved_mean']} (n={len(q_margins_unsolved)})"
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nDiagnostics saved to: {args.output}")


if __name__ == "__main__":
    main()
