
import os
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def shuffle_sudoku(board, solution, rng):
    digit_map = np.concatenate((
        np.array([0]),
        rng.permutation(np.arange(1, 10))
    ))

    transpose = rng.random() < 0.5

    bands = rng.permutation(3)
    rows = np.concatenate([
        b * 3 + rng.permutation(3) for b in bands
    ])

    stacks = rng.permutation(3)
    cols = np.concatenate([
        s * 3 + rng.permutation(3) for s in stacks
    ])

    mapping = np.array([
        rows[i // 9] * 9 + cols[i % 9]
        for i in range(81)
    ])

    def transform(x):
        if transpose:
            x = x.T

        return digit_map[
            x.flatten()[mapping].reshape(9, 9)
        ]

    return transform(board), transform(solution)


def download_csv(split):
    return hf_hub_download(
        repo_id="sapientinc/sudoku-extreme",
        filename=f"{split}.csv",
        repo_type="dataset",
    )


def process_split(
    split,
    output_dir,
    subsample=None,
    num_aug=0,
    seed=42,
):
    print(f"\nProcessing {split} split...")

    path = download_csv(split)
    rng = np.random.default_rng(seed)

    save_dir = os.path.join(output_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    inputs_path = os.path.join(save_dir, "all__inputs.npy")
    labels_path = os.path.join(save_dir, "all__labels.npy")

    # Count source rows.
    with open(path, newline="") as f:
        total_rows = sum(1 for _ in csv.reader(f)) - 1

    if split == "train" and subsample is not None:
        selected_count = min(subsample, total_rows)
        selected_indices = np.sort(
            rng.choice(total_rows, selected_count, replace=False)
        )
        selected_set = set(selected_indices.tolist())
    else:
        selected_count = total_rows
        selected_set = None

    variants = num_aug + 1 if split == "train" else 1
    total_examples = selected_count * variants

    inputs = np.lib.format.open_memmap(
        inputs_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_examples, 81),
    )

    labels = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_examples, 81),
    )

    group_indices = np.lib.format.open_memmap(
        os.path.join(save_dir, "all__group_indices.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(selected_count + 1,),
    )

    group_indices[0] = 0

    output_index = 0
    selected_index = 0

    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)

        for row_index, row in enumerate(
            tqdm(reader, total=total_rows, desc="Processing")
        ):
            if selected_set is not None and row_index not in selected_set:
                continue

            source, q, a, rating = row

            board = np.array(
                [int(c) for c in q.replace(".", "0")],
                dtype=np.uint8,
            ).reshape(9, 9)

            solution = np.array(
                [int(c) for c in a],
                dtype=np.uint8,
            ).reshape(9, 9)

            for variant in range(variants):
                if variant == 0:
                    x, y = board, solution
                else:
                    x, y = shuffle_sudoku(board, solution, rng)

                inputs[output_index] = x.flatten() + 1
                labels[output_index] = y.flatten() + 1

                output_index += 1

            selected_index += 1
            group_indices[selected_index] = output_index

    inputs.flush()
    labels.flush()
    group_indices.flush()

    np.save(
        os.path.join(save_dir, "all__puzzle_indices.npy"),
        np.arange(total_examples + 1, dtype=np.int32),
    )

    np.save(
        os.path.join(save_dir, "all__puzzle_identifiers.npy"),
        np.zeros(total_examples, dtype=np.int32),
    )

    metadata = {
        "seq_len": 81,
        "vocab_size": 11,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": selected_count,
        "mean_puzzle_examples": variants,
        "sets": ["all"],
    }

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Original puzzles: {selected_count:,}")
    print(f"Generated examples: {total_examples:,}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="data/sudoku",
    )

    parser.add_argument(
        "--subsample-size",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--num-aug",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    process_split(
        "train",
        args.output_dir,
        args.subsample_size,
        args.num_aug,
    )

    # Preserve the existing test split.
    print("\nExisting test split left unchanged.")
