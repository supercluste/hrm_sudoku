import torch
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

config = PuzzleDatasetConfig(
    seed=42,
    dataset_path="data/sudoku",
    global_batch_size=4,
    test_set_mode=False,
    epochs_per_iter=1,
    rank=0,
    num_replicas=1,
)

dataset = PuzzleDataset(config, split="train")
name, batch, effective_size = next(iter(dataset))

print("Dataset:", name)
print("Inputs:", batch["inputs"].shape, batch["inputs"].dtype)
print("Labels:", batch["labels"].shape, batch["labels"].dtype)
print("Puzzle IDs:", batch["puzzle_identifiers"].shape)
print("Unique labels:", torch.unique(batch["labels"]).tolist())
print("Effective batch size:", effective_size)
print("OFFICIAL DATASET LOADER: SUCCESS")
