import torch
from torch.utils.data import DataLoader
from datasets.sudoku_dataset import SudokuDataset

train_dataset = SudokuDataset(split="train")
test_dataset = SudokuDataset(split="test")

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

batch = next(iter(train_loader))

print("Training examples:", len(train_dataset))
print("Testing examples:", len(test_dataset))

print("\nBatch information:")
print("Input shape:", batch["inputs"].shape)
print("Label shape:", batch["labels"].shape)

print("Input dtype:", batch["inputs"].dtype)
print("Label dtype:", batch["labels"].dtype)

print("\nFirst puzzle:")
print(batch["inputs"][0].reshape(9, 9))

print("\nFirst solution:")
print(batch["labels"][0].reshape(9, 9))
