
import torch
import torch.nn as nn

from models.hrm import HRMCore
from datasets.sudoku_dataset import SudokuDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
dataset = SudokuDataset(split="train")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

batch = next(iter(loader))

inputs = batch["inputs"].to(device)
labels = batch["labels"].to(device)

# Initialize HRM
model = HRMCore(
    vocab_size=11,
    hidden_size=256,
    num_heads=4,
    H_layers=2,
    L_layers=2,
    H_cycles=2,
    L_cycles=2
).to(device)

# Forward pass
logits = model(inputs)

print("Device:", device)
print("Input shape:", inputs.shape)
print("Output shape:", logits.shape)

# Loss
criterion = nn.CrossEntropyLoss()

loss = criterion(
    logits.reshape(-1, 11),
    labels.reshape(-1)
)

print("Loss:", loss.item())

# Backpropagation
loss.backward()

print("Backward pass: SUCCESS")

# Parameter count
params = sum(
    p.numel() for p in model.parameters()
)

print("Total parameters:", f"{params:,}")
print("GPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")
