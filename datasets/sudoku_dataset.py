
import numpy as np
import torch
from torch.utils.data import Dataset


class SudokuDataset(Dataset):

    def __init__(self, root="data/sudoku", split="train"):
        self.path = f"{root}/{split}"

        self.inputs = np.load(
            f"{self.path}/all__inputs.npy",
            mmap_mode="r"
        )

        self.labels = np.load(
            f"{self.path}/all__labels.npy",
            mmap_mode="r"
        )

        self.puzzle_identifiers = np.load(
            f"{self.path}/all__puzzle_identifiers.npy",
            mmap_mode="r"
        )

        assert len(self.inputs) == len(self.labels)
        assert len(self.inputs) == len(self.puzzle_identifiers)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        return {
            "inputs": torch.tensor(
                self.inputs[idx].astype(np.int64)
            ),
            "labels": torch.tensor(
                self.labels[idx].astype(np.int64)
            ),
            "puzzle_identifiers": torch.tensor(
                int(self.puzzle_identifiers[idx]),
                dtype=torch.long
            )
        }
