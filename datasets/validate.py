
import numpy as np

def validate_split(split):
    path = f"data/sudoku/{split}"

    inputs = np.load(f"{path}/all__inputs.npy")
    labels = np.load(f"{path}/all__labels.npy")

    valid_digits = set(range(2, 11))

    for i, (puzzle, solution) in enumerate(zip(inputs, labels)):
        grid = solution.reshape(9, 9)

        # Every row must contain digits 2-10
        for row in grid:
            if set(row) != valid_digits:
                raise ValueError(f"Invalid row in example {i}")

        # Every column must contain digits 2-10
        for col in grid.T:
            if set(col) != valid_digits:
                raise ValueError(f"Invalid column in example {i}")

        # Every 3x3 box must contain digits 2-10
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box = grid[r:r+3, c:c+3].flatten()
                if set(box) != valid_digits:
                    raise ValueError(f"Invalid box in example {i}")

        # Original clues must remain unchanged
        given = puzzle != 1
        if not np.array_equal(puzzle[given], solution[given]):
            raise ValueError(f"Clue mismatch in example {i}")

    print(f"{split}: {len(inputs)} valid Sudoku examples")


if __name__ == "__main__":
    validate_split("train")
    validate_split("test")
