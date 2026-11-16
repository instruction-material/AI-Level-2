import csv
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def finite(values):
    if not values or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) for value in values
    ):
        raise ValueError("Supply a nonempty sequence of finite numbers.")


def paired(x_values, y_values):
    finite(x_values)
    finite(y_values)
    if len(x_values) != len(y_values):
        raise ValueError("Each x value needs exactly one target.")


def mse(actual, predicted):
    paired(actual, predicted)
    return sum((a - b) ** 2 for a, b in zip(actual, predicted)) / len(actual)


def read_split(name):
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["x"]) for row in rows], [float(row["target"]) for row in rows]


def solve(matrix, target):
    """Small dense Gaussian elimination with partial pivoting for this fixture."""
    size = len(target)
    augmented = [list(row) + [value] for row, value in zip(matrix, target)]
    scale = max(1.0, max(abs(value) for row in matrix for value in row))
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12 * scale:
            raise ValueError("Training design is rank-deficient or poorly scaled for this demo solver.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiple = augmented[row][column]
            augmented[row] = [value - multiple * chosen
                              for value, chosen in zip(augmented[row], augmented[column])]
    result = [row[-1] for row in augmented]
    finite(result)
    return result


def fit_polynomial(x_values, y_values, degree):
    # TODO: implement fit_polynomial from the supplied regression contract.
    raise NotImplementedError("Complete fit_polynomial before fitting.")


def predict(coefficients, x_values):
    finite(coefficients)
    finite(x_values)
    result = [sum(coefficient * x ** power for power, coefficient in enumerate(coefficients))
              for x in x_values]
    finite(result)
    return result


def choose_degree(train_x, train_y, validation_x, validation_y):
    paired(validation_x, validation_y)
    candidates = []
    for degree in [1, 2]:
        coefficients = fit_polynomial(train_x, train_y, degree)
        error = mse(validation_y, predict(coefficients, validation_x))
        candidates.append((error, degree, coefficients))
    # An exact validation-loss tie chooses the simpler model.
    error, degree, coefficients = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return {"degree": degree, "coefficients": coefficients, "validation_mse": error}


def main():
    train_x, train_y = read_split("train.csv")
    validation_x, validation_y = read_split("validation.csv")
    chosen = choose_degree(train_x, train_y, validation_x, validation_y)
    # Open the final test labels only after model selection.
    test_x, test_y = read_split("test.csv")
    predicted = predict(chosen["coefficients"], test_x)
    print(json.dumps({
        "training_rows": len(train_x), "validation_rows": len(validation_x), "test_rows": len(test_x),
        "chosen_degree": chosen["degree"],
        "coefficients": [round(value, 8) for value in chosen["coefficients"]],
        "predictions": [round(value, 8) for value in predicted],
        "validation_mse": round(chosen["validation_mse"], 12),
        "test_mse": round(mse(test_y, predicted), 12),
    }, sort_keys=True))



if __name__ == "__main__":
    main()
