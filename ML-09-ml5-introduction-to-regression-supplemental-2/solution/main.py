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


def fit(x_values, y_values):
    paired(x_values, y_values)
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values)) / denominator
        if denominator else 0.0
    )
    intercept = mean_y - slope * mean_x
    finite([slope, intercept])
    return {"slope": slope, "intercept": intercept, "training_target_mean": mean_y}


def predict(model, x_values):
    finite(x_values)
    result = [model["slope"] * x + model["intercept"] for x in x_values]
    finite(result)
    return result


def main():
    train_x, train_y = read_split("train.csv")
    test_x, test_y = read_split("test.csv")
    model = fit(train_x, train_y)
    predicted = predict(model, test_x)
    print(json.dumps({
        "training_rows": len(train_x), "test_rows": len(test_x),
        "slope": model["slope"], "intercept": model["intercept"],
        "predictions": predicted, "test_mse": mse(test_y, predicted),
        "baseline_mse": mse(test_y, [model["training_target_mean"]] * len(test_y)),
    }, sort_keys=True))



if __name__ == "__main__":
    main()
