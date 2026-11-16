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
    # TODO: implement mse from the supplied regression contract.
    raise NotImplementedError("Complete mse before fitting.")


def read_split(name):
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["x"]) for row in rows], [float(row["target"]) for row in rows]


def fit(x_values, y_values):
    # TODO: implement fit from the supplied regression contract.
    raise NotImplementedError("Complete fit before fitting.")


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
