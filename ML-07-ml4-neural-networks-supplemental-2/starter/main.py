import csv
import json
import math
from pathlib import Path
import random

DATA = Path(__file__).resolve().parent / "data"
PARAMETERS = 3
DEFAULT_STEPS = 2000


def validate(points, labels=None):
    if not points or any(
        len(row) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1 for value in row
        ) for row in points
    ):
        raise ValueError("Supply nonempty two-feature measurements in [0,1].")
    if labels is not None and (len(labels) != len(points) or any(
        type(label) not in (int, float) or label not in (0, 1) for label in labels
    )):
        raise ValueError("Each row needs a numeric binary label 0 or 1.")


def validate_parameters(parameters):
    if len(parameters) != PARAMETERS or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) for value in parameters
    ):
        raise ValueError("Supply the declared number of finite parameters.")


def sigmoid(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("The logit must be finite.")
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def initial_parameters():
    return [0.0, 0.0, 0.0]


def forward(parameters, point):
    validate_parameters(parameters)
    validate([point])
    logit = parameters[0] * point[0] + parameters[1] * point[1] + parameters[2]
    return {"logit": logit, "probability": sigmoid(logit)}


def loss_and_gradient(parameters, points, labels):
    # TODO: implement loss_and_gradient using the supplied model equations.
    raise NotImplementedError("Complete loss_and_gradient before training.")



def fit(points, labels, steps=DEFAULT_STEPS, learning_rate=0.5):
    validate(points, labels)
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("Choose an integer from 1 through 10000 training steps.")
    if (isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1):
        raise ValueError("Choose a finite learning rate in (0,1].")
    parameters = initial_parameters()
    initial_loss, _ = loss_and_gradient(parameters, points, labels)
    for _ in range(steps):
        _, gradient = loss_and_gradient(parameters, points, labels)
        parameters = [value - learning_rate * slope
                      for value, slope in zip(parameters, gradient)]
    final_loss, _ = loss_and_gradient(parameters, points, labels)
    return {"parameters": parameters, "training_rows": len(points),
            "initial_loss": initial_loss, "final_loss": final_loss}


def predict(parameters, point):
    return int(forward(parameters, point)["probability"] >= 0.5)


def load_fixture():
    def read(name):
        with (DATA / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [(float(row["x"]), float(row["y"])) for row in rows], [
            int(row["label"]) for row in rows
        ]
    points, labels = read("train.csv")
    test, expected = read("test.csv")
    return points, labels, test, expected


def main():
    points, labels, test, expected = load_fixture()
    model = fit(points, labels)
    predicted = [predict(model["parameters"], point) for point in test]
    print(json.dumps({
        "training_rows": len(points), "test_rows": len(test),
        "initial_loss": round(model["initial_loss"], 6),
        "final_loss": round(model["final_loss"], 6),
        "predictions": predicted,
        "accuracy": sum(a == b for a, b in zip(predicted, expected)) / len(test),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
