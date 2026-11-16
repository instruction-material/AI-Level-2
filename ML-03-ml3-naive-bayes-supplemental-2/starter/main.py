import csv
import json
import math
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def validate_points(points):
    if not points or any(
        len(row) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in row
        ) for row in points
    ):
        raise ValueError("Supply nonempty finite two-feature measurements.")


def fit(points, labels, variance_floor=1e-6):
    # TODO: implement fit using the supplied training contract.
    raise NotImplementedError("Complete fit before predicting.")


def log_scores(model, query):
    # TODO: implement log_scores using the supplied training contract.
    raise NotImplementedError("Complete log_scores before predicting.")


def predict(model, query):
    scores = log_scores(model, query)
    return min(scores, key=lambda label: (-scores[label], label))


def load_fixture():
    def read(name):
        with (DATA / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [(float(row["x"]), float(row["y"])) for row in rows], [
            row["label"] for row in rows
        ]
    train, labels = read("train.csv")
    test, expected = read("test.csv")
    return train, labels, test, expected


def main():
    train, labels, test, expected = load_fixture()
    model = fit(train, labels)
    predicted = [predict(model, point) for point in test]
    print(json.dumps({
        "training_rows": len(train), "test_rows": len(test),
        "means": model["means"], "variances": model["variances"],
        "predictions": predicted,
        "accuracy": sum(a == b for a, b in zip(predicted, expected)) / len(test),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
