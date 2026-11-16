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
    """Fit class priors, means and diagonal population variances on training data."""
    validate_points(points)
    if len(labels) != len(points) or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError("Each training row needs a nonempty class label.")
    if (isinstance(variance_floor, bool)
            or not isinstance(variance_floor, (int, float))
            or not math.isfinite(variance_floor) or variance_floor <= 0):
        raise ValueError("The variance floor must be finite and positive.")
    classes = sorted(set(labels))
    counts = Counter(labels)
    means, variances = {}, {}
    for label in classes:
        rows = [row for row, actual in zip(points, labels) if actual == label]
        means[label] = [sum(row[j] for row in rows) / len(rows) for j in range(2)]
        variances[label] = [
            max(sum((row[j] - means[label][j]) ** 2 for row in rows) / len(rows),
                variance_floor) for j in range(2)
        ]
    return {
        "classes": classes, "means": means, "variances": variances,
        "priors": {label: counts[label] / len(labels) for label in classes},
    }


def log_scores(model, query):
    """Sum log prior and independent Gaussian feature log densities."""
    validate_points([query])
    result = {}
    for label in model["classes"]:
        score = math.log(model["priors"][label])
        for value, mean, variance in zip(
            query, model["means"][label], model["variances"][label]
        ):
            score -= 0.5 * (math.log(2 * math.pi) + math.log(variance)
                            + (value - mean) ** 2 / variance)
        result[label] = score
    return result


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
