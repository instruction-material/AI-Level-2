import csv
import json
import math
from pathlib import Path


def _rows(points):
    rows = [tuple(row) for row in points]
    if not rows or not rows[0]:
        raise ValueError("Provide at least one nonempty feature row")
    width = len(rows[0])
    for row in rows:
        if len(row) != width or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in row
        ):
            raise ValueError("Feature rows must have equal width and finite numbers")
    return rows


def nearest(points, labels, query, k):
    """Return k distance/label pairs, using a declared coordinate/label tie rule."""
    rows = _rows(points)
    _rows([query])
    if len(query) != len(rows[0]):
        raise ValueError("Query width must match training features")
    if len(labels) != len(rows) or any(
        not isinstance(label, str) or not label for label in labels
    ):
        raise ValueError("Provide one nonempty text label per training row")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(rows):
        raise ValueError("k must be an integer from one to the training row count")
    ranked = [
        (math.dist(point, query), point, label)
        for point, label in zip(rows, labels)
    ]
    if any(not math.isfinite(distance) for distance, _, _ in ranked):
        raise ValueError("Distances must remain finite")
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(distance, label) for distance, _, label in ranked[:k]]


def load_fixture():
    """Load the supplied fixed split relative to this file, from any directory."""
    path = Path(__file__).resolve().parent / "data" / "sample.csv"
    training, labels, held_out, expected = [], [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point = (float(row["x"]), float(row["y"]))
            if row["split"] == "train":
                training.append(point)
                labels.append(row["label"])
            elif row["split"] == "test":
                held_out.append(point)
                expected.append(row["label"])
            else:
                raise ValueError("Unknown fixture split")
    return training, labels, held_out, expected

def predict(points, labels, query, k=3, weights="distance"):
    """Vote uniformly or with inverse distances; exact matches vote by themselves."""
    raise NotImplementedError("TODO: validate weights, handle exact matches, and vote")


def main():
    training, labels, held_out, expected = load_fixture()
    predictions = {
        mode: [predict(training, labels, query, 3, mode) for query in held_out]
        for mode in ("uniform", "distance")
    }
    result = {"k": 3, **predictions}
    for mode, values in predictions.items():
        result[mode + "_accuracy"] = sum(a == b for a, b in zip(values, expected)) / len(expected)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
