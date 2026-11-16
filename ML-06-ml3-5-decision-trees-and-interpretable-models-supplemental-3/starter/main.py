import csv
import json
import math
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DEFAULT_DEPTH = 2


def validate(points, labels=None):
    if not points or any(
        len(row) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in row
        ) for row in points
    ):
        raise ValueError("Supply nonempty finite two-feature measurements.")
    if labels is not None and (len(labels) != len(points) or any(
        not isinstance(label, str) or not label.strip() for label in labels
    )):
        raise ValueError("Each row needs a nonempty class label.")


def gini(labels):
    if not labels:
        raise ValueError("A node must contain labels.")
    counts = Counter(labels)
    return 1 - sum((count / len(labels)) ** 2 for count in counts.values())


def best_split(points, labels):
    """Choose the lowest weighted child impurity; stable feature/threshold ties."""
    best = None
    for feature in range(2):
        values = sorted(set(point[feature] for point in points))
        for low, high in zip(values, values[1:]):
            threshold = low / 2 + high / 2
            if threshold >= high:
                threshold = low
            left = [i for i, row in enumerate(points) if row[feature] <= threshold]
            right = [i for i, row in enumerate(points) if row[feature] > threshold]
            if not left or not right:
                continue
            impurity = (
                len(left) * gini([labels[i] for i in left])
                + len(right) * gini([labels[i] for i in right])
            ) / len(points)
            if best is None or impurity < best["impurity"] - 1e-12:
                best = {"feature": feature, "threshold": threshold,
                        "left": left, "right": right, "impurity": impurity}
    return best


def grow(points, labels, depth, max_depth):
    # TODO: implement grow from the supplied tree contract.
    raise NotImplementedError("Complete grow before fitting.")


def fit(points, labels, max_depth=DEFAULT_DEPTH):
    validate(points, labels)
    if type(max_depth) is not int or not 0 <= max_depth <= 8:
        raise ValueError("Choose an integer depth from 0 through 8.")
    return grow(points, labels, 0, max_depth)


def trace(model, query):
    validate([query])
    node, path = model, []
    while "feature" in node:
        feature, threshold = node["feature"], node["threshold"]
        left = query[feature] <= threshold
        path.append({"feature": feature, "threshold": threshold,
                     "branch": "left" if left else "right"})
        node = node["left"] if left else node["right"]
    return {"prediction": node["prediction"], "path": path, "leaf_counts": node["counts"].copy()}


def predict(model, query):
    return trace(model, query)["prediction"]


def load_fixture():
    def read(name):
        with (DATA / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [(float(row["x"]), float(row["y"])) for row in rows], [
            row["label"] for row in rows
        ]
    points, labels = read("train.csv")
    test, expected = read("test.csv")
    return points, labels, test, expected


def main():
    points, labels, test, expected = load_fixture()
    results = {}
    for depth in [1, 2]:
        model = fit(points, labels, depth)
        predicted = [predict(model, query) for query in test]
        results[str(depth)] = {
            "root_feature": model.get("feature"), "root_threshold": model.get("threshold"),
            "predictions": predicted,
            "accuracy": sum(a == b for a, b in zip(predicted, expected)) / len(test),
        }
    print(json.dumps({"training_rows": len(points), "test_rows": len(test),
                      "depth_comparison": results}, sort_keys=True))


if __name__ == "__main__":
    main()
