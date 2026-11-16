import csv
import json
import math
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DEFAULT_DEPTH = 1


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
    # TODO: implement gini from the supplied tree contract.
    raise NotImplementedError("Complete gini before fitting.")


def best_split(points, labels):
    # TODO: implement best_split from the supplied tree contract.
    raise NotImplementedError("Complete best_split before fitting.")


def grow(points, labels, depth, max_depth):
    counts = dict(sorted(Counter(labels).items()))
    node = {"counts": counts, "prediction": min(counts, key=lambda label: (-counts[label], label)),
            "impurity": gini(labels)}
    if depth >= max_depth or len(counts) == 1:
        return node
    split = best_split(points, labels)
    if split is None or split["impurity"] >= node["impurity"] - 1e-12:
        return node
    node.update(feature=split["feature"], threshold=split["threshold"],
                gain=node["impurity"] - split["impurity"])
    for side in ["left", "right"]:
        indices = split[side]
        node[side] = grow([points[i] for i in indices], [labels[i] for i in indices],
                          depth + 1, max_depth)
    return node


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
