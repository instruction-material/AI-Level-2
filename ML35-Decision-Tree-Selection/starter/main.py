"""Compare fixed tree depths using validation before opening the test split."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from exercise import choose_depth
from tree_rules import fit, predict, trace

DATA = Path(__file__).resolve().parent / "data"
DEPTHS = (0, 1, 2)


def load_split(name):
    if name not in {"train", "validation", "test"}:
        raise ValueError("Unknown fixed split.")
    with (DATA / (name + ".csv")).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Every split must contain observations.")
    return [[float(row["x"]), float(row["y"])] for row in rows], [
        row["label"] for row in rows
    ]


def accuracy(actual, predicted):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Align nonempty labels and predictions.")
    return sum(a == b for a, b in zip(actual, predicted)) / len(actual)


def select_model(points, labels, validation, validation_labels):
    """No final-test inputs are available during candidate selection."""
    models = {depth: fit(points, labels, depth) for depth in DEPTHS}
    scores = {
        depth: accuracy(validation_labels, [predict(model, row) for row in validation])
        for depth, model in models.items()
    }
    selected = choose_depth(scores.copy())
    if type(selected) is not int or selected not in models:
        raise ValueError("Return one declared integer candidate depth.")
    return selected, models[selected], scores


def experiment():
    points, labels = load_split("train")
    validation, validation_labels = load_split("validation")
    depth, model, scores = select_model(points, labels, validation, validation_labels)
    # Open the reserved split only after the model choice is frozen.
    test, expected = load_split("test")
    predicted = [predict(model, row) for row in test]
    counts = Counter(labels)
    majority = min(counts, key=lambda label: (-counts[label], label))
    return {
        "training_rows": len(points),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_accuracy": scores,
        "selected_depth": depth,
        "selected_tree": model,
        "majority_label": majority,
        "test_predictions": predicted,
        "test_accuracy": accuracy(expected, predicted),
        "baseline_test_accuracy": accuracy(expected, [majority] * len(test)),
        "test_traces": [trace(model, row) for row in test],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the complete JSON report.")
    args = parser.parse_args()
    report = json.dumps(experiment(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
