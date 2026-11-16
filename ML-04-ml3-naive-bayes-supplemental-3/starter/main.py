import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def tokenize(text):
    if not isinstance(text, str):
        raise ValueError("A document must be text.")
    return re.findall(r"[a-z]+", text.lower())


def fit(documents, labels, alpha=1.0):
    # TODO: implement fit using the supplied training contract.
    raise NotImplementedError("Complete fit before predicting.")


def log_scores(model, text):
    # TODO: implement log_scores using the supplied training contract.
    raise NotImplementedError("Complete log_scores before predicting.")


def predict(model, text):
    scores = log_scores(model, text)
    return min(scores, key=lambda label: (-scores[label], label))


def probabilities(model, text):
    scores = log_scores(model, text)
    largest = max(scores.values())
    weights = {label: math.exp(score - largest) for label, score in scores.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def load_fixture():
    def read(name):
        with (DATA / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [row["text"] for row in rows], [row["label"] for row in rows]
    train, labels = read("train.csv")
    test, expected = read("test.csv")
    return train, labels, test, expected


def main():
    train, labels, test, expected = load_fixture()
    model = fit(train, labels)
    predicted = [predict(model, text) for text in test]
    print(json.dumps({
        "training_rows": len(train), "test_rows": len(test),
        "vocabulary_size": len(model["vocabulary"]),
        "predictions": predicted,
        "claim_prize_probabilities": probabilities(model, "claim prize"),
        "accuracy": sum(a == b for a, b in zip(predicted, expected)) / len(test),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
