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
    """Fit training-only token counts and class document priors."""
    if not documents or len(documents) != len(labels) or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError("Supply equally sized nonempty documents and labels.")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Smoothing alpha must be finite and positive.")
    tokens = [tokenize(document) for document in documents]
    vocabulary = set(token for document in tokens for token in document)
    if not vocabulary:
        raise ValueError("Training needs at least one alphabetic token.")
    classes = sorted(set(labels))
    counts = {label: Counter() for label in classes}
    for document, label in zip(tokens, labels):
        counts[label].update(document)
    class_documents = Counter(labels)
    return {
        "classes": classes, "vocabulary": vocabulary, "counts": counts,
        "totals": {label: sum(counts[label].values()) for label in classes},
        "priors": {label: class_documents[label] / len(labels) for label in classes},
        "alpha": alpha,
    }


def log_scores(model, text):
    """Use repeated token counts and Laplace-smoothed likelihoods in log space."""
    query = Counter(token for token in tokenize(text)
                    if token in model["vocabulary"])
    scores = {}
    for label in model["classes"]:
        denominator = model["totals"][label] + (
            model["alpha"] * len(model["vocabulary"])
        )
        score = math.log(model["priors"][label])
        for token, count in query.items():
            probability = (model["counts"][label][token] + model["alpha"]) / denominator
            score += count * math.log(probability)
        scores[label] = score
    return scores


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
