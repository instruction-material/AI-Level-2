"""Select the shallowest candidate within 0.02 of the best validation accuracy."""
import math


def choose_depth(scores):
    if not isinstance(scores, dict) or not scores or any(
        type(depth) is not int or not 0 <= depth <= 8
        or isinstance(score, bool) or not isinstance(score, (int, float))
        or not math.isfinite(score) or not 0 <= score <= 1
        for depth, score in scores.items()
    ):
        raise ValueError("Supply finite accuracies for integer depths 0 through 8.")
    best = max(scores.values())
    return min(depth for depth, score in scores.items() if best - score <= 0.02 + 1e-12)
