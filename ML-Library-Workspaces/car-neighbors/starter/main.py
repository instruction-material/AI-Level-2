"""Categorical car acceptability: one-hot neighbors and a training-only baseline."""
import argparse
import json
from pathlib import Path

FEATURES = ["buying", "maintenance", "doors", "persons", "lug boot", "safety"]


from exercise import fit_classifier


def metrics(truth, predicted, labels):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist()}


def main():
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("cars.csv")
    frame = pd.read_csv(source, dtype=str)
    if list(frame.columns) != FEATURES + ["acceptability"] or frame.isna().any().any():
        raise ValueError("Use the supplied complete categorical car table")
    features = frame[FEATURES]
    targets = frame["acceptability"].to_numpy()
    train_ids, test_ids = train_test_split(np.arange(len(frame)), test_size=0.25,
                                          random_state=100, stratify=targets)
    train_x, train_y = features.iloc[train_ids], targets[train_ids]
    test_x, test_y = features.iloc[test_ids], targets[test_ids]
    classifier = fit_classifier(train_x, train_y)
    predicted = classifier.predict(test_x)
    baseline = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    baseline_predictions = baseline.predict(test_x)
    labels = sorted(set(map(str, train_y)))
    query = pd.DataFrame([["vhigh", "vhigh", "2", "2", "small", "low"]], columns=FEATURES)
    unknown = query.copy()
    unknown.loc[0, "buying"] = "unknown"
    transformer = classifier.named_steps["encode"]
    encoder = transformer.named_transformers_["categories"]
    report = {
        "dataset": source.name, "rows": len(frame), "features": FEATURES,
        "train_ids": train_ids.tolist(), "test_ids": test_ids.tolist(),
        "train_rows": len(train_ids), "test_rows": len(test_ids),
        "random_state": 100, "neighbors": 7, "model_fixed_before_test": True,
        "fitted_rows": int(classifier.named_steps["neighbors"].n_samples_fit_),
        "fitted_categories": [list(map(str, values)) for values in encoder.categories_],
        "class_labels": labels, "test_truth": test_y.tolist(),
        "test_predictions": predicted.tolist(),
        "metrics": metrics(test_y, predicted, labels),
        "baseline_predictions": baseline_predictions.tolist(),
        "baseline_metrics": metrics(test_y, baseline_predictions, labels),
        "query": {"raw_features": query.iloc[0].to_dict(),
                  "prediction": str(classifier.predict(query)[0])},
        "unknown_probe": {"prediction": str(classifier.predict(unknown)[0]),
                          "encoded": transformer.transform(unknown)[0].tolist()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
        "limitation": "The supplied acceptability target comes from an evaluation model; this demonstration does not measure real vehicle safety.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["rows", "train_rows", "test_rows", "neighbors",
                                                 "metrics", "baseline_metrics", "query", "versions"]}))


if __name__ == "__main__":
    main()
