"""Classify supplied fictional messages with a training-only text pipeline."""
import argparse
import json
from pathlib import Path
import time


def fit_pipeline(messages, targets, binary=False):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("words", CountVectorizer(binary=binary)),
                         ("classifier", MultinomialNB(alpha=1.0))])
    pipeline.fit(messages, targets)
    return pipeline


def metrics(truth, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(truth, predictions)),
            "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "support": support.tolist(),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist()}


def main():
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("fictional-emails.csv")
    frame = pd.read_csv(source)
    raw_rows = len(frame)
    if set(frame.columns) != {"message_id", "text", "label_num"} or frame.isna().any().any():
        raise ValueError("Use the supplied complete fictional message table")
    if set(frame["label_num"]) != {0, 1} or frame.groupby("text")["label_num"].nunique().max() > 1:
        raise ValueError("Use consistent ham=0/spam=1 labels")
    frame = frame.drop_duplicates(subset=["text", "label_num"]).reset_index(drop=True)
    targets = frame["label_num"].to_numpy()
    development, test = train_test_split(np.arange(len(frame)), test_size=0.2,
                                        random_state=42, stratify=targets)
    train, validation = train_test_split(development, test_size=0.25,
                                         random_state=42, stratify=targets[development])
    messages = frame["text"]
    started = time.perf_counter()
    pipeline = fit_pipeline(messages.iloc[train], targets[train])
    training_seconds = time.perf_counter() - started
    binary = fit_pipeline(messages.iloc[train], targets[train], binary=True)
    validation_predictions = pipeline.predict(messages.iloc[validation])
    binary_predictions = binary.predict(messages.iloc[validation])
    # The default count model is fixed in advance; validation compares one preprocessing change.
    predictions = pipeline.predict(messages.iloc[test])
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), targets[train])
    baseline_predictions = baseline.predict(np.zeros((len(test), 1)))
    report = {
        "dataset": source.name, "fictional": True, "input_rows": raw_rows,
        "rows": len(frame), "duplicates_removed": raw_rows - len(frame),
        "random_state": 42, "train_ids": train.tolist(),
        "validation_ids": validation.tolist(), "test_ids": test.tolist(),
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "training_seconds": training_seconds, "class_labels": [0, 1],
        "model_fixed_before_test": True, "chosen_model": "default count vectorizer",
        "validation_metrics": metrics(targets[validation], validation_predictions),
        "binary_validation_metrics": metrics(targets[validation], binary_predictions),
        "vocabulary": sorted(pipeline.named_steps["words"].vocabulary_),
        "class_count": pipeline.named_steps["classifier"].class_count_.tolist(),
        "class_token_counts": pipeline.named_steps["classifier"].feature_count_.tolist(),
        "vocabulary_by_index": pipeline.named_steps["words"].get_feature_names_out().tolist(),
        "test_truth": targets[test].tolist(), "test_predictions": predictions.tolist(),
        "metrics": metrics(targets[test], predictions),
        "baseline_predictions": baseline_predictions.tolist(),
        "baseline_metrics": metrics(targets[test], baseline_predictions),
        "misclassified_row_ids": frame.iloc[test[predictions != targets[test]]]["message_id"].tolist(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "limitation": "The authored fixture has easy repeated patterns. These scores describe this classroom example, not general email filtering or private inbox messages.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spam-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["rows", "duplicates_removed", "train_rows", "validation_rows",
                                                 "test_rows", "training_seconds", "metrics", "baseline_metrics",
                                                 "misclassified_row_ids", "versions", "limitation"]}))


if __name__ == "__main__":
    main()
