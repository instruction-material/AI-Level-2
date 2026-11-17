"""Compare linear and quadratic models for supplied country-year observations."""
import argparse
import json
from pathlib import Path

TARGET = "Life expectancy "


from exercise import fit_model


def metrics(truth, predictions):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"mae_years": float(mean_absolute_error(truth, predictions)),
            "rmse_years": float(mean_squared_error(truth, predictions) ** 0.5),
            "r_squared": float(r2_score(truth, predictions))}


def choose_degree(validation_metrics):
    # Require a two-percent validation-MAE improvement to choose extra complexity.
    return 2 if validation_metrics[2]["mae_years"] < 0.98 * validation_metrics[1]["mae_years"] else 1


def evaluation_evidence(rows, truth, predicted, feature_names, year_threshold):
    import random
    import numpy as np
    errors = np.abs(np.asarray(predicted) - np.asarray(truth))
    years = rows["Year"].to_numpy(dtype=float)
    missing = rows[feature_names].isna().any(axis=1).to_numpy()
    masks = {
        "year_at_or_below_training_median": years <= year_threshold,
        "year_above_training_median": years > year_threshold,
        "complete_covariates": ~missing,
        "missing_covariates": missing,
    }
    slices = {name: {"rows": int(mask.sum()),
                     "mae_years": float(errors[mask].mean()) if mask.any() else None}
              for name, mask in masks.items()}
    country_names = rows["Country"].to_numpy()
    groups = {name: errors[country_names == name].tolist()
              for name in sorted(set(country_names))}
    names = list(groups)
    if not names:
        raise ValueError("Evaluation needs held-out country observations")
    rng = random.Random(42)
    values = []
    for _ in range(200):
        sample = [value for _ in names
                  for value in groups[rng.choice(names)]]
        values.append(sum(sample) / len(sample))
    interval = np.quantile(values, [0.025, 0.975]).tolist()
    return {"year_threshold_from_training": float(year_threshold), "slices": slices,
            "uncertainty": {"method": "country-block MAE resampling",
                            "seed": 42, "resamples": 200,
                            "countries": len(names), "central_fraction": 0.95,
                            "lower_mae_years": float(interval[0]),
                            "upper_mae_years": float(interval[1]),
                            "estimates_mae_years": values,
                            "limitation": "Conditional variation over these held-out country blocks; no future, causal, individual or external-population coverage claim."}}


def fit_experiment(raw):
    import numpy as np
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import GroupShuffleSplit
    frame = raw.dropna(subset=[TARGET]).copy()
    if frame[["Country", "Year"]].isna().any().any() or frame.duplicated(["Country", "Year"]).any():
        raise ValueError("Use one complete country-year identity per observation")
    features = [name for name in frame.columns if name not in ["Country", "Status", TARGET]]
    x, y = frame[features], frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Targets and observed numeric covariates must be finite")
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
        x, groups=frame["Country"]))
    local_train, local_validation = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(
        x.iloc[development], groups=frame.iloc[development]["Country"]))
    train, validation = development[local_train], development[local_validation]
    models = {degree: fit_model(x.iloc[train], y[train], degree) for degree in (1, 2)}
    validation_metrics = {degree: metrics(y[validation], model.predict(x.iloc[validation]))
                          for degree, model in models.items()}
    selected = choose_degree(validation_metrics)
    model = models[selected]
    predicted = model.predict(x.iloc[test])
    dummy = DummyRegressor(strategy="mean").fit(x.iloc[train], y[train])
    baseline = dummy.predict(x.iloc[test])
    report = {
        "data_unit": "country-year", "rows_read": len(raw), "rows": len(frame),
        "missing_targets_removed": len(raw) - len(frame), "features": features,
        "train_ids": frame.index[train].tolist(), "validation_ids": frame.index[validation].tolist(),
        "test_ids": frame.index[test].tolist(),
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "countries": {"train": sorted(set(frame.iloc[train]["Country"])),
                      "validation": sorted(set(frame.iloc[validation]["Country"])),
                      "test": sorted(set(frame.iloc[test]["Country"]))},
        "split_seeds": [42, 43], "validation_metrics": validation_metrics,
        "selection_policy": "quadratic Ridge alpha=1 only if validation MAE improves by more than two percent; otherwise linear",
        "selected_degree": selected, "model_frozen_before_test": True,
        "test_truth": y[test].tolist(), "test_predictions": predicted.tolist(),
        "metrics": metrics(y[test], predicted),
        "baseline_predictions": baseline.tolist(), "baseline_metrics": metrics(y[test], baseline),
        "training_mean_target": float(y[train].mean()),
        "fitted_state": {
            "imputation_medians": model.named_steps["fill"].statistics_.tolist(),
            "scaling_mean": model.named_steps["scale"].mean_.tolist(),
            "scaling_scale": model.named_steps["scale"].scale_.tolist(),
            "polynomial_powers": model.named_steps["polynomial"].powers_.tolist(),
            "coefficients": model.named_steps["regression"].coef_.tolist(),
            "intercept": float(model.named_steps["regression"].intercept_),
        },
        "limitation": "This evaluates recorded country-year covariates on held-out countries from the same historical period. It establishes neither individual predictions, future forecasting nor causal effects.",
    }
    report["candidate_models"] = {1: "linear", 2: "quadratic Ridge, alpha=1"}
    report["candidate_test_predictions"] = {degree: fitted.predict(x.iloc[test]).tolist() for degree, fitted in models.items()}
    report["candidate_test_metrics"] = {degree: metrics(y[test], predictions) for degree, predictions in report["candidate_test_predictions"].items()}
    report["evaluation"] = evaluation_evidence(frame.iloc[test], y[test], predicted,
        features, float(np.median(x.iloc[train]["Year"])))
    return report, model


def main():
    import numpy as np
    import pandas as pd
    import sklearn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("life_expectancy.csv")
    report, _ = fit_experiment(pd.read_csv(source))
    report["dataset"] = source.name
    report["versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "regression-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["data_unit", "rows_read", "rows", "missing_targets_removed",
                                                 "train_rows", "validation_rows", "test_rows", "validation_metrics",
                                                 "selected_degree", "metrics", "baseline_metrics", "versions", "limitation"]}))


if __name__ == "__main__":
    main()
