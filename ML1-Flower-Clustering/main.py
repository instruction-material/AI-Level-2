"""Scaled scikit-learn clustering with reproducible k/seed comparisons."""
import argparse
import json
import math
from pathlib import Path

DATASET_KIND = 'iris'


def fit_model(points, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("cluster", KMeans(n_clusters=k, random_state=seed, n_init=10,
                           max_iter=300, tol=1e-6))
    ])
    pipeline.fit(points)
    return pipeline


def load_points():
    import numpy as np
    import pandas as pd
    folder = Path(__file__).resolve().parent
    if DATASET_KIND == "movie":
        frame = pd.read_csv(folder / "disney.csv")
        years = pd.to_datetime(frame["release_date"], errors="raise").dt.year
        points = np.column_stack([years, frame["inflation_adjusted_gross"]]).astype(float)
        return points, ["Release year", "Inflation-adjusted revenue (dataset dollars)"], None
    frame = pd.read_csv(folder / "iris.csv")
    points = frame[["petal_length", "petal_width"]].to_numpy(dtype=float)
    return points, ["Petal length (cm)", "Petal width (cm)"], frame["species"].tolist()


def draw_clustered_graph(points, labels, centers, axis_labels, destination):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    for cluster in sorted(set(map(int, labels))):
        members = points[labels == cluster]
        if len(members):
            axes.scatter(members[:, 0], members[:, 1], s=14, label=f"Cluster {cluster}")
    axes.scatter(centers[:, 0], centers[:, 1], marker="*", s=180, c="black", label="Centroids")
    axes.set(xlabel=axis_labels[0], ylabel=axis_labels[1], title="Clusters fitted in standardized coordinates")
    axes.legend()
    figure.tight_layout()
    figure.savefig(destination)


def main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points, axis_labels, species = load_points()
    if not len(points) or not all(math.isfinite(float(value)) for point in points for value in point):
        raise ValueError("Use a nonempty finite feature matrix")
    choices = (3, 5) if DATASET_KIND == "movie" else (2, 3)
    results = []
    fitted = []
    for k in choices:
        for seed in (7, 19, 31):
            pipeline = fit_model(points, k, seed)
            scaler = pipeline.named_steps["scale"]
            model = pipeline.named_steps["cluster"]
            labels = model.labels_
            scaled = scaler.transform(points)
            # These centers and labels belong to that same single fit.
            raw_centers = scaler.inverse_transform(model.cluster_centers_)
            score = silhouette_score(scaled, labels) if 1 < len(set(labels)) < len(points) else None
            row = {"k": k, "seed": seed, "n_init": 10, "max_iter": 300,
                   "iterations": int(model.n_iter_), "inertia": float(model.inertia_),
                   "silhouette": score, "labels": labels.tolist(),
                   "centers_scaled": model.cluster_centers_.tolist(),
                   "centers_raw_units": raw_centers.tolist(),
                   "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
            if species is not None:
                row["species_agreement_ari"] = float(adjusted_rand_score(species, labels))
            results.append(row)
            fitted.append((pipeline, row))
    # Choose the best restart only within the displayed k; compare the k values
    # using the reported evidence and domain meaning, not inertia alone.
    shown_k = 3
    pipeline, shown = min((fit for fit in fitted if fit[1]["k"] == shown_k),
                          key=lambda fit: (fit[1]["inertia"], fit[1]["seed"]))
    model = pipeline.named_steps["cluster"]
    figure, axes = plt.subplots()
    axes.scatter(points[:, 0], points[:, 1], s=14)
    axes.set(xlabel=axis_labels[0], ylabel=axis_labels[1], title="Observations before clustering")
    figure.tight_layout()
    figure.savefig(args.output_dir / "before_clustering.png")
    centers = pipeline.named_steps["scale"].inverse_transform(model.cluster_centers_)
    draw_clustered_graph(points, model.labels_, centers, axis_labels,
                         args.output_dir / "clustered_graph.png")
    report = {"dataset_kind": DATASET_KIND, "rows": len(points), "feature_labels": axis_labels,
              "runs": results, "displayed_k": shown_k, "displayed_seed": shown["seed"],
              "species_labels_used_for_fitting": False,
              "limitation": "Clusters describe the chosen features and scaling. Species comparisons are post-fit diagnostics; movie clusters do not establish causal eras."}
    (args.output_dir / "clustering-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
