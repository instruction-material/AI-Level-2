"""Manual customer clustering, with a comparison on the same scaled inputs."""
import argparse
import json
import math
from pathlib import Path
import random


def calculate_dist(a, b):
    """Euclidean distance in the feature coordinates supplied to the model."""
    if len(a) != len(b) or not a:
        raise ValueError("Points must have the same positive dimension")
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def calculate_centroid(points):
    raise NotImplementedError('Complete calculate_centroid')


def assign(points, centers):
    """Equal distances use the earlier center index."""
    return [min(range(len(centers)),
                key=lambda index: (calculate_dist(point, centers[index]), index))
            for point in points]


def fit_kmeans(points, k, seed=7, max_iter=300, tol=1e-6, initial=None):
    """Lloyd updates with a finite budget and deterministic empty-group repair."""
    points = [list(map(float, point)) for point in points]
    if not points or not points[0] or any(
        len(point) != len(points[0]) or not all(map(math.isfinite, point))
        for point in points
    ):
        raise ValueError("Use a nonempty finite rectangular feature matrix")
    unique = list(dict.fromkeys(map(tuple, points)))
    if not isinstance(k, int) or not 1 <= k <= len(unique):
        raise ValueError("k must be between 1 and the number of distinct points")
    if max_iter < 1 or not math.isfinite(tol) or tol < 0:
        raise ValueError("Use a positive iteration budget and finite nonnegative tolerance")
    centers = (random.Random(seed).sample(unique, k) if initial is None else initial)
    centers = [list(map(float, center)) for center in centers]
    if len(centers) != k or any(
        len(center) != len(points[0]) or not all(map(math.isfinite, center))
        for center in centers
    ):
        raise ValueError("Initial centers must match k and the feature dimensions")
    repairs = 0
    converged = False
    for iteration in range(1, max_iter + 1):
        labels = assign(points, centers)
        groups = [[index for index, label in enumerate(labels) if label == cluster]
                  for cluster in range(k)]
        for empty in [cluster for cluster in range(k) if not groups[cluster]]:
            # Move a faraway point from a group that can spare one member.
            candidates = [index for index in range(len(points))
                          if len(groups[labels[index]]) > 1]
            moved = max(candidates, key=lambda index: (
                calculate_dist(points[index], centers[labels[index]]), -index))
            groups[labels[moved]].remove(moved)
            groups[empty].append(moved)
            labels[moved] = empty
            repairs += 1
        updated = [calculate_centroid([points[index] for index in group])
                   for group in groups]
        movement = max(calculate_dist(a, b) for a, b in zip(centers, updated))
        centers = updated
        if movement <= tol:
            converged = True
            break
    labels = assign(points, centers)
    inertia = sum(calculate_dist(point, centers[label]) ** 2
                  for point, label in zip(points, labels))
    return {"centers": centers, "labels": labels, "inertia": inertia,
            "iterations": iteration, "converged": converged,
            "empty_cluster_repairs": repairs, "seed": seed,
            "max_iter": max_iter, "tolerance": tol}


def draw_clustered_graph(points, labels, centers, destination):
    """Draw raw income/spending coordinates; clustering uses scaled features."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    for cluster in sorted(set(labels)):
        members = [point for point, label in zip(points, labels) if label == cluster]
        axes.scatter([point[0] for point in members], [point[1] for point in members],
                     s=14, label=f"Cluster {cluster}")
    axes.scatter([center[0] for center in centers], [center[1] for center in centers],
                 marker="*", s=180, c="black", label="Centroids")
    axes.set(xlabel="Annual income (dataset units)", ylabel="Spending score",
             title="Manual k-means: labels from standardized features")
    axes.legend()
    figure.tight_layout()
    figure.savefig(destination)


def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().with_name("customers.csv")
    frame = pd.read_csv(source)
    raw = frame[["Annual Income", "Spending Score"]].to_numpy(dtype=float)
    if not all(math.isfinite(float(value)) for point in raw for value in point):
        raise ValueError("Income and spending must be finite numbers")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw).tolist()
    results = []
    fits = []
    for k in (3, 5):
        for seed in (7, 19, 31):
            manual = fit_kmeans(scaled, k, seed)
            # Fit the comparison exactly once, on the identical scaled inputs.
            library = KMeans(n_clusters=k, random_state=seed, n_init=10,
                             max_iter=300, tol=1e-6).fit(scaled)
            labels = manual["labels"]
            score = silhouette_score(scaled, labels) if 1 < len(set(labels)) < len(raw) else None
            results.append({"k": k, **manual, "silhouette": score,
                            "library_inertia": float(library.inertia_),
                            "library_n_init": 10, "manual_n_init": 1,
                            "agreement_ari": float(adjusted_rand_score(labels, library.labels_))})
            fits.append(manual)
    # k remains a modelling choice. Compare each k separately; do not select it
    # merely because larger k decreases inertia.
    selected = min((fit for fit in fits if len(fit["centers"]) == 5),
                   key=lambda fit: (fit["inertia"], fit["seed"]))
    raw_centers = scaler.inverse_transform(selected["centers"]).tolist()
    figure, axes = plt.subplots()
    axes.scatter(raw[:, 0], raw[:, 1], s=14)
    axes.set(xlabel="Annual income (dataset units)", ylabel="Spending score",
             title="Customer observations before clustering")
    figure.tight_layout()
    figure.savefig(args.output_dir / "before_clustering.png")
    draw_clustered_graph(raw.tolist(), selected["labels"], raw_centers,
                         args.output_dir / "clustered_graph.png")
    report = {"dataset": source.name, "rows": len(raw), "features": ["Annual Income", "Spending Score"],
              "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
              "one_cluster_inertia": sum(sum(value * value for value in point) for point in scaled),
              "runs": results, "displayed_k": 5, "displayed_seed": selected["seed"],
              "displayed_centers_raw_units": raw_centers,
              "limitation": "Descriptive groups depend on the features, scaling and initialization; cluster numbers are arbitrary labels."}
    (args.output_dir / "clustering-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
