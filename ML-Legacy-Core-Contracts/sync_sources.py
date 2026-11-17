"""Keep each assigned clustering folder standalone with its original CSV."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MANUAL = (HERE / "manual_clustering.py").read_text()
LIBRARY = (HERE / "library_clustering.py").read_text()


def starter(source, function, exercise):
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    lines = source.splitlines(keepends=True)
    lines[node.body[0].lineno - 1:node.end_lineno] = [
        '    raise NotImplementedError(' + repr("Complete " + exercise) + ')\n'
    ]
    return "".join(lines)


projects = []
for folder, role, kind, exercise in [
    ("ML1-Customer-Segmentation-Updated", "solution", "customer", None),
    ("ML1-Customer-Segmentation", "solution", "customer", None),
    ("ML1-Customer-Segmentation-Starter-Updated", "starter", "customer", "fit_kmeans"),
    ("ML1-Customer-Segmentation-Starter", "starter", "customer", "fit_kmeans"),
    ("ML1-Customer-Segmentation-Interview-Starter", "starter", "customer", "calculate_centroid"),
    ("ML1-Disney-Movie-Clustering-Updated", "solution", "movie", None),
    ("ML1-Disney-Movie-Clustering", "solution", "movie", None),
    ("ML1-Disney-Movie-Clustering-Starter-Updated", "starter", "movie", "fit_model"),
    ("ML1-Disney-Movie-Clustering-Starter", "starter", "movie", "fit_model"),
    ("ML1-Flower-Clustering", "solution", "iris", None),
    ("ML1-Flower-Clustering-Starter", "starter", "iris", "fit_model"),
]:
    source = MANUAL if kind == "customer" else LIBRARY.replace('DATASET_KIND = "movie"', 'DATASET_KIND = ' + repr(kind))
    if exercise:
        source = starter(source, exercise, exercise)
    target = ROOT / folder
    (target / "main.py").write_text(source)
    (target / "requirements.txt").write_bytes((ROOT / "requirements-ml-legacy.txt").read_bytes())
    model = "manual centroid updates with a scikit-learn comparison" if kind == "customer" else "a StandardScaler/KMeans pipeline fitted once per k and seed"
    task = "Implement " + exercise + ". The completed reference supplies the same surrounding workflow." if exercise else "Trace and explain the complete worked reference before using it to check a starter."
    (target / "README.md").write_text(f"""# {folder}

This {role} uses {model} on the CSV beside main.py.

{task}

For a local run, create a Python 3.14 virtual environment and install this
folder's requirements.txt with python -m pip install --require-hashes -r
requirements.txt. Run python main.py from any directory. It writes before_clustering.png,
clustered_graph.png and clustering-report.json into the current directory;
--output-dir selects a different output directory.

Compare both reported k values across seeds 7, 19 and 31. Distances and inertia
use standardized features; plots and reported raw centroids retain the original
units. Cluster numbers can permute between runs. Read the inertia, silhouette,
and label-permutation-aware agreement together, then explain one uncertainty.
The displayed k is an example, not a statistically proved best number of groups.

{"Customer labels describe the supplied observations, not real customer types." if kind == "customer" else "Flower species are used only for a post-fit diagnostic, never as fit features." if kind == "iris" else "Movie features are year and inflation-adjusted revenue, without causal claims."}
""")
    if folder == "ML1-Customer-Segmentation-Interview-Starter":
        with (target / "README.md").open("a") as out:
            out.write("\nThe saved PNGs preview the completed reference from ML1-Customer-Segmentation-Updated. Complete the missing centroid function before reproducing them locally.\n")
    projects.append({"folder": folder, "role": role, "kind": kind, "exercise": exercise})
(HERE / "projects.json").write_text(json.dumps(projects, indent=2) + "\n")
print(json.dumps({"standaloneAssignedFolders": len(projects)}))
