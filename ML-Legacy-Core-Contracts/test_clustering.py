"""Independent mathematical, source-role and actual program contracts."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECTS = json.loads((HERE / "projects.json").read_text())
EXECUTIONS = []


def load(path):
    spec = importlib.util.spec_from_file_location("clustering_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manual = load(HERE / "manual_clustering.py")
library = load(HERE / "library_clustering.py")


def completed_source(source, reference, name):
    before, after = ast.parse(source), ast.parse(reference)
    old = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name)
    new = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name)
    lines, replacement = source.splitlines(keepends=True), reference.splitlines(keepends=True)
    lines[old.lineno - 1:old.end_lineno] = replacement[new.lineno - 1:new.end_lineno]
    return "".join(lines)


class ClusteringContracts(unittest.TestCase):
    def test_distance_and_coordinate_means_by_hand(self):
        self.assertEqual(manual.calculate_dist([1, 2], [4, 6]), 5)
        self.assertEqual(manual.calculate_centroid([[1, 2], [3, 6]]), [2, 4])

    def test_empty_centroid_is_explicit(self):
        with self.assertRaises(ValueError):
            manual.calculate_centroid([])

    def test_exact_lloyd_update_and_iteration_budget(self):
        result = manual.fit_kmeans([[0], [2], [10], [12]], 2,
                                   initial=[[0], [10]], max_iter=1)
        self.assertEqual(result["centers"], [[1], [11]])
        self.assertEqual(result["labels"], [0, 0, 1, 1])
        self.assertEqual(result["inertia"], 4)
        self.assertEqual(result["iterations"], 1)
        self.assertFalse(result["converged"])

    def test_duplicate_initialization_is_repaired(self):
        result = manual.fit_kmeans([[0], [1], [10], [11]], 2, initial=[[0], [0]])
        self.assertGreater(result["empty_cluster_repairs"], 0)
        self.assertEqual(sorted(result["centers"]), [[0.5], [10.5]])
        self.assertEqual(len(set(result["labels"])), 2)
        self.assertEqual(result["inertia"], 1)

    def test_final_labels_and_inertia_belong_to_returned_centers(self):
        points = [[0, 1], [2, 3], [5, 8], [8, 9]]
        result = manual.fit_kmeans(points, 2, max_iter=1)
        self.assertEqual(result["labels"], manual.assign(points, result["centers"]))
        expected = sum(manual.calculate_dist(point, result["centers"][label]) ** 2
                       for point, label in zip(points, result["labels"]))
        self.assertAlmostEqual(result["inertia"], expected)

    def test_seeded_manual_results_are_repeatable(self):
        points = [[0, 0], [1, 1], [10, 10], [11, 11]]
        self.assertEqual(manual.fit_kmeans(points, 2, 19), manual.fit_kmeans(points, 2, 19))

    def test_invalid_dimensions_nonfinite_and_unattainable_k(self):
        for points in ([], [[]], [[1], [2, 3]], [[math.nan]], [[math.inf]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                manual.fit_kmeans(points, 1)
        for k in (0, 2, 4):
            with self.subTest(k=k), self.assertRaises(ValueError):
                manual.fit_kmeans([[1], [1]], k)

    def test_scaler_fit_and_raw_centroids_by_hand(self):
        import numpy as np
        points = np.array([[10, 1000], [12, 1000], [100, 2000], [102, 2000]])
        fitted = library.fit_model(points, 2, 7)
        scaler = fitted.named_steps["scale"]
        self.assertEqual(scaler.mean_.tolist(), [56, 1500])
        raw = scaler.inverse_transform(fitted.named_steps["cluster"].cluster_centers_)
        np.testing.assert_allclose(sorted(raw.tolist()), [[11, 1000], [101, 2000]])

    def test_library_uses_one_fit_and_its_own_labels(self):
        from sklearn.cluster import KMeans
        original = KMeans.fit
        calls = []
        def observed(model, *args, **kwargs):
            calls.append(model)
            return original(model, *args, **kwargs)
        points = [[0, 0], [1, 1], [9, 9], [10, 10]]
        with patch.object(KMeans, "fit", observed):
            fitted = library.fit_model(points, 2, 7)
        self.assertEqual(len(calls), 1)
        self.assertEqual(fitted.predict(points).tolist(), fitted.named_steps["cluster"].labels_.tolist())

    def test_scaled_library_partition_survives_a_unit_change(self):
        from sklearn.metrics import adjusted_rand_score
        points = [[0, 0], [1, 2], [8, 10], [9, 11], [10, 9], [2, 1]]
        first = library.fit_model(points, 2, 19).predict(points)
        changed = [[x, y * 1000] for x, y in points]
        second = library.fit_model(changed, 2, 19).predict(changed)
        self.assertEqual(adjusted_rand_score(first, second), 1)

    def test_actual_sources_are_standalone_and_have_intentional_role_gaps(self):
        for project in PROJECTS:
            path = ROOT / project["folder"] / "main.py"
            source = path.read_text()
            with self.subTest(project=project["folder"]):
                compile(source, str(path), "exec")
                module = load(path)
                if project["exercise"]:
                    node = next(node for node in ast.parse(source).body
                                if isinstance(node, ast.FunctionDef) and node.name == project["exercise"])
                    self.assertIsInstance(node.body[0], ast.Raise)
                elif project["kind"] == "customer":
                    self.assertEqual(module.calculate_centroid([[1, 3], [3, 5]]), [2, 4])
                self.assertNotIn("ML-Source-Cases", source)
                self.assertEqual((path.parent / "requirements.txt").read_bytes(),
                                 (ROOT / "requirements-ml-legacy.txt").read_bytes())

    def test_every_actual_reference_and_completed_starter_executes(self):
        for project in PROJECTS:
            with self.subTest(project=project["folder"]), tempfile.TemporaryDirectory() as temp:
                target = Path(temp)
                folder = ROOT / project["folder"]
                source = (folder / "main.py").read_text()
                if project["exercise"]:
                    reference = (HERE / ("manual_clustering.py" if project["kind"] == "customer" else "library_clustering.py")).read_text()
                    source = completed_source(source, reference, project["exercise"])
                (target / "main.py").write_text(source)
                dataset = {"customer": "customers.csv", "movie": "disney.csv", "iris": "iris.csv"}[project["kind"]]
                (target / dataset).write_bytes((folder / dataset).read_bytes())
                result = subprocess.run([sys.executable, "-B", str(target / "main.py"), "--output-dir", str(target / "output")],
                                        cwd="/", capture_output=True, text=True, timeout=45, check=True)
                report = json.loads(result.stdout)
                self.assertEqual(len(report["runs"]), 6)
                self.assertEqual(len({row["k"] for row in report["runs"]}), 2)
                self.assertEqual({row["seed"] for row in report["runs"]}, {7, 19, 31})
                self.assertGreater(report["rows"], 100)
                for row in report["runs"]:
                    self.assertEqual(len(row["labels"]), report["rows"])
                    self.assertLessEqual(row["iterations"], 300)
                    self.assertTrue(math.isfinite(row["inertia"]))
                for filename in ("before_clustering.png", "clustered_graph.png"):
                    self.assertEqual((target / "output" / filename).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(json.loads((target / "output/clustering-report.json").read_text()), report)
                EXECUTIONS.append({"folder": project["folder"], "role": project["role"], "completedStarter": bool(project["exercise"]), "rows": report["rows"], "kSeedRuns": len(report["runs"]), "pngFiles": 2})

    def test_actual_unfinished_starters_stop_at_the_named_exercise(self):
        for project in [row for row in PROJECTS if row["exercise"]]:
            with self.subTest(project=project["folder"]), tempfile.TemporaryDirectory() as temp:
                result = subprocess.run([sys.executable, "-B", str(ROOT / project["folder"] / "main.py")],
                                        cwd=temp, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("NotImplementedError: Complete " + project["exercise"], result.stderr)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClusteringContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("ML_LEGACY_PROOF"):
        Path(os.environ["ML_LEGACY_PROOF"]).write_text(json.dumps({
            "accepted": result.wasSuccessful(), "tests": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "actualProgramExecutions": EXECUTIONS
        }, indent=2) + "\n")
    sys.exit(not result.wasSuccessful())
