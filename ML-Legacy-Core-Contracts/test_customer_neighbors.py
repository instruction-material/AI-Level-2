"""Verify real original programs against independent distance and metric arithmetic."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ["ML2-KNN-Customer-Segmentation-Classification",
           "ML2-KNN-Customer-Segmentation-Classification-Updated"]
EXECUTIONS = []


def load_program(folder):
    spec = importlib.util.spec_from_file_location("actual_customer_neighbors", ROOT / folder / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_metrics(truth, predicted, labels):
    matrix = [[sum(a == actual and b == guess for a, b in zip(truth, predicted))
               for guess in labels] for actual in labels]
    score = []
    for index in range(len(labels)):
        tp = matrix[index][index]
        denominator = sum(matrix[index]) + sum(row[index] for row in matrix)
        score.append(2 * tp / denominator if denominator else 0)
    return matrix, sum(matrix[i][i] for i in range(len(labels))) / len(truth), sum(score) / len(labels)


class OriginalCustomerNeighbors(unittest.TestCase):
    def test_both_real_programs_training_boundary_distances_and_reports(self):
        for folder in FOLDERS:
            with self.subTest(folder=folder), tempfile.TemporaryDirectory(prefix="ml-customer-neighbors-") as tmp:
                source = ROOT / folder / "main.py"
                csv = source.with_name("customers.csv")
                before = hashlib.sha256(csv.read_bytes()).hexdigest()
                output = Path(tmp)
                args = [sys.executable, "-I", "-B", str(source), "--output-dir", str(output)]
                started = time.time()
                child = subprocess.Popen(args, cwd=output, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, start_new_session=True)
                record = dict(folder=folder, parentTask="original-customer-neighbor-contract", parentPid=os.getpid(),
                              pid=child.pid, command=args, cwd=str(output), startTime=started, timeoutSeconds=45)
                try:
                    stdout, stderr = child.communicate(timeout=45)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.communicate()
                    record.update(endTime=time.time(), exitCode=child.returncode,
                                  childProcessGroupCleanup="terminated and reaped")
                    EXECUTIONS.append(record)
                    raise
                record.update(endTime=time.time(), exitCode=child.returncode, childProcessGroupCleanup="not needed")
                EXECUTIONS.append(record)
                self.assertEqual(child.returncode, 0, stderr.decode())
                report = json.loads((output / "classification-report.json").read_text())
                self.assertEqual(json.loads(stdout)["metrics"], report["metrics"])
                self.assertEqual(hashlib.sha256(csv.read_bytes()).hexdigest(), before)
                self.assertEqual((report["rows"], report["train_rows"], report["test_rows"]), (200, 150, 50))
                train, test = report["train_ids"], report["test_ids"]
                self.assertFalse(set(train) & set(test))
                self.assertEqual(sorted(train + test), list(range(200)))
                frame = pd.read_csv(csv)
                points = frame[report["features"]].to_numpy(dtype=float)
                means, variances = points[train].mean(axis=0), points[train].var(axis=0)
                np.testing.assert_allclose(report["training_means"], means)
                np.testing.assert_allclose(report["training_variances"], variances)
                self.assertEqual(report["cluster_fitted_rows"], 150)
                self.assertEqual(report["neighbor_fitted_rows"], 150)
                self.assertLessEqual(report["cluster_iterations"], 100)
                centers = np.array(report["scaled_cluster_centers"])
                scaled = (points - means) / np.sqrt(variances)
                assigned = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
                self.assertEqual(assigned[train].tolist(), report["train_labels"])
                self.assertEqual(assigned[test].tolist(), report["test_truth"])
                independent = []
                for point in scaled[test]:
                    distances = ((scaled[train] - point) ** 2).sum(axis=1)
                    nearest = sorted(range(len(train)), key=lambda index: (distances[index], index))[:3]
                    counts = Counter(report["train_labels"][index] for index in nearest)
                    independent.append(min(counts, key=lambda label: (-counts[label], label)))
                self.assertEqual(independent, report["test_predictions"])
                labels = report["class_labels"]
                for key, predictions in [("metrics", report["test_predictions"]),
                                         ("baseline_metrics", report["baseline_predictions"])]:
                    matrix, accuracy, f1 = independent_metrics(report["test_truth"], predictions, labels)
                    self.assertEqual(matrix, report[key]["confusion_matrix"])
                    self.assertAlmostEqual(accuracy, report[key]["accuracy"])
                    self.assertAlmostEqual(f1, report[key]["macro_f1"])
                counts = Counter(report["train_labels"])
                majority = min(counts, key=lambda label: (-counts[label], label))
                self.assertEqual(report["baseline_predictions"], [majority] * 50)
                self.assertIn("imitation", report["limitation"])
                for name in ["before_clustering.png", "clustered_graph.png"]:
                    self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                record.update(rows=200, trainRows=150, testRows=50,
                              independentlyRecomputedAllPredictions=True, trainingOnlyClusteringAndScaling=True,
                              macroF1=report["metrics"]["macro_f1"], exactCSVPreserved=True)

    def test_duplicate_observations_count_and_existing_manual_ties(self):
        module = load_program(FOLDERS[0])
        nearest = module.nearest_neighbors([[0], [0], [3]], ["A", "A", "B"], [0], 3)
        self.assertEqual(nearest, ["A", "A", "B"])
        self.assertEqual(module.majority_vote(Counter(nearest)), ("A", 2))
        self.assertEqual(module.majority_vote({8: 2, 3: 2}), (8, 2))
        with self.assertRaises(ValueError):
            module.nearest_neighbors([[0]], ["A"], [0], 2)

    def test_held_out_changes_leave_every_learned_training_state_unchanged(self):
        module = load_program(FOLDERS[0])
        frame = pd.read_csv(ROOT / FOLDERS[0] / "customers.csv")
        report, clusterer, classifier = module.fit_experiment(frame)
        changed = frame.copy()
        changed.loc[report["test_ids"], module.FEATURES] += 1000
        replay, replay_clusterer, replay_classifier = module.fit_experiment(changed)
        self.assertEqual(report["train_ids"], replay["train_ids"])
        for first, second in [(clusterer, replay_clusterer), (classifier, replay_classifier)]:
            np.testing.assert_array_equal(first.named_steps["scale"].mean_, second.named_steps["scale"].mean_)
            np.testing.assert_array_equal(first.named_steps["scale"].var_, second.named_steps["scale"].var_)
        np.testing.assert_array_equal(clusterer.named_steps["cluster"].cluster_centers_,
                                      replay_clusterer.named_steps["cluster"].cluster_centers_)
        np.testing.assert_array_equal(classifier.named_steps["neighbors"]._fit_X,
                                      replay_classifier.named_steps["neighbors"]._fit_X)
        self.assertEqual(report["train_labels"], replay["train_labels"])


if __name__ == "__main__":
    unittest.main()
