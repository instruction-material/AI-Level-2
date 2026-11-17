"""Check actual original car roles, categorical geometry and held-out reports."""
from collections import Counter
import ast
import csv
import hashlib
import importlib.util
import io
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
FOLDERS = ["ML2-KNN-Car-Classification", "ML2-KNN-Car-Classification-Starter",
           "ML2-KNN-Car-Classification-Updated"]
EXECUTIONS = []


def run_actual(path, output, role):
    args = [sys.executable, "-I", "-B", str(path), "--output-dir", str(output)]
    child = subprocess.Popen(args, cwd=output, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True,
                             env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    entry = dict(folder=path.parent.name, sourceRole=role, parentPid=os.getpid(), pid=child.pid,
                 parentTask=os.environ.get("CLASSES_FAMILY_TASK_ID", "original-car-contract"),
                 command=args, cwd=str(output), startTime=time.time(), timeoutSeconds=45)
    EXECUTIONS.append(entry)
    cleanup = "not needed"
    try:
        stdout, stderr = child.communicate(timeout=45)
    except BaseException:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
        cleanup = "process group terminated after timeout or cancellation"
        raise
    finally:
        entry.update(endTime=time.time(), exitCode=child.returncode, childProcessGroupCleanup=cleanup)
    return child.returncode, stdout, stderr, entry


class OriginalCarContracts(unittest.TestCase):
    def test_actual_encoder_geometry_neighbors_and_unknown_vocabulary(self):
        spec = importlib.util.spec_from_file_location("actual_car_reference", ROOT / FOLDERS[0] / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame = pd.DataFrame([["low"] + ["same"] * 5, ["low"] + ["same"] * 5,
                              ["high"] + ["same"] * 5, ["high"] + ["same"] * 5], columns=module.FEATURES)
        pipeline = module.fit_classifier(frame, ["A", "A", "B", "B"], neighbors=1)
        np.testing.assert_array_equal(pipeline.predict(frame.iloc[[0, 2]]), ["A", "B"])
        encoder = pipeline.named_steps["encode"]
        encoded = encoder.transform(frame)
        self.assertAlmostEqual(float(sum((encoded[0] - encoded[2]) ** 2)), 2)
        before = [row.copy() for row in encoder.categories_]
        unknown = frame.iloc[[0]].copy()
        unknown.loc[unknown.index[0], "buying"] = "unseen"
        vector = encoder.transform(unknown)[0]
        self.assertEqual(vector[:2].tolist(), [0, 0])
        self.assertIn(pipeline.predict(unknown)[0], ["A", "B"])
        for a, b in zip(before, encoder.categories_):
            np.testing.assert_array_equal(a, b)

    def test_actual_two_references_and_completed_named_starter(self):
        reference = (ROOT / FOLDERS[0] / "main.py").read_text()
        reference_node = next(row for row in ast.parse(reference).body
                              if isinstance(row, ast.FunctionDef) and row.name == "fit_classifier")
        body = reference.splitlines(keepends=True)[reference_node.body[0].lineno - 1:reference_node.end_lineno]
        original_csv = (ROOT / FOLDERS[0] / "cars.csv").read_bytes()
        data = list(csv.DictReader(io.StringIO(original_csv.decode())))
        self.assertEqual(len(data), 1728)
        for folder in FOLDERS:
            with self.subTest(folder=folder), tempfile.TemporaryDirectory(prefix="ml-car-reference-") as temp:
                temp = Path(temp)
                source = ROOT / folder
                self.assertEqual((source / "cars.csv").read_bytes(), original_csv)
                self.assertEqual((source / "requirements.txt").read_bytes(),
                                 (ROOT / "requirements-ml-legacy.txt").read_bytes())
                path = source / "main.py"
                role = "solution"
                if folder.endswith("-Starter"):
                    lines = path.read_text().splitlines(keepends=True)
                    node = next(row for row in ast.parse("".join(lines)).body
                                if isinstance(row, ast.FunctionDef) and row.name == "fit_classifier")
                    lines[node.body[0].lineno - 1:node.end_lineno] = body
                    workspace = temp / "completed"
                    workspace.mkdir()
                    path = workspace / "main.py"
                    path.write_text("".join(lines))
                    (workspace / "cars.csv").write_bytes(original_csv)
                    role = "completed named starter"
                before = hashlib.sha256((source / "cars.csv").read_bytes()).hexdigest()
                output = temp / "output"
                output.mkdir()
                code, stdout, stderr, entry = run_actual(path, output, role)
                self.assertEqual(code, 0, stderr)
                self.assertEqual(stderr, "")
                summary = json.loads(stdout)
                report = json.loads((output / "classification-report.json").read_text())
                self.assertEqual(summary["rows"], report["rows"])
                self.assertEqual(report["rows"], 1728)
                self.assertEqual((report["train_rows"], report["test_rows"], report["fitted_rows"]), (1296, 432, 1296))
                train, test = report["train_ids"], report["test_ids"]
                self.assertFalse(set(train) & set(test))
                self.assertEqual(set(train) | set(test), set(range(1728)))
                features = report["features"]
                self.assertEqual(len(features), 6)
                self.assertNotIn("acceptability", features)
                self.assertEqual(report["fitted_categories"],
                                 [sorted({data[index][field] for index in train}) for field in features])
                truth = [data[index]["acceptability"] for index in test]
                self.assertEqual(report["test_truth"], truth)
                labels = report["class_labels"]
                predicted = report["test_predictions"]
                matrix = [[sum(a == row and b == column for a, b in zip(truth, predicted))
                           for column in labels] for row in labels]
                self.assertEqual(matrix, report["metrics"]["confusion_matrix"])
                f1 = []
                for index in range(len(labels)):
                    tp = matrix[index][index]
                    fp = sum(row[index] for row in matrix) - tp
                    fn = sum(matrix[index]) - tp
                    f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
                self.assertAlmostEqual(report["metrics"]["macro_f1"], sum(f1) / len(labels), places=12)
                counts = Counter(data[index]["acceptability"] for index in train)
                majority = max(sorted(counts), key=counts.get)
                self.assertEqual(set(report["baseline_predictions"]), {majority})
                self.assertEqual(before, hashlib.sha256((source / "cars.csv").read_bytes()).hexdigest())
                self.assertEqual(report["neighbors"], 7)
                self.assertTrue(report["model_fixed_before_test"])
                entry.update(rows=1728, trainRows=1296, testRows=432, fittedRows=1296,
                             macroF1=report["metrics"]["macro_f1"], exactCSVPreserved=True,
                             majorityBaselineTrainingOnly=True)

    def test_unfinished_starter_stops_at_the_named_pipeline_exercise(self):
        with tempfile.TemporaryDirectory(prefix="ml-car-intentional-starter-") as temp:
            path = ROOT / FOLDERS[1] / "main.py"
            code, stdout, stderr, entry = run_actual(path, Path(temp), "intentional starter")
            self.assertNotEqual(code, 0)
            self.assertIn("NotImplementedError: Complete fit_classifier", stderr)
            self.assertFalse((Path(temp) / "classification-report.json").exists())
            entry.update(intentionalExercise="fit_classifier", intentionalStopAccepted=True)


if __name__ == "__main__":
    unittest.main()
