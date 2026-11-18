"""Independent model-selection, source-role and held-out-boundary checks."""
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "ML35-Decision-Tree-Selection"
EXECUTIONS = []


def load(path):
    spec = importlib.util.spec_from_file_location("tree_contract_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def experiment():
    previous = {name: sys.modules.get(name) for name in ("tree_rules", "exercise")}
    try:
        for name in previous:
            sys.modules[name] = load(PACK / "solution" / (name + ".py"))
        yield load(PACK / "solution/main.py")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TreeSelection(unittest.TestCase):
    def execute(self, role, completed=False):
        with tempfile.TemporaryDirectory(prefix="tree-selection-contract-") as tmp:
            folder = Path(tmp) / "export"
            shutil.copytree(PACK / role, folder)
            if completed:
                shutil.copyfile(PACK / "solution/exercise.py", folder / "exercise.py")
            output = Path(tmp) / "tree-report.json"
            args = [sys.executable, "-B", str(folder / "main.py"), "--output", str(output)]
            record = dict(parentTask=os.environ.get("CLASSES_FAMILY_TASK_ID", "tree-selection-contracts"),
                          parentPid=os.getpid(), cwd=tmp, command=args, startTime=time.time(),
                          timeoutSeconds=20, role=role, completedLearner=completed)
            child = subprocess.Popen(args, cwd=tmp, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, start_new_session=True)
            record["pid"] = child.pid
            cleanup = "not needed"
            try:
                stdout, stderr = child.communicate(timeout=20)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.communicate()
                cleanup = "owned process group terminated and reaped"
                raise
            finally:
                record.update(endTime=time.time(), exitCode=child.returncode,
                              childProcessGroupCleanup=cleanup)
                EXECUTIONS.append(record)
            saved = json.loads(output.read_text()) if output.exists() else None
            return child.returncode, stdout.decode(), stderr.decode(), saved

    def test_reference_export_executes_from_another_directory(self):
        code, stdout, stderr, saved = self.execute("solution")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout), saved)
        self.assertEqual(saved["selected_depth"], 2)
        self.assertEqual(saved["test_predictions"], ["A", "A", "B", "A"])

    def test_starter_stops_at_named_task_before_final_evaluation(self):
        code, stdout, stderr, saved = self.execute("starter")
        self.assertNotEqual(code, 0)
        self.assertIn("NotImplementedError: Complete choose_depth(scores).", stderr)
        self.assertEqual(stdout, "")
        self.assertIsNone(saved)

    def test_completed_learner_export_executes(self):
        code, stdout, stderr, saved = self.execute("starter", completed=True)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout), saved)
        self.assertEqual(saved["test_accuracy"], 1)
        self.assertEqual(saved["baseline_test_accuracy"], 0.75)

    def test_independent_worked_tree_and_validation_comparison(self):
        with experiment() as app:
            report = app.experiment()
        self.assertEqual(report["validation_accuracy"], {0: 0.5, 1: 0.75, 2: 1.0})
        self.assertEqual(report["majority_label"], "A")
        tree = report["selected_tree"]
        self.assertEqual(tree["counts"], {"A": 4, "B": 4})
        self.assertEqual((tree["feature"], tree["threshold"]), (0, 1.5))
        self.assertEqual(tree["left"]["counts"], {"A": 2})
        self.assertEqual(tree["right"]["counts"], {"A": 2, "B": 4})
        self.assertEqual(tree["right"]["threshold"], 5.5)
        self.assertEqual(tree["right"]["left"]["counts"], {"B": 4})
        self.assertEqual(tree["right"]["right"]["counts"], {"A": 2})

    def test_test_labels_change_only_final_metrics(self):
        with experiment() as app:
            original = app.experiment()
            real_loader = app.load_split
            opened = []
            def changed(name):
                opened.append(name)
                points, labels = real_loader(name)
                return points, ["B", "A", "B", "A"] if name == "test" else labels
            app.load_split = changed
            second = app.experiment()
        self.assertEqual(opened, ["train", "validation", "test"])
        for key in ("validation_accuracy", "selected_depth", "selected_tree",
                    "majority_label", "test_predictions", "test_traces"):
            self.assertEqual(second[key], original[key], key)
        self.assertEqual(second["test_accuracy"], 0.75)
        self.assertEqual(second["baseline_test_accuracy"], 0.5)

    def test_shallowest_inclusive_tolerance_and_invalid_scores(self):
        helper = load(PACK / "solution/exercise.py")
        self.assertEqual(helper.choose_depth({2: 1.0, 1: 0.98, 0: 0.5}), 1)
        self.assertEqual(helper.choose_depth({2: 1.0, 1: 0.9799}), 2)
        self.assertEqual(helper.choose_depth({2: 0.5, 0: 0.5, 1: 0.5}), 0)
        for value in ({}, [], {True: 0.5}, {1: True}, {9: 0.5}, {1: float("nan")},
                      {1: float("inf")}, {1: -0.1}, {1: 1.1}, {"1": 0.5}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                helper.choose_depth(value)

    def test_validation_can_change_selection_without_fitting_its_labels(self):
        with experiment() as app:
            points, labels = app.load_split("train")
            validation, expected = app.load_split("validation")
            selected, first, _ = app.select_model(points, labels, validation, expected)
            other, second, scores = app.select_model(points, labels, validation, ["B"] * 4)
        self.assertEqual((selected, other), (2, 1))
        self.assertEqual(scores, {0: 0, 1: 0.75, 2: 0.5})
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(first["counts"], {"A": 4, "B": 4})

    def test_fixed_tree_boundary_probe_retains_every_fitted_count(self):
        with experiment() as app:
            report = app.experiment()
            tree = report["selected_tree"]
            original = copy.deepcopy(tree)
            before = app.trace(tree, [5.5, 0])
            after = app.trace(tree, [5.51, 0])
            distractor = app.trace(tree, [5.5, 99])
        self.assertEqual(before["prediction"], "B")
        self.assertEqual(after["prediction"], "A")
        self.assertEqual([p["branch"] for p in before["path"]], ["right", "left"])
        self.assertEqual([p["branch"] for p in after["path"]], ["right", "right"])
        self.assertEqual(before, distractor)
        self.assertEqual(tree, original)


if __name__ == "__main__":
    unittest.main()
