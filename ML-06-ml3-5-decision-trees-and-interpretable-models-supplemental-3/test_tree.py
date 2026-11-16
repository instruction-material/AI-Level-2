import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
STUMP = ROOT.name.startswith("ML-05")


def load(role):
    spec = importlib.util.spec_from_file_location("tree_" + role, ROOT / role / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TreeContract(unittest.TestCase):
    def setUp(self):
        self.tree = load("solution")

    def test_gini_hand_calculations(self):
        self.assertEqual(self.tree.gini(["A"] * 4), 0)
        self.assertEqual(self.tree.gini(["A", "A", "B", "B"]), 0.5)
        self.assertAlmostEqual(self.tree.gini(["A", "A", "B"]), 4 / 9)
        with self.assertRaises(ValueError):
            self.tree.gini([])

    def test_selected_root_and_weighted_gain(self):
        points, labels, _, _ = self.tree.load_fixture()
        fitted = self.tree.fit(points, labels)
        self.assertEqual(fitted["feature"], 0)
        self.assertEqual(fitted["threshold"], 3.5 if STUMP else 1.5)
        self.assertAlmostEqual(fitted["gain"], 0.5 if STUMP else 1 / 6)

    def test_fixed_depth_predictions_and_script_from_another_directory(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            output = subprocess.run(
                [sys.executable, "-I", "-B", str(ROOT / "solution/main.py")],
                cwd=elsewhere, text=True, capture_output=True, check=True, timeout=15,
            )
        result = json.loads(output.stdout)
        self.assertEqual((result["training_rows"], result["test_rows"]), (8, 4))
        expected = ["A", "A", "B", "B"] if STUMP else ["A", "B", "B", "A"]
        self.assertEqual(result["depth_comparison"]["2"]["predictions"], expected)
        self.assertEqual(result["depth_comparison"]["2"]["accuracy"], 1)
        self.assertEqual(result["depth_comparison"]["1"]["accuracy"], 1 if STUMP else 0.75)

    def test_trace_records_exact_threshold_branch_and_leaf_counts(self):
        points, labels, _, _ = self.tree.load_fixture()
        fitted = self.tree.fit(points, labels)
        if STUMP:
            path = self.tree.trace(fitted, (3.5, 0))
            self.assertEqual(path, {"prediction": "A", "path": [{"feature": 0, "threshold": 3.5, "branch": "left"}], "leaf_counts": {"A": 4}})
            self.assertEqual(self.tree.predict(fitted, (3.50001, 0)), "B")
        else:
            path = self.tree.trace(fitted, (6.5, 1))
            self.assertEqual(path, {"prediction": "A", "path": [
                {"feature": 0, "threshold": 1.5, "branch": "right"},
                {"feature": 0, "threshold": 5.5, "branch": "right"},
            ], "leaf_counts": {"A": 2}})
            self.assertEqual(self.tree.predict(fitted, (5.5, 0)), "B")

    def test_no_split_when_features_identical_or_labels_pure(self):
        for points, labels, expected in [
            ([(1, 1), (1, 1)], ["A", "B"], "A"),
            ([(0, 0), (1, 1)], ["B", "B"], "B"),
        ]:
            model = self.tree.fit(points, labels)
            self.assertNotIn("feature", model)
            self.assertEqual(model["prediction"], expected)

    def test_positive_gain_rule_stops_on_xor(self):
        model = self.tree.fit([(0, 0), (0, 1), (1, 0), (1, 1)], ["A", "B", "B", "A"], 2)
        self.assertNotIn("feature", model)
        self.assertEqual(model["counts"], {"A": 2, "B": 2})

    def test_equal_feature_gain_chooses_lowest_feature(self):
        model = self.tree.fit([(0, 0), (1, 1), (2, 2), (3, 3)], ["A", "A", "B", "B"])
        self.assertEqual((model["feature"], model["threshold"]), (0, 1.5))

    def test_depth_zero_keeps_majority_and_alphabetic_tie(self):
        model = self.tree.fit([(0, 0), (1, 1), (2, 2)], ["A", "B", "B"], 0)
        self.assertNotIn("feature", model)
        self.assertEqual(self.tree.predict(model, (100, 100)), "B")
        tie = self.tree.fit([(0, 0), (1, 1)], ["Z", "A"], 0)
        self.assertEqual(tie["prediction"], "A")

    def test_reordering_and_predicting_leave_training_state_unchanged(self):
        points, labels, _, _ = self.tree.load_fixture()
        model = self.tree.fit(points, labels)
        self.assertEqual(self.tree.fit(points[::-1], labels[::-1]), model)
        before = copy.deepcopy(model)
        trace = self.tree.trace(model, (1000, -1000))
        trace["leaf_counts"]["injected"] = 123
        self.assertEqual(model, before)

    def test_invalid_measurements_and_labels(self):
        for points, labels in [([], []), ([(1, 2)], []), ([(1,)], ["A"]),
                                ([(1, math.inf)], ["A"]), ([(math.nan, 2)], ["A"]),
                                ([(False, 2)], ["A"]), ([(1, 2)], [""])]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.tree.fit(points, labels)
        model = self.tree.fit([(0, 0)], ["A"])
        for query in [(1,), (1, math.nan), (True, 2)]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.tree.predict(model, query)

    def test_invalid_depth(self):
        for depth in [-1, 9, True, 1.5, "2"]:
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.tree.fit([(0, 0)], ["A"], depth)

    def test_starter_marks_learning_task_unfinished(self):
        starter = load("starter")
        points, labels, _, _ = starter.load_fixture()
        with self.assertRaises(NotImplementedError):
            starter.fit(points, labels)


if __name__ == "__main__":
    unittest.main()
