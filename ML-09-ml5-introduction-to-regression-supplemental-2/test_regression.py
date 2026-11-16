import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
LINEAR = ROOT.name.startswith("ML-09")


def load(role):
    spec = importlib.util.spec_from_file_location("regression_" + role, ROOT / role / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegressionContract(unittest.TestCase):
    def setUp(self):
        self.regression = load("solution")

    def fit(self, xs, ys):
        return self.regression.fit(xs, ys) if LINEAR else self.regression.fit_polynomial(xs, ys, 2)

    def run_script(self, script):
        with tempfile.TemporaryDirectory() as elsewhere:
            result = subprocess.run([sys.executable, "-I", "-B", str(script)], cwd=elsewhere,
                                    text=True, capture_output=True, check=True, timeout=15)
        return json.loads(result.stdout)

    def test_mse_hand_calculation_and_alignment(self):
        self.assertEqual(self.regression.mse([0, 1], [1, 3]), 2.5)
        self.assertEqual(self.regression.mse([-2, 2], [-2, 2]), 0)
        for actual, predicted in [([], []), ([1], []), ([1], [math.nan])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                self.regression.mse(actual, predicted)

    def test_supplied_parameters_against_hand_calculation(self):
        xs, ys = self.regression.read_split("train.csv")
        model = self.fit(xs, ys)
        if LINEAR:
            self.assertEqual(model, {"slope": 2, "intercept": 1, "training_target_mean": 5})
        else:
            for actual, expected in zip(model, [2, 3, 1]):
                self.assertAlmostEqual(actual, expected, places=12)

    def test_fixture_baseline_or_validation_error(self):
        if LINEAR:
            self.assertEqual(self.regression.mse([4, 6], [5, 5]), 1)
        else:
            train_x, train_y = self.regression.read_split("train.csv")
            val_x, val_y = self.regression.read_split("validation.csv")
            linear = self.regression.fit_polynomial(train_x, train_y, 1)
            self.assertAlmostEqual(self.regression.mse(val_y, self.regression.predict(linear, val_x)), 1.5625)
            chosen = self.regression.choose_degree(train_x, train_y, val_x, val_y)
            self.assertEqual(chosen["degree"], 2)
            self.assertLess(chosen["validation_mse"], 1e-24)

    def test_constant_training_features_have_declared_behavior(self):
        if LINEAR:
            model = self.regression.fit([2, 2], [1, 5])
            self.assertEqual((model["slope"], model["intercept"]), (0, 3))
            self.assertEqual(self.regression.predict(model, [2, 100]), [3, 3])
        else:
            with self.assertRaises(ValueError):
                self.regression.fit_polynomial([2, 2], [1, 5], 2)
            self.assertEqual(self.regression.fit_polynomial([2, 2], [1, 5], 0), [3])

    def test_invalid_training_measurements(self):
        for xs, ys in [([], []), ([1], []), ([True], [1]), ([1], [math.inf]), ([math.nan], [1])]:
            with self.subTest(xs=xs), self.assertRaises(ValueError):
                self.fit(xs, ys)

    def test_starter_marks_model_unfinished(self):
        starter = load("starter")
        xs, ys = starter.read_split("train.csv")
        with self.assertRaises(NotImplementedError):
            starter.fit(xs, ys) if LINEAR else starter.fit_polynomial(xs, ys, 2)

    def test_prediction_preserves_fitted_state_and_inputs(self):
        xs, ys = self.regression.read_split("train.csv")
        model = self.fit(xs, ys)
        before = copy.deepcopy((xs, ys, model))
        self.regression.predict(model, [0.25, 0.75])
        self.assertEqual((xs, ys, model), before)

    def test_reordering_training_rows_preserves_parameters(self):
        xs, ys = self.regression.read_split("train.csv")
        first, reverse = self.fit(xs, ys), self.fit(xs[::-1], ys[::-1])
        left = list(first.values()) if LINEAR else first
        right = list(reverse.values()) if LINEAR else reverse
        for actual, expected in zip(left, right):
            self.assertAlmostEqual(actual, expected, places=11)

    def test_negative_slope_line_is_fitted(self):
        if LINEAR:
            model = self.regression.fit([-1, 0, 1], [5, 3, 1])
            self.assertEqual(self.regression.predict(model, [-0.5, 0.5]), [4, 2])
        else:
            coefficients = self.regression.fit_polynomial([-1, 0, 1], [5, 3, 1], 1)
            self.assertEqual(coefficients, [3, -2])
            chosen = self.regression.choose_degree([-1, 0, 1], [-1, 0, 1], [-0.5, 0.5], [-0.5, 0.5])
            self.assertEqual(chosen["degree"], 1)

    def test_invalid_query_or_degree(self):
        if LINEAR:
            model = self.regression.fit([0, 1], [1, 3])
            for query in [[], [True], [math.inf]]:
                with self.subTest(query=query), self.assertRaises(ValueError):
                    self.regression.predict(model, query)
        else:
            for degree in [-1, 3, True, 1.5]:
                with self.subTest(degree=degree), self.assertRaises(ValueError):
                    self.regression.fit_polynomial([-1, 0, 1], [1, 0, 1], degree)
            with self.assertRaises(ValueError):
                self.regression.solve([[1, 1], [2, 2]], [1, 2])

    def test_actual_cli_test_labels_never_influence_fitting_or_selection(self):
        before = self.run_script(ROOT / "solution/main.py")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            (target / "main.py").write_bytes((ROOT / "solution/main.py").read_bytes())
            (target / "data").mkdir()
            for file in (ROOT / "solution/data").glob("*.csv"):
                (target / "data" / file.name).write_bytes(file.read_bytes())
            test = target / "data/test.csv"
            with test.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["target"] = float(row["target"]) + 10
            with test.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["x", "target"])
                writer.writeheader();writer.writerows(rows)
            after = self.run_script(target / "main.py")
        fields = ["slope", "intercept", "predictions"] if LINEAR else [
            "chosen_degree", "coefficients", "validation_mse", "predictions"
        ]
        self.assertEqual({key: before[key] for key in fields}, {key: after[key] for key in fields})
        self.assertAlmostEqual(after["test_mse"], 100)

    def test_cli_from_another_directory(self):
        result = self.run_script(ROOT / "solution/main.py")
        if LINEAR:
            self.assertEqual(result, {"training_rows": 5, "test_rows": 2,
                                     "slope": 2, "intercept": 1, "predictions": [4, 6],
                                     "test_mse": 0, "baseline_mse": 1})
        else:
            self.assertEqual(result, {"training_rows": 5, "validation_rows": 2, "test_rows": 2,
                                     "chosen_degree": 2, "coefficients": [2, 3, 1],
                                     "predictions": [0.75, 8.75], "validation_mse": 0, "test_mse": 0})


if __name__ == "__main__":
    unittest.main()
