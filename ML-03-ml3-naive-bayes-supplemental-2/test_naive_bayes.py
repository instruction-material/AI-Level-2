import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def load(role):
    spec = importlib.util.spec_from_file_location("bayes_" + role, ROOT / role / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BayesContract:
    def setUp(self):
        self.model = load("solution")

    def script_result(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(ROOT / "solution/main.py")],
                cwd=elsewhere, text=True, capture_output=True, check=True, timeout=15,
            )
        return json.loads(result.stdout)

    def test_starter_algorithms_are_explicitly_unfinished(self):
        starter = load("starter")
        for name in ["fit", "log_scores"]:
            with self.subTest(function=name), self.assertRaises(NotImplementedError):
                getattr(starter, name)([], [])


class GaussianContract(BayesContract, unittest.TestCase):
    def test_parameters_from_supplied_training_rows(self):
        points, labels, _, _ = self.model.load_fixture()
        model = self.model.fit(points, labels)
        self.assertEqual(model["means"], {"A": [2, 2], "B": [8, 8]})
        self.assertEqual(model["variances"], {"A": [1, 1], "B": [1, 1]})
        self.assertEqual(model["priors"], {"A": 0.5, "B": 0.5})

    def test_density_hand_calculation_and_tie(self):
        points, labels, _, _ = self.model.load_fixture()
        model = self.model.fit(points, labels)
        scores = self.model.log_scores(model, (2, 2))
        self.assertAlmostEqual(scores["A"], math.log(0.5 / (2 * math.pi)))
        self.assertAlmostEqual(scores["A"] - scores["B"], 36)
        self.assertEqual(self.model.predict(model, (5, 5)), "A")
        self.assertEqual(self.model.predict(model, (2, 2)), "A")
        self.assertEqual(self.model.predict(model, (8, 8)), "B")

    def test_constant_features_use_declared_floor(self):
        model = self.model.fit([(2, 3), (8, 9)], ["A", "B"], variance_floor=0.01)
        self.assertEqual(model["variances"], {"A": [0.01, 0.01], "B": [0.01, 0.01]})
        self.assertTrue(all(math.isfinite(v) for v in self.model.log_scores(model, (2, 3)).values()))
        self.assertEqual(self.model.predict(model, (8, 9)), "B")

    def test_class_prior_breaks_equal_density(self):
        model = self.model.fit([(0, 0)] * 3, ["A", "B", "B"])
        self.assertEqual(self.model.predict(model, (0, 0)), "B")
        scores = self.model.log_scores(model, (0, 0))
        self.assertAlmostEqual(scores["B"] - scores["A"], math.log(2))

    def test_input_reordering_and_prediction_leave_fit_unchanged(self):
        points, labels, _, _ = self.model.load_fixture()
        fitted = self.model.fit(points, labels)
        before = copy.deepcopy(fitted)
        self.model.predict(fitted, (1000, -1000))
        self.assertEqual(fitted, before)
        self.assertEqual(self.model.fit(points[::-1], labels[::-1]), fitted)

    def test_invalid_data_and_floor(self):
        for points, labels in [([], []), ([(1, 2)], []), ([(1,)], ["A"]),
                                ([(True, 2)], ["A"]), ([(math.nan, 2)], ["A"]),
                                ([(1, 2)], [""])]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.model.fit(points, labels)
        for floor in [0, -1, math.nan, math.inf, True, "x"]:
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                self.model.fit([(1, 2)], ["A"], floor)

    def test_invalid_queries(self):
        model = self.model.fit([(1, 2)], ["A"])
        for query in [(1,), (1, 2, 3), (1, math.inf), (False, 2)]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.model.predict(model, query)

    def test_all_fixture_predictions_and_absolute_script_path(self):
        self.assertEqual(self.script_result(), {
            "training_rows": 8, "test_rows": 4,
            "means": {"A": [2, 2], "B": [8, 8]},
            "variances": {"A": [1, 1], "B": [1, 1]},
            "predictions": ["A", "B", "A", "B"], "accuracy": 1.0,
        })


if __name__ == "__main__":
    unittest.main()
