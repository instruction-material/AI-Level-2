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
    spec = importlib.util.spec_from_file_location("knn_" + role, ROOT / role / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NeighborContract:
    def setUp(self):
        self.model = load("solution")

    def test_explicit_distance_and_boundary_order(self):
        self.assertEqual(
            self.model.nearest([(2, 0), (0, 0)], ["A", "B"], (1, 0), 1),
            [(1.0, "B")],
        )
        for order in itertools.permutations(range(2)):
            points = [(0, 0), (2, 0)]
            labels = ["B", "A"]
            self.assertEqual(
                self.model.nearest([points[i] for i in order],
                                   [labels[i] for i in order], (1, 0), 1),
                [(1.0, "B")],
            )

    def test_invalid_k(self):
        for k in [0, -1, 3, True, 1.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.model.nearest([(0, 0), (2, 0)], ["A", "B"], (1, 0), k)

    def test_invalid_training_or_query(self):
        cases = [
            ([], [], (1, 0)),
            ([(0, 0)], [], (1, 0)),
            ([(0, 0)], [""], (1, 0)),
            ([(0, 0)], ["A"], (1,)),
            ([(0, 0), (1,)], ["A", "B"], (1, 0)),
            ([(math.nan, 0)], ["A"], (1, 0)),
            ([(0, 0)], ["A"], (math.inf, 0)),
            ([(False, 0)], ["A"], (1, 0)),
        ]
        for points, labels, query in cases:
            with self.subTest(points=points, query=query), self.assertRaises(ValueError):
                self.model.nearest(points, labels, query, 1)

    def test_starter_marks_the_algorithm_unfinished(self):
        starter = load("starter")
        with self.assertRaises(NotImplementedError):
            starter.predict([(0, 0)], ["A"], (0, 0), 1)

    def run_script(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            result = subprocess.run(
                [sys.executable, str(ROOT / "solution" / "main.py")],
                cwd=elsewhere, capture_output=True, text=True, check=True,
                timeout=15,
            )
        return json.loads(result.stdout)

class ScalingAndUniformVote(NeighborContract, unittest.TestCase):
    def test_training_statistics_and_no_refit(self):
        points, _, test, _ = self.model.load_fixture()
        means, scales = self.model.fit_scale(points)
        self.assertEqual(means, [5.0, 5.0])
        for scale in scales:
            self.assertAlmostEqual(scale, math.sqrt(17))
        before = (means[:], scales[:])
        self.model.transform(test + [(10000, -10000)], means, scales)
        self.assertEqual((means, scales), before)

    def test_constant_columns(self):
        means, scales = self.model.fit_scale([(2, 4), (2, 4)])
        self.assertEqual(scales, [1.0, 1.0])
        self.assertEqual(self.model.transform([(2, 4)], means, scales), [(0, 0)])

    def test_declared_label_vote_tie(self):
        self.assertEqual(self.model.predict([(0, 0), (2, 0)], ["B", "A"], (1, 0), 2), "A")

    def test_supplied_predictions_and_script_directory(self):
        result = self.run_script()
        self.assertEqual(result, {
            "training_rows": 8, "test_rows": 4,
            "means": [5.0, 5.0], "scales": [4.123106, 4.123106],
            "k1": ["A", "B", "A", "B"], "k3": ["A", "B", "A", "B"],
            "accuracy_k3": 1.0,
        })

    def test_query_geometry_affects_predictions(self):
        points = [(0, 0), (1, 0), (8, 0), (9, 0)]
        labels = ["A", "A", "B", "B"]
        self.assertEqual(self.model.predict(points, labels, (0.2, 0), 1), "A")
        self.assertEqual(self.model.predict(points, labels, (8.8, 0), 1), "B")

    def test_invalid_scale_contract(self):
        for means, scales in [([0], [1]), ([0, 0], [0, 1]), ([0, 0], [math.inf, 1])]:
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                self.model.transform([(0, 0)], means, scales)


if __name__ == "__main__":
    unittest.main()
