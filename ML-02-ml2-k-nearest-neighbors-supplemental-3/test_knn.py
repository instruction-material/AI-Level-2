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

class DistanceVote(NeighborContract, unittest.TestCase):
    def test_hand_computed_disagreement(self):
        points, labels = [(0, 0), (2, 0), (3, 0)], ["A", "A", "B"]
        self.assertEqual(self.model.predict(points, labels, (2.9, 0), 3, "uniform"), "A")
        self.assertEqual(self.model.predict(points, labels, (2.9, 0), 3, "distance"), "B")

    def test_exact_matches_exclude_more_distant_votes(self):
        self.assertEqual(
            self.model.predict([(0, 0), (1, 0), (2, 0)],
                               ["B", "A", "A"], (0, 0), 3), "B",
        )

    def test_multiple_exact_matches_and_label_tie(self):
        self.assertEqual(
            self.model.predict([(0, 0)] * 3, ["A", "B", "B"], (0, 0), 3), "B",
        )
        self.assertEqual(
            self.model.predict([(0, 0)] * 2, ["B", "A"], (0, 0), 2), "A",
        )

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.model.predict([(0, 0)], ["A"], (1, 0), 1, "unknown")

    def test_tiny_positive_distances_keep_the_weight_ratio(self):
        self.assertEqual(
            self.model.predict([(1e-320, 0), (2e-320, 0), (3e-320, 0)],
                               ["B", "A", "A"], (0, 0), 3), "B",
        )

    def test_supplied_predictions_and_script_directory(self):
        self.assertEqual(self.run_script(), {
            "k": 3,
            "uniform": ["A", "A", "A", "A"],
            "distance": ["B", "A", "A", "B"],
            "uniform_accuracy": 0.5, "distance_accuracy": 1.0,
        })


if __name__ == "__main__":
    unittest.main()
