import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SINGLE = ROOT.name.startswith("ML-07")


def load(role):
    spec = importlib.util.spec_from_file_location("network_" + role, ROOT / role / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NetworkContract(unittest.TestCase):
    def setUp(self):
        self.network = load("solution")

    def test_sigmoid_at_zero_and_extreme_logits(self):
        self.assertEqual(self.network.sigmoid(0), 0.5)
        self.assertEqual(self.network.sigmoid(1000), 1)
        self.assertEqual(self.network.sigmoid(-1000), 0)
        for value in [math.inf, math.nan, "x"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.network.sigmoid(value)

    def test_forward_pass_against_hand_calculation(self):
        if SINGLE:
            result = self.network.forward([2, -1, 0.5], (0.25, 0.5))
        else:
            parameters = [0.0] * 13
            parameters[2] = math.atanh(0.5)
            parameters[9], parameters[12] = 2, -0.5
            result = self.network.forward(parameters, (0.25, 0.5))
            self.assertAlmostEqual(result["hidden"][0], 0.5)
            self.assertEqual(result["hidden"][1:], [0, 0])
        self.assertAlmostEqual(result["logit"], 0.5)
        self.assertAlmostEqual(result["probability"], 1 / (1 + math.exp(-0.5)))

    def test_zero_initialization_hand_gradient(self):
        points, labels, _, _ = self.network.load_fixture()
        loss, gradient = self.network.loss_and_gradient([0.0] * self.network.PARAMETERS, points, labels)
        self.assertAlmostEqual(loss, math.log(2))
        self.assertEqual(gradient, [-0.25] * 3 if SINGLE else [0.0] * 13)
        if SINGLE:
            fitted = self.network.fit(points, labels, steps=1)
            self.assertEqual(fitted["parameters"], [0.125] * 3)

    def test_every_gradient_matches_centered_finite_difference(self):
        parameters = [-0.2 + 0.05 * i for i in range(self.network.PARAMETERS)]
        points, labels, _, _ = self.network.load_fixture()
        _, gradient = self.network.loss_and_gradient(parameters, points, labels)
        epsilon = 1e-6
        for index in range(len(parameters)):
            plus, minus = parameters[:], parameters[:]
            plus[index] += epsilon
            minus[index] -= epsilon
            high, _ = self.network.loss_and_gradient(plus, points, labels)
            low, _ = self.network.loss_and_gradient(minus, points, labels)
            with self.subTest(parameter=index):
                self.assertAlmostEqual(gradient[index], (high - low) / (2 * epsilon), delta=2e-9)

    def test_one_step_reduces_loss(self):
        points, labels, _, _ = self.network.load_fixture()
        fitted = self.network.fit(points, labels, steps=1)
        self.assertLess(fitted["final_loss"], fitted["initial_loss"])

    def test_default_training_and_held_out_predictions(self):
        points, labels, held, expected = self.network.load_fixture()
        model = self.network.fit(points, labels)
        self.assertLess(model["final_loss"], 0.02)
        self.assertEqual([self.network.predict(model["parameters"], point) for point in points], labels)
        self.assertEqual([self.network.predict(model["parameters"], point) for point in held], expected)
        self.assertEqual(expected, [0, 1, 1, 1] if SINGLE else [0, 1, 1, 0])

    def test_seeded_training_preserves_inputs_and_global_random_state(self):
        points, labels, _, _ = self.network.load_fixture()
        before = copy.deepcopy((points, labels))
        state = random.getstate()
        first = self.network.fit(points, labels, steps=10)
        self.assertEqual(first, self.network.fit(points, labels, steps=10))
        self.assertEqual(random.getstate(), state)
        self.assertEqual((points, labels), before)
        parameters = first["parameters"][:]
        self.network.predict(parameters, (0.1, 0.9))
        self.assertEqual(parameters, first["parameters"])

    def test_confident_wrong_prediction_has_finite_cross_entropy(self):
        parameters = [0.0] * self.network.PARAMETERS
        parameters[-1] = 1000
        loss, gradient = self.network.loss_and_gradient(parameters, [(0, 0)], [0])
        self.assertEqual(loss, 1000)
        self.assertEqual(gradient[-1], 1)
        self.assertTrue(all(math.isfinite(value) for value in gradient))

    def test_invalid_points_and_labels(self):
        for points, labels in [([], []), ([(0, 0)], []), ([(0,)], [0]),
                                ([(2, 0)], [0]), ([(True, 0)], [0]),
                                ([(math.nan, 0)], [0]), ([(0, 0)], [2]),
                                ([(0, 0)], [True])]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.network.fit(points, labels)

    def test_invalid_parameters_and_training_controls(self):
        for parameters in [[], [0] * (self.network.PARAMETERS + 1),
                           [math.inf] * self.network.PARAMETERS]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.network.forward(parameters, (0, 0))
        for steps in [0, -1, 10001, True, 1.5]:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.network.fit([(0, 0)], [0], steps=steps)
        for rate in [0, -1, 2, math.nan, True, "x"]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.network.fit([(0, 0)], [0], steps=1, learning_rate=rate)

    def test_starter_marks_learning_functions_unfinished(self):
        starter = load("starter")
        points, labels, _, _ = starter.load_fixture()
        with self.assertRaises(NotImplementedError):
            starter.fit(points, labels, steps=1)

    def test_cli_from_another_directory(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(ROOT / "solution/main.py")],
                cwd=elsewhere, text=True, capture_output=True, check=True, timeout=15,
            )
        output = json.loads(result.stdout)
        self.assertEqual(output["predictions"], [0, 1, 1, 1] if SINGLE else [0, 1, 1, 0])
        self.assertEqual((output["training_rows"], output["test_rows"], output["accuracy"]), (4, 4, 1))
        self.assertLess(output["final_loss"], 0.02)


if __name__ == "__main__":
    unittest.main()
