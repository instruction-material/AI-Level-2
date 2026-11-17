"""Independent known-value, stability and learner-role contracts for original ML4 tasks."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def load(folder):
    path = ROOT / folder / "main.py"
    spec = importlib.util.spec_from_file_location(folder.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NeuronSourceContracts(unittest.TestCase):
    def test_imports_preserve_quiet_reusable_modules(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for folder in ["ML4-Neuron-Implementation", "ML4-Simple-Neural-Network",
                           "ML4-Neuron-Implementation-Starter", "ML4-Simple-Neural-Network-Starter"]:
                load(folder)
        self.assertEqual(out.getvalue(), "")

    def test_original_single_neuron_hand_value(self):
        module = load("ML4-Neuron-Implementation")
        neuron = module.Neuron(1, 2, module.sigmoid)
        self.assertEqual((neuron.w1, neuron.w2), (1, 2))
        self.assertIs(neuron.activation_function, module.sigmoid)
        self.assertAlmostEqual(neuron.run(0, 1, 1), 1 / (1 + math.exp(-3)), places=15)
        self.assertEqual(module.Neuron(1, 2, lambda z: max(0, z)).run(-2, 0, 0), 0)

    def test_large_logits_are_stable_in_every_supplied_role(self):
        for folder in ["ML4-Neuron-Implementation", "ML4-Simple-Neural-Network",
                       "ML4-Neuron-Implementation-Starter", "ML4-Simple-Neural-Network-Starter"]:
            module = load(folder)
            self.assertEqual(module.sigmoid(-1000), 0)
            self.assertEqual(module.sigmoid(1000), 1)
            self.assertEqual(module.sigmoid(0), 0.5)

    def test_active_and_inactive_weight_perturbations(self):
        module = load("ML4-Neuron-Implementation")
        self.assertEqual(module.Neuron(9, 2, module.sigmoid).run(0, 1, 1),
                         module.Neuron(1, 2, module.sigmoid).run(0, 1, 1))
        self.assertAlmostEqual(module.Neuron(1, 1, module.sigmoid).run(0, 1, 1),
                               1 / (1 + math.exp(-2)), places=15)
        self.assertAlmostEqual(module.Neuron(1, 2, module.sigmoid).run(0, 1, 2),
                               1 / (1 + math.exp(-4)), places=15)

    def test_three_neuron_forward_pass_and_changed_input(self):
        module = load("ML4-Simple-Neural-Network")
        for x2 in [1, 0, -1]:
            h1 = 1 / (1 + math.exp(-(2 * x2 + 1)))
            h2 = 1 / (1 + math.exp(-(x2 + 1)))
            expected = (h1, h2, max(0, 3 * h1 + 5 * h2 + 1))
            actual = module.simulate(0, x2, 1)
            self.assertEqual(len(actual), 3)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, places=14)
        self.assertEqual(module.simulate(), module.simulate())

    def test_distinct_starters_stop_at_named_learning_tasks(self):
        module = load("ML4-Neuron-Implementation-Starter")
        with self.assertRaisesRegex(NotImplementedError, "Store the Neuron"):
            module.Neuron(1, 2, module.sigmoid)
        module = load("ML4-Simple-Neural-Network-Starter")
        with self.assertRaisesRegex(NotImplementedError, "Connect the two hidden"):
            module.simulate()


if __name__ == "__main__":
    unittest.main()
