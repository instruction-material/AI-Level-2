"""Execute the original Iris program and independently check its fitted likelihoods."""
from collections import Counter
from contextlib import redirect_stdout
import io
import math
import os
from pathlib import Path
import runpy
import tempfile
import time
import unittest

import numpy as np
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[1]
EXECUTIONS = []


class OriginalGaussianIris(unittest.TestCase):
    def test_actual_program_training_only_parameters_metrics_and_density_prediction(self):
        source = ROOT / "ML3-Naive-Bayes-Iris-Flowers-Classification"
        self.assertEqual((source / "requirements.txt").read_bytes(),
                         (ROOT / "requirements-ml-legacy.txt").read_bytes())
        previous = Path.cwd()
        output = io.StringIO()
        started = time.time()
        try:
            with tempfile.TemporaryDirectory(prefix="ml-original-iris-") as temp:
                os.chdir(temp)
                with redirect_stdout(output):
                    state = runpy.run_path(str(source / "main.py"), run_name="__main__")
        finally:
            os.chdir(previous)
        model = state["model"]
        self.assertIsInstance(model, GaussianNB)
        train_x, train_y = state["x_train"], state["y_train"]
        test_x, test_y = state["x_test"], state["y_test"]
        self.assertEqual((train_x.shape, test_x.shape), ((112, 4), (38, 4)))
        self.assertEqual(set(Counter(test_y).values()), {12, 13})
        labels = sorted(set(train_y))
        self.assertEqual(labels, list(model.classes_))
        means = np.array([train_x[train_y == label].mean(axis=0) for label in labels])
        epsilon = 1e-9 * train_x.var(axis=0).max()
        variances = np.array([train_x[train_y == label].var(axis=0) + epsilon for label in labels])
        priors = np.array([sum(train_y == label) / len(train_y) for label in labels])
        np.testing.assert_allclose(model.theta_, means, rtol=1e-12)
        np.testing.assert_allclose(model.var_, variances, rtol=1e-12)
        np.testing.assert_allclose(model.class_prior_, priors, rtol=1e-12)
        query = state["custom_vectors"][0]
        scores = [math.log(prior) - 0.5 * sum(
            math.log(2 * math.pi * variance) + (value - mean) ** 2 / variance
            for value, mean, variance in zip(query, center, spread))
            for prior, center, spread in zip(priors, means, variances)]
        self.assertEqual(model.predict([query])[0], labels[int(np.argmax(scores))])
        predicted = state["y_pred"]
        matrix = [[sum(actual == row and guessed == column
                       for actual, guessed in zip(test_y, predicted))
                   for column in labels] for row in labels]
        np.testing.assert_array_equal(state["matrix"], matrix)
        f1s = []
        for index in range(len(labels)):
            tp = matrix[index][index]
            fp = sum(row[index] for row in matrix) - tp
            fn = sum(matrix[index]) - tp
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        self.assertAlmostEqual(state["macro_f1"], sum(f1s) / len(labels), places=12)
        counts = Counter(train_y)
        majority = max(labels, key=lambda label: counts[label])
        self.assertEqual(set(state["baseline_predictions"]), {majority})
        self.assertIn("GaussianNB macro F1:", output.getvalue())
        self.assertIn("Dummy macro F1:", output.getvalue())
        EXECUTIONS.append(dict(folder=source.name, executionMethod="actual runpy program",
                               parentTask=os.environ.get("CLASSES_FAMILY_TASK_ID", "original-iris-contract"),
                               pid=os.getpid(), startTime=started, endTime=time.time(), exitCode=0,
                               rows=150, trainRows=112, testRows=38, GaussianFamily=True,
                               independentDensityWinner=str(labels[int(np.argmax(scores))]),
                               manualConfusionMatrix=[[int(value) for value in row] for row in matrix], macroF1=float(state["macro_f1"]),
                               trainingOnlyBaseline=True))


if __name__ == "__main__":
    unittest.main()
