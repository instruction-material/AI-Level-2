"""Check the exact references linked by the current linear/polynomial lessons."""
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXECUTIONS = []


def least_squares(x, y, degree):
    x, y = list(map(Fraction, x)), [Fraction(str(value)) for value in y]
    matrix = [[sum(value ** (row + column) for value in x) for column in range(degree + 1)]
              + [sum(value ** row * target for value, target in zip(x, y))] for row in range(degree + 1)]
    for column in range(degree + 1):
        pivot = next(row for row in range(column, degree + 1) if matrix[row][column])
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        denominator = matrix[column][column]
        matrix[column] = [value / denominator for value in matrix[column]]
        for row in range(degree + 1):
            if row != column:
                multiple = matrix[row][column]
                matrix[row] = [a - multiple * b for a, b in zip(matrix[row], matrix[column])]
    return [row[-1] for row in matrix]


class CurrentLinkedRegressionReferences(unittest.TestCase):
    def test_actual_current_references_match_exact_rational_fits_and_printed_query(self):
        cases = [("ML-Simple-Linear-Regression-Updated", [4, 5, 6, 3, 5, 7, 3, 9, 13, 14], 1),
                 ("ML-Simple-Polynomial-Regression-Updated", [10, 8, 7.5, 7, 6, 6, 7, 9, 8.5, 11], 2)]
        wrapper = """import json,runpy,sys
n=runpy.run_path(sys.argv[1]);np=n['np'];query=np.array([[7]])
if 'features' in n:query=n['features'].transform(query)
report={'x':n['x'].ravel().tolist(),'y':n['y'].tolist(),'intercept':float(n['b']),
        'coefficients':np.asarray(n['coefs'] if 'coefs' in n else [n['m']]).tolist(),
        'query':n['test'].ravel().tolist(),'prediction':float(n['model'].predict(query)[0]),
        'fitted_predictions':n['model'].predict(n['x_modified'] if 'x_modified' in n else n['x']).tolist()}
open(sys.argv[2],'w').write(json.dumps(report))
"""
        for folder, original_y, degree in cases:
            with self.subTest(folder=folder), tempfile.TemporaryDirectory(prefix="ml-current-regression-") as tmp:
                cwd = Path(tmp)
                path = ROOT / folder / "main.py"
                args = [sys.executable, "-I", "-B", "-c", wrapper, str(path), str(cwd / "actual.json")]
                child = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                record = dict(parentTask="current-linked-regression-contract", parentPid=os.getpid(), command=args,
                              cwd=str(cwd), pid=child.pid, source=folder, startTime=time.time(), timeoutSeconds=45)
                try:
                    stdout, stderr = child.communicate(timeout=45)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.communicate()
                    record.update(endTime=time.time(), exitCode=child.returncode, childProcessGroupCleanup="terminated and reaped")
                    EXECUTIONS.append(record)
                    raise
                record.update(endTime=time.time(), exitCode=child.returncode, childProcessGroupCleanup="not needed")
                EXECUTIONS.append(record)
                self.assertEqual(child.returncode, 0, stderr.decode())
                result = json.loads((cwd / "actual.json").read_text())
                self.assertEqual(result["x"], list(range(1, 11)))
                self.assertEqual(result["y"], original_y)
                self.assertEqual(result["query"], [7])
                coefficients = least_squares(result["x"], result["y"], degree)
                for actual, expected in zip([result["intercept"]] + result["coefficients"], coefficients):
                    self.assertAlmostEqual(actual, float(expected), places=8)
                predicted = float(sum(value * 7 ** power for power, value in enumerate(coefficients)))
                self.assertAlmostEqual(result["prediction"], predicted, places=8)
                text = stdout.decode()
                printed_predictions = list(map(float, re.findall(r"Predicted value of y for x = 7: ([^\n]+)", text)))
                self.assertEqual(len(printed_predictions), degree)
                for candidate, value in enumerate(printed_predictions, 1):
                    fitted = least_squares(result["x"], result["y"], candidate)
                    self.assertAlmostEqual(value, float(sum(coef * 7 ** power for power, coef in enumerate(fitted))), places=8)
                self.assertNotIn("Correlation Between x and y", text)
                if degree == 1:
                    residual = sum((a - b) ** 2 for a, b in zip(result["y"], result["fitted_predictions"]))
                    mean = sum(result["y"]) / 10
                    total = sum((value - mean) ** 2 for value in result["y"])
                    r_squared = float(re.search(r"Training R-squared: ([^\n]+)", text)[1])
                    correlation = float(re.search(r"Pearson correlation: ([^\n]+)", text)[1])
                    self.assertAlmostEqual(r_squared, 1 - residual / total)
                    self.assertAlmostEqual(correlation, math.sqrt(r_squared))
                else:
                    self.assertEqual(text.count("Training R-squared:"), 2)
                    self.assertEqual(text.count("Training MAE:"), 2)
                    self.assertEqual(text.count("Training RMSE:"), 2)
                record.update(exactOriginalArraysPreserved=True, independentFractionFit=True, printedQueryMatchesActual7=True)


if __name__ == "__main__":
    unittest.main()
