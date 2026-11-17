"""Check the actual ten-point introductory regression program against hand arithmetic."""
from fractions import Fraction
import math
import os
import signal
import time
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXECUTIONS = []


class OriginalRegressionStatistics(unittest.TestCase):
    def test_actual_program_names_and_reports_distinct_training_statistics(self):
        x = list(map(Fraction, range(1, 11)))
        y = list(map(Fraction, [4, 5, 6, 3, 5, 7, 3, 9, 13, 14]))
        mean_x, mean_y = sum(x) / 10, sum(y) / 10
        sxx = sum((value - mean_x) ** 2 for value in x)
        syy = sum((value - mean_y) ** 2 for value in y)
        sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
        slope = sxy / sxx
        intercept = mean_y - slope * mean_x
        residual = sum((b - (slope * a + intercept)) ** 2 for a, b in zip(x, y))
        expected_r2 = float(1 - residual / syy)
        expected_r = float(sxy) / math.sqrt(float(sxx * syy))
        with tempfile.TemporaryDirectory(prefix="ml-original-regression-") as temp:
            args = [sys.executable, "-I", "-B", str(ROOT / "ML5-Simple-Linear-Regression/main.py")]
            child = subprocess.Popen(args, cwd=temp, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, start_new_session=True,
                                     env={**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            process = dict(folder="ML5-Simple-Linear-Regression", parentPid=os.getpid(), pid=child.pid,
                           parentTask=os.environ.get("CLASSES_FAMILY_TASK_ID", "original-regression-contract"),
                           command=args, cwd=temp, startTime=time.time(), timeoutSeconds=30)
            EXECUTIONS.append(process)
            cleanup = "not needed"
            try:
                stdout, stderr = child.communicate(timeout=30)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.communicate()
                cleanup = "process group terminated after timeout or cancellation"
                raise
            finally:
                process.update(endTime=time.time(), exitCode=child.returncode,
                               childProcessGroupCleanup=cleanup)
            self.assertEqual(child.returncode, 0, stderr)
            self.assertEqual(stderr, "")
            r2 = float(re.search(r"Training R-squared: ([-+0-9.eE]+)", stdout)[1])
            correlation = float(re.search(r"Pearson correlation: ([-+0-9.eE]+)", stdout)[1])
            self.assertAlmostEqual(r2, expected_r2, places=12)
            self.assertAlmostEqual(correlation, expected_r, places=12)
            self.assertNotAlmostEqual(r2, correlation, places=2)
            self.assertAlmostEqual(r2, correlation ** 2, places=12)
            self.assertNotIn("Correlation Between x and y:", stdout)
            predicted = float(re.search(r"Predicted value of y for x = 7: ([-+0-9.eE]+)", stdout)[1])
            self.assertAlmostEqual(predicted, float(slope * 7 + intercept), places=12)
            for name in ["before.png", "after.png"]:
                self.assertTrue((Path(temp) / name).read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a")))
        process.update(trainingRSquared=r2,
                               pearsonCorrelation=correlation, predictionAt7=predicted,
                               independentHeldOutPerformanceClaimed=False)


if __name__ == "__main__":
    unittest.main()
