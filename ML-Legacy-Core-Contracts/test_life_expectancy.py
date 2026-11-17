"""Check actual country-year programs, frozen selection and independent errors."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "ML5-Predicting-Life-Expectancy"
STARTER = ROOT / "ML5-Predicting-Life-Expectancy-Starter"
EXECUTIONS = []


def execute(path, cwd, output, stop=False, notebook=False):
    actual = path
    if notebook:
        nb = json.loads(path.read_text())
        actual = cwd / "actual-notebook-cells.py"
        actual.write_text("\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"))
    args = [sys.executable, "-I", "-B", str(actual)]
    if not notebook:
        args += ["--output-dir", str(output)]
    child = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    record = dict(source=path.name, parentTask="original-country-year-regression-contract", parentPid=os.getpid(),
                  command=args, cwd=str(cwd), pid=child.pid, startTime=time.time(), timeoutSeconds=45)
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
    record.update(endTime=time.time(), exitCode=child.returncode, childProcessGroupCleanup="not needed",
                  expectedIntentionalStarterStop=stop, actualNotebookCells=notebook)
    EXECUTIONS.append(record)
    if stop:
        assert child.returncode != 0 and b"NotImplementedError" in stderr and b"fit_model" in stderr, stderr.decode()
        return None
    assert child.returncode == 0, stderr.decode()
    report = json.loads((output / "regression-report.json").read_text())
    assert json.loads(stdout)["metrics"] == report["metrics"]
    assert b"Accuracy" not in stdout and b" %" not in stdout
    record.update(rows=report["rows"], trainRows=report["train_rows"], validationRows=report["validation_rows"],
                  testRows=report["test_rows"], selectedDegree=report["selected_degree"], metrics=report["metrics"])
    return report


def load_program():
    spec = importlib.util.spec_from_file_location("actual_country_year_regression", REFERENCE / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalCountryYearRegression(unittest.TestCase):
    def check_report(self, report):
        frame = pd.read_csv(REFERENCE / "life_expectancy.csv")
        self.assertEqual((report["rows_read"], report["rows"], report["missing_targets_removed"]), (2938, 2928, 10))
        self.assertEqual(report["data_unit"], "country-year")
        train, validation, test = (report[name + "_ids"] for name in ("train", "validation", "test"))
        self.assertFalse(set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
        self.assertEqual(sorted(train + validation + test), list(frame.dropna(subset=["Life expectancy "]).index))
        groups = [set(frame.iloc[ids]["Country"]) for ids in (train, validation, test)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual([sorted(values) for values in groups], [report["countries"][key] for key in ("train", "validation", "test")])
        features = report["features"]
        self.assertEqual(len(features), 19)
        self.assertNotIn("Country", features)
        self.assertNotIn("Status", features)
        self.assertNotIn("Life expectancy ", features)
        training = frame.iloc[train][features].to_numpy(dtype=float)
        medians = np.nanmedian(training, axis=0)
        medians = np.where(np.isnan(medians), 0, medians)
        imputed = np.where(np.isnan(training), medians, training)
        state = report["fitted_state"]
        np.testing.assert_allclose(state["imputation_medians"], medians)
        np.testing.assert_allclose(state["scaling_mean"], imputed.mean(axis=0))
        np.testing.assert_allclose(state["scaling_scale"], imputed.std(axis=0))
        test_x = frame.iloc[test][features].to_numpy(dtype=float)
        filled = np.where(np.isnan(test_x), medians, test_x)
        scaled = (filled - state["scaling_mean"]) / state["scaling_scale"]
        powers = np.array(state["polynomial_powers"])
        terms = np.prod(scaled[:, None, :] ** powers[None, :, :], axis=2)
        independently_predicted = terms @ np.array(state["coefficients"]) + state["intercept"]
        np.testing.assert_allclose(report["test_predictions"], independently_predicted, rtol=1e-8, atol=1e-6)
        actual = frame.iloc[test]["Life expectancy "].to_numpy(dtype=float)
        self.assertEqual(report["test_truth"], actual.tolist())
        mean = float(frame.iloc[train]["Life expectancy "].mean())
        np.testing.assert_allclose(report["baseline_predictions"], [mean] * len(test))
        for key, predicted in [("metrics", report["test_predictions"]), ("baseline_metrics", report["baseline_predictions"])]:
            errors = [float(a) - float(b) for a, b in zip(actual, predicted)]
            squared = math.fsum(error * error for error in errors)
            center = math.fsum(actual) / len(actual)
            total = math.fsum((value - center) ** 2 for value in actual)
            self.assertAlmostEqual(report[key]["mae_years"], math.fsum(abs(error) for error in errors) / len(errors))
            self.assertAlmostEqual(report[key]["rmse_years"], math.sqrt(squared / len(errors)))
            self.assertAlmostEqual(report[key]["r_squared"], 1 - squared / total)
        validation_metrics = report["validation_metrics"]
        expected = 2 if validation_metrics["2"]["mae_years"] < 0.98 * validation_metrics["1"]["mae_years"] else 1
        self.assertEqual(report["selected_degree"], expected)

    def test_actual_reference_script_and_notebook_grouped_rows_and_predictions(self):
        csv = REFERENCE / "life_expectancy.csv"
        before = hashlib.sha256(csv.read_bytes()).hexdigest()
        for name in ["Life-Expectancy-Linear-Regression.py", "LifeExpectancyLinearRegression.ipynb"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="ml-country-year-reference-") as tmp:
                cwd = Path(tmp)
                (cwd / "life_expectancy.csv").write_bytes(csv.read_bytes())
                report = execute(REFERENCE / name, cwd, cwd / "output", notebook=name.endswith("ipynb"))
                self.check_report(report)
                EXECUTIONS[-1].update(independentAllPredictionsAndMetrics=True, countryDisjoint=True,
                                      trainingOnlyImputationScaling=True, exactCSVPreserved=True)
        self.assertEqual(hashlib.sha256(csv.read_bytes()).hexdigest(), before)

    def test_named_starter_scripts_and_notebook_then_completed_starter(self):
        for path in [REFERENCE / "Life-Expectancy-Linear-Regression-Starter.py", STARTER / "main.py", REFERENCE / "LifeExpectancyLinearRegressionStarter.ipynb"]:
            with self.subTest(name=path.name), tempfile.TemporaryDirectory(prefix="ml-country-year-unfinished-") as tmp:
                cwd = Path(tmp)
                (cwd / "life_expectancy.csv").write_bytes((REFERENCE / "life_expectancy.csv").read_bytes())
                execute(path, cwd, cwd / "output", stop=True, notebook=path.suffix == ".ipynb")
        with tempfile.TemporaryDirectory(prefix="ml-country-year-completed-") as tmp:
            cwd = Path(tmp)
            original, complete = (STARTER / "main.py").read_text(), (REFERENCE / "main.py").read_text()
            function = complete[complete.index("def fit_model("):complete.index("\n\ndef metrics(")]
            modified = original[:original.index("def fit_model(")] + function + original[original.index("\n\ndef metrics("):]
            path = cwd / "main.py"
            path.write_text(modified)
            (cwd / "life_expectancy.csv").write_bytes((STARTER / "life_expectancy.csv").read_bytes())
            self.check_report(execute(path, cwd, cwd / "output"))

    def test_held_out_mutations_preserve_fitted_state_and_negative_r_squared_is_valid(self):
        module = load_program()
        self.assertEqual(module.metrics([2, 4], [1, 5]), {"mae_years": 1.0, "rmse_years": 1.0, "r_squared": 0.0})
        self.assertAlmostEqual(module.metrics([0, 1], [2, 2])["r_squared"], -9)
        self.assertEqual(module.choose_degree({1: {"mae_years": 1}, 2: {"mae_years": 0.99}}), 1)
        self.assertEqual(module.choose_degree({1: {"mae_years": 1}, 2: {"mae_years": 0.9}}), 2)
        frame = pd.read_csv(REFERENCE / "life_expectancy.csv")
        report, _ = module.fit_experiment(frame)
        changed = frame.copy()
        changed.loc[report["test_ids"], "Life expectancy "] += 10000
        changed.loc[report["test_ids"], "GDP"] += 1000000
        replay, _ = module.fit_experiment(changed)
        for key in ["train_ids", "validation_ids", "test_ids", "selected_degree", "fitted_state", "validation_metrics"]:
            self.assertEqual(report[key], replay[key])


if __name__ == "__main__":
    unittest.main()
