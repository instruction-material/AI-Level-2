"""Execute the assigned library roles and check their fitted models independently."""
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "ML-Library-Workspaces"
MANIFEST = json.loads((PACKS / "manifest.json").read_text())
EXECUTIONS = []


def role(project, selected="solution"):
    return next(row for row in MANIFEST["resources"]
                if row["project"] == project and row["role"] == selected)


def module_at(path):
    name = "library_role_" + path.parent.parent.name.replace("-", "_") + "_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def experiment_module(project):
    folder = ROOT / role(project)["folder"]
    previous = sys.modules.get("exercise")
    sys.modules["exercise"] = module_at(folder / "exercise.py")
    try:
        yield module_at(folder / "main.py")
    finally:
        if previous is None:
            sys.modules.pop("exercise", None)
        else:
            sys.modules["exercise"] = previous


class LibraryWorkspaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ml-seven-library-")
        cls.output = Path(cls.temporary.name)
        cls.references = {}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def execute(self, metadata, label, completed=False):
        target = self.output / label
        target.mkdir()
        for file in metadata["files"]:
            shutil.copyfile(ROOT / file["path"], target / file["name"])
        if completed:
            shutil.copyfile(ROOT / role(metadata["project"])["folder"] / "exercise.py",
                            target / "exercise.py")
        args = [sys.executable, "-B", str(target / "main.py")]
        if "--output-dir" in metadata["command"]:
            args += ["--output-dir", str(target)]
        env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1")
        record = dict(parentTask=env.get("CLASSES_FAMILY_TASK_ID", "seven-library-contracts"),
                      parentPid=os.getpid(), cwd=str(target), command=args,
                      startTime=time.time(), timeoutSeconds=60, project=metadata["project"],
                      role=metadata["role"], completedLearner=completed)
        child = subprocess.Popen(args, cwd=target, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, start_new_session=True)
        record["pid"] = child.pid
        cleanup = "not needed"
        try:
            stdout, stderr = child.communicate(timeout=60)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
            cleanup = "owned process group terminated and reaped"
            raise
        finally:
            record.update(endTime=time.time(), exitCode=child.returncode,
                          childProcessGroupCleanup=cleanup)
            EXECUTIONS.append(record)
        return target, child.returncode, stdout.decode(), stderr.decode()

    def reference(self, project):
        if project not in self.references:
            result = self.execute(role(project), project + "-solution")
            self.assertEqual(result[1], 0, result[3])
            self.references[project] = result
        return self.references[project]

    def test_all_seven_reference_programs_execute(self):
        for project in sorted({row["project"] for row in MANIFEST["resources"]}):
            with self.subTest(project=project):
                target, _, stdout, _ = self.reference(project)
                self.assertTrue(stdout.strip())
                expected = {
                    "customer-neighbors": "classification-report.json",
                    "car-neighbors": "classification-report.json",
                    "email-naive-bayes": "spam-report.json",
                    "country-year-regression": "regression-report.json",
                    "linear-regression": "after.png",
                    "polynomial-regression": "polynomial.png",
                }.get(project)
                if expected:
                    self.assertGreater((target / expected).stat().st_size, 0)
                if project == "iris-naive-bayes":
                    self.assertIn("GaussianNB macro F1:", stdout)

    def test_all_seven_unfinished_learners_stop_at_the_named_task(self):
        for metadata in MANIFEST["resources"]:
            if metadata["role"] != "starter":
                continue
            with self.subTest(project=metadata["project"]):
                _, code, _, stderr = self.execute(metadata, metadata["project"] + "-starter")
                self.assertNotEqual(code, 0)
                self.assertIn("NotImplementedError", stderr)
                self.assertIn(metadata["taskFunction"], stderr)

    def test_all_seven_completed_learners_execute(self):
        for metadata in MANIFEST["resources"]:
            if metadata["role"] != "starter":
                continue
            with self.subTest(project=metadata["project"]):
                _, code, stdout, stderr = self.execute(
                    metadata, metadata["project"] + "-completed", completed=True)
                self.assertEqual(code, 0, stderr)
                self.assertTrue(stdout.strip())

    def test_customer_scaling_uses_training_rows_only(self):
        helper = module_at(ROOT / role("customer-neighbors")["folder"] / "exercise.py")
        features = np.array([[0., 0.], [2., 2.], [4., 4.]])
        model = helper.fit_classifier(features, [0, 0, 1], neighbors=1)
        scale = model.named_steps["scale"]
        np.testing.assert_allclose(scale.mean_, [2., 2.])
        np.testing.assert_allclose(scale.var_, [8 / 3, 8 / 3])
        self.assertEqual(model.predict([[0., 0.]])[0], 0)
        model.predict([[100., 100.]])
        np.testing.assert_allclose(scale.mean_, [2., 2.])
        self.assertEqual(model.named_steps["neighbors"].n_samples_fit_, 3)

    def test_car_encoder_never_learns_an_unknown_test_category(self):
        helper = module_at(ROOT / role("car-neighbors")["folder"] / "exercise.py")
        features = pd.DataFrame([["low"] * 6, ["high"] * 6], columns=helper.FEATURES)
        model = helper.fit_classifier(features, ["A", "B"], neighbors=1)
        encoder = model.named_steps["encode"].named_transformers_["categories"]
        original = [values.tolist() for values in encoder.categories_]
        unknown = features.iloc[[0]].copy()
        unknown.loc[:, "buying"] = "unseen"
        encoded = model.named_steps["encode"].transform(unknown)[0]
        self.assertEqual(sum(encoded[:2]), 0)
        self.assertEqual([values.tolist() for values in encoder.categories_], original)
        self.assertEqual(model.predict(features).tolist(), ["A", "B"])

    def test_iris_gaussian_parameters_match_independent_moments(self):
        helper = module_at(ROOT / role("iris-naive-bayes")["folder"] / "exercise.py")
        features = np.array([[1.], [3.], [7.], [9.]])
        model = helper.fit_classifier(features, ["A", "A", "B", "B"])
        np.testing.assert_allclose(model.theta_, [[2.], [8.]])
        np.testing.assert_allclose(model.var_, [[1. + 1e-9 * features.var()]] * 2)
        np.testing.assert_allclose(model.class_prior_, [.5, .5])
        self.assertEqual(model.predict([[2.]])[0], "A")

    def test_email_count_probabilities_and_unknown_vocabulary(self):
        helper = module_at(ROOT / role("email-naive-bayes")["folder"] / "exercise.py")
        model = helper.fit_pipeline(["meeting meeting note", "prize prize note"], [0, 1])
        np.testing.assert_allclose(model.predict_proba(["meeting note"])[0], [.75, .25])
        vocabulary = dict(model.named_steps["words"].vocabulary_)
        model.predict(["unseen"])
        self.assertEqual(model.named_steps["words"].vocabulary_, vocabulary)
        binary = helper.fit_pipeline(["meeting meeting note", "prize prize note"], [0, 1], binary=True)
        self.assertEqual(binary.named_steps["words"].transform(["meeting meeting"]).sum(), 1)

    def test_linear_fit_matches_exact_line(self):
        helper = module_at(ROOT / role("linear-regression")["folder"] / "exercise.py")
        model = helper.fit_model([[0.], [1.], [2.]], [1., 3., 5.])
        self.assertAlmostEqual(model.intercept_, 1.)
        self.assertAlmostEqual(model.coef_[0], 2.)
        self.assertAlmostEqual(model.predict([[3.]])[0], 7.)

    def test_polynomial_fit_reuses_the_training_transform(self):
        helper = module_at(ROOT / role("polynomial-regression")["folder"] / "exercise.py")
        values = np.arange(-2., 3.)
        transform, model = helper.fit_model(values.reshape(-1, 1), 2 + 3 * values + values ** 2)
        np.testing.assert_allclose(transform.transform([[7.]]), [[7., 49.]])
        np.testing.assert_allclose(model.coef_, [3., 1.], atol=1e-12)
        self.assertAlmostEqual(model.intercept_, 2.)
        self.assertAlmostEqual(model.predict(transform.transform([[7.]]))[0], 72.)

    def test_country_ridge_matches_independent_regularized_normal_equations(self):
        helper = module_at(ROOT / role("country-year-regression")["folder"] / "exercise.py")
        features = pd.DataFrame({"Year": [2000., 2001., 2002., 2003., 2004., 2005.],
                                 "value": [1., np.nan, 2., 4., 3., 5.]})
        targets = np.array([10., 11., 12., 13., 14., 15.])
        model = helper.fit_model(features, targets, 2)
        np.testing.assert_allclose(model.named_steps["fill"].statistics_, [2002.5, 3.])
        filled = model.named_steps["fill"].transform(features)
        scaled = model.named_steps["scale"].transform(filled)
        design = model.named_steps["polynomial"].transform(scaled)
        centered = design - design.mean(axis=0)
        expected = np.linalg.solve(centered.T @ centered + np.eye(design.shape[1]),
                                   centered.T @ (targets - targets.mean()))
        fitted = model.named_steps["regression"]
        self.assertEqual(type(fitted).__name__, "Ridge")
        self.assertEqual(fitted.alpha, 1.)
        np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)
        np.testing.assert_allclose(fitted.intercept_, targets.mean() - design.mean(axis=0) @ expected)

    def test_country_final_reports_cover_both_frozen_candidates_and_disjoint_groups(self):
        target, _, _, _ = self.reference("country-year-regression")
        report = json.loads((target / "regression-report.json").read_text())
        groups = [set(report["countries"][key]) for key in ["train", "validation", "test"]]
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i)))
        self.assertEqual(set(report["candidate_test_metrics"]), {"1", "2"})
        self.assertEqual(set(report["candidate_test_predictions"]), {"1", "2"})
        self.assertTrue(report["model_frozen_before_test"])
        for degree, predicted in report["candidate_test_predictions"].items():
            errors = np.asarray(predicted) - np.asarray(report["test_truth"])
            measured = report["candidate_test_metrics"][degree]
            self.assertAlmostEqual(measured["mae_years"], float(np.abs(errors).mean()))
            self.assertAlmostEqual(measured["rmse_years"], float(np.sqrt((errors ** 2).mean())))
        self.assertEqual(set(report["evaluation"]["slices"]), {
            "year_at_or_below_training_median", "year_above_training_median",
            "complete_covariates", "missing_covariates"})

    def test_country_resampling_keeps_each_country_block_together(self):
        with experiment_module("country-year-regression") as module:
            rows = pd.DataFrame({"Country": ["A", "A", "B", "B"],
                                 "Year": [1999., 2000., 2001., 2002.],
                                 "value": [np.nan, 2., 3., 4.]})
            evidence = module.evaluation_evidence(rows, [0., 0., 0., 0.],
                                                 [0., 0., 2., 2.], ["Year", "value"], 2000.)
        distribution = evidence["uncertainty"]["estimates_mae_years"]
        self.assertEqual(len(distribution), 200)
        self.assertEqual(set(distribution), {0., 1., 2.})
        self.assertEqual(evidence["slices"]["year_at_or_below_training_median"]["mae_years"], 0.)
        self.assertEqual(evidence["slices"]["year_above_training_median"]["mae_years"], 2.)
        self.assertEqual(evidence["slices"]["missing_covariates"]["rows"], 1)

    def test_changed_final_test_targets_never_refit_or_reselect_the_country_models(self):
        target, _, _, _ = self.reference("country-year-regression")
        original = json.loads((target / "regression-report.json").read_text())
        data = pd.read_csv(ROOT / role("country-year-regression")["folder"] / "life_expectancy.csv")
        data.loc[original["test_ids"], "Life expectancy "] += 1000.
        with experiment_module("country-year-regression") as module:
            changed, _ = module.fit_experiment(data)
        self.assertEqual(original["selected_degree"], changed["selected_degree"])
        self.assertEqual(original["validation_metrics"],
                         {str(k): v for k, v in changed["validation_metrics"].items()})
        self.assertEqual(original["fitted_state"], changed["fitted_state"])
        np.testing.assert_allclose(original["test_predictions"], changed["test_predictions"])
        self.assertGreater(changed["metrics"]["mae_years"], 900.)

    def test_roles_are_complete_hash_bound_and_use_the_existing_runtime_lock(self):
        import hashlib
        self.assertEqual(MANIFEST["projects"], 7)
        self.assertEqual(len(MANIFEST["resources"]), 14)
        for metadata in MANIFEST["resources"]:
            files = {item["name"]: item for item in metadata["files"]}
            self.assertTrue({"main.py", "exercise.py", "README.md", "requirements.txt", "SOURCES.md"} <= files.keys())
            self.assertEqual((ROOT / files["requirements.txt"]["path"]).read_bytes(),
                             (ROOT / "requirements-ml-legacy.txt").read_bytes())
            self.assertEqual("NotImplementedError" in (ROOT / files["exercise.py"]["path"]).read_text(),
                             metadata["role"] == "starter")
            for file in files.values():
                data = (ROOT / file["path"]).read_bytes()
                self.assertEqual(len(data), file["byteLength"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), file["sha256"])
            if metadata["dataset"]:
                self.assertEqual((ROOT / files[metadata["dataset"]]["path"]).read_bytes(),
                                 (ROOT / metadata["originalFolder"] / metadata["dataset"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
