"""Run original scripts/notebooks and independently recompute their count model."""
from collections import Counter
import csv
import importlib.util
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
REFERENCE = ROOT / "ML3-Email-Spam-Classification"
STARTER = ROOT / "ML3-Email-Spam-Classification-Starter"
EXECUTIONS = []


def execute(path, cwd, output, stop=False, notebook=False):
    actual = path
    if notebook:
        nb = json.loads(path.read_text())
        code = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        actual = cwd / "actual-notebook-cells.py"
        actual.write_text(code)
    args = [sys.executable, "-I", "-B", str(actual)]
    if not notebook:
        args += ["--output-dir", str(output)]
    child = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    record = dict(source=path.name, parentTask="original-email-pipeline-contract", parentPid=os.getpid(),
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
        assert child.returncode != 0 and b"NotImplementedError" in stderr and b"fit_pipeline" in stderr, stderr.decode()
        return None
    assert child.returncode == 0, stderr.decode()
    report = json.loads((output / "spam-report.json").read_text())
    assert json.loads(stdout)["metrics"] == report["metrics"]
    assert b"Project meeting" not in stdout and b"Prize offer" not in stdout
    record.update(rows=report["rows"], trainingRows=report["train_rows"],
                  validationRows=report["validation_rows"], testRows=report["test_rows"],
                  macroF1=report["metrics"]["macro_f1"])
    return report


class OriginalEmailPipeline(unittest.TestCase):
    def check_report(self, report):
        with (REFERENCE / "fictional-emails.csv").open(newline="") as file:
            raw = list(csv.DictReader(file))
        rows = list({(row["text"], row["label_num"]): row for row in raw}.values())
        self.assertEqual((report["input_rows"], report["rows"], report["duplicates_removed"]), (66, 64, 2))
        train, validation, test = (report[name + "_ids"] for name in ("train", "validation", "test"))
        self.assertEqual((len(train), len(validation), len(test)), (38, 13, 13))
        self.assertEqual(sorted(train + validation + test), list(range(64)))
        self.assertFalse(set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
        tokenize = lambda value: re.findall(r"(?u)\b\w\w+\b", value.lower())
        vocabulary = sorted(set(token for index in train for token in tokenize(rows[index]["text"])))
        self.assertEqual(report["vocabulary"], vocabulary)
        self.assertEqual(report["vocabulary_by_index"], vocabulary)
        class_counts = Counter(int(rows[index]["label_num"]) for index in train)
        token_counts = [Counter(token for index in train if int(rows[index]["label_num"]) == label
                                for token in tokenize(rows[index]["text"])) for label in (0, 1)]
        self.assertEqual(report["class_count"], [class_counts[label] for label in (0, 1)])
        self.assertEqual(report["class_token_counts"], [[counts[token] for token in vocabulary] for counts in token_counts])
        predicted = []
        for index in test:
            words = Counter(tokenize(rows[index]["text"]))
            scores = [math.log(class_counts[label] / len(train)) + sum(
                count * math.log((token_counts[label][token] + 1) / (sum(token_counts[label].values()) + len(vocabulary)))
                for token, count in words.items() if token in vocabulary) for label in (0, 1)]
            predicted.append(0 if scores[0] >= scores[1] else 1)
        truth = [int(rows[index]["label_num"]) for index in test]
        self.assertEqual(report["test_truth"], truth)
        self.assertEqual(report["test_predictions"], predicted)
        majority = min((0, 1), key=lambda label: (-class_counts[label], label))
        self.assertEqual(report["baseline_predictions"], [majority] * len(test))
        for key, guesses in [("metrics", predicted), ("baseline_metrics", report["baseline_predictions"])]:
            matrix = [[sum(a == label and b == guess for a, b in zip(truth, guesses)) for guess in (0, 1)] for label in (0, 1)]
            self.assertEqual(report[key]["confusion_matrix"], matrix)
            self.assertAlmostEqual(report[key]["accuracy"], (matrix[0][0] + matrix[1][1]) / len(test))
            f1s = []
            for label in (0, 1):
                tp = matrix[label][label]
                actual, called = sum(matrix[label]), sum(row[label] for row in matrix)
                precision, recall = (tp / called if called else 0), (tp / actual if actual else 0)
                f1 = 2 * tp / (actual + called) if actual + called else 0
                self.assertAlmostEqual(report[key]["precision"][label], precision)
                self.assertAlmostEqual(report[key]["recall"][label], recall)
                self.assertAlmostEqual(report[key]["f1"][label], f1)
                f1s.append(f1)
            self.assertAlmostEqual(report[key]["macro_f1"], sum(f1s) / 2)

    def test_actual_reference_script_and_notebook_use_only_training_vocabulary(self):
        for name in ["Email-Spam-Classifier.py", "EmailSpamClassifier.ipynb"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="ml-email-reference-") as tmp:
                cwd = Path(tmp)
                (cwd / "fictional-emails.csv").write_bytes((REFERENCE / "fictional-emails.csv").read_bytes())
                report = execute(REFERENCE / name, cwd, cwd / "output", notebook=name.endswith("ipynb"))
                self.check_report(report)
                EXECUTIONS[-1].update(independentAllPredictionsAndMetrics=True, fittedVocabularyTrainingOnly=True)

    def test_named_starter_scripts_and_notebook_then_completed_standalone_starter(self):
        for path in [REFERENCE / "Email-Spam-Classifier-Starter.py", STARTER / "main.py", REFERENCE / "EmailSpamClassifierStarter.ipynb"]:
            with self.subTest(path=path.name), tempfile.TemporaryDirectory(prefix="ml-email-unfinished-") as tmp:
                cwd = Path(tmp)
                (cwd / "fictional-emails.csv").write_bytes((REFERENCE / "fictional-emails.csv").read_bytes())
                execute(path, cwd, cwd / "output", stop=True, notebook=path.suffix == ".ipynb")
        with tempfile.TemporaryDirectory(prefix="ml-email-completed-") as tmp:
            cwd = Path(tmp)
            original = (STARTER / "main.py").read_text()
            complete = (REFERENCE / "main.py").read_text()
            function = complete[complete.index("def fit_pipeline("):complete.index("\n\ndef metrics(")]
            modified = original[:original.index("def fit_pipeline(")] + function + original[original.index("\n\ndef metrics("):]
            path = cwd / "main.py"
            path.write_text(modified)
            (cwd / "fictional-emails.csv").write_bytes((STARTER / "fictional-emails.csv").read_bytes())
            self.check_report(execute(path, cwd, cwd / "output"))

    def test_smoothed_hand_model_and_unseen_held_out_word(self):
        spec = importlib.util.spec_from_file_location("actual_email_pipeline", REFERENCE / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.fit_pipeline(["meeting meeting note", "prize prize note"], [0, 1])
        self.assertAlmostEqual(model.predict_proba(["meeting note"])[0][0], 3 / 4)
        self.assertAlmostEqual(model.predict_proba(["meeting note"])[0][1], 1 / 4)
        words = model.named_steps["words"]
        before = words.vocabulary_.copy()
        self.assertEqual(words.transform(["quasarzzonly"]).sum(), 0)
        model.predict(["quasarzzonly"])
        self.assertEqual(words.vocabulary_, before)
        self.assertEqual(module.metrics([0, 0, 1], [0, 1, 1])["confusion_matrix"], [[1, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
