import copy
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
    spec = importlib.util.spec_from_file_location("bayes_" + role, ROOT / role / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BayesContract:
    def setUp(self):
        self.model = load("solution")

    def script_result(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(ROOT / "solution/main.py")],
                cwd=elsewhere, text=True, capture_output=True, check=True, timeout=15,
            )
        return json.loads(result.stdout)

    def test_starter_algorithms_are_explicitly_unfinished(self):
        starter = load("starter")
        for name in ["fit", "log_scores"]:
            with self.subTest(function=name), self.assertRaises(NotImplementedError):
                getattr(starter, name)([], [])


class MultinomialContract(BayesContract, unittest.TestCase):
    def fitted(self, alpha=1):
        docs, labels, _, _ = self.model.load_fixture()
        return self.model.fit(docs, labels, alpha)

    def test_training_token_counts_and_document_priors(self):
        model = self.fitted()
        self.assertEqual(model["vocabulary"], {"team", "meeting", "project", "claim", "prize", "offer"})
        self.assertEqual(model["totals"], {"ham": 6, "spam": 6})
        self.assertEqual(dict(model["counts"]["spam"]), {"claim": 2, "prize": 2, "offer": 2})
        self.assertEqual(model["priors"], {"ham": 0.5, "spam": 0.5})

    def test_laplace_hand_calculation(self):
        model = self.fitted()
        scores = self.model.log_scores(model, "claim prize")
        self.assertAlmostEqual(math.exp(scores["spam"]), 1 / 32)
        self.assertAlmostEqual(math.exp(scores["ham"]), 1 / 288)
        probability = self.model.probabilities(model, "claim prize")
        self.assertAlmostEqual(probability["spam"], 0.9)
        self.assertAlmostEqual(probability["ham"], 0.1)
        self.assertAlmostEqual(sum(probability.values()), 1)

    def test_repeated_words_remain_counted(self):
        model = self.fitted()
        one = self.model.log_scores(model, "prize")
        two = self.model.log_scores(model, "prize prize")
        self.assertAlmostEqual(two["spam"] - one["spam"], math.log(1 / 4))
        self.assertAlmostEqual(two["ham"] - one["ham"], math.log(1 / 12))

    def test_long_message_stays_finite_despite_product_underflow(self):
        model = self.fitted()
        self.assertEqual((1 / 4) ** 2000, 0.0)
        text = "prize " * 2000
        self.assertTrue(all(math.isfinite(value) for value in self.model.log_scores(model, text).values()))
        self.assertEqual(self.model.predict(model, text), "spam")
        self.assertAlmostEqual(self.model.probabilities(model, text)["spam"], 1.0)

    def test_unknown_tokens_never_refit_vocabulary(self):
        model = self.fitted()
        before = copy.deepcopy(model)
        self.assertEqual(self.model.predict(model, "unseen team"), "ham")
        self.assertNotIn("unseen", model["vocabulary"])
        self.assertEqual(model, before)
        self.assertEqual(self.model.log_scores(model, "unseen team"), self.model.log_scores(model, "team"))

    def test_empty_and_unknown_use_priors_and_label_tie_rule(self):
        model = self.model.fit(["a", "b", "b"], ["A", "B", "B"])
        self.assertEqual(self.model.predict(model, ""), "B")
        self.assertEqual(self.model.predict(model, "unknown"), "B")
        self.assertEqual(self.model.predict(self.fitted(), ""), "ham")

    def test_declared_tokenization_and_smoothing_change(self):
        self.assertEqual(self.model.tokenize("TEAM, prize! 123 café"), ["team", "prize", "caf"])
        self.assertAlmostEqual(self.model.probabilities(self.fitted(2), "claim prize")["spam"], 0.8)

    def test_invalid_training_and_alpha(self):
        for docs, labels in [([], []), (["a"], []), (["a"], [""]), (["123"], ["A"]), ([None], ["A"])]:
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                self.model.fit(docs, labels)
        for alpha in [0, -1, math.nan, math.inf, True, "x"]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                self.model.fit(["a"], ["A"], alpha)
        with self.assertRaises(ValueError):
            self.model.log_scores(self.fitted(), None)

    def test_all_fixture_predictions_and_absolute_script_path(self):
        result = self.script_result()
        self.assertEqual(result["predictions"], ["ham", "spam", "ham", "spam"])
        self.assertEqual((result["training_rows"], result["test_rows"], result["vocabulary_size"]), (6, 4, 6))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["claim_prize_probabilities"]["spam"], 0.9)


if __name__ == "__main__":
    unittest.main()
