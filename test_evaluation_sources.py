"""Independent hand-worked contracts and actual exported-case entry points."""
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import copy
import struct
import time
import zlib
ROOT = Path(__file__).resolve().parent

def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
c = load('ML-Source-Cases/classification_model.py', 'classification_reference')
r = load('ML-Source-Cases/regression_model.py', 'regression_reference')

def driver_for(folder):
    old = sys.modules.get('model')
    try:
        sys.modules['model'] = load(folder + '/solution/model.py', 'active_case_model')
        return load(folder + '/solution/main.py', 'active_case_driver')
    finally:
        if old is None:
            sys.modules.pop('model', None)
        else:
            sys.modules['model'] = old

class EvaluationContracts(unittest.TestCase):

    def test_hand_confusion_and_metrics(self):
        metrics = c.classification_metrics(['A', 'A', 'B'], ['A', 'B', 'B'], ['A', 'B'])
        self.assertEqual(metrics['confusion_matrix'], [[1, 1], [0, 1]])
        self.assertAlmostEqual(metrics['accuracy'], 2 / 3)
        self.assertAlmostEqual(metrics['macro_f1'], 2 / 3)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 3 / 4)

    def test_imbalanced_majority_does_not_hide_failures(self):
        truth = ['A'] * 18 + ['B'] * 4 + ['C'] * 2
        metrics = c.classification_metrics(truth, ['A'] * 24, ['A', 'B', 'C'])
        self.assertEqual(metrics['accuracy'], 0.75)
        self.assertAlmostEqual(metrics['macro_f1'], 2 / 7)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 1 / 3)
        self.assertEqual(metrics['per_class']['C']['recall'], 0)

    def test_absent_truth_class_is_declared(self):
        metrics = c.classification_metrics(['A'], ['A'], ['A', 'B'])
        self.assertEqual(metrics['macro_f1'], 0.5)
        self.assertEqual(metrics['balanced_accuracy'], 1)

    def test_invalid_metric_inputs(self):
        for actual, pred, labels in [([], [], ['A']), (['A'], [], ['A']), (['B'], ['A'], ['A']), (['A'], ['A'], ['A', 'A'])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                c.classification_metrics(actual, pred, labels)

    def test_nominal_vocab_is_fitted_on_training_only(self):
        train = [{'categories': ['red', 'small']}, {'categories': ['blue', 'small']}]
        vocab = c.fit_encoding(train)
        before = copy.deepcopy(vocab)
        self.assertEqual(vocab, [['blue', 'red'], ['small']])
        encoded = c.encode([{'categories': ['green', 'large']}], vocab)
        self.assertEqual(encoded, [[0, 0, 1, 0, 1]])
        self.assertEqual(vocab, before)

    def test_weighted_exact_match_and_tie(self):
        self.assertEqual(c.vote_neighbors([(0, '1', 'B'), (0, '2', 'A'), (0.0001, '3', 'C')], 'distance'), 'A')

    def test_weighted_and_uniform_can_differ(self):
        neighbors = [(0.01, '1', 'B'), (1, '2', 'A'), (1, '3', 'A')]
        self.assertEqual(c.vote_neighbors(neighbors, 'uniform'), 'A')
        self.assertEqual(c.vote_neighbors(neighbors, 'distance'), 'B')

    def test_declared_simplicity_rule(self):
        scores = {'dummy': 0.5, 'stump': 0.79, 'knn': 0.8}
        self.assertEqual(c.choose_candidate(scores, ['dummy', 'stump', 'knn']), 'stump')

    def test_all_constant_training_fit(self):
        rows = [{'id': '1', 'x': [3, 7], 'label': 'A'}, {'id': '2', 'x': [3, 7], 'label': 'B'}]
        model = c.fit_model(rows, 'stump')
        self.assertEqual(model['preprocessing']['scales'], [1, 1])
        self.assertEqual(c.predict(model, [{'x': [3, 7]}]), ['A'])

    def test_numeric_query_shape_is_validated(self):
        model = c.fit_model([{'id': '1', 'x': [1, 2], 'label': 'A'}], 'knn-1')
        with self.assertRaises(ValueError):
            c.predict(model, [{'x': [1, 2, 3]}])

    def test_every_grouped_fold_is_disjoint(self):
        driver = driver_for(next(ROOT.glob('ML-16-*')).name)
        rows, _, _ = driver.load_rows()
        train = c.partitions(rows)['train']
        seen = []
        for fit, validation in c.group_folds(train):
            self.assertFalse({r['group'] for r in fit} & {r['group'] for r in validation})
            self.assertEqual({r['label'] for r in fit}, {'amber', 'blue', 'green'})
            self.assertEqual({r['label'] for r in validation}, {'amber', 'blue', 'green'})
            seen.extend((r['id'] for r in validation))
        self.assertCountEqual(seen, [r['id'] for r in train])
        self.assertEqual(len(seen), len(set(seen)))

    def test_cross_partition_group_is_rejected(self):
        rows = [{'id': '1', 'group': 'g', 'split': 'train'}, {'id': '2', 'group': 'h', 'split': 'validation'}, {'id': '3', 'group': 'g', 'split': 'test'}]
        with self.assertRaises(ValueError):
            c.partitions(rows)

    def test_test_labels_and_features_cannot_select_or_fit(self):
        driver = driver_for(next(ROOT.glob('ML-16-*')).name)
        rows, _, _ = driver.load_rows()
        a = c.compare(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['split'] == 'test':
                row['label'] = 'amber'
                row['x'] = [9999, -9999]
        b = c.compare(changed)
        self.assertEqual(a['chosen'], b['chosen'])
        self.assertEqual(a['cross_validation'], b['cross_validation'])
        self.assertEqual(a['frozen_fit'], b['frozen_fit'])

    def test_car_case_uses_observed_four_classes(self):
        driver = driver_for(next(ROOT.glob('ML-29-*')).name)
        rows, _, source = driver.load_rows()
        self.assertEqual(source, 1728)
        self.assertEqual(len(rows), 160)
        parts = c.partitions(rows)
        self.assertEqual({name: len(v) for name, v in parts.items()}, {'train': 96, 'validation': 32, 'test': 32})
        self.assertEqual({row['label'] for row in rows}, {'acc', 'good', 'unacc', 'vgood'})

    def test_all_classification_references_and_starters(self):
        for case in json.loads((ROOT / 'ML-Source-Cases/classification-cases.json').read_text()):
            folder = ROOT / case['folder']
            with self.subTest(case=case['number']), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'report.json'
                result = subprocess.run([sys.executable, '-B', str(folder / 'solution/main.py'), '--output', str(output)], cwd=tmp, capture_output=True, text=True, timeout=20, check=True)
                report = json.loads(result.stdout)
                self.assertIn(report['chosen'], report['cross_validation'])
                self.assertTrue(output.with_suffix('.model-card.md').is_file())
                result = subprocess.run([sys.executable, '-B', str(folder / 'starter/main.py')], cwd=tmp, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('NotImplementedError: Complete ' + case['todo'], result.stderr)

    def test_precomputed_metric_exercise_is_real(self):
        folder = next(ROOT.glob('ML-15-*'))
        result = subprocess.run([sys.executable, '-B', str(folder / 'solution/main.py')], capture_output=True, text=True, check=True, timeout=10)
        report = json.loads(result.stdout)
        self.assertEqual(report['chosen_before_test'], 'complex')
        self.assertAlmostEqual(report['validation']['simple']['macro_f1'], 2 / 7)

    def test_polynomial_feature_order(self):
        self.assertEqual(r.polynomial_features([2, 3], 2), [1, 2, 3, 4, 6, 9])

    def test_regression_metrics_by_hand(self):
        metrics = r.metrics([1, 3], [2, 2])
        self.assertEqual((metrics['mae'], metrics['rmse'], metrics['r2']), (1, 1, 0))
        self.assertIsNone(r.metrics([1, 1], [2, 2])['r2'])

    def test_pivoted_solver_has_known_answer(self):
        self.assertEqual(r.solve([[0, 2], [1, 1]], [4, 3]), [1, 2])

    def test_quadratic_fit_recovers_known_surface(self):
        x = [[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)]
        y = [2 + 3 * a + 4 * b + 5 * a * a + 6 * a * b + 7 * b * b for a, b in x]
        fitted = r.fit(x, y, 2)
        pred = r.predict(fitted, x)
        self.assertLess(max((abs(a - b) for a, b in zip(y, pred))), 0.0001)

    def test_native_regression_test_targets_do_not_fit_or_select(self):
        source = next(ROOT.glob('ML-18-*')) / 'solution'
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            target = Path(tmp) / 'case'
            shutil.copytree(source, target)

            def execute():
                return json.loads(subprocess.run([sys.executable, '-B', str(target / 'main.py')], cwd=tmp, capture_output=True, text=True, check=True, timeout=15).stdout)
            before = execute()
            path = target / 'data/energy.csv'
            with path.open(newline='') as h:
                rows = list(csv.DictReader(h))
            for row in rows:
                if row['split'] == 'test':
                    row['energy_units'] = str(float(row['energy_units']) + 9999)
            with path.open('w', newline='') as h:
                writer = csv.DictWriter(h, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            after = execute()
            self.assertEqual(before['chosen'], after['chosen'])
            self.assertEqual(before['frozen_fits'], after['frozen_fits'])
if __name__ == '__main__':
    unittest.main()
