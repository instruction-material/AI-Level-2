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
ROOT = Path(__file__).resolve().parent

def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
m = load('ML-Source-Cases/cluster_model.py', 'cluster_reference')

class ClusteringContracts(unittest.TestCase):

    def test_centroid_by_hand(self):
        self.assertEqual(m.centroid([[1, 2], [3, 4]]), [2, 3])

    def test_scaler_and_constant(self):
        self.assertEqual(m.fit_scale([[1, 7], [3, 7]]), ([2, 7], [1, 1]))

    def test_transform_reuses_fit(self):
        fit = m.fit_scale([[1], [3]])
        self.assertEqual(m.transform([[101]], fit), [[99]])
        self.assertEqual(fit, ([2], [1]))

    def test_tie_uses_center_index(self):
        self.assertEqual(m.assign([[2, 3]], [[0, 0], [4, 6]]), [0])

    def test_one_iteration_is_exact(self):
        r = m.kmeans([[0], [2], [10], [12]], 2, max_iter=1, initial=[[0], [10]])
        self.assertEqual(r['centers'], [[1], [11]])
        self.assertEqual(r['labels'], [0, 0, 1, 1])
        self.assertEqual(r['inertia'], 4)
        self.assertEqual(r['stop_reason'], 'max_iter')

    def test_empty_group_is_repaired(self):
        r = m.kmeans([[0], [1], [10], [11]], 2, initial=[[0], [0]])
        self.assertGreater(r['empty_cluster_repairs'], 0)
        self.assertEqual(len(set(r['labels'])), 2)
        self.assertTrue(all((math.isfinite(v) for c in r['centers'] for v in c)))

    def test_final_labels_match_final_centers(self):
        x = [[0, 0], [2, 1], [4, 7], [8, 9], [10, 10]]
        r = m.kmeans(x, 3, max_iter=1)
        self.assertEqual(r['labels'], m.assign(x, r['centers']))

    def test_objective_descends_without_empty_repair(self):
        r = m.kmeans([[i, i % 3] for i in range(20)], 3, 7)
        values = [h['inertia'] for h in r['history']]
        self.assertTrue(all((a >= b - 1e-10 for a, b in zip(values, values[1:]))))

    def test_restarts_choose_observed_minimum(self):
        x = [[0], [2], [10], [12], [30], [32]]
        r, all_runs = m.best_restart(x, 3, (7, 19, 31))
        self.assertEqual((r['inertia'], r['seed']), min(((v['inertia'], v['seed']) for v in all_runs)))

    def test_initialization_is_seeded(self):
        x = [[i] for i in range(10)]
        self.assertEqual(m.initialize(x, 3, 7), m.initialize(x, 3, 7))
        self.assertEqual(len(set(map(tuple, m.initialize(x, 3, 7)))), 3)

    def test_bad_k_including_explicit_centers(self):
        for k in (0, 3, 17):
            with self.subTest(k=k), self.assertRaises(ValueError):
                m.kmeans([[1], [2]], k, initial=[[1]] * max(1, k))

    def test_invalid_matrix(self):
        for x in ([], [[]], [[1], [1, 2]], [[float('nan')]], [[float('inf')]]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                m.fit_scale(x)

    def test_ari_is_label_permutation_invariant(self):
        self.assertEqual(m.adjusted_rand([0, 0, 1, 1], [9, 9, 4, 4]), 1)

    def test_ari_cross_partition_by_hand(self):
        self.assertAlmostEqual(m.adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)

    def test_ari_degenerate_partitions(self):
        self.assertEqual(m.adjusted_rand([], []), 1)
        self.assertEqual(m.adjusted_rand([0, 0], [2, 2]), 1)
        self.assertEqual(m.adjusted_rand([0, 0], [0, 1]), 0)

    def test_ari_length(self):
        with self.assertRaises(ValueError):
            m.adjusted_rand([1], [1, 2])

    def test_silhouette_hand_distances(self):
        score = m.silhouette([[0], [2], [10], [12]], [0, 0, 1, 1])
        self.assertAlmostEqual(score, (9 / 11 + 7 / 9) / 2)

    def test_silhouette_singleton_contributes_zero(self):
        self.assertAlmostEqual(m.silhouette([[0], [1], [10]], [0, 0, 1]), (0.9 + 8 / 9) / 3)

    def test_silhouette_undefined_partitions(self):
        self.assertIsNone(m.silhouette([[0], [1]], [0, 0]))
        self.assertIsNone(m.silhouette([[0], [1]], [0, 1]))

    def test_nominal_encoding_and_unknown(self):
        self.assertEqual(m.one_hot(['Action', 'Drama', 'new'], ['Action', 'Drama']), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_log_zero_and_original_units(self):
        rows = [{'year': 2000, 'revenue': 0}, {'year': 2005, 'revenue': 99}]
        self.assertEqual(m.numeric_features(rows), [[2000, 0], [2005, 99]])
        self.assertAlmostEqual(m.numeric_features(rows, True)[1][1], math.log(100))

    def test_profiles_are_raw_means(self):
        self.assertEqual(m.profile_clusters([[1, 10], [3, 14], [9, 20]], [5, 5, 8]), [{'cluster': 5, 'count': 2, 'raw_centroid': [2, 12]}, {'cluster': 8, 'count': 1, 'raw_centroid': [9, 20]}])

    def test_movie_parser_real_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'movies.csv'
            path.write_text('movie_title,release_date,genre,inflation_adjusted_gross\nA,2000-01-01,,0\nB,bad,Action,2\nC,2001-01-01,Drama,-2\nD,2001-01-01,Drama,nan\nE,2005-01-01,Drama,99\n')
            rows, info = m.load_movies(path, 12)
            self.assertEqual(info, {'valid_source_rows': 2, 'rejected_source_lines': [3, 4, 5], 'used_rows': 2, 'row_limit': 12})
            self.assertEqual({r['title'] for r in rows}, {'A', 'E'})
            self.assertEqual(next((r for r in rows if r['title'] == 'A'))['genre'], 'unknown')

    def test_all_ten_actual_reference_programs(self):
        cases = json.loads((ROOT / 'ML-Source-Cases/clustering-cases.json').read_text())
        for case in cases:
            folder = next(ROOT.glob('ML-' + str(case['number']) + '-*'))
            with self.subTest(case=case['number']), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run([sys.executable, '-B', str(folder / 'solution/main.py'), '--output', str(Path(tmp) / 'report.json')], cwd=tmp, capture_output=True, text=True, timeout=20, check=True)
                report = json.loads(result.stdout)
                self.assertEqual(len(report['labels']), report['source']['used_rows'])
                self.assertTrue(-1 <= report['silhouette'] <= 1)
                self.assertTrue((Path(tmp) / 'report.svg').read_text().startswith('<svg'))
                self.assertEqual(sum((p['count'] for p in report['raw_profiles'])), len(report['labels']))
                if case['number'] == 24:
                    self.assertAlmostEqual(report['scaled_unit_change_ari'], 1)
                if case['number'] == 28:
                    self.assertTrue(report['query_preserves_fit'])

    def test_all_ten_starters_reach_the_named_exercise(self):
        for case in json.loads((ROOT / 'ML-Source-Cases/clustering-cases.json').read_text()):
            folder = next(ROOT.glob('ML-' + str(case['number']) + '-*'))
            with self.subTest(case=case['number']):
                result = subprocess.run([sys.executable, '-B', str(folder / 'starter/main.py')], cwd=ROOT, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('NotImplementedError: Complete ' + case['todo'], result.stderr)

    def test_actual_solution_sources_are_standalone(self):
        for number in range(19, 29):
            folder = next(ROOT.glob('ML-' + str(number) + '-*'))
            for role in ('starter', 'solution'):
                self.assertTrue((folder / role / 'model.py').is_file())
                self.assertEqual(len(list((folder / role / 'data').glob('*.csv'))), 1)
                self.assertNotIn('summarize(values)', (folder / role / 'main.py').read_text())
if __name__ == '__main__':
    unittest.main()
