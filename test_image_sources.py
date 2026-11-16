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
im = load('ML-Source-Cases/image_model.py', 'image_reference')
gen = load('ML-Source-Cases/generate_weather.py', 'weather_generator')

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

class ImageContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = 'ML7-Offline-Weather-Classifier'
        cls.driver = driver_for(cls.folder)
        cls.workspace = tempfile.TemporaryDirectory()
        cls.rows = cls.driver.load_rows(cls.workspace.name)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_png_pixels_by_hand(self):
        raw = gen.png([(17, 34, 51)] * 144)
        pixels = im.decode_png(raw)
        self.assertEqual(len(pixels), 432)
        self.assertEqual(pixels[:3], [17 / 255, 34 / 255, 51 / 255])

    def test_png_crc_and_truncation_are_rejected(self):
        raw = bytearray(gen.png([(17, 34, 51)] * 144))
        raw[20] ^= 1
        for value in (bytes(raw), bytes(raw[:20])):
            with self.assertRaises(ValueError):
                im.decode_png(value)

    def test_png_expansion_is_bounded(self):

        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 4294967295)
        raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 12, 12, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'\x00' * 100000)) + chunk(b'IEND', b'')
        with self.assertRaises(ValueError):
            im.decode_png(raw)

    def test_actual_images_and_grouped_directories(self):
        self.assertEqual(len(self.rows), 32)
        self.assertEqual(len({r['sha256'] for r in self.rows}), 32)
        for split, count in (('train', 16), ('validation', 8), ('test', 8)):
            self.assertEqual(sum((r['split'] == split for r in self.rows)), count)
            self.assertEqual(len(list((Path(self.workspace.name) / split).rglob('*.png'))), count)
        groups = {split: {r['source_group'] for r in self.rows if r['split'] == split} for split in ('train', 'validation', 'test')}
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])

    def test_generator_reproduces_all_input_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = gen.generate(tmp)
            for row in records:
                self.assertEqual((Path(tmp) / 'images' / row['image']).read_bytes(), (ROOT / self.folder / 'solution/images' / row['image']).read_bytes())

    def test_declared_parameter_counts(self):

        def count(model):
            return sum((len(row) for key in ('kernel', 'output') for row in model[key])) + len(model['conv_bias']) + len(model['bias'])
        self.assertEqual(count(im.initialize('cnn', 7)), 260)
        self.assertEqual(count(im.initialize('dense', 7)), 1732)

    def test_all_cnn_gradients_by_finite_differences(self):
        model = im.initialize('cnn', 13)
        pixels = self.rows[0]['pixels']
        loss, g = im.loss_gradient(model, pixels, 1)
        checked = 0
        for key in ('kernel', 'output', 'conv_bias', 'bias'):
            arrays = model[key] if key in ('kernel', 'output') else [model[key]]
            gradients = g[key] if key in ('kernel', 'output') else [g[key]]
            for values, derivatives in zip(arrays, gradients):
                for i, value in enumerate(values):
                    values[i] = value + 1e-06
                    plus = im.loss_gradient(model, pixels, 1)[0]
                    values[i] = value - 1e-06
                    minus = im.loss_gradient(model, pixels, 1)[0]
                    values[i] = value
                    self.assertAlmostEqual(derivatives[i], (plus - minus) / 2e-06, delta=2e-06)
                    checked += 1
        self.assertEqual(checked, 260)

    def test_dense_gradient_by_finite_differences(self):
        model = im.initialize('dense', 13)
        pixels = self.rows[0]['pixels']
        _, g = im.loss_gradient(model, pixels, 1)
        for cindex in range(4):
            for j in (0, 17, 431):
                old = model['output'][cindex][j]
                model['output'][cindex][j] = old + 1e-06
                plus = im.loss_gradient(model, pixels, 1)[0]
                model['output'][cindex][j] = old - 1e-06
                minus = im.loss_gradient(model, pixels, 1)[0]
                model['output'][cindex][j] = old
                self.assertAlmostEqual(g['output'][cindex][j], (plus - minus) / 2e-06, delta=2e-06)

    def test_small_gradient_step_reduces_same_example_loss(self):
        model = im.initialize('cnn', 7)
        pixels = self.rows[0]['pixels']
        before, g = im.loss_gradient(model, pixels, 0)
        im.update(model, g, 0.001)
        self.assertLess(im.loss_gradient(model, pixels, 0)[0], before)

    def test_training_augmentation_preserves_originals(self):
        import random
        rows = [r for r in self.rows if r['split'] == 'train']
        before = copy.deepcopy(rows)
        a = im.augment_training(rows, random.Random(7))
        b = im.augment_training(rows, random.Random(7))
        self.assertEqual(rows, before)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0]['pixels'], rows[0]['pixels'])
        with self.assertRaises(ValueError):
            im.augment_training([next((r for r in self.rows if r['split'] == 'test'))], random.Random(7))

    def test_controlled_horizontal_flip(self):

        class Fixed:

            def random(self):
                return 0

            def uniform(self, a, b):
                return 1
        row = {'split': 'train', 'pixels': [x / 11 for y in range(12) for x in range(12) for c in range(3)]}
        image = im.augment_training([row], Fixed())[0]['pixels']
        self.assertEqual(image[:3], [1, 1, 1])
        self.assertEqual(image[33:36], [0, 0, 0])

    def test_image_metrics_are_four_class_metrics(self):
        metrics = im.image_metrics(['cloudy', 'rainy'], ['cloudy', 'cloudy'])
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertAlmostEqual(metrics['macro_f1'], 1 / 6)
        self.assertEqual(metrics['per_class_recall']['rainy'], 0)

    def test_cpu_stop_reports_untrained_initial_checkpoint(self):
        training = [r for r in self.rows if r['split'] == 'train']
        val = [r for r in self.rows if r['split'] == 'validation']
        result = im.train(training, val, 'cnn', 7, 6, 4, time.process_time() - 1)
        self.assertEqual((result['completed_epochs'], result['update_steps'], result['best_epoch']), (0, 0, 0))
        self.assertEqual(result['stop_reason'], 'cpu_budget')

    def test_image_caps_and_split_roles(self):
        training = [r for r in self.rows if r['split'] == 'train']
        val = [r for r in self.rows if r['split'] == 'validation']
        for epochs, batch in ((0, 4), (13, 4), (1, 0), (1, 9)):
            with self.assertRaises(ValueError):
                im.train(training, val, 'cnn', 7, epochs, batch, time.process_time() + 1)
        with self.assertRaises(ValueError):
            im.train(val, training, 'cnn', 7, 1, 4, time.process_time() + 1)

    def test_test_labels_never_choose_or_train_image_models(self):
        before, _ = self.driver.execute(self.rows, epochs=2)
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row['split'] == 'test':
                row['label'] = 'cloudy'
        after, _ = self.driver.execute(changed, epochs=2)
        self.assertEqual(before['chosen'], after['chosen'])
        self.assertEqual(before['chosen_seed'], after['chosen_seed'])
        self.assertEqual(before['frozen_model'], after['frozen_model'])
        self.assertEqual(before['training'], after['training'])

    def test_all_actual_image_references_starters_and_file_limit(self):
        for case in json.loads((ROOT / 'ML-Source-Cases/image-cases.json').read_text()):
            folder = ROOT / case['folder']
            with self.subTest(case=case['number']), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'result.json'
                completed = subprocess.run([sys.executable, '-B', str(folder / 'solution/main.py'), '--output', str(output)], cwd=tmp, capture_output=True, text=True, timeout=30, check=True)
                result = json.loads(completed.stdout)
                self.assertEqual(len(result['truth']), 8)
                self.assertIn(result['chosen'], result['validation_macro_f1'])
                self.assertEqual(output.with_suffix('.svg').read_text().count('<image '), 8)
                for role in ('starter', 'solution'):
                    self.assertEqual(len([p for p in (folder / role).rglob('*') if p.is_file()]), 36)
                attempted = subprocess.run([sys.executable, '-B', str(folder / 'starter/main.py')], cwd=tmp, capture_output=True, text=True, timeout=10)
                self.assertIn('NotImplementedError: Complete ' + case['todo'], attempted.stderr)
if __name__ == '__main__':
    unittest.main()
