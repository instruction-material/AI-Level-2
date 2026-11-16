"""Train and evaluate the supplied real pixel models within fixed local budgets."""
import platform
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import tempfile
import time
import model
EXPERIMENT = 'evaluation'
ROOT = Path(__file__).resolve().parent

def load_rows(workspace):
    with (ROOT / 'metadata.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32 or len({r['image'] for r in rows}) != 32:
        raise ValueError('Use the supplied 32 distinct originals')
    groups = {}
    for row in rows:
        name = row['image']
        if Path(name).name != name or '\\' in name or (not name.endswith('.png')) or (row['label'] not in model.LABELS) or (row['split'] not in ('train', 'validation', 'test')):
            raise ValueError('Invalid fixture metadata')
        if row['source_group'] in groups and groups[row['source_group']] != row['split']:
            raise ValueError('Related originals cannot cross splits')
        groups[row['source_group']] = row['split']
        source = ROOT / 'images' / name
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Source image checksum changed')
        directory = Path(workspace) / row['split'] / row['label']
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, directory / name)
        row['pixels'] = model.decode_png((directory / name).read_bytes())
    for split in ('train', 'validation', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(model.LABELS):
            raise ValueError('Each split needs every class')
    return rows

def execute(rows, epochs=6, batch_size=4, seconds=8):
    if not 0 < seconds <= 12:
        raise ValueError('Bound the shared training CPU budget to 12 seconds')
    parts = {split: [r for r in rows if r['split'] == split] for split in ('train', 'validation', 'test')}
    if len(rows) > 32:
        raise ValueError('Maximum 32 original images')
    deadline = time.process_time() + seconds
    started = time.process_time()
    training = parts['train']
    validation = parts['validation']
    test = parts['test']
    histogram = model.fit_histogram(training)
    majority = min(model.LABELS, key=lambda label: (-sum((r['label'] == label for r in training)), label))
    val_truth = [r['label'] for r in validation]
    baseline_validation = {'majority': model.image_metrics(val_truth, [majority] * len(validation)), 'color_histogram': model.image_metrics(val_truth, model.histogram_predict(histogram, validation))}
    fits = {}
    kinds = ['dense', 'cnn'] + (['cnn_no_augmentation'] if EXPERIMENT == 'augmentation' else [])
    for kind in kinds:
        fits[kind] = {}
        for seed in (7, 19):
            fits[kind][seed] = model.train(training, validation, 'cnn' if kind.startswith('cnn') else 'dense', seed, epochs, batch_size, deadline, augment=kind != 'cnn_no_augmentation')
    validation_metrics = {kind: {seed: model.image_metrics(val_truth, model.predict(record['model'], validation)) for seed, record in records.items()} for kind, records in fits.items()}
    scores = {name: score['macro_f1'] for name, score in baseline_validation.items()}
    scores.update({name: sum((row['macro_f1'] for row in records.values())) / len(records) for name, records in validation_metrics.items()})
    order = ['majority', 'color_histogram', 'dense', 'cnn_no_augmentation', 'cnn']
    chosen = next((name for name in order if name in scores and scores[name] >= max(scores.values()) - 0.02))
    selected_seed = min(fits[chosen], key=lambda seed: (fits[chosen][seed]['best_validation_loss'], seed)) if chosen in fits else None
    # Architectures, checkpoints, candidate and seed choices are frozen.
    test_truth = [row['label'] for row in test]
    baseline_predictions = {'majority': [majority] * len(test), 'color_histogram': model.histogram_predict(histogram, test)}
    test_metrics = {name: model.image_metrics(test_truth, predictions) for name, predictions in baseline_predictions.items()}
    neural_test = {kind: {seed: model.image_metrics(test_truth, model.predict(record['model'], test)) for seed, record in records.items()} for kind, records in fits.items()}
    predictions = model.predict(fits[chosen][selected_seed]['model'], test) if chosen in fits else baseline_predictions[chosen]
    errors = {label: [r['image'] for r, p in zip(test, predictions) if r['label'] == label and p != label][:2] for label in model.LABELS}
    metadata = {kind: {seed: {k: v for k, v in record.items() if k != 'model'} for seed, record in records.items()} for kind, records in fits.items()}
    result = {'experiment': EXPERIMENT, 'chosen': chosen, 'chosen_seed': selected_seed, 'validation_macro_f1': scores, 'selection_rule': 'fixed preference within 0.02 mean validation macro F1; neural seed by validation cross-entropy', 'baseline_test': test_metrics, 'neural_test_by_seed': neural_test, 'training': metadata, 'split_images': {name: [r['image'] for r in part] for name, part in parts.items()}, 'split_groups': {name: sorted({r['source_group'] for r in part}) for name, part in parts.items()}, 'source_image_hashes': {r['image']: r['sha256'] for r in rows}, 'predictions': predictions, 'truth': test_truth, 'errors_by_class': errors, 'frozen_model': fits[chosen][selected_seed]['model'] if chosen in fits else {'kind': chosen, 'histograms': histogram, 'majority': majority}, 'limits': {'originals': 32, 'width': 12, 'height': 12, 'epochs': epochs, 'batch_size': batch_size, 'training_cpu_seconds': seconds}, 'training_and_evaluation_cpu_seconds': time.process_time() - started, 'architecture': 'RGB 3x3 valid convolution,2 tanh filters,2x2 average pool,50 flattened features,4-class softmax;260 parameters', 'dense_comparison': '432 flattened RGB inputs and a 4-class softmax;1732 parameters', 'limitation': 'Authored weather-like patterns,not photographs or evidence of real weather recognition'}
    return (result, test)

def contact_sheet(test, predictions, path):
    images = []
    for i, (row, prediction) in enumerate(zip(test, predictions)):
        x = 12 + i % 4 * 150
        y = 30 + i // 4 * 160
        raw = (ROOT / 'images' / row['image']).read_bytes()
        data = base64.b64encode(raw).decode()
        images.append(f'''<image x="{x}" y="{y}" width="120" height="120" image-rendering="pixelated" href="data:image/png;base64,{data}"/><text x="{x}" y="{y + 136}" font-size="11">{html.escape(row['label'] + ' → ' + prediction)}</text>''')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="620" height="360"><rect width="620" height="360" fill="white"/><text x="12" y="18">Authored pattern test images: actual → predicted</text>' + ''.join(images) + '</svg>')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='weather-case-') as workspace:
        rows = load_rows(workspace)
        result, test = execute(rows, args.epochs, args.batch_size, args.seconds)
        result['python_version'] = platform.python_version()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            with args.output.with_suffix('.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['image', 'actual', 'predicted'])
                writer.writerows(zip([r['image'] for r in test], result['truth'], result['predictions']))
            contact_sheet(test, result['predictions'], args.output.with_suffix('.svg'))
            args.output.with_suffix('.model-card.md').write_text('# Authored weather-pattern classifier\n\nIntended use: inspect a complete bounded image-learning workflow. Input: 32 authored 12x12 RGB patterns in four source-disjoint weather classes,not photographs. Training/validation/test=16/8/8;related original variants remain together. Augmentation uses training only;checkpoints and candidate/seed choice use validation only.\n\nChosen: ' + result['chosen'] + '. Two fixed seeds are reported with class recall, macro F1, confusion matrices, errors and training histories. Default budgets: 6 epochs, batch 4, 8 training CPU seconds. Each fit reports actual updates, completed epochs, checkpoint and stop reason. Colors and renderer shortcuts can dominate this tiny fixture;no real-world visual-recognition claim follows. Historical Colab/Mendeley material remains a separate optional source audit.\n')
        print(json.dumps(result, sort_keys=True))
if __name__ == '__main__':
    main()
