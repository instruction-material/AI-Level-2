import platform
import argparse, csv, hashlib, json, statistics
from pathlib import Path
import model
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    path = ROOT / 'data/energy.csv'
    with path.open(newline='') as h:
        rows = list(csv.DictReader(h))
    parts = {name: [r for r in rows if r['split'] == name] for name in ('train', 'validation', 'test')}
    features = lambda rs: [[float(r['temperature']), float(r['active_hours'])] for r in rs]
    targets = lambda rs: [float(r['energy_units']) for r in rs]
    x = features(parts['train'])
    y = targets(parts['train'])
    baseline = statistics.median(y)
    fits = {name: model.fit(x, y, degree) for name, degree in [('linear', 1), ('quadratic', 2)]}
    prediction = lambda name, rs: [baseline] * len(rs) if name == 'median' else model.predict(fits[name], features(rs))
    order = ['median', 'linear', 'quadratic']
    validation = {name: model.metrics(targets(parts['validation']), prediction(name, parts['validation'])) for name in order}
    best = min((row['mae'] for row in validation.values()))
    chosen = next((name for name in order if validation[name]['mae'] <= best + 0.25))
    predictions = {name: prediction(name, parts['test']) for name in order}
    test = {name: model.metrics(targets(parts['test']), predictions[name]) for name in order}
    slices = {}
    for name, condition in [('low_activity', lambda r: float(r['active_hours']) < 6), ('high_activity', lambda r: float(r['active_hours']) >= 6)]:
        indices = [i for i, r in enumerate(parts['test']) if condition(r)]
        slices[name] = {'rows': len(indices), 'mae': sum((abs(test[chosen]['residuals'][i]) for i in indices)) / len(indices) if indices else None}
    result = {'chosen': chosen, 'selection': 'validation MAE;prefer simplest within 0.25 fictional units', 'baseline_training_median': baseline, 'frozen_fits': fits, 'validation': validation, 'test': test, 'test_error_slices': slices, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'split_ids': {name: [r['id'] for r in part] for name, part in parts.items()}}
    result['python_version'] = platform.python_version()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        with args.output.with_suffix('.csv').open('w', newline='') as h:
            writer = csv.writer(h)
            writer.writerow(['id', 'actual', 'predicted', 'residual'])
            writer.writerows(zip(result['split_ids']['test'], targets(parts['test']), predictions[chosen], test[chosen]['residuals']))
        args.output.with_suffix('.model-card.md').write_text('# Synthetic energy comparison\n\nIntended use: compare a median baseline, linear fit and quadratic fit on authored fictional energy units. Training-only scaling; fixed 72/24/24 split; validation-only selection, simplest within 0.25 units. Chosen: ' + chosen + '. Test MAE: ' + str(test[chosen]['mae']) + '.\n\nLimitations: generated data; no real building forecast or causal claim. Inspect residuals and both activity slices. No fitted preprocessing or model choice uses final test targets. Reproduce from the included input hash and source.\n')
    print(json.dumps(result, sort_keys=True))
if __name__ == '__main__':
    main()
