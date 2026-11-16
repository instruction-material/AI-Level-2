import platform
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import model
CASE = {'title': 'Complete reproducible classification capstone', 'cars': False, 'weighted': False, 'stress': True}
ROOT = Path(__file__).resolve().parent

def load_rows():
    path = ROOT / 'data' / ('cars.csv' if CASE['cars'] else 'tokens.csv')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        raw = list(csv.DictReader(handle))
    if not CASE['cars']:
        rows = [{'id': r['row_id'], 'group': r['batch'], 'split': r['split'], 'x': [float(r['sensor_a']), float(r['sensor_b'])], 'label': r['label']} for r in raw]
    else:
        by_class = {}
        for i, row in enumerate(raw):
            r = {'id': f'car-{i + 2:04d}', 'group': '|'.join((row[k] for k in ('buying', 'maintenance', 'doors', 'persons', 'lug boot', 'safety'))), 'categories': [row[k] for k in ('buying', 'maintenance', 'doors', 'persons', 'lug boot', 'safety')], 'label': row['acceptability']}
            by_class.setdefault(r['label'], []).append(r)
        rng = random.Random(42)
        rows = []
        for label, members in sorted(by_class.items()):
            rng.shuffle(members)
            for i, row in enumerate(members[:40]):
                rows.append(dict(row, split='train' if i < 24 else 'validation' if i < 32 else 'test'))
    return (rows, path, len(raw))

def main():
    parser = argparse.ArgumentParser(description=CASE['title'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows, path, original_rows = load_rows()
    candidates = ('dummy', 'knn-3', 'knn-1') if CASE['cars'] else ('dummy', 'stump', 'knn-3', 'knn-1')
    if CASE['weighted']:
        candidates = ('dummy', 'knn-3', 'knn-distance-3')
    result = model.compare(rows, candidates, 'onehot' if CASE['cars'] else 'numeric', stress=CASE['stress'])
    result.update(case=CASE['title'], input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_rows=original_rows, used_rows=len(rows), random_seed=42, intended_use='Reproduce and explain a bounded low-stakes classification workflow', limitations='A small authored token fixture; no real-world recognition claim' if not CASE['cars'] else 'Balanced 160-row demonstration sample; not the original class prevalence or a driving decision')
    result['python_version'] = platform.python_version()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        with args.output.with_suffix('.csv').open('w', newline='') as h:
            writer = csv.writer(h)
            writer.writerow(['row_id', 'actual', 'predicted'])
            writer.writerows(zip(result['split_ids']['test'], result['test_labels'], result['test_predictions']))
        score = result['test'][result['chosen']]
        card = '# Model card: ' + CASE['title'] + '\n\nIntended use: ' + result['intended_use'] + '\n\nSource SHA256: ' + result['input_sha256'] + '\n\nSelection: training-only grouped CV, macro F1, declared preference within 0.02. ' + 'The separate validation set is reported as a check and does not revise selection. All fitted preprocessing uses training rows.\n\n' + 'Frozen candidate: ' + result['chosen'] + '. Test macro F1: ' + str(score['macro_f1']) + '; balanced accuracy: ' + str(score['balanced_accuracy']) + '.\n\nLimits: ' + result['limitations'] + '. Results include class supports/confusion matrices and supplied error slices or unknown-category evidence. ' + 'A feature-corruption probe is a stress demonstration, not a causal counterfactual. Reproduce from the included source and same fixed seeds.\n'
        args.output.with_suffix('.model-card.md').write_text(card)
    print(json.dumps(result, sort_keys=True))
if __name__ == '__main__':
    main()
