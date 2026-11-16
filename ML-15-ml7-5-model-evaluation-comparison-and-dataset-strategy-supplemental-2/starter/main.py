import platform
import argparse, csv, hashlib, json
from pathlib import Path
import model
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    path = ROOT / 'data/predictions.csv'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    validation = [r for r in rows if r['split'] == 'validation']
    metrics = {name: model.classification_metrics([r['actual'] for r in validation], [r[name] for r in validation], ['A', 'B', 'C']) for name in ('simple', 'complex')}
    chosen = model.choose_candidate({name: row['macro_f1'] for name, row in metrics.items()}, ['simple', 'complex'])
    test = [r for r in rows if r['split'] == 'test']
    results = {name: model.classification_metrics([r['actual'] for r in test], [r[name] for r in test], ['A', 'B', 'C']) for name in ('simple', 'complex')}
    result = {'validation': metrics, 'chosen_before_test': chosen, 'test': results, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scope': 'Precomputed prediction evaluation; no model fitting'}
    result['python_version'] = platform.python_version()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
if __name__ == '__main__':
    main()
