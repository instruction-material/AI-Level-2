"""Run this supplied, bounded clustering case from any working directory."""
import platform
import argparse
import csv
import hashlib
import json
from pathlib import Path
import model
CASE = {'title': 'Assign a new movie with a frozen fit', 'movie': True, 'experiment': 'frozen'}
ROOT = Path(__file__).resolve().parent

def run():
    path = ROOT / 'data' / ('movies.csv' if CASE['movie'] else 'visits.csv')
    if CASE['movie']:
        rows, source = model.load_movies(path, 120)
        raw = model.numeric_features(rows, CASE['experiment'] == 'log')
    else:
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        raw = [[float(row['visits']), float(row['basket_units'])] for row in rows]
        source = {'used_rows': len(rows)}
    scale = model.fit_scale(raw)
    x = model.transform(raw, scale)
    vocabulary = sorted({row['genre'] for row in rows}) if CASE['experiment'] == 'genre' else []
    if vocabulary:
        hot = model.one_hot([row['genre'] for row in rows], vocabulary)
        x = [a + b for a, b in zip(x, hot)]
    fitted, restarts = model.best_restart(x, 3)
    labels = fitted['labels']
    result = {'case': CASE['title'], 'source': source, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scaler': {'means': scale[0], 'scales': scale[1]}, 'fitted_rows': [row['id'] if CASE['movie'] else row['visit_id'] for row in rows], 'k': 3, 'inertia': fitted['inertia'], 'iterations': fitted['iterations'], 'stop_reason': fitted['stop_reason'], 'restarts': restarts, 'silhouette': model.silhouette(x, labels), 'raw_profiles': model.profile_clusters(raw, labels), 'labels': labels}
    mode = CASE['experiment']
    if mode == 'stability':
        result['seed_stability'] = [{'seed': seed, 'adjusted_rand': model.adjusted_rand(labels, model.kmeans(x, 3, seed)['labels'])} for seed in (2, 3, 5)]
    if mode == 'choose_k':
        scores = []
        for k in (2, 3, 4):
            candidate, _ = model.best_restart(x, k)
            scores.append({'k': k, 'inertia': candidate['inertia'], 'silhouette': model.silhouette(x, candidate['labels'])})
        best = max((row['silhouette'] for row in scores))
        result['k_comparison'] = scores
        result['chosen_k'] = min((row['k'] for row in scores if row['silhouette'] >= best - 0.03))
    if mode == 'units':
        changed = [[row[0], row[1] * 1000] for row in raw]
        other = model.transform(changed, model.fit_scale(changed))
        changed_fit, _ = model.best_restart(other, 3)
        raw_a = model.kmeans(raw, 3, 42)
        raw_b = model.kmeans(changed, 3, 42)
        result['scaled_unit_change_ari'] = model.adjusted_rand(labels, changed_fit['labels'])
        result['unscaled_unit_change_ari'] = model.adjusted_rand(raw_a['labels'], raw_b['labels'])
    if mode == 'log':
        linear = model.transform(model.numeric_features(rows, False), model.fit_scale(model.numeric_features(rows, False)))
        base, _ = model.best_restart(linear, 3)
        result['linear_vs_log_ari'] = model.adjusted_rand(labels, base['labels'])
        result['raw_profiles_in_original_units'] = model.profile_clusters(model.numeric_features(rows, False), labels)
    if mode == 'frozen':
        before = json.dumps({'scale': scale, 'centers': fitted['centers']}, sort_keys=True)
        query = {'year': 2005, 'revenue': 125000000.0}
        q = model.transform(model.numeric_features([query]), scale)
        result['new_query'] = dict(query, cluster=model.assign(q, fitted['centers'])[0])
        result['query_preserves_fit'] = before == json.dumps({'scale': scale, 'centers': fitted['centers']}, sort_keys=True)
    return (result, x, labels, fitted)

def main():
    parser = argparse.ArgumentParser(description=CASE['title'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result, x, labels, fitted = run()
    result['python_version'] = platform.python_version()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        with args.output.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['row_id', 'cluster'])
            writer.writerows(zip(result['fitted_rows'], labels))
        model.scatter_svg(x, labels, args.output.with_suffix('.svg'))
        args.output.with_suffix('.centers.json').write_text(json.dumps({'centers': fitted['centers'], 'scaler': result['scaler']}, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
if __name__ == '__main__':
    main()
