"""Small explicit clustering algorithms for bounded, supplied teaching cases."""
import csv
import datetime
import hashlib
import html
import math
import random

def matrix(values):
    rows = [list(map(float, row)) for row in values]
    if not rows or not rows[0] or len(rows) > 600 or (len(rows[0]) > 32):
        raise ValueError('Use 1..600 rows and 1..32 numeric features')
    if any((len(row) != len(rows[0]) or not all(map(math.isfinite, row)) for row in rows)):
        raise ValueError('Expected a finite rectangular matrix')
    return rows

def fit_scale(values):
    """Population standard deviation; a constant column has scale one."""
    raise NotImplementedError('Complete fit_scale; see README.md')

def transform(values, scale):
    means, scales = scale
    rows = matrix(values)
    if len(means) != len(rows[0]) or len(scales) != len(means) or any((s <= 0 for s in scales)):
        raise ValueError('Scaler dimensions and positive scales must match')
    return [[(x - m) / s for x, m, s in zip(row, means, scales)] for row in rows]

def distance2(a, b):
    return sum(((x - y) ** 2 for x, y in zip(a, b)))

def assign(values, centers):
    rows = matrix(values)
    centers = matrix(centers)
    if len(rows[0]) != len(centers[0]):
        raise ValueError('Feature widths must match')
    return [min(range(len(centers)), key=lambda i: (distance2(row, centers[i]), i)) for row in rows]

def centroid(rows):
    if not rows:
        raise ValueError('An empty group has no arithmetic mean')
    return [sum(col) / len(rows) for col in zip(*rows)]

def initialize(values, k, seed):
    """k-means++ samples subsequent centers with squared-distance weights."""
    rows = matrix(values)
    if not 1 <= k <= min(16, len(set(map(tuple, rows)))):
        raise ValueError('k must fit the distinct rows and be at most 16')
    rng = random.Random(seed)
    centers = [rows[rng.randrange(len(rows))][:]]
    while len(centers) < k:
        weights = [min((distance2(row, c) for c in centers)) for row in rows]
        target = rng.random() * sum(weights)
        cumulative = 0.0
        index = len(rows) - 1
        for i, weight in enumerate(weights):
            cumulative += weight
            if weight > 0 and cumulative > target:
                index = i
                break
        centers.append(rows[index][:])
    return centers

def kmeans(values, k, seed=42, max_iter=40, tolerance=1e-08, initial=None):
    rows = matrix(values)
    if not 1 <= k <= min(16, len(set(map(tuple, rows)))):
        raise ValueError('k must fit distinct rows')
    if not 1 <= max_iter <= 100 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Bound iterations to 1..100 and use a finite nonnegative tolerance')
    centers = initialize(rows, k, seed) if initial is None else matrix(initial)
    if len(centers) != k or len(centers[0]) != len(rows[0]):
        raise ValueError('Initial centers must match k and feature width')
    history = []
    previous = None
    reason = 'max_iter'
    repairs = 0
    for iteration in range(max_iter):
        labels = assign(rows, centers)
        groups = [[row for row, label in zip(rows, labels) if label == j] for j in range(k)]
        updated = []
        reserved = set()
        empty = False
        for group in groups:
            if group:
                updated.append(centroid(group))
                continue
            empty = True
            repairs += 1
            index = max((i for i in range(len(rows)) if i not in reserved), key=lambda i: (distance2(rows[i], centers[labels[i]]), -i))
            reserved.add(index)
            updated.append(rows[index][:])
        shift = max((distance2(a, b) for a, b in zip(centers, updated)))
        centers = updated
        final_labels = assign(rows, centers)
        inertia = sum((distance2(row, centers[label]) for row, label in zip(rows, final_labels)))
        history.append({'iteration': iteration + 1, 'inertia': inertia, 'shift_squared': shift})
        if not empty and (labels == previous or shift <= tolerance * tolerance):
            reason = 'converged'
            break
        previous = labels
    labels = assign(rows, centers)
    return {'centers': centers, 'labels': labels, 'inertia': sum((distance2(row, centers[label]) for row, label in zip(rows, labels))), 'iterations': len(history), 'stop_reason': reason, 'empty_cluster_repairs': repairs, 'history': history}

def best_restart(values, k, seeds=(7, 19, 31)):
    if not 1 <= len(seeds) <= 10:
        raise ValueError('Use 1..10 fixed restarts')
    runs = [dict(kmeans(values, k, seed), seed=seed) for seed in seeds]
    chosen = min(runs, key=lambda run: (run['inertia'], run['seed']))
    return (chosen, [{'seed': r['seed'], 'inertia': r['inertia'], 'iterations': r['iterations']} for r in runs])

def profile_clusters(values, labels):
    rows = matrix(values)
    if len(rows) != len(labels):
        raise ValueError('One label is required for every row')
    return [{'cluster': label, 'count': labels.count(label), 'raw_centroid': centroid([row for row, l in zip(rows, labels) if l == label])} for label in sorted(set(labels))]

def adjusted_rand(a, b):
    if len(a) != len(b):
        raise ValueError('Partitions must describe the same ordered rows')
    if len(a) < 2:
        return 1.0
    choose = lambda n: n * (n - 1) / 2
    cells = {}
    left = {}
    right = {}
    for x, y in zip(a, b):
        cells[x, y] = cells.get((x, y), 0) + 1
        left[x] = left.get(x, 0) + 1
        right[y] = right.get(y, 0) + 1
    both = sum(map(choose, cells.values()))
    sa = sum(map(choose, left.values()))
    sb = sum(map(choose, right.values()))
    expected = sa * sb / choose(len(a))
    denominator = (sa + sb) / 2 - expected
    return (both - expected) / denominator if denominator else 1.0

def silhouette(values, labels, sample_limit=120):
    rows = matrix(values)
    if len(rows) != len(labels):
        raise ValueError('Labels and rows must match')
    groups = sorted(set(labels))
    if len(groups) < 2 or len(groups) >= len(rows):
        return None
    if not 1 <= sample_limit <= 600:
        raise ValueError('Use a bounded silhouette sample')
    indices = sorted({min(len(rows) - 1, int(i * len(rows) / min(sample_limit, len(rows)))) for i in range(min(sample_limit, len(rows)))})
    scores = []
    for i in indices:
        own = [j for j in range(len(rows)) if j != i and labels[j] == labels[i]]
        if not own:
            scores.append(0.0)
            continue
        a = sum((math.sqrt(distance2(rows[i], rows[j])) for j in own)) / len(own)
        other = []
        for group in groups:
            if group == labels[i]:
                continue
            members = [j for j, l in enumerate(labels) if l == group]
            other.append(sum((math.sqrt(distance2(rows[i], rows[j])) for j in members)) / len(members))
        b = min(other)
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return sum(scores) / len(scores)

def one_hot(values, vocabulary):
    index = {name: i for i, name in enumerate(vocabulary)}
    result = []
    for value in values:
        row = [0.0] * (len(vocabulary) + 1)
        row[index.get(value, len(vocabulary))] = 1.0
        result.append(row)
    return result

def numeric_features(rows, log_revenue=False):
    return [[float(row['year']), math.log1p(float(row['revenue'])) if log_revenue else float(row['revenue'])] for row in rows]

def load_movies(path, limit=120):
    if not 12 <= limit <= 600:
        raise ValueError('Bound movie rows to 12..600')
    valid = []
    rejected = []
    with path.open(newline='', encoding='utf-8-sig') as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                year = datetime.date.fromisoformat(row['release_date']).year
                revenue = float(row['inflation_adjusted_gross'])
                if not math.isfinite(revenue) or revenue < 0:
                    raise ValueError('Invalid gross')
            except (ValueError, KeyError, TypeError):
                rejected.append(number)
                continue
            valid.append({'id': 'movie-' + str(number), 'title': row['movie_title'], 'year': year, 'revenue': revenue, 'genre': row.get('genre') or 'unknown'})
    chosen = sorted(valid, key=lambda r: hashlib.sha256((r['id'] + r['title']).encode()).hexdigest())[:limit]
    return (sorted(chosen, key=lambda r: r['id']), {'valid_source_rows': len(valid), 'rejected_source_lines': rejected, 'used_rows': len(chosen), 'row_limit': limit})

def scatter_svg(values, labels, path):
    xs = [row[0] for row in values]
    ys = [row[1] for row in values]
    lo = [min(xs), min(ys)]
    span = [max(xs) - lo[0] or 1, max(ys) - lo[1] or 1]
    colors = ['#2563eb', '#b45309', '#166534', '#be123c', '#7c3aed']
    dots = []
    for row, label in zip(values, labels):
        x = 45 + 310 * (row[0] - lo[0]) / span[0]
        y = 250 - 210 * (row[1] - lo[1]) / span[1]
        dots.append('<circle cx="' + str(x) + '" cy="' + str(y) + '" r="3" fill="' + colors[label % len(colors)] + '"/>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="440" height="310" viewBox="0 0 440 310"><rect width="440" height="310" fill="white"/><text x="20" y="24">First two fitted features; cluster labels</text><path d="M40 40V260H370" fill="none" stroke="black"/>' + ''.join(dots) + '<text x="150" y="294">Feature 1</text><text x="4" y="150">F2</text></svg>')
