"""Explicit train-only classification, grouped CV, metrics and categorical KNN."""
import csv
import math
import random
from collections import Counter

def classification_metrics(truth, predictions, labels):
    if not truth or len(truth) != len(predictions) or (not labels) or (len(set(labels)) != len(labels)):
        raise ValueError('Use nonempty aligned predictions and unique declared labels')
    index = {label: i for i, label in enumerate(labels)}
    if any((v not in index for v in list(truth) + list(predictions))):
        raise ValueError('All observed labels must belong to the declared label set')
    confusion = [[0] * len(labels) for _ in labels]
    for actual, predicted in zip(truth, predictions):
        confusion[index[actual]][index[predicted]] += 1
    per_class = {}
    for i, label in enumerate(labels):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted = sum((row[i] for row in confusion))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[label] = {'support': support, 'precision': precision, 'recall': recall, 'f1': f1}
    observed = [row['recall'] for row in per_class.values() if row['support']]
    return {'accuracy': sum((a == b for a, b in zip(truth, predictions))) / len(truth), 'macro_f1': sum((v['f1'] for v in per_class.values())) / len(labels), 'balanced_accuracy': sum(observed) / len(observed), 'labels': labels, 'confusion_matrix': confusion, 'per_class': per_class}

def majority(labels):
    counts = Counter(labels)
    if not counts:
        raise ValueError('Training labels cannot be empty')
    return min(counts, key=lambda label: (-counts[label], label))

def fit_encoding(rows):
    if not rows:
        raise ValueError('Encoding needs training rows')
    width = len(rows[0]['categories'])
    if not width or any((len(row['categories']) != width for row in rows)):
        raise ValueError('Categorical rows must be rectangular')
    return [sorted({row['categories'][column] for row in rows}) for column in range(width)]

def encode(rows, vocabularies):
    result = []
    for row in rows:
        if len(row['categories']) != len(vocabularies):
            raise ValueError('Category widths differ')
        encoded = []
        for value, vocabulary in zip(row['categories'], vocabularies):
            one = [0.0] * (len(vocabulary) + 1)
            one[vocabulary.index(value) if value in vocabulary else len(vocabulary)] = 1.0
            encoded.extend(one)
        result.append(encoded)
    return result

def fit_scale(x):
    means = [sum(col) / len(x) for col in zip(*x)]
    scales = [math.sqrt(sum(((v - m) ** 2 for v in col)) / len(x)) or 1.0 for col, m in zip(zip(*x), means)]
    return {'means': means, 'scales': scales}

def scaled(x, state):
    if any((len(row) != len(state['means']) for row in x)):
        raise ValueError('Query widths must match the training fit')
    return [[(v - m) / s for v, m, s in zip(row, state['means'], state['scales'])] for row in x]

def gini(labels):
    n = len(labels)
    return 1 - sum(((count / n) ** 2 for count in Counter(labels).values())) if n else 0.0

def fit_stump(x, y):
    best = None
    for column in range(len(x[0])):
        values = sorted({row[column] for row in x})
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left = [label for row, label in zip(x, y) if row[column] <= threshold]
            right = [label for row, label in zip(x, y) if row[column] > threshold]
            score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            key = (score, column, threshold)
            if best is None or key < best[0]:
                best = (key, {'column': column, 'threshold': threshold, 'left': majority(left), 'right': majority(right)})
    return best[1] if best else {'leaf': majority(y)}

def fit_model(rows, algorithm, representation='numeric'):
    if not 1 <= len(rows) <= 400:
        raise ValueError('Bound training to 1..400 rows')
    if representation == 'onehot':
        state = {'kind': 'onehot', 'vocabularies': fit_encoding(rows)}
        x = encode(rows, state['vocabularies'])
    elif representation == 'numeric':
        x = [row['x'] for row in rows]
        if not x[0] or any((len(row) != len(x[0]) for row in x)):
            raise ValueError('Invalid numeric matrix')
        state = {'kind': 'numeric', **fit_scale(x)}
        x = scaled(x, state)
    else:
        raise ValueError('Unknown representation')
    if len(x[0]) > 64 or any((not math.isfinite(v) for row in x for v in row)):
        raise ValueError('Use at most 64 finite encoded features')
    y = [row['label'] for row in rows]
    model = {'algorithm': algorithm, 'preprocessing': state, 'training_ids': [r['id'] for r in rows], 'labels': sorted(set(y)), 'majority': majority(y)}
    if algorithm == 'stump':
        model['stump'] = fit_stump(x, y)
    elif algorithm.startswith('knn-'):
        k = int(algorithm.rsplit('-', 1)[1])
        if not 1 <= k <= len(rows):
            raise ValueError('k must fit training rows')
        model.update(k=k, x=x, y=y, weights='distance' if 'distance' in algorithm else 'uniform')
    elif algorithm != 'dummy':
        raise ValueError('Unknown candidate')
    return model

def vote_neighbors(neighbors, weights):
    if not neighbors or weights not in ('uniform', 'distance'):
        raise ValueError('Invalid neighbor vote')
    chosen = [row for row in neighbors if row[0] == 0] if weights == 'distance' and neighbors[0][0] == 0 else neighbors
    votes = {}
    for distance, identifier, label in chosen:
        weight = 1.0 if weights == 'uniform' or distance == 0 else 1.0 / distance
        votes[label] = votes.get(label, 0.0) + weight
    return min(votes, key=lambda label: (-votes[label], label))

def predict(model, rows):
    state = model['preprocessing']
    x = encode(rows, state['vocabularies']) if state['kind'] == 'onehot' else scaled([r['x'] for r in rows], state)
    output = []
    for query in x:
        if len(query) != (sum((len(v) + 1 for v in state['vocabularies'])) if state['kind'] == 'onehot' else len(state['means'])):
            raise ValueError('Query dimensions must match the frozen fit')
        if not all(map(math.isfinite, query)):
            raise ValueError('Query values must be finite')
        if model['algorithm'] == 'dummy':
            label = model['majority']
        elif model['algorithm'] == 'stump':
            rule = model['stump']
            label = rule['leaf'] if 'leaf' in rule else rule['left'] if query[rule['column']] <= rule['threshold'] else rule['right']
        else:
            neighbors = sorted(((math.sqrt(sum(((a - b) ** 2 for a, b in zip(query, row)))), identifier, label) for row, identifier, label in zip(model['x'], model['training_ids'], model['y'])))[:model['k']]
            label = vote_neighbors(neighbors, model['weights'])
        output.append(label)
    return output

def group_folds(rows, folds=4, seed=42):
    if not 2 <= folds <= 8:
        raise ValueError('Use 2..8 grouped folds')
    by_group = {}
    for row in rows:
        by_group.setdefault(row['group'], []).append(row)
    by_label = {}
    for group, members in sorted(by_group.items()):
        labels = {row['label'] for row in members}
        if len(labels) != 1:
            raise ValueError('This stratified teaching fixture has one class per source group')
        by_label.setdefault(next(iter(labels)), []).append(group)
    assigned = [set() for _ in range(folds)]
    rng = random.Random(seed)
    for label, groups in sorted(by_label.items()):
        if len(groups) < folds:
            raise ValueError('Each class needs at least one group per fold')
        rng.shuffle(groups)
        for i, group in enumerate(groups):
            assigned[i % folds].add(group)
    return [([r for r in rows if r['group'] not in groups], [r for r in rows if r['group'] in groups]) for groups in assigned]

def choose_candidate(scores, order, margin=0.02):
    if not scores or set(scores) != set(order) or len(order) != len(set(order)) or (not 0 <= margin <= 1):
        raise ValueError('Scores and declared preference order must agree')
    if any((not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values())):
        raise ValueError('Metric values must be finite and in [0,1]')
    best = max(scores.values())
    return next((name for name in order if scores[name] >= best - margin - 1e-12))

def partitions(rows):
    result = {name: [r for r in rows if r['split'] == name] for name in ('train', 'validation', 'test')}
    if sum(map(len, result.values())) != len(rows) or any((not part for part in result.values())):
        raise ValueError('Every row needs one of three nonempty partitions')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Row IDs must be unique')
    names = list(result)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if {r['group'] for r in result[a]} & {r['group'] for r in result[b]}:
                raise ValueError('Related source groups cannot cross partitions')
    return result

def error_slices(rows, predictions):
    raise NotImplementedError('Complete error_slices; see README.md')

def compare(rows, candidates=('dummy', 'stump', 'knn-3', 'knn-1'), representation='numeric', stress=False):
    parts = partitions(rows)
    train = parts['train']
    labels = sorted({r['label'] for r in train})
    cv = group_folds(train)
    cv_results = {}
    fold_evidence = []
    for i, (fit, validation) in enumerate(cv):
        fold_evidence.append({'fold': i, 'training_ids': [r['id'] for r in fit], 'validation_ids': [r['id'] for r in validation], 'training_groups': sorted({r['group'] for r in fit}), 'validation_groups': sorted({r['group'] for r in validation})})
    for candidate in candidates:
        scores = []
        for fit, validation in cv:
            model = fit_model(fit, candidate, representation)
            scores.append(classification_metrics([r['label'] for r in validation], predict(model, validation), labels)['macro_f1'])
        cv_results[candidate] = {'fold_macro_f1': scores, 'mean_macro_f1': sum(scores) / len(scores)}
    chosen = choose_candidate({name: row['mean_macro_f1'] for name, row in cv_results.items()}, list(candidates))
    fitted = {name: fit_model(train, name, representation) for name in candidates}
    validation = {name: classification_metrics([r['label'] for r in parts['validation']], predict(model, parts['validation']), labels) for name, model in fitted.items()}
    # All candidate and fitted-state decisions are frozen before test prediction.
    predictions = {name: predict(model, parts['test']) for name, model in fitted.items()}
    tests = {name: classification_metrics([r['label'] for r in parts['test']], predictions[name], labels) for name in candidates}
    result = {'primary_metric': 'macro_f1', 'secondary_metric': 'balanced_accuracy', 'selection_source': 'training_grouped_CV', 'preference_order': list(candidates), 'simplicity_margin': 0.02, 'chosen': chosen, 'cross_validation': cv_results, 'folds': fold_evidence, 'split_ids': {name: [r['id'] for r in part] for name, part in parts.items()}, 'validation': validation, 'test': tests, 'frozen_fit': fitted[chosen], 'test_predictions': predictions[chosen], 'test_labels': [r['label'] for r in parts['test']]}
    if representation == 'numeric':
        result['error_slices'] = error_slices(parts['test'], predictions[chosen])
        if stress:
            probes = [dict(row, x=[row['x'][0] + 1.0, *row['x'][1:]]) for row in parts['test']]
            result['fixed_feature_corruption_probe'] = classification_metrics(result['test_labels'], predict(fitted[chosen], probes), labels)
    else:
        probe = dict(parts['test'][0], categories=['unseen-prototype', *parts['test'][0]['categories'][1:]])
        result['unknown_category_probe'] = {'prediction': predict(fitted[chosen], [probe])[0], 'reserved_unknown_indicator': True}
    return result
