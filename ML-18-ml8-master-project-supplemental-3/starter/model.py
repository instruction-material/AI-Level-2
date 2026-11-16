"""Train-only polynomial least squares for a supplied low-stakes capstone."""
import math
import statistics

def polynomial_features(row, degree):
    raise NotImplementedError('Complete polynomial_features; see README.md')

def fit_scale(x):
    means = [statistics.mean(col) for col in zip(*x)]
    scales = [statistics.pstdev(col) or 1.0 for col in zip(*x)]
    return (means, scales)

def transform(x, state):
    means, scales = state
    if any((len(row) != len(means) for row in x)):
        raise ValueError('Feature width changed')
    return [[(v - m) / s for v, m, s in zip(row, means, scales)] for row in x]

def solve(a, b):
    n = len(b)
    matrix = [list(row) + [target] for row, target in zip(a, b)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            raise ValueError('Singular normal equations')
        matrix[column], matrix[pivot] = (matrix[pivot], matrix[column])
        value = matrix[column][column]
        matrix[column] = [v / value for v in matrix[column]]
        for row in range(n):
            if row == column:
                continue
            weight = matrix[row][column]
            matrix[row] = [a - weight * b for a, b in zip(matrix[row], matrix[column])]
    return [row[-1] for row in matrix]

def fit(x, y, degree):
    if not x or len(x) != len(y) or len(x) > 400:
        raise ValueError('Use aligned bounded training data')
    scale = fit_scale(x)
    features = [polynomial_features(row, degree) for row in transform(x, scale)]
    n = len(features[0])
    gram = [[sum((row[i] * row[j] for row in features)) + (1e-06 if i == j and i else 0) for j in range(n)] for i in range(n)]
    target = [sum((row[i] * value for row, value in zip(features, y))) for i in range(n)]
    return {'scale': scale, 'degree': degree, 'weights': solve(gram, target)}

def predict(model, x):
    return [sum((w * v for w, v in zip(model['weights'], polynomial_features(row, model['degree'])))) for row in transform(x, model['scale'])]

def metrics(y, p):
    if not y or len(y) != len(p):
        raise ValueError('Use aligned nonempty outputs')
    if not all((math.isfinite(v) for v in list(y) + list(p))):
        raise ValueError('Targets and predictions must be finite')
    residuals = [a - b for a, b in zip(y, p)]
    sse = sum((v * v for v in residuals))
    den = sum(((v - statistics.mean(y)) ** 2 for v in y))
    return {'mae': sum(map(abs, residuals)) / len(y), 'rmse': math.sqrt(sse / len(y)), 'r2': 1 - sse / den if den else None, 'residuals': residuals}
