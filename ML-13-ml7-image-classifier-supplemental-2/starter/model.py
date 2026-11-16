"""A bounded, explicit RGB CNN and dense comparison for authored 12x12 PNGs."""
import copy
import math
import random
import struct
import time
import zlib
SIZE = 12
LABELS = ['cloudy', 'rainy', 'sunny', 'sunrise']

def decode_png(raw):
    if not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw) > 4096:
        raise ValueError('Use the supplied bounded RGB PNG fixtures')
    offset = 8
    header = None
    compressed = []
    ended = False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError('Truncated PNG')
        length = struct.unpack('>I', raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        payload = raw[offset + 8:offset + 8 + length]
        if len(payload) != length or offset + 12 + length > len(raw):
            raise ValueError('Truncated PNG chunk')
        checksum = struct.unpack('>I', raw[offset + 8 + length:offset + 12 + length])[0]
        if zlib.crc32(kind + payload) & 4294967295 != checksum:
            raise ValueError('PNG CRC mismatch')
        if kind == b'IHDR':
            if header is not None or len(payload) != 13:
                raise ValueError('Invalid header')
            header = struct.unpack('>IIBBBBB', payload)
        elif kind == b'IDAT':
            if header is None:
                raise ValueError('PNG header must precede pixel data')
            compressed.append(payload)
        elif kind == b'IEND':
            if length or not compressed:
                raise ValueError('Invalid PNG end')
            ended = True
            offset += 12 + length
            break
        else:
            raise ValueError('This teaching decoder accepts only its generated PNG subset')
        offset += 12 + length
    if not ended or offset != len(raw) or header != (SIZE, SIZE, 8, 2, 0, 0, 0):
        raise ValueError('Expected a complete non-interlaced 12x12 RGB8 PNG')
    decoder = zlib.decompressobj()
    expected = SIZE * (1 + SIZE * 3)
    data = decoder.decompress(b''.join(compressed), expected + 1)
    if len(data) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError('Invalid or over-budget image data')
    pixels = []
    for y in range(SIZE):
        start = y * (1 + SIZE * 3)
        if data[start] != 0:
            raise ValueError("Only the fixture's unfiltered PNG scanlines are supported")
        pixels.extend((v / 255.0 for v in data[start + 1:start + 1 + SIZE * 3]))
    return pixels

def initialize(kind, seed):
    if kind not in ('cnn', 'dense'):
        raise ValueError('Unknown architecture')
    rng = random.Random(seed)
    width = 50 if kind == 'cnn' else SIZE * SIZE * 3
    return {'kind': kind, 'kernel': [[rng.uniform(-0.15, 0.15) for _ in range(27)] for _ in range(2)] if kind == 'cnn' else [], 'conv_bias': [0.0, 0.0] if kind == 'cnn' else [], 'output': [[rng.uniform(-0.15, 0.15) for _ in range(width)] for _ in LABELS], 'bias': [0.0] * len(LABELS)}

def forward(model, pixels):
    if len(pixels) != SIZE * SIZE * 3 or not all((math.isfinite(v) and 0 <= v <= 1 for v in pixels)):
        raise ValueError('Expected 432 rescaled RGB values')
    cache = {}
    if model['kind'] == 'dense':
        features = pixels
    else:
        patches = [[pixels[((y + dy) * SIZE + x + dx) * 3 + c] for dy in range(3) for dx in range(3) for c in range(3)] for y in range(10) for x in range(10)]
        activations = [[math.tanh(b + sum((w * v for w, v in zip(kernel, patch)))) for patch in patches] for kernel, b in zip(model['kernel'], model['conv_bias'])]
        features = []
        for values in activations:
            for y in range(5):
                for x in range(5):
                    i = 2 * y * 10 + 2 * x
                    features.append((values[i] + values[i + 1] + values[i + 10] + values[i + 11]) / 4)
        cache = {'patches': patches, 'activations': activations}
    logits = [b + sum((w * v for w, v in zip(weights, features))) for weights, b in zip(model['output'], model['bias'])]
    exps = [math.exp(value - max(logits)) for value in logits]
    total = sum(exps)
    return ([v / total for v in exps], dict(cache, features=features))

def zero_gradient(model):
    return {'kernel': [[0.0] * len(row) for row in model['kernel']], 'conv_bias': [0.0] * len(model['conv_bias']), 'output': [[0.0] * len(row) for row in model['output']], 'bias': [0.0] * len(model['bias'])}

def loss_gradient(model, pixels, label_index):
    if not 0 <= label_index < len(LABELS):
        raise ValueError('Unknown class')
    probabilities, cache = forward(model, pixels)
    g = zero_gradient(model)
    delta = probabilities[:]
    delta[label_index] -= 1.0
    g['bias'] = delta
    for c in range(4):
        g['output'][c] = [delta[c] * value for value in cache['features']]
    if model['kind'] == 'cnn':
        feature_delta = [sum((model['output'][c][j] * delta[c] for c in range(4))) for j in range(50)]
        for f in range(2):
            for i, value in enumerate(cache['activations'][f]):
                pool = f * 25 + i // 10 // 2 * 5 + i % 10 // 2
                d = feature_delta[pool] * (1 - value * value) / 4
                g['conv_bias'][f] += d
                for p, v in enumerate(cache['patches'][i]):
                    g['kernel'][f][p] += d * v
    return (-math.log(max(probabilities[label_index], 1e-300)), g)

def update(model, gradient, rate):
    for name in ('kernel', 'output'):
        for row, grow in zip(model[name], gradient[name]):
            for j in range(len(row)):
                row[j] -= rate * grow[j]
    for name in ('conv_bias', 'bias'):
        for j in range(len(model[name])):
            model[name][j] -= rate * gradient[name][j]

def augment_training(rows, rng):
    raise NotImplementedError('Complete augment_training; see README.md')

def image_metrics(truth, predictions):
    if not truth or len(truth) != len(predictions) or any((v not in LABELS for v in list(truth) + list(predictions))):
        raise ValueError('Use aligned nonempty weather labels')
    table = [[0] * 4 for _ in LABELS]
    for a, b in zip(truth, predictions):
        table[LABELS.index(a)][LABELS.index(b)] += 1
    recall = {}
    f1 = []
    for i, label in enumerate(LABELS):
        support = sum(table[i])
        predicted = sum((row[i] for row in table))
        recall[label] = table[i][i] / support if support else 0.0
        f1.append(2 * table[i][i] / (support + predicted) if support + predicted else 0.0)
    return {'accuracy': sum((a == b for a, b in zip(truth, predictions))) / len(truth), 'macro_f1': sum(f1) / 4, 'per_class_recall': recall, 'class_support': {label: sum(table[i]) for i, label in enumerate(LABELS)}, 'labels': LABELS[:], 'confusion_matrix': table}

def predict(model, rows):
    output = []
    for row in rows:
        probabilities, _ = forward(model, row['pixels'])
        output.append(LABELS[max(range(4), key=lambda i: probabilities[i])])
    return output

def validation_loss(model, rows):
    return sum((-math.log(max(forward(model, row['pixels'])[0][LABELS.index(row['label'])], 1e-300)) for row in rows)) / len(rows)

def train(train_rows, validation, kind, seed, epochs, batch_size, deadline, augment=True):
    if not 1 <= epochs <= 12 or not 1 <= batch_size <= 8 or (not 1 <= len(train_rows) <= 32):
        raise ValueError('Enforce the declared image/epoch/batch budgets')
    if any((r['split'] != 'train' for r in train_rows)) or any((r['split'] != 'validation' for r in validation)):
        raise ValueError('Training and validation evidence have separate roles')
    current = initialize(kind, seed)
    best = copy.deepcopy(current)
    best_loss = validation_loss(current, validation)
    history = []
    best_epoch = 0
    stale = 0
    updates = 0
    reason = 'epoch_limit'
    order_rng = random.Random(seed + 1000)
    augment_rng = random.Random(seed + 2000)
    for epoch in range(epochs):
        ordered = list(train_rows)
        order_rng.shuffle(ordered)
        total = 0.0
        seen = 0
        stopped = False
        for start in range(0, len(ordered), batch_size):
            if time.process_time() >= deadline:
                reason = 'cpu_budget'
                stopped = True
                break
            batch = ordered[start:start + batch_size]
            if augment:
                batch = augment_training(batch, augment_rng)
            accumulated = zero_gradient(current)
            for row in batch:
                loss, g = loss_gradient(current, row['pixels'], LABELS.index(row['label']))
                total += loss
                seen += 1
                for name in ('kernel', 'output'):
                    for dest, source in zip(accumulated[name], g[name]):
                        for j in range(len(dest)):
                            dest[j] += source[j]
                for name in ('conv_bias', 'bias'):
                    for j in range(len(accumulated[name])):
                        accumulated[name][j] += g[name][j]
            update(current, accumulated, 0.12 / len(batch))
            updates += 1
        if stopped:
            break
        score = validation_loss(current, validation)
        history.append({'epoch': epoch + 1, 'training_cross_entropy': total / seen, 'validation_cross_entropy': score})
        if score < best_loss - 1e-10:
            best_loss = score
            best = copy.deepcopy(current)
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                reason = 'validation_patience'
                break
    return {'model': best, 'best_epoch': best_epoch, 'best_validation_loss': best_loss, 'completed_epochs': len(history), 'update_steps': updates, 'stop_reason': reason, 'history': history}

def histogram(row):
    return [sum(row['pixels'][c::3]) / (SIZE * SIZE) for c in range(3)]

def fit_histogram(rows):
    return {label: [sum((histogram(row)[c] for row in rows if row['label'] == label)) / sum((row['label'] == label for row in rows)) for c in range(3)] for label in LABELS}

def histogram_predict(fitted, rows):
    return [min(LABELS, key=lambda label: (sum(((a - b) ** 2 for a, b in zip(histogram(row), fitted[label]))), label)) for row in rows]
