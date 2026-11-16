"""Reproduce the 32 authored, non-personal weather-pattern PNG fixtures."""
import csv
import hashlib
import math
from pathlib import Path
import random
import struct
import zlib
LABELS = ['cloudy', 'rainy', 'sunny', 'sunrise']

def png(pixels):

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 4294967295)
    raw = b''.join((b'\x00' + bytes((channel for pixel in pixels[y * 12:(y + 1) * 12] for channel in pixel)) for y in range(12)))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 12, 12, 8, 2, 0, 0, 0)) + chunk(b'IDAT', (b'\x78\x01' + b'\x01' + struct.pack('<HH', len(raw), 65535 - len(raw)) + raw + struct.pack('>I', zlib.adler32(raw) & 0xffffffff))) + chunk(b'IEND', b'')

def generate(target):
    target = Path(target)
    (target / 'images').mkdir(parents=True, exist_ok=True)
    records = []
    for label in LABELS:
        for family in range(4):
            rng = random.Random(1000 + LABELS.index(label) * 50 + family)
            shift = rng.choice([-1, 0, 1])
            brightness = rng.uniform(0.83, 1.17)
            for variant in range(2):
                pixels = []
                for y in range(12):
                    for x in range(12):
                        xx = 11 - x if variant else x
                        if label == 'cloudy':
                            color = (125, 150, 175)
                            if any((((xx - cx - shift) / 3) ** 2 + ((y - cy) / 2) ** 2 <= 1 for cx, cy in [(3, 5), (6, 4), (9, 5)])):
                                color = (210, 210, 210)
                        elif label == 'rainy':
                            color = (55, 75, 110)
                            if 2 <= y <= 5 and 1 <= xx <= 10:
                                color = (135, 140, 150)
                            if y >= 7 and (xx + y + shift) % 4 == 0:
                                color = (80, 170, 245)
                        elif label == 'sunny':
                            color = (65, 160, 230)
                            if (xx - 8 - shift) ** 2 + (y - 3) ** 2 <= 5:
                                color = (255, 230, 40)
                        else:
                            color = (230 - int(y * 6), 100 + int(y * 5), 65 + int(y * 4))
                            if (xx - 6 - shift) ** 2 + (y - 8) ** 2 <= 10:
                                color = (255, 215, 65)
                            if y >= 9:
                                color = (70, 65, 85)
                        pixels.append(tuple((max(0, min(255, round(value * brightness + variant * 2))) for value in color)))
                raw = png(pixels)
                name = f'{label}-g{family}-v{variant}.png'
                (target / 'images' / name).write_bytes(raw)
                records.append({'image': name, 'label': label, 'source_group': f'{label}-g{family}', 'split': 'train' if family < 2 else 'validation' if family == 2 else 'test', 'sha256': hashlib.sha256(raw).hexdigest()})
    with (target / 'metadata.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    return records
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    generate(args.destination)
