#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_value(raw: str):
    raw = raw.strip().rstrip(',')
    if raw.startswith('np.float64(') and raw.endswith(')'):
        raw = raw[len('np.float64('):-1]
    elif raw.startswith('np.int64(') and raw.endswith(')'):
        raw = raw[len('np.int64('):-1]
    elif raw.startswith('np.int32(') and raw.endswith(')'):
        raw = raw[len('np.int32('):-1]
    elif raw.startswith('torch.Size('):
        return raw
    if raw in {'None', 'null'}:
        return None
    if raw in {'True', 'False'}:
        return raw == 'True'
    try:
        if raw.isdigit() or (raw.startswith('-') and raw[1:].isdigit()):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def extract_metrics(text: str) -> dict[str, object]:
    step_lines = [line for line in text.splitlines() if 'step:1 - ' in line or re.search(r'\bstep:\d+ - ', line)]
    if not step_lines:
        raise ValueError('No step metrics line found')
    line = step_lines[-1]
    if 'step:' in line:
        line = line[line.index('step:'):]
    metrics: dict[str, object] = {}
    for part in line.split(' - '):
        if ':' not in part:
            continue
        key, value = part.split(':', 1)
        metrics[key.strip()] = parse_value(value)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()

    text = Path(args.input).read_text(errors='replace')
    metrics = extract_metrics(text)
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + '\n')
    else:
        print(payload)


if __name__ == '__main__':
    main()
